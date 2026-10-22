"""worker 主循环：MongoDB 原子领取任务 + 租约心跳 + 崩溃恢复。

- 查询优先：每轮先领查询，再领导入；
- 心跳每 20 秒延长租约；租约丢失后停止写入，不覆盖新持有者；
- 启动时把租约过期的 running 查询补记为 interrupted，不重新调用模型。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import uuid
from typing import Any

from ..domain.enums import (
    Availability,
    ImportStage,
    ImportStatus,
    PublicationState,
    QueryState,
    VectorState,
)
from ..domain.models import Error
from ..ingestion.pipeline import IngestionState, LeaseLost as IngestionLeaseLost, StageFailure, build_pipeline
from ..retrieval.query_pipeline import LeaseLost as QueryLeaseLost
from ..retrieval.query_pipeline import QueryPipeline
from ..runtime import Runtime


class Worker:
    def __init__(self, runtime: Runtime | None = None) -> None:
        self.runtime = runtime or Runtime()
        self.settings = self.runtime.settings
        self.config = self.settings.config
        self.worker_id = f"worker-{uuid.uuid4()}"
        self._stopping = asyncio.Event()
        self._pipeline = build_pipeline(
            self.settings,
            self.runtime.store,
            self.runtime.objects,
            self.runtime.vectors,
            self.runtime.graph,
            self.runtime.embedder,
        )
        self._query_pipeline = QueryPipeline(
            self.settings,
            self.runtime.store,
            self.runtime.vectors,
            self.runtime.graph,
            self.runtime.embedder,
            self.runtime.answer_model,
            self.runtime.aux_model,
            self.runtime.reranker,
            self.runtime.web,
            self.runtime.visibility,
        )

    async def _warm_models(self) -> None:
        """启动时预热嵌入模型。

        BGE-M3 首次加载需要数十秒，若不预热，第一个查询的向量检索会超过
        timeouts.ordinary_seconds 而被记为 timeout（真实发生的故障，不是理论风险）。
        """
        try:
            await self.runtime.embedder.encode(["预热"])
            print("嵌入模型已预热", flush=True)
        except Exception as exc:  # 预热失败不应阻断 worker，但需如实输出
            print(f"嵌入模型预热失败，首个查询可能触发检索路径超时：{exc}", flush=True)

    async def start(self) -> None:
        await self.runtime.startup()
        await self.runtime.store.recover_expired_queries()
        await self._warm_models()
        loops = [
            asyncio.create_task(self._query_loop(), name="query-loop"),
            asyncio.create_task(self._import_loop(), name="import-loop"),
            asyncio.create_task(self._maintenance_loop(), name="maintenance-loop"),
        ]
        await self._stopping.wait()
        for task in loops:
            task.cancel()
        for task in loops:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.runtime.shutdown()

    def stop(self) -> None:
        self._stopping.set()

    @contextlib.asynccontextmanager
    async def _heartbeat(self, kind: str, task_id: str):
        """独立心跳：租约到期前持续续期；主任务结束后取消。"""
        stop = asyncio.Event()

        async def beat() -> None:
            while not stop.is_set():
                try:
                    if kind == "query":
                        await self.runtime.store.extend_query_lease(
                            task_id, self.worker_id, self.config.runtime.lease_seconds
                        )
                    else:
                        await self.runtime.store.extend_task_lease(
                            task_id, self.worker_id, self.config.runtime.lease_seconds
                        )
                except Exception:
                    pass
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self.config.runtime.heartbeat_seconds)

        heartbeat = asyncio.create_task(beat())
        try:
            yield
        finally:
            stop.set()
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _query_loop(self) -> None:
        while not self._stopping.is_set():
            query = await self.runtime.store.claim_query(self.worker_id, self.config.runtime.lease_seconds)
            if query is None:
                await asyncio.sleep(0.5)
                continue
            await self._handle_query(query)

    async def _handle_query(self, query: dict[str, Any]) -> None:
        query_id = query["query_id"]
        async with self._heartbeat("query", query_id):
            try:
                await self._query_pipeline.run(query_id, self.worker_id)
            except QueryLeaseLost:
                return
            except Exception as exc:
                # 兜底：任何未处理异常都不能让查询停在 running
                await self._fail_query(query_id, "QUERY_FAILED", str(exc))

    async def _fail_query(self, query_id: str, code: str, message: str) -> None:
        doc = await self.runtime.store.get_query(query_id)
        if doc is None or doc.get("status") != QueryState.RUNNING.value:
            return
        seq = int(doc.get("last_event_seq", 0)) + 1
        from ..storage.mongo import utcnow

        await self.runtime.store.query_cas(
            query_id,
            lease_token=self.worker_id,
            revision=int(doc.get("revision", 0)),
            patch={
                "status": QueryState.FAILED.value,
                "finished_at": utcnow(),
                "warnings": list(doc.get("warnings") or []) + [code],
            },
            push_events=[
                {
                    "seq": seq,
                    "event": "failed",
                    "payload": {"code": code, "message": message, "status": QueryState.FAILED.value},
                    "created_at": utcnow(),
                }
            ],
        )

    async def _import_loop(self) -> None:
        while not self._stopping.is_set():
            task = await self.runtime.store.claim_task(self.worker_id, self.config.runtime.lease_seconds)
            if task is None:
                await asyncio.sleep(0.5)
                continue
            await self._handle_import(task)

    async def _handle_import(self, task: dict[str, Any]) -> None:
        task_id = task["task_id"]
        state: IngestionState = {
            "task_id": task_id,
            "document_id": task["document_id"],
            "version_id": task["version_id"],
            # 以任务文档中的实际租约令牌为准（测试/重启接管时都成立）
            "lease_token": task.get("lease_token") or self.worker_id,
            "stage_success": list(task.get("stage_success") or []),
            "artifacts": dict(task.get("stage_artifacts") or {}),
            "warnings": list(task.get("warnings") or []),
            "quality_flags": [],
            "vector_state": task.get("vector_state", VectorState.PENDING.value),
            "graph_state": task.get("graph_state", "pending"),
            "availability": task.get("availability", Availability.UNAVAILABLE.value),
        }
        async with self._heartbeat("task", task_id):
            try:
                await self._pipeline.ainvoke(state, config={"recursion_limit": 50})
            except IngestionLeaseLost:
                return
            except StageFailure as exc:
                await self._fail_task(task_id, exc)
            except Exception as exc:
                await self._fail_task(
                    task_id,
                    StageFailure("IMPORT_FAILED", f"导入失败：{exc}", retryable=True),
                )

    async def _fail_task(self, task_id: str, failure: StageFailure) -> None:
        task = await self.runtime.store.get_task(task_id)
        if task is None:
            return
        attempt = int(task.get("attempt", 1))
        max_attempts = self.config.ingestion.max_attempts
        retryable = bool(failure.retryable) and attempt < max_attempts
        error = Error(
            code=failure.code,
            message=failure.message,
            request_id=uuid.UUID(str(task.get("request_id") or uuid.uuid4())),
            retryable=retryable,
            details=failure.details,
        )
        status = (
            ImportStatus.FAILED.value
            if failure.code not in {"CANCELLED"}
            else ImportStatus.CANCELLED.value
        )
        warnings = list(task.get("warnings") or [])
        for warning in failure.details.get("warnings", []):
            if warning not in warnings:
                warnings.append(warning)
        await self.runtime.store.update_task_any(
            task_id,
            {
                "status": status,
                "error": error.model_dump(mode="json"),
                "retryable": retryable,
                "warnings": warnings,
                "lease_token": None,
                "lease_until": None,
            },
        )

    async def _maintenance_loop(self) -> None:
        """定期恢复失联查询、清理过期事件日志。"""
        while not self._stopping.is_set():
            await asyncio.sleep(30)
            try:
                await self.runtime.store.recover_expired_queries()
                await self._expire_event_logs()
            except Exception:
                continue

    async def _expire_event_logs(self) -> None:
        from ..storage.mongo import utcnow

        now = utcnow()
        await self.runtime.store.collection("queries").update_many(
            {"events_expired": {"$ne": True}, "events_expires_at": {"$lt": now}},
            {"$set": {"events_expired": True}, "$unset": {"event_log": ""}},
        )


async def main() -> None:
    worker = Worker()
    loop = asyncio.get_running_loop()
    for signal_name in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_name, worker.stop)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
