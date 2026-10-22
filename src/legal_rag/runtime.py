"""进程级依赖装配：API 与 worker 各自持有一套适配器实例。

启动阶段只建立连接与索引，不加载嵌入模型（由 worker 首次使用时加载）；
依赖缺失不阻塞进程启动，由 /health/ready 如实报告 unready/degraded。
"""

from __future__ import annotations

import asyncio
from typing import Any

from .adapters.embedding import BGEM3Embedder
from .adapters.llm import AnswerModelClient, AuxModelClient
from .adapters.reranker import Reranker
from .adapters.web_search import WebSearchClient
from .services.budget import BudgetService
from .services.idempotency import IdempotencyService
from .services.visibility import VisibilityService
from .settings import Settings, get_settings
from .storage.graph import GraphStore
from .storage.mongo import MongoStore
from .storage.objects import ObjectStore
from .storage.vector import VectorStore


class Runtime:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = MongoStore(self.settings)
        self.objects = ObjectStore(self.settings)
        self.vectors = VectorStore(self.settings)
        self.graph = GraphStore(self.settings)
        self.embedder = BGEM3Embedder(self.settings)
        self.answer_model = AnswerModelClient(self.settings)
        self.aux_model = AuxModelClient(self.settings)
        self.reranker = Reranker(self.settings)
        self.web = WebSearchClient(self.settings)
        self.visibility = VisibilityService(self.store)
        self.budget = BudgetService(self.store, self.settings)
        self.idempotency = IdempotencyService(self.store)
        self._startup_errors: dict[str, str] = {}

    async def startup(self, *, ensure_indexes: bool = True) -> None:
        try:
            await self.store.connect()
        except Exception as exc:
            self._startup_errors["mongo"] = str(exc)[:200]
            return
        if ensure_indexes:
            try:
                await self.store.ensure_indexes()
            except Exception as exc:
                self._startup_errors["mongo_indexes"] = str(exc)[:200]
        try:
            await self.objects.ensure_bucket()
        except Exception as exc:
            self._startup_errors["object_store"] = str(exc)[:200]
        try:
            await asyncio.wait_for(self.vectors.ensure_collection(), timeout=15)
        except Exception as exc:
            self._startup_errors["milvus"] = str(exc)[:200]
        try:
            await self.budget.ensure_account()
        except Exception as exc:
            self._startup_errors["budget"] = str(exc)[:200]

    async def shutdown(self) -> None:
        await self.answer_model.close()
        await self.aux_model.close()
        await self.graph.close()
        await self.store.close()

    async def collection_count(self, name: str, query: dict[str, Any] | None = None) -> int:
        """测试与维护脚本常用：按集合计数，避免直接依赖仓储内部结构。"""
        return await self.store.collection_count(name, query)

    async def health(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        results["mongo"] = "ok" if await self.store.health() else "unavailable"
        results["milvus"] = "ok" if (await self.vectors.health()).get("ok") else "unavailable"
        results["neo4j"] = "ok" if (await self.graph.health()).get("ok") else "unavailable"
        object_health = await self.objects.health()
        results["object_store"] = "ok" if object_health.get("ok") else "unavailable"
        results["answer_model"] = (
            "ok" if (await self.answer_model.health()).get("ok") else ("not_configured" if not self.answer_model.configured else "unavailable")
        )
        results["aux_model"] = "ok" if self.aux_model.configured else "not_configured"
        results["rerank"] = "ok" if self.reranker.configured else "not_configured"
        results["web"] = "ok" if self.web.configured else "not_configured"
        results["embedding"] = "ok" if (await self.embedder.health()).get("ok") else "unavailable"
        if results["mongo"] != "ok":
            status = "unready"
        elif any(results[name] == "unavailable" for name in ("milvus", "neo4j", "object_store")):
            status = "unready"
        elif any(value in {"not_configured", "unavailable"} for value in results.values()):
            status = "degraded"
        else:
            status = "ready"
        return {"status": status, "checks": results, "meta": self.settings.snapshot(), "startup_errors": self._startup_errors}
