"""MongoDB 仓储：元数据、切片正文、任务、查询事件与预算的唯一权威来源。

所有写操作遵循调用方传入的 CAS 条件（lease_token / revision），
旧租约持有者无法覆盖新持有者，这是任务恢复与单文档事件日志一致性的基础。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from ..domain.enums import TERMINAL_QUERY_STATES, ImportStage, ImportStatus, QueryState
from ..settings import Settings

WORKSPACE = "default"
SCHEMA_REVISION = "0.4"

# 内部字段：不得出现在 HTTP 响应中
DOCUMENT_INTERNAL = {"_id", "workspace_id", "schema_revision"}
VERSION_INTERNAL = {
    "_id",
    "workspace_id",
    "scope",
    "session_id",
    "stage_artifacts",
    "doc_type",
    "chunk_count",
    "schema_revision",
}
QUERY_INTERNAL = {
    "_id",
    "request_id",
    "input_hash",
    "request",
    "snapshot_version_ids",
    "cancel_requested",
    "lease_token",
    "lease_until",
    "revision",
    "event_log",
    "events_expires_at",
    "events_expired",
    "workspace_id",
    "snapshot",
    "generation_raw_output",
    "schema_revision",
}
TASK_INTERNAL = {
    "_id",
    "request_id",
    "input_hash",
    "stage_success",
    "lease_token",
    "lease_until",
    "heartbeat_at",
    "cancel_requested",
    "created_at",
    "scope",
    "session_id",
    "replace_document_id",
    "sha256",
    "object_key",
    "filename",
    "mime",
    "size_bytes",
    "doc_type",
    "title",
    "source_url",
    "workspace_id",
    "pipeline_revision",
    "schema_revision",
    "graph_failed_stage",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def strip_internal(doc: dict[str, Any] | None, internal: set[str]) -> dict[str, Any] | None:
    if doc is None:
        return None
    return {key: value for key, value in doc.items() if key not in internal}


class MongoStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: AsyncMongoClient | None = None
        self._db: Any = None

    # ---------- 生命周期 ----------
    async def connect(self) -> None:
        if self._client is not None:
            return
        self._client = AsyncMongoClient(
            self._settings.env.mongo_uri, serverSelectionTimeoutMS=2500, tz_aware=True
        )
        self._db = self._client[self._settings.env.mongo_db]
        await self._client.admin.command("ping")

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._db = None

    @property
    def db(self) -> Any:
        if self._db is None:
            raise RuntimeError("MongoDB 尚未连接")
        return self._db

    def collection(self, name: str) -> Any:
        return self.db[name]

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    async def ensure_indexes(self) -> None:
        db = self.db
        await db.documents.create_index([("document_id", ASCENDING)], unique=True)
        await db.documents.create_index(
            [("scope", ASCENDING), ("session_id", ASCENDING), ("tombstoned", ASCENDING), ("updated_at", DESCENDING)]
        )
        await db.document_versions.create_index([("version_id", ASCENDING)], unique=True)
        await db.document_versions.create_index(
            [("document_id", ASCENDING), ("sha256", ASCENDING), ("pipeline_revision", ASCENDING)],
            unique=True,
        )
        await db.content_registry.create_index(
            [
                ("scope", ASCENDING),
                ("session_id", ASCENDING),
                ("sha256", ASCENDING),
                ("pipeline_revision", ASCENDING),
            ],
            unique=True,
        )
        await db.chunks.create_index([("chunk_id", ASCENDING)], unique=True)
        await db.chunks.create_index([("version_id", ASCENDING), ("ordinal", ASCENDING)], unique=True)
        await db.import_tasks.create_index([("task_id", ASCENDING)], unique=True)
        await db.import_tasks.create_index([("scope", ASCENDING), ("request_id", ASCENDING)], unique=True)
        await db.import_tasks.create_index(
            [("status", ASCENDING), ("lease_until", ASCENDING), ("created_at", ASCENDING)]
        )
        await db.sessions.create_index([("session_id", ASCENDING)], unique=True)
        await db.sessions.create_index([("deleted", ASCENDING), ("updated_at", DESCENDING)])
        await db.messages.create_index([("query_id", ASCENDING), ("role", ASCENDING)], unique=True)
        await db.messages.create_index(
            [("session_id", ASCENDING), ("created_at", ASCENDING), ("message_id", ASCENDING)]
        )
        await db.queries.create_index([("query_id", ASCENDING)], unique=True)
        await db.queries.create_index([("session_id", ASCENDING), ("request_id", ASCENDING)], unique=True)
        await db.queries.create_index([("status", ASCENDING), ("created_at", ASCENDING)])
        await db.graph_unresolved.create_index(
            [("source_chunk_id", ASCENDING), ("citation_hash", ASCENDING)], unique=True
        )
        await db.cost_ledger.create_index([("entry_id", ASCENDING)], unique=True)
        await db.cost_ledger.create_index([("external_receipt_id", ASCENDING)], unique=True, sparse=True)
        await db.idempotency.create_index([("scope", ASCENDING), ("request_id", ASCENDING)], unique=True)

    # ---------- 资料与版本 ----------
    async def insert_document(self, doc: dict[str, Any]) -> None:
        payload = dict(doc)
        payload.update({"_id": payload["document_id"], "workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("documents").insert_one(payload)

    async def get_document(self, document_id: uuid.UUID | str) -> dict[str, Any] | None:
        return await self.collection("documents").find_one({"document_id": str(document_id)})

    async def find_document_by_identity(
        self, scope: str, session_id: uuid.UUID | None, sha256: str, pipeline_revision: str
    ) -> dict[str, Any] | None:
        """显式替换：只在目标资料内查找同摘要版本，避免把其他资料身份当结果。"""
        version = await self.collection("document_versions").find_one(
            {"sha256": sha256, "pipeline_revision": pipeline_revision}
        )
        if not version:
            return None
        return await self.get_document(version["document_id"])

    async def list_documents(
        self,
        *,
        scope: str = "knowledge",
        include_tombstoned: bool = False,
        doc_type: str | None = None,
        availability: str | None = None,
        q: str | None = None,
        cursor: tuple[datetime, str] | None = None,
        limit: int = 20,
        session_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        """列表过滤在数据库侧完成（含当前版本可用性），避免逐行请求详情。

        可用性通过 $lookup 当前版本计算，因此筛选语义是“当前发布版本”，不是任一历史版本。
        """
        match: dict[str, Any] = {"scope": scope, "workspace_id": WORKSPACE}
        if session_id is not None:
            match["session_id"] = str(session_id)
        if not include_tombstoned:
            match["tombstoned"] = False
        if doc_type:
            match["doc_type"] = doc_type
        if cursor:
            created_at, document_id = cursor
            match["$or"] = [
                {"created_at": {"$lt": created_at}},
                {"created_at": created_at, "document_id": {"$lt": document_id}},
            ]
        pipeline: list[dict[str, Any]] = [
            {"$match": match},
            {
                "$lookup": {
                    "from": "document_versions",
                    "localField": "current_version_id",
                    "foreignField": "version_id",
                    "as": "current_version",
                }
            },
            {"$unwind": {"path": "$current_version", "preserveNullAndEmptyArrays": True}},
        ]
        if availability:
            pipeline.append(
                {
                    "$match": {
                        "$expr": {
                            "$eq": [
                                {
                                    "$cond": [
                                        {"$eq": ["$tombstoned", True]},
                                        "unavailable",
                                        {"$ifNull": ["$current_version.availability", "unavailable"]},
                                    ]
                                },
                                availability,
                            ]
                        }
                    }
                }
            )
        if q:
            # 字面子串匹配：对用户输入转义正则元字符，避免被当作表达式执行
            escaped = q.replace("\\", "\\\\")
            for ch in [".", "^", "$", "*", "+", "?", "(", ")", "[", "]", "{", "}", "|"]:
                escaped = escaped.replace(ch, "\\" + ch)
            pattern = {"$regex": escaped, "$options": "i"}
            pipeline.append(
                {
                    "$match": {
                        "$or": [
                            {"title": pattern},
                            {"current_version.metadata.document_number": pattern},
                            {"current_version.metadata.case_number": pattern},
                        ]
                    }
                }
            )
        pipeline += [
            {"$sort": {"created_at": DESCENDING, "document_id": DESCENDING}},
            {"$limit": limit},
        ]
        rows: list[dict[str, Any]] = []
        cursor = await self.collection("documents").aggregate(pipeline)
        async for row in cursor:
            rows.append(row)
        return rows

    async def list_session_documents(self, session_id: uuid.UUID) -> list[dict[str, Any]]:
        cursor = self.collection("documents").find(
            {"scope": "session", "session_id": str(session_id), "workspace_id": WORKSPACE}
        )
        return [doc async for doc in cursor]

    async def set_tombstoned(self, document_id: str, tombstoned: bool) -> dict[str, Any] | None:
        return await self.collection("documents").find_one_and_update(
            {"document_id": document_id},
            {"$set": {"tombstoned": tombstoned, "updated_at": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )

    async def update_document_current_version(
        self, document_id: str, version_id: str, *, touch: bool = True
    ) -> None:
        update: dict[str, Any] = {"current_version_id": version_id}
        if touch:
            update["updated_at"] = utcnow()
        await self.collection("documents").update_one({"document_id": document_id}, {"$set": update})

    async def insert_version(self, version: dict[str, Any]) -> None:
        payload = dict(version)
        payload.update({"_id": payload["version_id"], "workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("document_versions").insert_one(payload)

    async def get_version(self, version_id: str) -> dict[str, Any] | None:
        return await self.collection("document_versions").find_one({"version_id": version_id})

    async def update_version(self, version_id: str, patch: dict[str, Any]) -> None:
        await self.collection("document_versions").update_one({"version_id": version_id}, {"$set": patch})

    async def list_versions(
        self, document_id: str, *, cursor: tuple[datetime, str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"document_id": document_id}
        if cursor:
            created_at, version_id = cursor
            query["$or"] = [
                {"created_at": {"$lt": created_at}},
                {"created_at": created_at, "version_id": {"$lt": version_id}},
            ]
        cursor_obj = (
            self.collection("document_versions")
            .find(query)
            .sort([("created_at", DESCENDING), ("version_id", DESCENDING)])
            .limit(limit)
        )
        return [doc async for doc in cursor_obj]

    async def published_versions(self, *, scope: str = "knowledge", session_id: str | None = None) -> list[dict[str, Any]]:
        """已发布且未撤销的可见版本白名单（查询快照与图/向量过滤共用）。"""
        query: dict[str, Any] = {"publication_state": "published", "workspace_id": WORKSPACE}
        if scope == "knowledge":
            query["scope"] = "knowledge"
        else:
            query["scope"] = "session"
            query["session_id"] = session_id
        cursor = self.collection("document_versions").find(query)
        return [doc async for doc in cursor]

    # ---------- 内容登记与切片 ----------
    async def register_content(
        self,
        *,
        scope: str,
        session_id: uuid.UUID | None,
        sha256: str,
        pipeline_revision: str,
        document_id: str,
        version_id: str,
        task_id: str,
    ) -> dict[str, Any] | None:
        """并发默认上传的身份登记：唯一键冲突时返回已存在登记。

        调用方先读后写都会冲突，因此以 DuplicateKeyError 作为“别人已登记”的信号，
        并按已分配 ID 恢复，不重复创建资料。
        """
        now = utcnow()
        doc = {
            "_id": f"{scope}:{session_id or '-'}:{sha256}:{pipeline_revision}",
            "scope": scope,
            "session_id": str(session_id) if session_id else None,
            "sha256": sha256,
            "pipeline_revision": pipeline_revision,
            "document_id": document_id,
            "version_id": version_id,
            "task_id": task_id,
            "creation_state": "registered",
            "created_at": now,
            "schema_revision": SCHEMA_REVISION,
        }
        try:
            await self.collection("content_registry").insert_one(doc)
            return None
        except DuplicateKeyError:
            return await self.collection("content_registry").find_one({"_id": doc["_id"]})

    async def find_content_registration(
        self, *, scope: str, session_id: uuid.UUID | None, sha256: str, pipeline_revision: str
    ) -> dict[str, Any] | None:
        return await self.collection("content_registry").find_one(
            {
                "scope": scope,
                "session_id": str(session_id) if session_id else None,
                "sha256": sha256,
                "pipeline_revision": pipeline_revision,
            }
        )

    async def upsert_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        for chunk in chunks:
            payload = dict(chunk)
            payload.update({"workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
            payload.setdefault("_id", payload["chunk_id"])
            await self.collection("chunks").replace_one(
                {"chunk_id": payload["chunk_id"]}, payload, upsert=True
            )

    async def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return await self.collection("chunks").find_one({"chunk_id": chunk_id})

    async def get_chunks(self, chunk_ids: Iterable[str]) -> list[dict[str, Any]]:
        ids = list(dict.fromkeys(chunk_ids))
        if not ids:
            return []
        cursor = self.collection("chunks").find({"chunk_id": {"$in": ids}})
        return [doc async for doc in cursor]

    async def count_chunks(self, version_ids: Sequence[str] | None = None) -> int:
        query: dict[str, Any] = {}
        if version_ids is not None:
            query["version_id"] = {"$in": list(version_ids)}
        return await self.collection("chunks").count_documents(query)

    async def collection_count(self, name: str, query: dict[str, Any] | None = None) -> int:
        return await self.collection(name).count_documents(query or {})

    async def count_documents_knowledge(self) -> int:
        return await self.collection("documents").count_documents(
            {"scope": "knowledge", "tombstoned": False}
        )

    # ---------- 导入任务 ----------
    async def create_task(self, task: dict[str, Any]) -> None:
        payload = dict(task)
        payload.update({"_id": payload["task_id"], "workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("import_tasks").insert_one(payload)

    async def get_task(self, task_id: uuid.UUID | str) -> dict[str, Any] | None:
        return await self.collection("import_tasks").find_one({"task_id": str(task_id)})

    async def find_task_by_request(self, scope: str, request_id: uuid.UUID | str) -> dict[str, Any] | None:
        return await self.collection("import_tasks").find_one(
            {"scope": scope, "request_id": str(request_id)}
        )

    async def latest_task_for_document(self, document_id: str) -> dict[str, Any] | None:
        cursor = (
            self.collection("import_tasks")
            .find({"document_id": document_id})
            .sort([("created_at", DESCENDING), ("task_id", DESCENDING)])
            .limit(1)
        )
        docs = [doc async for doc in cursor]
        return docs[0] if docs else None

    async def latest_tasks_for_documents(self, document_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """批量取每个资料的最新任务（按 created_at、task_id 倒序），避免列表页面 N+1 查询。"""
        if not document_ids:
            return {}
        pipeline = [
            {"$match": {"document_id": {"$in": list(document_ids)}}},
            {"$sort": {"created_at": DESCENDING, "task_id": DESCENDING}},
            {"$group": {"_id": "$document_id", "doc": {"$first": "$$ROOT"}}},
        ]
        result: dict[str, dict[str, Any]] = {}
        cursor = await self.collection("import_tasks").aggregate(pipeline)
        async for row in cursor:
            result[row["_id"]] = row["doc"]
        return result

    async def claim_task(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        """原子领取：只领 queued，或租约已过期且仍在 running 的任务（崩溃恢复）。"""
        now = utcnow()
        lease_until = now + timedelta(seconds=lease_seconds)
        return await self.collection("import_tasks").find_one_and_update(
            {
                "$or": [
                    {"status": ImportStatus.QUEUED.value},
                    {"status": ImportStatus.RUNNING.value, "lease_until": {"$lt": now}},
                ]
            },
            {
                "$set": {
                    "status": ImportStatus.RUNNING.value,
                    "lease_token": worker_id,
                    "lease_until": lease_until,
                    "heartbeat_at": now,
                    "updated_at": now,
                }
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    async def extend_task_lease(self, task_id: str, lease_token: str, lease_seconds: int) -> bool:
        now = utcnow()
        result = await self.collection("import_tasks").update_one(
            {"task_id": task_id, "lease_token": lease_token, "status": ImportStatus.RUNNING.value},
            {"$set": {"lease_until": now + timedelta(seconds=lease_seconds), "heartbeat_at": now}},
        )
        return result.modified_count == 1

    async def update_task_cas(
        self, task_id: str, lease_token: str, patch: dict[str, Any], *, unset: dict[str, Any] | None = None
    ) -> bool:
        update: dict[str, Any] = {"$set": {**patch, "updated_at": utcnow()}}
        if unset:
            update["$unset"] = unset
        result = await self.collection("import_tasks").update_one(
            {"task_id": task_id, "lease_token": lease_token}, update
        )
        return result.modified_count == 1

    async def update_task_any(self, task_id: str, patch: dict[str, Any]) -> None:
        await self.collection("import_tasks").update_one(
            {"task_id": task_id}, {"$set": {**patch, "updated_at": utcnow()}}
        )

    async def requeue_task(self, task_id: str, attempt: int) -> bool:
        """重试：failed → queued，清空租约；只允许可重试任务。"""
        result = await self.collection("import_tasks").update_one(
            {
                "task_id": task_id,
                "status": {"$in": [ImportStatus.FAILED.value, ImportStatus.SUCCEEDED_WITH_WARNINGS.value]},
                "retryable": True,
            },
            {
                "$set": {
                    "status": ImportStatus.QUEUED.value,
                    "attempt": attempt,
                    "lease_token": None,
                    "lease_until": None,
                    "cancel_requested": False,
                    "updated_at": utcnow(),
                }
            },
        )
        return result.modified_count == 1

    # ---------- 会话与消息 ----------
    async def insert_session(self, session: dict[str, Any]) -> None:
        payload = dict(session)
        payload.update({"_id": payload["session_id"], "workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("sessions").insert_one(payload)

    async def get_session(self, session_id: uuid.UUID | str, *, include_deleted: bool = False) -> dict[str, Any] | None:
        query: dict[str, Any] = {"session_id": str(session_id)}
        if not include_deleted:
            query["deleted"] = False
        return await self.collection("sessions").find_one(query)

    async def list_sessions(
        self, *, cursor: tuple[datetime, str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"deleted": False}
        if cursor:
            updated_at, session_id = cursor
            query["$or"] = [
                {"updated_at": {"$lt": updated_at}},
                {"updated_at": updated_at, "session_id": {"$lt": session_id}},
            ]
        cursor_obj = (
            self.collection("sessions")
            .find(query)
            .sort([("updated_at", DESCENDING), ("session_id", DESCENDING)])
            .limit(limit)
        )
        return [doc async for doc in cursor_obj]

    async def touch_session(self, session_id: str, last_query_id: str | None = None) -> None:
        patch: dict[str, Any] = {"updated_at": utcnow()}
        if last_query_id:
            patch["last_query_id"] = last_query_id
        await self.collection("sessions").update_one({"session_id": session_id}, {"$set": patch})

    async def soft_delete_session(self, session_id: str) -> bool:
        result = await self.collection("sessions").update_one(
            {"session_id": session_id, "deleted": False},
            {"$set": {"deleted": True, "updated_at": utcnow()}},
        )
        return result.modified_count == 1

    async def upsert_message(self, message: dict[str, Any]) -> None:
        payload = dict(message)
        payload.update({"workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("messages").replace_one(
            {"query_id": payload["query_id"], "role": payload["role"]}, payload, upsert=True
        )

    async def list_messages(
        self, session_id: uuid.UUID | str, *, cursor: tuple[datetime, str] | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"session_id": str(session_id)}
        if cursor:
            created_at, message_id = cursor
            query["$or"] = [
                {"created_at": {"$gt": created_at}},
                {"created_at": created_at, "message_id": {"$gt": message_id}},
            ]
        cursor_obj = (
            self.collection("messages")
            .find(query)
            .sort([("created_at", ASCENDING), ("message_id", ASCENDING)])
            .limit(limit)
        )
        return [doc async for doc in cursor_obj]

    async def messages_for_queries(self, query_ids: Sequence[str]) -> list[dict[str, Any]]:
        if not query_ids:
            return []
        cursor = self.collection("messages").find({"query_id": {"$in": list(query_ids)}})
        return [doc async for doc in cursor]

    # ---------- 查询 ----------
    async def insert_query(self, query: dict[str, Any]) -> None:
        payload = dict(query)
        payload.update({"_id": payload["query_id"], "workspace_id": WORKSPACE, "schema_revision": SCHEMA_REVISION})
        await self.collection("queries").insert_one(payload)

    async def get_query(self, query_id: uuid.UUID | str) -> dict[str, Any] | None:
        return await self.collection("queries").find_one({"query_id": str(query_id)})

    async def find_query_by_request(self, session_id: str, request_id: str) -> dict[str, Any] | None:
        return await self.collection("queries").find_one(
            {"session_id": session_id, "request_id": request_id}
        )

    async def claim_query(self, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        now = utcnow()
        return await self.collection("queries").find_one_and_update(
            {
                "$or": [
                    {"status": QueryState.QUEUED.value},
                    {"status": QueryState.RUNNING.value, "lease_until": {"$lt": now}},
                ]
            },
            {
                "$set": {
                    "status": QueryState.RUNNING.value,
                    "lease_token": worker_id,
                    "lease_until": now + timedelta(seconds=lease_seconds),
                    "updated_at": now,
                }
            },
            sort=[("created_at", ASCENDING)],
            return_document=ReturnDocument.AFTER,
        )

    async def extend_query_lease(self, query_id: str, lease_token: str, lease_seconds: int) -> bool:
        result = await self.collection("queries").update_one(
            {"query_id": query_id, "lease_token": lease_token, "status": QueryState.RUNNING.value},
            {"$set": {"lease_until": utcnow() + timedelta(seconds=lease_seconds)}},
        )
        return result.modified_count == 1

    async def query_cas(
        self,
        query_id: str,
        *,
        lease_token: str,
        revision: int,
        patch: dict[str, Any],
        push_events: Sequence[dict[str, Any]] = (),
        require_running: bool = True,
        unset: Sequence[str] = (),
    ) -> bool:
        """单文档 CAS：答案、证据、事件、revision 在同一原子更新中提交。

        过滤条件包含 lease_token 与 revision，保证并发取消/完成只有一个终止结果。
        """
        now = utcnow()
        # last_event_seq 只由 $inc 维护，避免与 $set 同字段冲突
        set_patch = {k: v for k, v in {**patch, "revision": revision + 1, "updated_at": now}.items() if k != "last_event_seq"}
        update: dict[str, Any] = {"$set": set_patch, "$inc": {"last_event_seq": len(push_events)}}
        if push_events:
            update["$push"] = {"event_log": {"$each": list(push_events)}}
        if unset:
            update["$unset"] = {key: "" for key in unset}
        filters: dict[str, Any] = {
            "query_id": query_id,
            "lease_token": lease_token,
            "revision": revision,
        }
        if require_running:
            filters["status"] = QueryState.RUNNING.value
        result = await self.collection("queries").update_one(filters, update)
        return result.modified_count == 1

    async def append_query_events(
        self, query_id: str, lease_token: str, revision: int, events: Sequence[dict[str, Any]]
    ) -> bool:
        now = utcnow()
        result = await self.collection("queries").update_one(
            {"query_id": query_id, "lease_token": lease_token, "revision": revision},
            {
                "$push": {"event_log": {"$each": list(events)}},
                "$inc": {"last_event_seq": len(events)},
                "$set": {"revision": revision + 1, "updated_at": now},
            },
        )
        return result.modified_count == 1

    async def set_query_cancel_requested(self, query_id: str) -> dict[str, Any] | None:
        return await self.collection("queries").find_one_and_update(
            {"query_id": query_id},
            {"$set": {"cancel_requested": True, "updated_at": utcnow()}},
            return_document=ReturnDocument.AFTER,
        )

    async def recover_expired_queries(self) -> list[str]:
        """租约过期的 running 查询：补记 interrupted 终止事件，不重新调用模型。"""
        now = utcnow()
        recovered: list[str] = []
        cursor = self.collection("queries").find(
            {"status": QueryState.RUNNING.value, "lease_until": {"$lt": now}},
            {"query_id": 1, "last_event_seq": 1, "revision": 1, "answer": 1},
        )
        async for doc in cursor:
            seq = int(doc.get("last_event_seq", 0)) + 1
            event = {
                "seq": seq,
                "event": "failed",
                "payload": {
                    "code": "QUERY_INTERRUPTED",
                    "message": "worker 中断，已保留已提交的部分答案",
                    "status": QueryState.INTERRUPTED.value,
                },
                "created_at": now,
            }
            result = await self.collection("queries").update_one(
                {
                    "query_id": doc["query_id"],
                    "status": QueryState.RUNNING.value,
                    "lease_until": {"$lt": now},
                },
                {
                    "$set": {
                        "status": QueryState.INTERRUPTED.value,
                        "finished_at": now,
                        "updated_at": now,
                        "lease_token": None,
                        "lease_until": None,
                        "events_expires_at": now + timedelta(days=self._settings.config.session.event_retention_days),
                    },
                    "$push": {"event_log": event},
                    "$inc": {"last_event_seq": 1, "revision": 1},
                },
            )
            if result.modified_count:
                recovered.append(doc["query_id"])
        return recovered

    async def mark_tasks_cancelled_for_session(self, session_id: str) -> int:
        now = utcnow()
        result = await self.collection("import_tasks").update_many(
            {
                "session_id": session_id,
                "status": {"$in": [ImportStatus.QUEUED.value, ImportStatus.RUNNING.value]},
            },
            {"$set": {"status": ImportStatus.CANCELLED.value, "cancel_requested": True, "updated_at": now}},
        )
        return result.modified_count

    # ---------- 幂等 ----------
    async def get_idempotency(self, scope: str, request_id: str) -> dict[str, Any] | None:
        return await self.collection("idempotency").find_one({"scope": scope, "request_id": request_id})

    async def put_idempotency(self, doc: dict[str, Any]) -> bool:
        payload = dict(doc)
        payload.setdefault("_id", f"{payload['scope']}:{payload['request_id']}")
        try:
            await self.collection("idempotency").insert_one(payload)
            return True
        except DuplicateKeyError:
            return False

    # ---------- 图未解析引用 ----------
    async def upsert_unresolved(self, doc: dict[str, Any]) -> None:
        await self.collection("graph_unresolved").replace_one(
            {"source_chunk_id": doc["source_chunk_id"], "citation_hash": doc["citation_hash"]},
            doc,
            upsert=True,
        )

    async def list_unresolved(self, limit: int = 50) -> list[dict[str, Any]]:
        cursor = self.collection("graph_unresolved").find({}).sort("source_chunk_id", 1).limit(limit)
        return [doc async for doc in cursor]

    # ---------- 成本 ----------
    async def get_budget_account(self) -> dict[str, Any] | None:
        return await self.collection("budget_account").find_one({"_id": "project"})

    async def insert_cost_entry(self, entry: dict[str, Any]) -> None:
        await self.collection("cost_ledger").insert_one(dict(entry))

    async def list_cost_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection("cost_ledger").find({}).sort("created_at", DESCENDING).limit(limit)
        return [doc async for doc in cursor]

    async def health(self) -> bool:
        try:
            await self.db.command("ping")
            return True
        except PyMongoError:
            return False
