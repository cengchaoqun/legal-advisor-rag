"""Milvus 适配：稠密与稀疏向量混合检索（本地普通路径内部使用）。

契约要点：
- 集合名 legal_chunks_v1，模型切换必须建新集合，不能覆写原维度；
- 检索必须携带 snapshot_version_ids 与 scope/session 限制，禁止先全库召回再页面过滤；
- 正文权威副本在 MongoDB，这里只存向量与过滤字段。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Sequence

from pymilvus import DataType, MilvusClient

from ..settings import Settings

COLLECTION = "legal_chunks_v1"
DENSE_DIM = 1024


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float


def _quote_list(values: Sequence[str]) -> str:
    # 标识均为十六进制或 UUID 字符串，用 JSON 引号包裹即可安全嵌入过滤表达式
    return "[" + ",".join(json.dumps(value) for value in values) + "]"


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MilvusClient | None = None

    def _client_or_raise(self) -> MilvusClient:
        if self._client is None:
            self._client = MilvusClient(uri=self._settings.env.milvus_uri)
        return self._client

    async def ensure_collection(self) -> None:
        def _run() -> None:
            client = self._client_or_raise()
            if client.has_collection(COLLECTION):
                return
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("version_id", DataType.VARCHAR, max_length=64)
            schema.add_field("document_id", DataType.VARCHAR, max_length=36)
            schema.add_field("workspace_id", DataType.VARCHAR, max_length=32)
            schema.add_field("scope", DataType.VARCHAR, max_length=16)
            schema.add_field("session_id", DataType.VARCHAR, max_length=36)
            schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
            schema.add_field("scenario_tags", DataType.VARCHAR, max_length=256)
            schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DENSE_DIM)
            schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
            index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            client.create_collection(COLLECTION, schema=schema, index_params=index_params)

        await asyncio.to_thread(_run)

    async def upsert(
        self,
        rows: Sequence[dict[str, Any]],
        dense_vectors: Sequence[Sequence[float]],
        sparse_vectors: Sequence[dict[int, float]],
    ) -> None:
        if not rows:
            return
        await self.ensure_collection()

        def _run() -> None:
            client = self._client_or_raise()
            payload = []
            for row, dense, sparse in zip(rows, dense_vectors, sparse_vectors):
                if len(dense) != DENSE_DIM:
                    raise ValueError(f"稠密向量维度 {len(dense)} 与契约 {DENSE_DIM} 不一致")
                payload.append({**row, "dense": list(dense), "sparse": dict(sparse)})
            client.upsert(COLLECTION, payload)

        await asyncio.to_thread(_run)

    def _filter(self, version_ids: Sequence[str], scope: str, session_id: str | None) -> str:
        if not version_ids:
            # 无可见版本时必须返回空结果，不能退化成全库检索
            return 'chunk_id in [""]'
        parts = [f"version_id in {_quote_list(version_ids)}", f"scope == {json.dumps(scope)}"]
        if scope == "session":
            parts.append(f"session_id == {json.dumps(session_id or '-')}")
        return " and ".join(parts)

    async def search_dense(
        self, vector: Sequence[float], version_ids: Sequence[str], scope: str, session_id: str | None, limit: int
    ) -> list[VectorHit]:
        return await self._search("dense", list(vector), version_ids, scope, session_id, limit, {"metric_type": "COSINE"})

    async def search_sparse(
        self, vector: dict[int, float], version_ids: Sequence[str], scope: str, session_id: str | None, limit: int
    ) -> list[VectorHit]:
        return await self._search("sparse", dict(vector), version_ids, scope, session_id, limit, {"metric_type": "IP"})

    async def _search(
        self,
        field: str,
        vector: Any,
        version_ids: Sequence[str],
        scope: str,
        session_id: str | None,
        limit: int,
        metric: dict[str, Any],
    ) -> list[VectorHit]:
        expr = self._filter(version_ids, scope, session_id)
        if not version_ids:
            return []

        def _run() -> list[VectorHit]:
            client = self._client_or_raise()
            if not client.has_collection(COLLECTION):
                return []
            res = client.search(
                COLLECTION,
                data=[vector],
                anns_field=field,
                search_params={**metric, "params": {}},
                limit=limit,
                filter=expr,
                output_fields=["chunk_id"],
            )
            hits: list[VectorHit] = []
            for row in res[0] if res else []:
                entity = row.get("entity", {})
                chunk_id = entity.get("chunk_id") or row.get("id")
                if chunk_id:
                    hits.append(VectorHit(chunk_id=str(chunk_id), score=float(row.get("distance", 0.0))))
            return hits

        return await asyncio.to_thread(_run)

    async def delete_version(self, version_id: str) -> None:
        def _run() -> None:
            client = self._client_or_raise()
            if client.has_collection(COLLECTION):
                client.delete(COLLECTION, filter=f"version_id == {json.dumps(version_id)}")

        await asyncio.to_thread(_run)

    async def health(self) -> dict[str, Any]:
        def _run() -> dict[str, Any]:
            client = self._client_or_raise()
            collections = client.list_collections()
            return {"ok": True, "collection": COLLECTION, "present": COLLECTION in collections, "collections": len(collections)}

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160]}
