"""知识关联接口：中文实体搜索 + 有界关系路径 + 支持原文。

闭环要求：nodes 覆盖全部 path.node_ids，support_chunks 覆盖全部 support_chunk_ids，
页面据此直接展示中文标签与原文，不借用咨询 query 的 E 编号。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from ..domain.enums import EntityType, PathStatusValue
from ..domain.ids import path_id as make_path_id
from ..domain.models import Chunk, GraphEntity, Location, Path
from .errors import AppError, decode_cursor, encode_cursor, parse_limit

router = APIRouter(prefix="/api/graph", tags=["graph"])

CHUNK_FIELDS = (
    "chunk_id",
    "document_id",
    "version_id",
    "ordinal",
    "content",
    "location",
    "scenario_tags",
    "embedding_revision",
    "quality_flags",
)


def _runtime(request: Request):
    return request.app.state.runtime


async def _visible_version_ids(runtime) -> list[str]:
    versions = await runtime.store.published_versions(scope="knowledge")
    visible: list[str] = []
    for version in versions:
        doc = await runtime.store.get_document(version["document_id"])
        if doc and not doc.get("tombstoned"):
            visible.append(version["version_id"])
    return visible


@router.get("/entities")
async def search_entities(
    request: Request, q: str | None = None, cursor: str | None = None, limit: int = 20
) -> dict[str, Any]:
    runtime = _runtime(request)
    limit = parse_limit(limit)
    if q is not None:
        q = q.strip()
        if not q:
            raise AppError(422, "VALIDATION_ERROR", "q 不能为空白字符串", details={"field": "q"})
        if len(q) > 100:
            raise AppError(422, "VALIDATION_ERROR", "q 最长 100 字符", details={"field": "q"})
    raw = decode_cursor(cursor)
    parsed = (str(raw[0]), str(raw[1])) if raw and len(raw) == 2 else None
    visible = await _visible_version_ids(runtime)
    try:
        rows = await runtime.graph.entities(q, visible, cursor=parsed, limit=limit + 1)
    except Exception as exc:
        raise AppError(503, "DEPENDENCY_UNAVAILABLE", f"图服务不可用：{exc}") from exc
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = []
    for row in rows:
        if row.get("entity_type") not in {item.value for item in EntityType}:
            continue
        items.append(
            GraphEntity(
                entity_id=row["entity_id"],
                label=row.get("label") or row["entity_id"],
                entity_type=EntityType(row["entity_type"]),
                version_ids=[vid for vid in (row.get("version_ids") or []) if vid in set(visible)],
            ).model_dump(mode="json")
        )
    next_cursor = encode_cursor([rows[-1]["label"], rows[-1]["entity_id"]]) if has_more and rows else None
    return {"items": items, "next_cursor": next_cursor}


@router.get("/relations")
async def relations(request: Request, entity_id: str, max_hops: int = 2) -> dict[str, Any]:
    runtime = _runtime(request)
    if not 1 <= max_hops <= 2:
        raise AppError(422, "VALIDATION_ERROR", "max_hops 取值 1..2", details={"field": "max_hops"})
    visible = await _visible_version_ids(runtime)
    try:
        entity = await runtime.graph.entity(entity_id, visible)
    except Exception as exc:
        raise AppError(503, "DEPENDENCY_UNAVAILABLE", f"图服务不可用：{exc}") from exc
    if entity is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "实体不存在或当前不可见")

    try:
        neighborhood = await runtime.graph.neighborhood(
            entity_id,
            visible,
            max_hops=max_hops,
            max_paths=runtime.settings.config.retrieval.graph_max_paths,
        )
    except Exception as exc:
        raise AppError(503, "DEPENDENCY_UNAVAILABLE", f"图服务不可用：{exc}") from exc

    chunks = await runtime.store.get_chunks(neighborhood.support_chunk_ids)
    visible_set = set(visible)
    chunk_models = []
    available_chunk_ids = set()
    for chunk in chunks:
        if chunk["version_id"] not in visible_set:
            continue
        model = Chunk.model_validate({field: chunk.get(field) for field in CHUNK_FIELDS})
        chunk_models.append(model)
        available_chunk_ids.add(model.chunk_id)

    paths = []
    node_map = {node["entity_id"]: node for node in neighborhood.nodes}
    for item in neighborhood.paths:
        support = [cid for cid in item.support_chunk_ids if cid in available_chunk_ids]
        if not support:
            continue
        if any(node_id not in node_map for node_id in item.node_ids):
            continue
        versions = [vid for vid in item.version_ids if vid in visible_set]
        paths.append(
            Path(
                path_id=make_path_id(item.node_ids, item.relation_types),
                node_ids=item.node_ids,
                relation_types=item.relation_types,
                support_chunk_ids=support,
                status=PathStatusValue.SOURCE_SUPPORTED,
                version_ids=versions,
            )
        )

    used_node_ids = {node_id for path in paths for node_id in path.node_ids}
    nodes = [
        GraphEntity(
            entity_id=node["entity_id"],
            label=node["label"],
            entity_type=EntityType(node["entity_type"])
            if node["entity_type"] in {item.value for item in EntityType}
            else EntityType.PROVISION,
            version_ids=[vid for vid in node.get("version_ids", []) if vid in visible_set],
        )
        for node in neighborhood.nodes
        if node["entity_id"] in used_node_ids
    ]
    used_chunk_ids = {cid for path in paths for cid in path.support_chunk_ids}
    support_chunks = [model for model in chunk_models if model.chunk_id in used_chunk_ids]

    warnings: list[str] = []
    if neighborhood.truncated:
        warnings.append("GRAPH_RESULT_TRUNCATED")
    return {
        "paths": [item.model_dump(mode="json") for item in paths],
        "warnings": warnings,
        "nodes": [item.model_dump(mode="json") for item in nodes],
        "support_chunks": [item.model_dump(mode="json") for item in support_chunks],
    }
