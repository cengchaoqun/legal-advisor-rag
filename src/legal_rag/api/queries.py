"""咨询接口：提交问题、查询快照、取消、证据详情、原件下载。"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..domain.enums import (
    Availability,
    Behavior,
    CitationValidation,
    PathState,
    QueryState,
    ReasoningCode,
    Role,
    VectorState,
)
from ..domain.models import PathStatus, QueryRequest
from ..retrieval.policy import extract_facts
from ..storage.mongo import utcnow
from .errors import AppError
from .serializers import as_query

router = APIRouter(prefix="/api", tags=["queries"])


def _runtime(request: Request):
    return request.app.state.runtime


@router.post("/sessions/{session_id}/queries", status_code=202)
async def create_query(session_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    session = await runtime.store.get_session(session_id)
    if session is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "会话不存在")
    try:
        body = await request.json()
    except Exception as exc:
        raise AppError(422, "VALIDATION_ERROR", "请求体不是合法 JSON") from exc
    try:
        payload = QueryRequest.model_validate(body)
    except Exception as exc:
        raise AppError(422, "VALIDATION_ERROR", "请求字段校验失败", details={"error": str(exc)[:200]}) from exc

    existing = await runtime.store.find_query_by_request(str(session_id), str(payload.request_id))
    if existing is not None:
        return {
            "query_id": existing["query_id"],
            "status": existing["status"],
            "events_url": f"/api/queries/{existing['query_id']}/events",
        }

    # 知识资料范围校验：显式指定的资料必须可见
    for document_id in payload.document_ids:
        doc = await runtime.store.get_document(document_id)
        if doc is None or doc.get("tombstoned") or doc.get("scope") != "knowledge":
            raise AppError(404, "RESOURCE_NOT_FOUND", "指定的资料不存在或不可见", details={"document_id": str(document_id)})

    # 附件必须在当前会话、已发布且向量就绪，否则 409
    for attachment_id in payload.attachment_ids:
        doc = await runtime.store.get_document(attachment_id)
        if doc is None or doc.get("scope") != "session" or doc.get("session_id") != str(session_id) or doc.get("tombstoned"):
            raise AppError(404, "RESOURCE_NOT_FOUND", "附件不存在或不属于当前会话")
        version = await runtime.store.get_version(doc.get("current_version_id") or "")
        if (
            version is None
            or version.get("publication_state") != "published"
            or version.get("vector_state") != VectorState.READY.value
            or version.get("availability") == Availability.UNAVAILABLE.value
        ):
            raise AppError(409, "ATTACHMENT_NOT_READY", "附件尚未完成入库，暂不可用于提问")

    facts = extract_facts(
        payload.question,
        scenario=None,
        location=payload.location,
        relevant_date=payload.relevant_date,
    )
    now = utcnow()
    query_id = uuid.uuid4()
    await runtime.store.insert_query(
        {
            "query_id": str(query_id),
            "session_id": str(session_id),
            "request_id": str(payload.request_id),
            "input_hash": payload.question,
            "request": payload.model_dump(mode="json"),
            "snapshot_version_ids": [],
            "facts": facts.model_dump(mode="json"),
            "path_status": PathStatus.all_pending().model_dump(mode="json"),
            "cancel_requested": False,
            "lease_token": None,
            "lease_until": None,
            "revision": 0,
            "event_log": [],
            "events_expires_at": None,
            "events_expired": False,
            "status": QueryState.QUEUED.value,
            "behavior": None,
            "answer": "",
            "citation_validation": CitationValidation.PENDING.value,
            "warnings": [],
            "evidence": [],
            "last_event_seq": 0,
            "created_at": now,
            "finished_at": None,
            "generation_model_id": None,
            "generation_artifact_sha256": None,
            "reasoning_code": None,
        }
    )
    # 用户消息在查询创建时唯一写入；finalize 只写 assistant，重放 SSE 不重复写历史
    await runtime.store.upsert_message(
        {
            "message_id": str(uuid.uuid4()),
            "query_id": str(query_id),
            "session_id": str(session_id),
            "role": Role.USER.value,
            "content": payload.question,
            "status": "complete",
            "facts": facts.model_dump(mode="json"),
            "created_at": now,
        }
    )
    await runtime.store.touch_session(str(session_id), str(query_id))
    return {"query_id": str(query_id), "status": QueryState.QUEUED.value, "events_url": f"/api/queries/{query_id}/events"}


@router.get("/queries/{query_id}")
async def get_query(query_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    doc = await runtime.store.get_query(query_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "查询不存在")
    return as_query(doc).model_dump(mode="json")


@router.post("/queries/{query_id}/cancel")
async def cancel_query(query_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    doc = await runtime.store.get_query(query_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "查询不存在")
    if doc.get("status") in {state.value for state in {QueryState.COMPLETED, QueryState.COMPLETED_WITH_WARNINGS, QueryState.FAILED, QueryState.INTERRUPTED, QueryState.CANCELLED}}:
        return {"query_id": doc["query_id"], "cancel_requested": False, "status": doc["status"]}
    updated = await runtime.store.set_query_cancel_requested(doc["query_id"])
    if updated is None:
        raise AppError(409, "INVALID_STATE", "查询状态已变化")
    return {
        "query_id": updated["query_id"],
        "cancel_requested": True,
        "status": updated["status"],
    }


@router.get("/queries/{query_id}/evidence/{evidence_id}")
async def get_evidence(query_id: uuid.UUID, evidence_id: str, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    if not evidence_id.startswith("E") or not evidence_id[1:].isdigit():
        raise AppError(422, "VALIDATION_ERROR", "evidence_id 形如 E1")
    doc = await runtime.store.get_query(query_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "查询不存在")
    evidence = next((item for item in (doc.get("evidence") or []) if item.get("evidence_id") == evidence_id), None)
    if evidence is None:
        raise AppError(404, "EVIDENCE_NOT_FOUND", "本轮查询中不存在该证据")
    # 撤销资料不继续公开原文：证据快照仍可审计，但原文入口关闭
    version_id = evidence.get("version_id")
    if version_id:
        version = await runtime.store.get_version(version_id)
        document = await runtime.store.get_document(version["document_id"]) if version else None
        if document and document.get("tombstoned"):
            raise AppError(410, "SOURCE_WITHDRAWN", "资料已撤销，不再提供原文", details={"evidence_id": evidence_id})
    return evidence


@router.get("/versions/{version_id}/file")
async def download_version(version_id: str, request: Request) -> StreamingResponse:
    runtime = _runtime(request)
    version = await runtime.store.get_version(version_id)
    if version is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "版本不存在")
    document = await runtime.store.get_document(version["document_id"])
    if document is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "资料不存在")
    if document.get("tombstoned"):
        raise AppError(410, "SOURCE_WITHDRAWN", "资料已撤销，不再提供原件")
    if document.get("scope") == "session":
        # 附件只允许其所属会话访问；首期无登录，因此以资料归属会话作为边界
        session_hint = request.query_params.get("session_id")
        if not session_hint or session_hint != document.get("session_id"):
            raise AppError(404, "RESOURCE_NOT_FOUND", "附件不存在或不属于当前会话")
    import io

    data = await runtime.objects.get_bytes(version["object_key"])
    filename = f"{document.get('title', 'document')}.{('pdf' if version['object_key'].endswith('.pdf') else 'md')}"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/pdf" if filename.endswith(".pdf") else "text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
