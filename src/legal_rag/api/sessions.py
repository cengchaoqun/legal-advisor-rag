"""会话接口：创建/列表/软删除、历史消息、会话附件上传。"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..domain.enums import Role, Scope
from ..domain.models import Facts, Session
from ..storage.mongo import utcnow
from .documents import _create_document
from .errors import AppError, decode_cursor, datetime_cursor, encode_cursor, parse_limit
from .serializers import as_query

router = APIRouter(prefix="/api", tags=["sessions"])


def _runtime(request: Request):
    return request.app.state.runtime


@router.post("/sessions", status_code=201)
async def create_session(request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    body: dict[str, Any] = {}
    raw = await request.body()
    if raw:
        try:
            import json

            body = json.loads(raw)
        except Exception as exc:
            raise AppError(422, "VALIDATION_ERROR", "请求体不是合法 JSON") from exc
    if body and set(body) - {"title"}:
        raise AppError(422, "VALIDATION_ERROR", "存在未定义字段", details={"fields": sorted(set(body) - {"title"})})
    title = (body.get("title") or "新咨询").strip()[:80] or "新咨询"
    now = utcnow()
    session_id = uuid.uuid4()
    await runtime.store.insert_session(
        {
            "session_id": str(session_id),
            "title": title,
            "created_at": now,
            "updated_at": now,
            "deleted": False,
            "last_query_id": None,
        }
    )
    model = Session(
        session_id=session_id, title=title, created_at=now, updated_at=now, deleted=False
    )
    return model.model_dump(mode="json")


@router.get("/sessions")
async def list_sessions(request: Request, cursor: str | None = None, limit: int = 20) -> dict[str, Any]:
    runtime = _runtime(request)
    limit = parse_limit(limit)
    raw = decode_cursor(cursor)
    parsed = (datetime_cursor(raw[0]), str(raw[1])) if raw and len(raw) == 2 else None
    rows = await runtime.store.list_sessions(cursor=parsed, limit=limit + 1)
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [
        Session(
            session_id=uuid.UUID(row["session_id"]),
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted=bool(row.get("deleted")),
        ).model_dump(mode="json")
        for row in rows
    ]
    next_cursor = (
        encode_cursor([rows[-1]["updated_at"].isoformat(), rows[-1]["session_id"]]) if has_more and rows else None
    )
    return {"items": items, "next_cursor": next_cursor}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    session = await runtime.store.get_session(session_id)
    if session is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "会话不存在")
    await runtime.store.soft_delete_session(str(session_id))
    # 会话删除先关闭可见性，再标记未完成查询取消、附件撤销
    await runtime.store.mark_tasks_cancelled_for_session(str(session_id))
    for doc in await runtime.store.list_session_documents(session_id):
        await runtime.store.set_tombstoned(doc["document_id"], True)
    cursor = runtime.store.collection("queries").find(
        {"session_id": str(session_id), "status": {"$in": ["queued", "running"]}}
    )
    async for doc in cursor:
        await runtime.store.set_query_cancel_requested(doc["query_id"])
    return {"session_id": str(session_id), "deleted": True}


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: uuid.UUID, request: Request, cursor: str | None = None, limit: int = 20
) -> dict[str, Any]:
    runtime = _runtime(request)
    limit = parse_limit(limit)
    session = await runtime.store.get_session(session_id)
    if session is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "会话不存在")
    raw = decode_cursor(cursor)
    parsed = (datetime_cursor(raw[0]), str(raw[1])) if raw and len(raw) == 2 else None
    rows = await runtime.store.list_messages(session_id, cursor=parsed, limit=limit + 1)
    has_more = len(rows) > limit
    rows = rows[:limit]

    # 终态已提交但消息投影缺失时，从 queries 回补，避免完整轮次消失
    query_ids = [row["query_id"] for row in rows]
    present = {row["query_id"] for row in rows if row["role"] == Role.ASSISTANT.value}
    missing = [qid for qid in query_ids if qid not in present]
    for query_id in missing:
        query = await runtime.store.get_query(query_id)
        if query and query.get("status") in {"completed", "completed_with_warnings", "failed", "interrupted"}:
            await runtime.store.upsert_message(
                {
                    "message_id": str(uuid.uuid4()),
                    "query_id": query_id,
                    "session_id": str(session_id),
                    "role": Role.ASSISTANT.value,
                    "content": query.get("answer", ""),
                    "status": "partial" if query.get("status") == "interrupted" else "complete",
                    "facts": query.get("facts"),
                    "created_at": query.get("finished_at") or query.get("created_at"),
                }
            )
    if missing:
        rows = await runtime.store.list_messages(session_id, cursor=parsed, limit=limit + 1)
        has_more = len(rows) > limit
        rows = rows[:limit]

    items = []
    for row in rows:
        facts = Facts.model_validate(row["facts"])
        items.append(
            {
                "message_id": row["message_id"],
                "query_id": row["query_id"],
                "role": row["role"],
                "content": row["content"],
                "status": row["status"],
                "created_at": row["created_at"].isoformat().replace("+00:00", "Z"),
                "facts": facts.model_dump(mode="json"),
            }
        )
    next_cursor = (
        encode_cursor([rows[-1]["created_at"].isoformat(), rows[-1]["message_id"]]) if has_more and rows else None
    )
    return {"items": items, "next_cursor": next_cursor}


@router.post("/sessions/{session_id}/attachments", status_code=202)
async def upload_attachment(
    session_id: uuid.UUID,
    request: Request,
    file: UploadFile = File(...),
    request_id: uuid.UUID = Form(...),
) -> dict[str, Any]:
    runtime = _runtime(request)
    session = await runtime.store.get_session(session_id)
    if session is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "会话不存在")
    result = await _create_document(
        request,
        file=file,
        request_id=request_id,
        title=file.filename,
        doc_type="unknown",
        source_url=None,
        replace_document_id=None,
        scope=Scope.SESSION,
        session_id=session_id,
    )
    return {
        "attachment_id": result["document_id"],
        "task_id": result["task_id"],
        "status": "queued" if not result.get("duplicate") else "succeeded",
        "duplicate": bool(result.get("duplicate")),
    }
