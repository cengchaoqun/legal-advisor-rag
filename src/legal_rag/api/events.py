"""SSE 事件流：只重放已持久化事件，断线重连不重复文本、不重新调用模型。

- 事件 id 为 query_id:seq；Last-Event-ID 与 after_seq 同时出现必须一致；
- 游标规则：query_id 必须与 URL 一致，seq 不得超过当前 last_event_seq；
- 事件日志过期（7 天）返回 410，前端改为读取 GET /api/queries/{id} 快照；
- 建流之后的执行故障使用已持久化的 failed/cancelled 事件表达，不混入 JSON 错误包。
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Header, Request
from fastapi.responses import StreamingResponse

from ..domain.enums import TERMINAL_QUERY_STATES
from .errors import AppError

router = APIRouter(prefix="/api", tags=["queries"])


def _runtime(request: Request):
    return request.app.state.runtime


def _parse_last_event_id(value: str | None, query_id: str) -> int | None:
    if not value:
        return None
    if ":" not in value:
        raise AppError(400, "INVALID_CURSOR", "Last-Event-ID 形如 query_id:seq")
    raw_query, _, raw_seq = value.rpartition(":")
    if raw_query != query_id:
        raise AppError(400, "INVALID_CURSOR", "Last-Event-ID 的 query_id 与请求路径不一致")
    try:
        seq = int(raw_seq)
    except ValueError as exc:
        raise AppError(400, "INVALID_CURSOR", "Last-Event-ID 的 seq 必须是整数") from exc
    if seq < 0:
        raise AppError(400, "INVALID_CURSOR", "seq 不能为负")
    return seq


def _format(query_id: str, event: dict) -> str:
    payload = {
        "query_id": query_id,
        "seq": event["seq"],
        "event": event["event"],
        "payload": event.get("payload") or {},
    }
    return (
        f"id: {query_id}:{event['seq']}\n"
        f"event: {event['event']}\n"
        f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    )


@router.get("/queries/{query_id}/events")
async def stream_events(
    query_id: uuid.UUID,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    after_seq: int | None = None,
):
    runtime = _runtime(request)
    doc = await runtime.store.get_query(query_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "查询不存在")
    if doc.get("events_expired"):
        raise AppError(410, "EVENT_HISTORY_EXPIRED", "事件日志已过期，请读取查询快照")

    cursor = _parse_last_event_id(last_event_id, str(query_id))
    if after_seq is not None:
        if after_seq < 0:
            raise AppError(400, "INVALID_CURSOR", "after_seq 不能为负")
        if cursor is not None and cursor != after_seq:
            raise AppError(400, "INVALID_CURSOR", "Last-Event-ID 与 after_seq 不一致")
        cursor = after_seq
    cursor = cursor or 0
    if cursor > int(doc.get("last_event_seq", 0)):
        raise AppError(400, "INVALID_CURSOR", "游标超过当前 last_event_seq")

    async def generator():
        sent = cursor
        last_heartbeat = asyncio.get_running_loop().time()
        while True:
            if await request.is_disconnected():
                return
            current = await runtime.store.get_query(query_id)
            if current is None:
                return
            events = [item for item in (current.get("event_log") or []) if int(item.get("seq", 0)) > sent]
            for event in events:
                sent = int(event["seq"])
                yield _format(str(query_id), event)
            status = current.get("status")
            if status in {state.value for state in TERMINAL_QUERY_STATES} and sent >= int(current.get("last_event_seq", 0)):
                return
            now = asyncio.get_running_loop().time()
            if now - last_heartbeat >= runtime.settings.config.session.sse_heartbeat_seconds:
                last_heartbeat = now
                yield ": heartbeat\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
