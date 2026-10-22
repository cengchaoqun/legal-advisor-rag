"""HTTP 错误封套与游标编码：错误响应统一复用 components.responses 形状。"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(RuntimeError):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


def error_body(request: Request, exc: AppError) -> dict[str, Any]:
    request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
    return {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
            "retryable": exc.retryable,
            "details": exc.details,
        }
    }


def error_response(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=error_body(request, exc))


def encode_cursor(values: list[Any]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_cursor(cursor: str | None) -> list[Any] | None:
    if not cursor:
        return None
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise AppError(400, "INVALID_CURSOR", "游标无效或与当前过滤条件不匹配", details={"cursor": cursor}) from exc
    if not isinstance(data, list) or not data:
        raise AppError(400, "INVALID_CURSOR", "游标格式不正确")
    return data


def datetime_cursor(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def parse_limit(limit: int, *, maximum: int = 100) -> int:
    if limit < 1 or limit > maximum:
        raise AppError(422, "VALIDATION_ERROR", f"limit 必须在 1..{maximum} 之间")
    return limit
