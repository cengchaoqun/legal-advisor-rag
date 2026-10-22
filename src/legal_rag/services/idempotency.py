"""幂等：同一 request_id 相同输入复用结果，不同输入返回 409。

- 上传类操作的身份参数（文件摘要、scope、session、replace_document_id、标题等）
  全部参与 input_hash；任何一项变化都视为冲突，而不是“再传一次”。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from ..domain.ids import stable_hash
from ..storage.mongo import MongoStore, utcnow


class IdempotencyConflict(RuntimeError):
    code = "IDEMPOTENCY_CONFLICT"


@dataclass(slots=True)
class IdempotencyRecord:
    resource_id: str
    creation_response: dict[str, Any]


class IdempotencyService:
    def __init__(self, store: MongoStore) -> None:
        self._store = store

    def input_hash(self, payload: dict[str, Any]) -> str:
        return stable_hash(payload)

    async def begin(self, scope: str, request_id: uuid.UUID | str, payload: dict[str, Any]) -> IdempotencyRecord | None:
        """返回已存在记录（重放）；不存在返回 None 由调用方继续创建。"""
        digest = self.input_hash(payload)
        existing = await self._store.get_idempotency(scope, str(request_id))
        if existing is None:
            return None
        if existing.get("input_hash") != digest:
            raise IdempotencyConflict("同一 request_id 使用了不同的输入参数")
        response = existing.get("creation_response") or {}
        return IdempotencyRecord(resource_id=existing.get("resource_id", ""), creation_response=response)

    async def commit(
        self,
        scope: str,
        request_id: uuid.UUID | str,
        payload: dict[str, Any],
        *,
        resource_id: str,
        response: dict[str, Any],
    ) -> IdempotencyRecord:
        digest = self.input_hash(payload)
        inserted = await self._store.put_idempotency(
            {
                "scope": scope,
                "request_id": str(request_id),
                "input_hash": digest,
                "resource_id": resource_id,
                "creation_response": response,
                "created_at": utcnow(),
            }
        )
        if not inserted:
            # 并发提交：以先写入者为准，若输入不同则冲突
            existing = await self._store.get_idempotency(scope, str(request_id))
            if existing and existing.get("input_hash") == digest:
                return IdempotencyRecord(
                    resource_id=existing.get("resource_id", resource_id),
                    creation_response=existing.get("creation_response", response),
                )
            raise IdempotencyConflict("同一 request_id 已被不同输入占用")
        return IdempotencyRecord(resource_id=resource_id, creation_response=response)
