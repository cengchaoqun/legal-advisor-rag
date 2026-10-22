"""统一可见性：知识资料、会话附件、版本、原件下载共用同一套规则。

- 知识检索只使用 knowledge 范围内已发布且未撤销的版本；
- 会话附件只属于创建它的会话，越界统一 404，不泄露存在性；
- 明确 related_date 时按生效区间选择历史版本，日期未知时不被最新版覆盖。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from ..storage.mongo import MongoStore


class NotVisible(LookupError):
    """资源不存在或不可见；接口层统一转换为 404。"""


@dataclass(slots=True)
class Snapshot:
    version_ids: list[str]
    versions: list[dict[str, Any]]
    attachment_version_ids: list[str]

    @property
    def all_version_ids(self) -> list[str]:
        return list(dict.fromkeys(self.version_ids + self.attachment_version_ids))


def _effective_contains(metadata: dict[str, Any], relevant_date: date) -> bool | None:
    start = metadata.get("effective_from")
    end = metadata.get("effective_to")
    if start is None:
        return None
    start_date = date.fromisoformat(start) if isinstance(start, str) else start
    if start_date > relevant_date:
        return False
    if end is None:
        return True
    end_date = date.fromisoformat(end) if isinstance(end, str) else end
    return relevant_date < end_date


class VisibilityService:
    def __init__(self, store: MongoStore) -> None:
        self._store = store

    async def knowledge_versions(
        self,
        *,
        document_ids: Sequence[uuid.UUID] | None = None,
        relevant_date: date | None = None,
    ) -> list[dict[str, Any]]:
        docs = await self._store.list_documents(scope="knowledge", limit=100)
        allowed = {str(item) for item in document_ids} if document_ids else None
        versions: list[dict[str, Any]] = []
        for doc in docs:
            if allowed is not None and doc["document_id"] not in allowed:
                continue
            versions.extend(await self._select_versions_for_document(doc, relevant_date))
        return versions

    async def _select_versions_for_document(
        self, doc: dict[str, Any], relevant_date: date | None
    ) -> list[dict[str, Any]]:
        current_id = doc.get("current_version_id")
        current = await self._store.get_version(current_id) if current_id else None
        if current and current.get("publication_state") != "published":
            current = None
        if relevant_date is None or current is None:
            return [current] if current else []

        # 明确相关日期：优先取生效区间覆盖该日期的历史版本
        candidates = await self._store.list_versions(doc["document_id"], limit=100)
        matched: list[dict[str, Any]] = []
        for version in candidates:
            if version.get("publication_state") != "published":
                continue
            covered = _effective_contains(version.get("metadata") or {}, relevant_date)
            if covered:
                matched.append(version)
        if matched:
            # 同区间多版本时保留当前版本优先、其余按发布时间倒序
            matched.sort(key=lambda item: (item["version_id"] != current_id, item.get("published_at") or item["created_at"]), reverse=False)
            return matched
        return [current]

    async def attachment_versions(self, session_id: uuid.UUID) -> list[dict[str, Any]]:
        docs = await self._store.list_session_documents(session_id)
        versions: list[dict[str, Any]] = []
        for doc in docs:
            if doc.get("tombstoned"):
                continue
            version_id = doc.get("current_version_id")
            if not version_id:
                continue
            version = await self._store.get_version(version_id)
            if version and version.get("publication_state") == "published" and version.get("availability") != "unavailable":
                versions.append(version)
        return versions

    async def snapshot(
        self,
        *,
        session_id: uuid.UUID,
        document_ids: Sequence[uuid.UUID] | None = None,
        attachment_ids: Sequence[uuid.UUID] | None = None,
        relevant_date: date | None = None,
    ) -> Snapshot:
        versions = await self.knowledge_versions(document_ids=document_ids, relevant_date=relevant_date)
        attachments = await self.attachment_versions(session_id)
        if attachment_ids:
            # 附件 ID 必须属于当前会话，越界直接 404，不泄露其他会话存在性
            wanted = {str(item) for item in attachment_ids}
            attachments = [item for item in attachments if str(item["document_id"]) in wanted]
            found = {str(item["document_id"]) for item in attachments}
            if found != wanted:
                raise NotVisible("附件不存在或不属于当前会话")
        return Snapshot(
            version_ids=[item["version_id"] for item in versions],
            versions=versions,
            attachment_version_ids=[item["version_id"] for item in attachments],
        )

    async def require_document(self, document_id: uuid.UUID, *, session_id: uuid.UUID | None = None) -> dict[str, Any]:
        doc = await self._store.get_document(document_id)
        if doc is None:
            raise NotVisible("资料不存在")
        if doc.get("scope") == "session" and str(doc.get("session_id")) != str(session_id):
            raise NotVisible("附件不存在或不属于当前会话")
        return doc

    async def require_version_file(self, version_id: str, *, session_id: uuid.UUID | None = None) -> dict[str, Any]:
        version = await self._store.get_version(version_id)
        if version is None:
            raise NotVisible("版本不存在")
        doc = await self._store.get_document(version["document_id"])
        if doc is None:
            raise NotVisible("资料不存在")
        if doc.get("tombstoned"):
            raise NotVisible("资料已撤销")
        if doc.get("scope") == "session" and str(doc.get("session_id")) != str(session_id):
            raise NotVisible("附件不存在或不属于当前会话")
        return version

    async def filter_visible_versions(self, version_ids: Sequence[str], *, session_id: uuid.UUID | None) -> set[str]:
        """输出前复核：剔除已撤销/不可见的版本，防止撤销后继续引用。"""
        visible: set[str] = set()
        for version_id in set(version_ids):
            version = await self._store.get_version(version_id)
            if version is None or version.get("publication_state") != "published":
                continue
            doc = await self._store.get_document(version["document_id"])
            if doc is None or doc.get("tombstoned"):
                continue
            if doc.get("scope") == "session" and str(doc.get("session_id")) != str(session_id):
                continue
            visible.add(version_id)
        return visible
