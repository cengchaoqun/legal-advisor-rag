"""对外序列化：显式字段投影，HTTP 不暴露内部租约、事件日志、对象键以外的恢复字段。"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from ..domain.enums import Availability, DocType, PublicationState, VectorState, GraphState
from ..domain.models import (
    Document,
    DocumentListItem,
    QueryStatus,
    TaskStatus,
    Version,
)

DOCUMENT_FIELDS = (
    "document_id",
    "title",
    "doc_type",
    "scope",
    "session_id",
    "current_version_id",
    "tombstoned",
    "created_at",
    "updated_at",
)

VERSION_FIELDS = (
    "version_id",
    "document_id",
    "sha256",
    "source_url",
    "object_key",
    "metadata",
    "publication_state",
    "availability",
    "vector_state",
    "graph_state",
    "quality_flags",
    "created_at",
    "pipeline_revision",
    "published_at",
)

TASK_FIELDS = (
    "task_id",
    "document_id",
    "version_id",
    "status",
    "stage",
    "availability",
    "vector_state",
    "graph_state",
    "warnings",
    "retryable",
    "attempt",
    "error",
    "updated_at",
)

QUERY_FIELDS = (
    "query_id",
    "session_id",
    "status",
    "behavior",
    "answer",
    "citation_validation",
    "warnings",
    "evidence",
    "last_event_seq",
    "created_at",
    "finished_at",
    "generation_model_id",
    "generation_artifact_sha256",
    "facts",
    "path_status",
    "reasoning_code",
)


def project(doc: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: doc.get(field) for field in fields}


def as_document(doc: dict[str, Any]) -> dict[str, Any]:
    return project(doc, DOCUMENT_FIELDS)


def as_version(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    return project(doc, VERSION_FIELDS) if doc else None


def as_task(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    return project(doc, TASK_FIELDS) if doc else None


def as_query(doc: dict[str, Any]) -> QueryStatus:
    return QueryStatus.model_validate(project(doc, QUERY_FIELDS))


def as_list_item(
    document: dict[str, Any], version: dict[str, Any] | None, task: dict[str, Any] | None
) -> DocumentListItem:
    if document.get("tombstoned") or version is None:
        availability = Availability.UNAVAILABLE
    else:
        availability = Availability(version.get("availability", "unavailable"))
    return DocumentListItem(
        document=Document.model_validate(as_document(document)),
        version=Version.model_validate(as_version(version)) if version else None,
        availability=availability,
        latest_task=TaskStatus.model_validate(as_task(task)) if task else None,
    )


def derived_availability(document: dict[str, Any], version: dict[str, Any] | None) -> str:
    if document.get("tombstoned") or version is None:
        return Availability.UNAVAILABLE.value
    return str(version.get("availability", "unavailable"))


def new_uuid() -> uuid.UUID:
    return uuid.uuid4()
