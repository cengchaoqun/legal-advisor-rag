"""资料接口：上传（自动入库）、列表（DocumentListItem）、详情、撤销、版本历史。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile

from ..adapters.parser import detect_mime
from ..domain.enums import Availability, DocType, ImportStage, ImportStatus, PublicationState, Scope, VectorState
from ..domain.ids import content_sha256, version_id as make_version_id
from ..domain.models import LegalMetadata, Version
from ..services.idempotency import IdempotencyConflict
from ..settings import ConfigError
from ..storage.mongo import utcnow
from .errors import AppError, decode_cursor, datetime_cursor, encode_cursor, parse_limit
from .serializers import as_list_item, as_task, as_version, derived_availability

router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_SUFFIXES = {"pdf", "md", "markdown"}
MAX_BYTES = 20 * 1024 * 1024


def _runtime(request: Request):
    return request.app.state.runtime


def _require_embedding_revision(runtime) -> str:
    try:
        return runtime.settings.require_embedding_revision()
    except ConfigError as exc:
        raise AppError(503, "PRICING_NOT_CONFIGURED", str(exc)) from exc


async def _build_items(runtime, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """列表批量关联当前版本与最新任务，禁止逐行 HTTP 详情请求。"""
    document_ids = [row["document_id"] for row in rows]
    tasks = await runtime.store.latest_tasks_for_documents(document_ids)
    items = []
    for row in rows:
        version = row.pop("current_version", None)
        if version is None and row.get("current_version_id"):
            version = await runtime.store.get_version(row["current_version_id"])
        task = tasks.get(row["document_id"])
        item = as_list_item(row, version, task)
        items.append(item.model_dump(mode="json"))
    return items


@router.get("/documents")
async def list_documents(
    request: Request,
    cursor: str | None = None,
    limit: int = 20,
    doc_type: str | None = None,
    availability: str | None = None,
    q: str | None = None,
) -> dict[str, Any]:
    runtime = _runtime(request)
    limit = parse_limit(limit)
    if doc_type and doc_type not in {item.value for item in DocType}:
        raise AppError(422, "VALIDATION_ERROR", "doc_type 取值不合法", details={"field": "doc_type"})
    if availability and availability not in {item.value for item in Availability}:
        raise AppError(422, "VALIDATION_ERROR", "availability 取值不合法", details={"field": "availability"})
    if q is not None:
        q = q.strip()
        if not q:
            raise AppError(422, "VALIDATION_ERROR", "q 不能为空白字符串", details={"field": "q"})

    raw_cursor = decode_cursor(cursor)
    parsed_cursor = None
    if raw_cursor:
        if len(raw_cursor) != 2:
            raise AppError(400, "INVALID_CURSOR", "游标与当前排序不匹配")
        parsed_cursor = (datetime_cursor(raw_cursor[0]), str(raw_cursor[1]))

    rows = await runtime.store.list_documents(
        scope="knowledge",
        doc_type=doc_type,
        availability=availability,
        q=q,
        cursor=parsed_cursor,
        limit=limit + 1,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = await _build_items(runtime, rows)
    next_cursor = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = encode_cursor([last["created_at"].isoformat(), last["document_id"]])
    return {"items": items, "next_cursor": next_cursor}


@router.post("/documents", status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    request_id: uuid.UUID = Form(...),
    title: str | None = Form(None),
    doc_type: str = Form("unknown"),
    source_url: str | None = Form(None),
    replace_document_id: uuid.UUID | None = Form(None),
) -> dict[str, Any]:
    return await _create_document(
        request,
        file=file,
        request_id=request_id,
        title=title,
        doc_type=doc_type,
        source_url=source_url,
        replace_document_id=replace_document_id,
        scope=Scope.KNOWLEDGE,
        session_id=None,
    )


async def _create_document(
    request: Request,
    *,
    file: UploadFile,
    request_id: uuid.UUID,
    title: str | None,
    doc_type: str,
    source_url: str | None,
    replace_document_id: uuid.UUID | None,
    scope: Scope,
    session_id: uuid.UUID | None,
) -> dict[str, Any]:
    runtime = _runtime(request)
    filename = file.filename or "未命名"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_SUFFIXES:
        raise AppError(400, "INVALID_FILE_TYPE", "仅支持文本型 PDF 与 Markdown")
    if doc_type not in {item.value for item in DocType}:
        raise AppError(422, "VALIDATION_ERROR", "doc_type 取值不合法", details={"field": "doc_type"})

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise AppError(413, "FILE_TOO_LARGE", f"文件超过 {MAX_BYTES // (1024 * 1024)} MiB")
    if not data:
        raise AppError(400, "INVALID_FILE_TYPE", "文件为空")

    sha256 = content_sha256(data)
    revision = _require_embedding_revision(runtime)
    from ..services.pipeline_revision import pipeline_revision

    fingerprint = pipeline_revision(runtime.settings, revision)
    resolved_title = (title or filename)[:200]

    idem_payload = {
        "scope": scope.value,
        "session_id": str(session_id) if session_id else None,
        "sha256": sha256,
        "pipeline_revision": fingerprint,
        "replace_document_id": str(replace_document_id) if replace_document_id else None,
        "title": resolved_title,
        "doc_type": doc_type,
        "source_url": source_url,
    }
    try:
        existing = await runtime.idempotency.begin(f"{scope.value}:upload", request_id, idem_payload)
    except IdempotencyConflict as exc:
        raise AppError(409, "IDEMPOTENCY_CONFLICT", str(exc)) from exc
    if existing is not None:
        # 同一 request_id 的重放属于重复提交：复用资源但标记 duplicate
        return {**existing.creation_response, "duplicate": True}

    if replace_document_id is not None:
        target = await runtime.store.get_document(replace_document_id)
        if target is None or target.get("tombstoned") or target.get("scope") != scope.value:
            raise AppError(404, "RESOURCE_NOT_FOUND", "替换目标不存在或不可见")
        if session_id and target.get("session_id") != str(session_id):
            raise AppError(404, "RESOURCE_NOT_FOUND", "替换目标不存在或不可见")
        document_id = str(target["document_id"])
        document = target
        duplicate_version = await runtime.store.collection("document_versions").find_one(
            {"document_id": document_id, "sha256": sha256, "pipeline_revision": fingerprint}
        )
        if duplicate_version is not None:
            task = await runtime.store.collection("import_tasks").find_one(
                {"document_id": document_id, "version_id": duplicate_version["version_id"]}
            )
            response = {
                "document_id": document_id,
                "version_id": duplicate_version["version_id"],
                "task_id": (task or {}).get("task_id", str(uuid.uuid4())),
                "duplicate": True,
            }
            await runtime.idempotency.commit(
                f"{scope.value}:upload", request_id, idem_payload, resource_id=document_id, response=response
            )
            return response
    else:
        registration = await runtime.store.find_content_registration(
            scope=scope.value, session_id=session_id, sha256=sha256, pipeline_revision=fingerprint
        )
        if registration is not None:
            task = await runtime.store.get_task(registration["task_id"])
            response = {
                "document_id": registration["document_id"],
                "version_id": registration["version_id"],
                "task_id": registration["task_id"],
                "duplicate": True,
            }
            await runtime.idempotency.commit(
                f"{scope.value}:upload",
                request_id,
                idem_payload,
                resource_id=registration["document_id"],
                response=response,
            )
            return response
        document_id = str(uuid.uuid4())
        document = None

    version_id = make_version_id(document_id, sha256, fingerprint)
    object_key = f"originals/{document_id}/{sha256}.{ 'pdf' if suffix == 'pdf' else 'md' }"
    now = utcnow()

    if document is None:
        meta = LegalMetadata(
            jurisdiction="unknown",
            issued_date=None,
            effective_from=None,
            effective_to=None,
            effectivity="unknown",
            effectivity_source_chunk_id=None,
            case_number=None,
            issuer=None,
            document_number=None,
        ).model_dump(mode="json")
        await runtime.store.insert_document(
            {
                "document_id": document_id,
                "title": resolved_title,
                "doc_type": doc_type,
                "scope": scope.value,
                "session_id": str(session_id) if session_id else None,
                "current_version_id": None,
                "tombstoned": False,
                "created_at": now,
                "updated_at": now,
                "metadata_hint": meta,
            }
        )

    await runtime.store.insert_version(
        {
            "version_id": version_id,
            "document_id": document_id,
            "sha256": sha256,
            "source_url": source_url,
            "object_key": object_key,
            "metadata": LegalMetadata(
                jurisdiction="unknown",
                issued_date=None,
                effective_from=None,
                effective_to=None,
                effectivity="unknown",
                effectivity_source_chunk_id=None,
                case_number=None,
                issuer=None,
                document_number=None,
            ).model_dump(mode="json"),
            "publication_state": PublicationState.STAGED.value,
            "availability": Availability.UNAVAILABLE.value,
            "vector_state": VectorState.PENDING.value,
            "graph_state": "pending",
            "quality_flags": [],
            "created_at": now,
            "pipeline_revision": fingerprint,
            "published_at": None,
            "scope": scope.value,
            "session_id": str(session_id) if session_id else None,
            "doc_type": doc_type,
            "stage_artifacts": {},
        }
    )
    task_id = str(uuid.uuid4())
    await runtime.store.create_task(
        {
            "task_id": task_id,
            "document_id": document_id,
            "version_id": version_id,
            "status": ImportStatus.QUEUED.value,
            "stage": ImportStage.VALIDATE.value,
            "availability": Availability.UNAVAILABLE.value,
            "vector_state": VectorState.PENDING.value,
            "graph_state": "pending",
            "warnings": [],
            "retryable": False,
            "attempt": 1,
            "error": None,
            "updated_at": now,
            "created_at": now,
            "request_id": str(request_id),
            "input_hash": sha256,
            "stage_success": [],
            "stage_artifacts": {},
            "lease_token": None,
            "lease_until": None,
            "cancel_requested": False,
            "scope": scope.value,
            "session_id": str(session_id) if session_id else None,
            "replace_document_id": str(replace_document_id) if replace_document_id else None,
            "sha256": sha256,
            "object_key": object_key,
            "filename": filename,
            "mime": detect_mime(filename),
            "size_bytes": len(data),
            "doc_type": doc_type,
            "title": resolved_title,
            "source_url": source_url,
            "pipeline_revision": fingerprint,
        }
    )
    await runtime.objects.put_bytes(object_key, data, detect_mime(filename))
    if scope == Scope.KNOWLEDGE:
        await runtime.store.register_content(
            scope=scope.value,
            session_id=None,
            sha256=sha256,
            pipeline_revision=fingerprint,
            document_id=document_id,
            version_id=version_id,
            task_id=task_id,
        )

    response = {
        "document_id": document_id,
        "version_id": version_id,
        "task_id": task_id,
        "duplicate": False,
    }
    await runtime.idempotency.commit(
        f"{scope.value}:upload", request_id, idem_payload, resource_id=document_id, response=response
    )
    return response


@router.get("/documents/{document_id}")
async def get_document(document_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    doc = await runtime.store.get_document(document_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "资料不存在")
    version = await runtime.store.get_version(doc["current_version_id"]) if doc.get("current_version_id") else None
    task = await runtime.store.latest_task_for_document(doc["document_id"])
    return as_list_item(doc, version, task).model_dump(mode="json")


@router.delete("/documents/{document_id}", status_code=202)
async def withdraw_document(document_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    doc = await runtime.store.get_document(document_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "资料不存在")
    await runtime.store.set_tombstoned(doc["document_id"], True)
    return {"document_id": doc["document_id"], "tombstoned": True}


@router.get("/documents/{document_id}/versions")
async def list_versions(
    document_id: uuid.UUID, request: Request, cursor: str | None = None, limit: int = 20
) -> dict[str, Any]:
    runtime = _runtime(request)
    limit = parse_limit(limit)
    doc = await runtime.store.get_document(document_id)
    if doc is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "资料不存在")
    raw = decode_cursor(cursor)
    parsed = (datetime_cursor(raw[0]), str(raw[1])) if raw and len(raw) == 2 else None
    rows = await runtime.store.list_versions(doc["document_id"], cursor=parsed, limit=limit + 1)
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [as_version(row) for row in rows]
    next_cursor = (
        encode_cursor([rows[-1]["created_at"].isoformat(), rows[-1]["version_id"]]) if has_more and rows else None
    )
    return {"items": items, "next_cursor": next_cursor}


@router.get("/import-tasks/{task_id}")
async def get_task(task_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    task = await runtime.store.get_task(task_id)
    if task is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "任务不存在")
    return as_task(task)


@router.post("/import-tasks/{task_id}/retry", status_code=202)
async def retry_task(task_id: uuid.UUID, request: Request) -> dict[str, Any]:
    runtime = _runtime(request)
    body = await request.json() if await request.body() else {}
    if not body.get("request_id"):
        raise AppError(422, "VALIDATION_ERROR", "request_id 必填", details={"field": "request_id"})
    task = await runtime.store.get_task(task_id)
    if task is None:
        raise AppError(404, "RESOURCE_NOT_FOUND", "任务不存在")
    if not task.get("retryable"):
        raise AppError(409, "INVALID_STATE", "当前任务不可重试", details={"status": task.get("status")})
    attempt = int(task.get("attempt", 1)) + 1
    if attempt > runtime.settings.config.ingestion.max_attempts:
        raise AppError(409, "INVALID_STATE", "重试次数已达上限")
    requeued = await runtime.store.requeue_task(task["task_id"], attempt)
    if not requeued:
        raise AppError(409, "INVALID_STATE", "任务状态已变化，无法重试")
    return {"task_id": task["task_id"], "status": ImportStatus.QUEUED.value}
