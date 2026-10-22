"""导入流水线：validate → save_original → parse → split → extract → write_vectors →
write_graph → publish，用 LangGraph 串成单向图，阶段产物持久化以支持断点恢复。

失败语义：
- 非法输入/未支持格式/页数超限：不可重试；
- 向量失败：版本不可用，任务 failed，可重试；
- 图失败但向量成功：发布 limited 并 succeeded_with_warnings，可重试仅恢复图阶段。
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..adapters.embedding import BGEM3Embedder, EmbeddingUnavailable
from ..adapters.parser import ParserError, detect_mime, parse
from ..domain.enums import (
    Availability,
    GraphState,
    ImportStage,
    ImportStatus,
    PublicationState,
    QualityFlag,
    VectorState,
)
from ..domain.ids import sha256_hex
from ..settings import ConfigError, Settings
from ..storage.graph import GraphStore
from ..storage.mongo import MongoStore, utcnow
from ..storage.objects import ObjectStore, ObjectStoreError
from ..storage.vector import VectorStore
from .extractor import (
    case_entity_id,
    extract_document,
    issue_entity_id,
    organization_entity_id,
    resolve_citations,
)
from .splitter import split_document, to_chunk_document


class StageFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = True, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details or {}


class LeaseLost(RuntimeError):
    """租约被新 worker 接管后必须停止写入。"""


class IngestionState(TypedDict, total=False):
    task_id: str
    document_id: str
    version_id: str
    lease_token: str
    stage_success: list[str]
    artifacts: dict[str, Any]
    warnings: list[str]
    availability: str
    vector_state: str
    graph_state: str
    quality_flags: list[str]
    chunk_count: int


def build_pipeline(
    settings: Settings,
    store: MongoStore,
    objects: ObjectStore,
    vectors: VectorStore,
    graph_store: GraphStore,
    embedder: BGEM3Embedder,
):
    config = settings.config

    async def persist_stage(
        state: IngestionState, stage: ImportStage, patch: dict[str, Any], *, mark_done: bool = True
    ) -> dict[str, Any]:
        """阶段产物写回任务文档，CAS 绑定租约；失败说明租约已丢失，必须中止。

        mark_done=False 用于“图阶段失败但仍可发布受限版本”：不把该阶段记为成功，
        以便重试时只恢复图阶段。
        """
        stages = list(state.get("stage_success", []))
        if mark_done and stage.value not in stages:
            stages.append(stage.value)
        ok = await store.update_task_cas(
            state["task_id"],
            state["lease_token"],
            {
                "stage": stage.value,
                "stage_success": stages,
                "vector_state": patch.get("vector_state", state.get("vector_state", VectorState.PENDING.value)),
                "graph_state": patch.get("graph_state", state.get("graph_state", GraphState.PENDING.value)),
                "availability": patch.get("availability", state.get("availability", Availability.UNAVAILABLE.value)),
                "warnings": patch.get("warnings", state.get("warnings", [])),
            },
        )
        if not ok:
            raise LeaseLost("任务租约已失效，停止写入")
        return patch

    async def node_validate(state: IngestionState) -> dict[str, Any]:
        if ImportStage.VALIDATE.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        if task is None:
            raise StageFailure("RESOURCE_NOT_FOUND", "任务不存在", retryable=False)
        if task.get("cancel_requested"):
            raise StageFailure("CANCELLED", "任务已取消", retryable=False)
        if task.get("status") == ImportStatus.CANCELLED.value:
            raise StageFailure("CANCELLED", "任务已取消", retryable=False)
        try:
            settings.require_embedding_revision()
        except ConfigError as exc:
            raise StageFailure("EMBEDDING_REVISION_UNLOCKED", str(exc), retryable=False) from exc
        filename = task.get("filename", "")
        snapshot = task.get("mime") or detect_mime(filename)
        if snapshot not in {"application/pdf", "text/markdown"}:
            raise StageFailure("INVALID_FILE_TYPE", "仅支持文本型 PDF 与 Markdown", retryable=False)
        if task.get("scope") == "knowledge":
            count = await store.count_documents_knowledge()
            if count > config.ingestion.max_documents:
                raise StageFailure(
                    "DOCUMENT_LIMIT_EXCEEDED",
                    f"知识资料数量 {count} 已超过上限 {config.ingestion.max_documents}",
                    retryable=False,
                )
        return await persist_stage(state, ImportStage.VALIDATE, {})

    async def node_save_original(state: IngestionState) -> dict[str, Any]:
        if ImportStage.SAVE_ORIGINAL.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        if task is None:
            raise StageFailure("RESOURCE_NOT_FOUND", "任务不存在", retryable=False)
        key = task["object_key"]
        try:
            await objects.get_bytes(key)
        except ObjectStoreError as exc:
            raise StageFailure("OBJECT_MISSING", f"原件不可读：{exc}", retryable=True) from exc
        return await persist_stage(
            state, ImportStage.SAVE_ORIGINAL, {"artifacts": {**state.get("artifacts", {}), "object_key": key}}
        )

    async def node_parse(state: IngestionState) -> dict[str, Any]:
        artifacts = dict(state.get("artifacts", {}))
        if ImportStage.PARSE.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        if task is None:
            raise StageFailure("RESOURCE_NOT_FOUND", "任务不存在", retryable=False)
        try:
            data = await objects.get_bytes(task["object_key"])
        except ObjectStoreError as exc:
            raise StageFailure("OBJECT_MISSING", f"原件不可读：{exc}") from exc
        try:
            parsed = parse(task["filename"], data, max_pages=config.ingestion.max_pdf_pages)
        except ParserError as exc:
            # 页数超限在解析阶段发现：写 TaskStatus.error，不再补发 HTTP 413
            raise StageFailure(exc.code, exc.message, retryable=False) from exc
        text_key = f"parsed/{state['version_id']}.txt"
        pages_key = f"parsed/{state['version_id']}.pages.json"
        await objects.put_bytes(text_key, parsed.text.encode("utf-8"), "text/plain; charset=utf-8")
        await objects.put_bytes(
            pages_key,
            json.dumps([{"page": page.page, "text": page.text} for page in parsed.pages], ensure_ascii=False).encode("utf-8"),
            "application/json",
        )
        artifacts.update(
            {
                "text_object": text_key,
                "pages_object": pages_key,
                "parser": parsed.parser,
                "partial": parsed.partial,
                "page_count": len(parsed.pages),
            }
        )
        flags = list(state.get("quality_flags", []))
        if parsed.partial and QualityFlag.PARSER_PARTIAL.value not in flags:
            flags.append(QualityFlag.PARSER_PARTIAL.value)
        patch = await persist_stage(state, ImportStage.PARSE, {"artifacts": artifacts, "quality_flags": flags})
        await store.update_task_cas(state["task_id"], state["lease_token"], {"stage_artifacts": artifacts})
        return patch

    async def node_split(state: IngestionState) -> dict[str, Any]:
        if ImportStage.SPLIT.value in state.get("stage_success", []):
            return {}
        artifacts = state.get("artifacts", {})
        text_key = artifacts.get("text_object")
        if not text_key:
            raise StageFailure("STAGE_ARTIFACT_MISSING", "缺少解析产物，无法切分")
        text = (await objects.get_bytes(text_key)).decode("utf-8")
        from ..adapters.parser import PageText, ParsedDocument  # 局部导入避免环

        pages: list[PageText] = []
        pages_key = artifacts.get("pages_object")
        if pages_key:
            raw_pages = json.loads((await objects.get_bytes(pages_key)).decode("utf-8"))
            pages = [PageText(page=int(item["page"]), text=str(item["text"])) for item in raw_pages]
        parsed = ParsedDocument(text=text, pages=pages, partial=artifacts.get("partial", False))
        drafts = split_document(
            parsed,
            version_id=state["version_id"],
            target_min=config.ingestion.chunk_target_min_chars,
            target_max=config.ingestion.chunk_target_max_chars,
            soft_max=config.ingestion.chunk_soft_max_chars,
            overlap=config.ingestion.chunk_overlap_chars,
        )
        if not drafts:
            raise StageFailure("PARSER_UNSUPPORTED", "未切分出有效正文", retryable=False)
        if len(drafts) > config.ingestion.max_chunks:
            raise StageFailure(
                "CHUNK_LIMIT_EXCEEDED",
                f"切片数 {len(drafts)} 超过上限 {config.ingestion.max_chunks}",
                retryable=False,
            )
        embedding_revision = settings.require_embedding_revision()
        task = await store.get_task(state["task_id"])
        assert task is not None
        docs = [
            to_chunk_document(
                draft,
                version_id=state["version_id"],
                document_id=state["document_id"],
                embedding_revision=embedding_revision,
                quality_flags=state.get("quality_flags", []),
            )
            for draft in drafts
        ]
        for doc in docs:
            doc.update(
                {
                    "workspace_id": "default",
                    "scope": task.get("scope", "knowledge"),
                    "session_id": task.get("session_id"),
                    "parser_revision": artifacts.get("parser", "pypdf_text"),
                }
            )
        await store.upsert_chunks(docs)
        return await persist_stage(state, ImportStage.SPLIT, {"chunk_count": len(docs)})

    async def node_extract(state: IngestionState) -> dict[str, Any]:
        if ImportStage.EXTRACT.value in state.get("stage_success", []):
            return {}
        artifacts = state.get("artifacts", {})
        text_key = artifacts.get("text_object")
        text = (await objects.get_bytes(text_key)).decode("utf-8") if text_key else ""
        task = await store.get_task(state["task_id"])
        assert task is not None
        chunks = await store.collection("chunks").find({"version_id": state["version_id"]}).sort("ordinal", 1).to_list(length=5000)
        result = extract_document(
            title=task.get("title", task["filename"]),
            doc_type=task.get("doc_type", "unknown"),
            text=text,
            chunks=chunks,
            version_id=state["version_id"],
            document_id=state["document_id"],
        )
        metadata = result.metadata.as_dict()
        flags = set(state.get("quality_flags", []))
        if result.metadata.issued_date is None or result.metadata.effective_from is None:
            flags.add(QualityFlag.METADATA_MISSING.value)
        if result.metadata.effectivity == "unknown":
            flags.add(QualityFlag.EFFECTIVITY_UNKNOWN.value)
        await store.update_version(
            state["version_id"],
            {
                "metadata": _jsonable(metadata),
                "quality_flags": sorted(flags),
            },
        )
        extraction_artifact = {
            "provisions": result.provisions,
            "issues": result.issues,
            "case_number": result.metadata.case_number,
            "issuer": result.metadata.issuer,
            "citation_count": len(result.citations),
        }
        artifacts.update({"extraction": extraction_artifact})
        # 引用候选写入对象存储，图阶段单独解析，避免 extract 依赖全库状态
        citation_key = f"extracted/{state['version_id']}.citations.json"
        await objects.put_bytes(
            citation_key,
            json.dumps([asdict(c) for c in result.citations], ensure_ascii=False).encode("utf-8"),
            "application/json",
        )
        artifacts["citation_object"] = citation_key
        return await persist_stage(
            state, ImportStage.EXTRACT, {"artifacts": artifacts, "quality_flags": sorted(flags)}
        )

    async def node_write_vectors(state: IngestionState) -> dict[str, Any]:
        if ImportStage.WRITE_VECTORS.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        assert task is not None
        chunks = await store.collection("chunks").find({"version_id": state["version_id"]}).sort("ordinal", 1).to_list(length=5000)
        if not chunks:
            raise StageFailure("PARSER_UNSUPPORTED", "没有可索引的切片", retryable=False)
        revision = settings.require_embedding_revision()
        try:
            await vectors.ensure_collection()
            batch = config.models.embedding_batch_size
            for index in range(0, len(chunks), batch):
                slice_ = chunks[index : index + batch]
                dense, sparse = await embedder.encode([chunk["content"] for chunk in slice_], revision)
                rows = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "version_id": chunk["version_id"],
                        "document_id": chunk["document_id"],
                        "workspace_id": "default",
                        "scope": chunk.get("scope", "knowledge"),
                        "session_id": chunk.get("session_id") or "-",
                        "doc_type": task.get("doc_type", "unknown"),
                        "scenario_tags": ",".join(chunk.get("scenario_tags") or []),
                    }
                    for chunk in slice_
                ]
                await vectors.upsert(rows, dense, sparse)
                # 每批 4 个切片后让出执行机会，避免长时间占用事件循环
        except Exception as exc:  # 向量不可用即不可问答
            await store.update_version(state["version_id"], {"vector_state": VectorState.FAILED.value, "availability": Availability.UNAVAILABLE.value})
            raise StageFailure("VECTOR_INDEX_FAILED", f"向量索引失败：{exc}", retryable=True) from exc
        await store.update_version(
            state["version_id"], {"vector_state": VectorState.READY.value, "embedding_revision": revision}
        )
        return await persist_stage(state, ImportStage.WRITE_VECTORS, {"vector_state": VectorState.READY.value})

    async def node_write_graph(state: IngestionState) -> dict[str, Any]:
        if ImportStage.WRITE_GRAPH.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        assert task is not None
        artifacts = state.get("artifacts", {})
        extraction = artifacts.get("extraction", {})
        chunks = await store.collection("chunks").find({"version_id": state["version_id"]}).sort("ordinal", 1).to_list(length=5000)
        try:
            versions = await store.published_versions(scope="knowledge")
            known_documents: list[dict[str, Any]] = []
            for version in versions:
                doc = await store.get_document(version["document_id"])
                if doc and not doc.get("tombstoned"):
                    known_documents.append(
                        {"document_id": doc["document_id"], "title": doc["title"], "version_id": version["version_id"]}
                    )
            citations: list[dict[str, Any]] = []
            resolved_any = False
            citation_key = artifacts.get("citation_object")
            if citation_key:
                raw = json.loads((await objects.get_bytes(citation_key)).decode("utf-8"))
                from .extractor import CitationCandidate

                candidates = [CitationCandidate(**item) for item in raw]
                case_key = extraction.get("case_number") or task.get("source_url") or state["document_id"]
                source_entity = case_entity_id(case_key, state["version_id"])
                resolution = resolve_citations(
                    candidates, source_entity=source_entity, version_id=state["version_id"], known_documents=known_documents
                )
                citations = resolution.resolved
                resolved_any = bool(citations)
                for item in resolution.unresolved:
                    await store.upsert_unresolved(
                        {
                            **item,
                            "extractor_revision": "rule_citation_v1",
                            "version_id": state["version_id"],
                            "created_at": utcnow(),
                        }
                    )
                if resolution.unresolved and not citations:
                    flags = set(state.get("quality_flags", []))
                    flags.add(QualityFlag.CITATION_UNRESOLVED.value)
                    await store.update_version(state["version_id"], {"quality_flags": sorted(flags)})

            case = None
            if task.get("doc_type") == "case":
                case_key = extraction.get("case_number") or task.get("source_url") or state["document_id"]
                case = {
                    "entity_id": case_entity_id(case_key, state["version_id"]),
                    "case_identity_key": case_key,
                    "label": task.get("title", task["filename"]),
                    "chunk_id": chunks[0]["chunk_id"] if chunks else None,
                }

            issues = []
            for scenario in extraction.get("issues", []):
                from .extractor import _ISSUE_LABELS  # 同一模块内的展示名映射

                issues.append(
                    {
                        "entity_id": issue_entity_id(scenario),
                        "label": _ISSUE_LABELS.get(scenario, scenario),
                        "scenario": scenario,
                        "assertion_id": sha256_hex("CONCERNS", state["version_id"], scenario),
                    }
                )
            organization = None
            if extraction.get("issuer"):
                organization = {
                    "entity_id": organization_entity_id(extraction["issuer"]),
                    "label": extraction["issuer"],
                    "assertion_id": sha256_hex("ISSUED_BY", state["version_id"], extraction["issuer"]),
                }
            await graph_store.write_version_graph(
                {
                    "document": {
                        "document_id": state["document_id"],
                        "title": task.get("title", task["filename"]),
                        "doc_type": task.get("doc_type", "unknown"),
                    },
                    "version": {"version_id": state["version_id"], "availability": "limited"},
                    "chunks": [
                        {
                            "chunk_id": chunk["chunk_id"],
                            "ordinal": chunk["ordinal"],
                            "label": f"{task.get('title', '')} · 第{chunk['ordinal'] + 1}片",
                        }
                        for chunk in chunks
                    ],
                    "provisions": extraction.get("provisions", []),
                    "case": case,
                    "issues": issues,
                    "organization": organization,
                    "citations": citations,
                }
            )
            graph_state = GraphState.READY.value if (citations or case or extraction.get("provisions")) else GraphState.EMPTY.value
            await store.update_version(state["version_id"], {"graph_state": graph_state})
            return await persist_stage(state, ImportStage.WRITE_GRAPH, {"graph_state": graph_state})
        except Exception as exc:
            # 图失败可发布受限版本：向量可用即可检索，图状态与警告如实记录
            await store.update_version(state["version_id"], {"graph_state": GraphState.FAILED.value})
            warnings = list(state.get("warnings", []))
            if QualityFlag.GRAPH_UNAVAILABLE.value not in warnings:
                warnings.append(QualityFlag.GRAPH_UNAVAILABLE.value)
            # 图失败不阻断发布：向量已就绪即可 limited 使用；该阶段不记成功，重试只补图
            return await persist_stage(
                state,
                ImportStage.WRITE_GRAPH,
                {"graph_state": GraphState.FAILED.value, "warnings": warnings, "artifacts": artifacts},
                mark_done=False,
            )

    async def node_publish(state: IngestionState) -> dict[str, Any]:
        if ImportStage.PUBLISH.value in state.get("stage_success", []):
            return {}
        task = await store.get_task(state["task_id"])
        assert task is not None
        version = await store.get_version(state["version_id"])
        if version is None:
            raise StageFailure("RESOURCE_NOT_FOUND", "版本不存在", retryable=False)
        doc = await store.get_document(state["document_id"])
        if doc is None or doc.get("tombstoned"):
            raise StageFailure("CANCELLED", "资料已撤销，不再发布", retryable=False)

        vector_state = version.get("vector_state")
        graph_state = version.get("graph_state")
        flags = set(version.get("quality_flags") or [])
        if vector_state != VectorState.READY.value:
            availability = Availability.UNAVAILABLE.value
        elif graph_state == GraphState.FAILED.value or QualityFlag.METADATA_MISSING.value in flags:
            availability = Availability.LIMITED.value
        elif QualityFlag.CITATION_UNRESOLVED.value in flags:
            availability = Availability.LIMITED.value
        elif QualityFlag.EFFECTIVITY_UNKNOWN.value in flags:
            # 效力未知属于关键元数据不完整：正文可用，但不得标为“可用”
            availability = Availability.LIMITED.value
        else:
            availability = Availability.AVAILABLE.value

        published_at = version.get("published_at") or utcnow()
        await store.update_version(
            state["version_id"],
            {
                "publication_state": PublicationState.PUBLISHED.value,
                "availability": availability,
                "published_at": published_at,
                "quality_flags": sorted(flags),
            },
        )
        await store.update_document_current_version(state["document_id"], state["version_id"])

        warnings = list(state.get("warnings", []))
        for flag in flags:
            if flag not in warnings:
                warnings.append(flag)
        patch = await persist_stage(
            state, ImportStage.PUBLISH, {"availability": availability, "warnings": warnings}
        )
        status = ImportStatus.SUCCEEDED_WITH_WARNINGS.value if warnings else ImportStatus.SUCCEEDED.value
        retryable = graph_state == GraphState.FAILED.value
        await store.update_task_any(
            state["task_id"],
            {
                "stage": ImportStage.DONE.value,
                "status": status,
                "availability": availability,
                "warnings": warnings,
                "retryable": retryable,
                "lease_token": None,
                "lease_until": None,
            },
        )
        return patch

    builder = StateGraph(IngestionState)
    builder.add_node(ImportStage.VALIDATE.value, node_validate)
    builder.add_node(ImportStage.SAVE_ORIGINAL.value, node_save_original)
    builder.add_node(ImportStage.PARSE.value, node_parse)
    builder.add_node(ImportStage.SPLIT.value, node_split)
    builder.add_node(ImportStage.EXTRACT.value, node_extract)
    builder.add_node(ImportStage.WRITE_VECTORS.value, node_write_vectors)
    builder.add_node(ImportStage.WRITE_GRAPH.value, node_write_graph)
    builder.add_node(ImportStage.PUBLISH.value, node_publish)

    builder.add_edge(START, ImportStage.VALIDATE.value)
    builder.add_edge(ImportStage.VALIDATE.value, ImportStage.SAVE_ORIGINAL.value)
    builder.add_edge(ImportStage.SAVE_ORIGINAL.value, ImportStage.PARSE.value)
    builder.add_edge(ImportStage.PARSE.value, ImportStage.SPLIT.value)
    builder.add_edge(ImportStage.SPLIT.value, ImportStage.EXTRACT.value)
    builder.add_edge(ImportStage.EXTRACT.value, ImportStage.WRITE_VECTORS.value)
    builder.add_edge(ImportStage.WRITE_VECTORS.value, ImportStage.WRITE_GRAPH.value)
    builder.add_edge(ImportStage.WRITE_GRAPH.value, ImportStage.PUBLISH.value)
    builder.add_edge(ImportStage.PUBLISH.value, END)
    return builder.compile()


def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        result[key] = value.isoformat() if hasattr(value, "isoformat") else value
    return result
