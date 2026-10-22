"""咨询流水线：validate→history_and_facts→route→snapshot→retrieve→fuse→rerank→
evidence_policy→generate→finalize。

关键约束：
- 所有检索路径都携带冻结的版本白名单；
- 答案、证据快照、SSE 事件、终态在同一个 queries 文档中按 revision CAS 提交；
- 回答模型只使用已部署的微调模型，不可用时 failed，不静默改用辅助云模型；
- 澄清/依据不足是正常 completed，不调用回答模型，也不谎称已生成。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Sequence

from ..adapters.embedding import BGEM3Embedder
from ..adapters.llm import AnswerModelClient, AuxModelClient, AuxModelUnavailable, TokenCounter
from ..adapters.reranker import Reranker, RerankUnavailable
from ..adapters.web_search import WebSearchClient, WebSearchUnavailable
from ..domain.enums import (
    Behavior,
    CitationValidation,
    PathState,
    QualityFlag,
    QueryState,
    ReasoningCode,
    SourceType,
)
from ..domain.ids import web_source_id
from ..domain.models import (
    Candidate,
    Evidence,
    Facts,
    LegalMetadata,
    Location,
    PathStatus,
)
from ..services.visibility import VisibilityService
from ..settings import Settings
from ..storage.graph import GraphStore
from ..storage.mongo import MongoStore, utcnow
from ..storage.vector import VectorStore
from .fusion import FusedItem, ScoredItem, cliff_filter, weighted_rrf
from .policy import (
    ContextOverflow,
    build_messages,
    clarify_text,
    decide,
    fit_context,
    insufficient_text,
    should_clarify,
    validate_citations,
)


class LeaseLost(RuntimeError):
    pass


@dataclass
class EventWriter:
    """单文档事件日志写入器：一次 CAS 同时提交事件与业务快照。"""

    store: MongoStore
    query_id: str
    lease_token: str
    revision: int
    last_seq: int
    retention_days: int

    async def emit(self, event: str, payload: dict[str, Any], *, patch: dict[str, Any] | None = None) -> None:
        record = {"seq": self.last_seq + 1, "event": event, "payload": payload, "created_at": utcnow()}
        ok = await self.store.query_cas(
            self.query_id,
            lease_token=self.lease_token,
            revision=self.revision,
            patch=patch or {},
            push_events=[record],
        )
        if not ok:
            raise LeaseLost("查询租约已失效，停止写入事件")
        self.revision += 1
        self.last_seq += 1

    async def finish(self, status: QueryState, patch: dict[str, Any], payload: dict[str, Any]) -> None:
        event = (
            "done"
            if status in {QueryState.COMPLETED, QueryState.COMPLETED_WITH_WARNINGS}
            else ("cancelled" if status == QueryState.CANCELLED else "failed")
        )
        record = {"seq": self.last_seq + 1, "event": event, "payload": payload, "created_at": utcnow()}
        ok = await self.store.query_cas(
            self.query_id,
            lease_token=self.lease_token,
            revision=self.revision,
            patch={
                **patch,
                "status": status.value,
                "finished_at": utcnow(),
                "lease_token": None,
                "lease_until": None,
                "events_expires_at": utcnow() + timedelta(days=self.retention_days),
            },
            push_events=[record],
        )
        if not ok:
            raise LeaseLost("查询租约已失效，无法提交终态")
        self.revision += 1
        self.last_seq += 1


@dataclass
class Retrieved:
    candidates: dict[str, Candidate] = field(default_factory=dict)
    ranking: dict[str, list[str]] = field(default_factory=dict)
    path_status: dict[str, str] = field(default_factory=dict)
    # source_id -> document_id：Candidate 属于契约对象不能加字段，文档身份单独携带，
    # 供证据策略判断“是否为同一份资料的多个版本”。
    document_by_source: dict[str, str] = field(default_factory=dict)


class QueryPipeline:
    def __init__(
        self,
        settings: Settings,
        store: MongoStore,
        vectors: VectorStore,
        graph: GraphStore,
        embedder: BGEM3Embedder,
        answer_model: AnswerModelClient,
        aux_model: AuxModelClient,
        reranker: Reranker,
        web: WebSearchClient,
        visibility: VisibilityService,
    ) -> None:
        self.settings = settings
        self.store = store
        self.vectors = vectors
        self.graph = graph
        self.embedder = embedder
        self.answer_model = answer_model
        self.aux_model = aux_model
        self.reranker = reranker
        self.web = web
        self.visibility = visibility
        self.config = settings.config
        # 计数器读取本地tokenizer文件，不能把远程HTTP地址误当作文件目录。
        self.counter = TokenCounter(settings.env.model_tokenizer_path or None)

    async def run(self, query_id: str, lease_token: str) -> None:
        query = await self.store.get_query(query_id)
        if query is None:
            return
        writer = EventWriter(
            store=self.store,
            query_id=query_id,
            lease_token=lease_token,
            revision=int(query.get("revision", 0)),
            last_seq=int(query.get("last_event_seq", 0)),
            retention_days=self.config.session.event_retention_days,
        )
        session_id = uuid.UUID(query["session_id"])
        request = query.get("request") or {}
        question = request.get("question", "")
        facts = Facts.model_validate(query["facts"])
        path_status = PathStatus.all_pending()

        try:
            await writer.emit("started", {"status": QueryState.RUNNING.value})
            await writer.emit("stage", {"name": "route", "path_status": path_status.model_dump(mode="json")})

            # route：规则识别场景与必要事实；需要澄清时短路，不执行四路检索
            if should_clarify(facts):
                await self._short_circuit(
                    writer,
                    query,
                    facts,
                    Behavior.CLARIFY,
                    ReasoningCode.FACTS_MISSING,
                    clarify_text(facts),
                    PathStatus(
                        ordinary=PathState.SKIPPED,
                        hyde=PathState.SKIPPED,
                        graph=PathState.SKIPPED,
                        web=PathState.SKIPPED,
                        rerank=PathState.SKIPPED,
                    ),
                )
                return

            snapshot = await self.visibility.snapshot(
                session_id=session_id,
                document_ids=[uuid.UUID(item) for item in request.get("document_ids", [])],
                attachment_ids=[uuid.UUID(item) for item in request.get("attachment_ids", [])],
                relevant_date=facts.relevant_date,
            )
            version_ids = snapshot.all_version_ids
            await self.store.query_cas(
                writer.query_id,
                lease_token=writer.lease_token,
                revision=writer.revision,
                patch={"snapshot_version_ids": version_ids},
            )
            writer.revision += 1

            await writer.emit("stage", {"name": "retrieve"})
            retrieved = await self._retrieve(question, facts, version_ids, writer)
            path_status = PathStatus(**retrieved.path_status)

            await writer.emit("stage", {"name": "fuse", "path_status": path_status.model_dump(mode="json")})
            fused = self._fuse(retrieved)

            await writer.emit("stage", {"name": "rerank"})
            ranked, rerank_status = await self._rerank(question, fused, retrieved, writer)
            retrieved.path_status["rerank"] = rerank_status
            path_status = PathStatus(**retrieved.path_status)
            await self.store.query_cas(
                writer.query_id,
                lease_token=writer.lease_token,
                revision=writer.revision,
                patch={"path_status": path_status.model_dump(mode="json")},
            )
            writer.revision += 1

            await writer.emit(
                "stage", {"name": "evidence_policy", "path_status": path_status.model_dump(mode="json")}
            )
            evidence = await self._enrich_metadata(self._build_evidence(ranked, retrieved.candidates, query_id))
            # 证据对象不带 document_id（契约固定），这里把文档身份单独传给策略
            document_ids = {
                item.evidence_id: retrieved.document_by_source.get(item.source_id, item.title)
                for item in evidence
            }
            decision = decide(facts=facts, evidence=evidence, question=question, document_ids=document_ids)

            if decision.behavior in {Behavior.CLARIFY, Behavior.INSUFFICIENT}:
                text = (
                    clarify_text(facts)
                    if decision.behavior == Behavior.CLARIFY
                    else insufficient_text(decision.reasoning_code)
                )
                await self._short_circuit(
                    writer,
                    query,
                    facts,
                    decision.behavior,
                    decision.reasoning_code,
                    text,
                    path_status,
                    warnings=decision.warnings,
                )
                return

            await self._generate(writer, query, question, facts, evidence, decision, path_status)
        except LeaseLost:
            return
        except ContextOverflow as exc:
            await self._fail(writer, "INPUT_TOO_LONG", str(exc))
        except Exception as exc:  # 未预期异常也必须落终态，避免查询永远 running
            await self._fail(writer, "QUERY_FAILED", f"查询执行失败：{exc}")

    # ---------- 检索 ----------
    async def _retrieve(
        self, question: str, facts: Facts, version_ids: Sequence[str], writer: EventWriter
    ) -> Retrieved:
        config = self.config
        retrieved = Retrieved(
            path_status={
                "ordinary": PathState.PENDING.value,
                "hyde": PathState.PENDING.value,
                "graph": PathState.PENDING.value,
                "web": PathState.PENDING.value,
                "rerank": PathState.PENDING.value,
            }
        )

        async def ordinary() -> tuple[list[str], str]:
            if not version_ids:
                return [], PathState.EMPTY.value
            try:
                dense, sparse = await self.embedder.encode([question])
                hits_dense, hits_sparse = await asyncio.gather(
                    self.vectors.search_dense(
                        dense[0], version_ids, "knowledge", None, config.retrieval.ordinary_dense_k
                    ),
                    self.vectors.search_sparse(
                        sparse[0], version_ids, "knowledge", None, config.retrieval.ordinary_sparse_k
                    ),
                )
                fused = weighted_rrf(
                    {
                        "dense": [hit.chunk_id for hit in hits_dense],
                        "sparse": [hit.chunk_id for hit in hits_sparse],
                    },
                    {
                        "dense": config.retrieval.ordinary_weights[0],
                        "sparse": config.retrieval.ordinary_weights[1],
                    },
                    k=config.retrieval.rrf_k,
                    rank_start=config.retrieval.rrf_rank_start,
                )
                ids = [item.source_id for item in fused[: config.retrieval.ordinary_result_k]]
                return ids, PathState.SUCCEEDED.value if ids else PathState.EMPTY.value
            except Exception:
                return [], PathState.FAILED.value

        async def hyde() -> tuple[list[str], str]:
            if not self.aux_model.configured:
                return [], PathState.UNAVAILABLE.value
            if not version_ids:
                return [], PathState.EMPTY.value
            try:
                passage = await self.aux_model.complete(
                    [
                        {
                            "role": "user",
                            "content": "根据以下问题写一段不超过300字的假定法律要点，仅用于检索，不要下结论："
                            + question,
                        }
                    ],
                    max_tokens=300,
                    timeout_seconds=config.timeouts.hyde_seconds,
                )
                passage = passage[: config.retrieval.hyde_max_chars]
                dense, sparse = await self.embedder.encode([passage])
                hits_dense, hits_sparse = await asyncio.gather(
                    self.vectors.search_dense(
                        dense[0], version_ids, "knowledge", None, config.retrieval.hyde_result_k
                    ),
                    self.vectors.search_sparse(
                        sparse[0], version_ids, "knowledge", None, config.retrieval.hyde_result_k
                    ),
                )
                merged = weighted_rrf(
                    {
                        "dense": [hit.chunk_id for hit in hits_dense],
                        "sparse": [hit.chunk_id for hit in hits_sparse],
                    },
                    {"dense": 0.7, "sparse": 0.3},
                    k=config.retrieval.rrf_k,
                )
                ids = [item.source_id for item in merged[: config.retrieval.hyde_result_k]]
                return ids, PathState.SUCCEEDED.value if ids else PathState.EMPTY.value
            except (asyncio.TimeoutError, AuxModelUnavailable):
                return [], PathState.TIMEOUT.value
            except Exception:
                return [], PathState.FAILED.value

        async def graph_path() -> tuple[list[str], str]:
            if not version_ids:
                return [], PathState.EMPTY.value
            try:
                ids = await self.graph.scenario_chunks(
                    facts.scenario.value, version_ids, config.retrieval.graph_result_k
                )
                return ids, PathState.SUCCEEDED.value if ids else PathState.EMPTY.value
            except asyncio.TimeoutError:
                return [], PathState.TIMEOUT.value
            except Exception:
                return [], PathState.FAILED.value

        async def web_path() -> tuple[list[dict[str, Any]], str]:
            web_enabled = bool((await self.store.get_query(writer.query_id) or {}).get("request", {}).get("web_enabled"))
            if not web_enabled:
                return [], PathState.SKIPPED.value
            if not self.web.configured:
                return [], PathState.UNAVAILABLE.value
            try:
                results = await asyncio.wait_for(
                    self.web.search(
                        question, limit=config.retrieval.web_max_results, timeout=config.timeouts.web_seconds
                    ),
                    timeout=config.timeouts.web_seconds,
                )
                return [dict(item.__dict__) for item in results], (
                    PathState.SUCCEEDED.value if results else PathState.EMPTY.value
                )
            except asyncio.TimeoutError:
                return [], PathState.TIMEOUT.value
            except WebSearchUnavailable:
                return [], PathState.UNAVAILABLE.value

        tasks = [
            asyncio.create_task(asyncio.wait_for(ordinary(), timeout=config.timeouts.ordinary_seconds + 0.5)),
            asyncio.create_task(asyncio.wait_for(hyde(), timeout=config.timeouts.hyde_seconds + 0.5)),
            asyncio.create_task(asyncio.wait_for(graph_path(), timeout=config.timeouts.graph_seconds + 0.5)),
            asyncio.create_task(asyncio.wait_for(web_path(), timeout=config.timeouts.web_seconds + 0.5)),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        def unpack(result: Any) -> tuple[Any, str]:
            if isinstance(result, asyncio.TimeoutError):
                return [], PathState.TIMEOUT.value
            if isinstance(result, BaseException):
                return [], PathState.FAILED.value
            return result

        ordinary_ids, ordinary_status = unpack(results[0])
        hyde_ids, hyde_status = unpack(results[1])
        graph_ids, graph_status = unpack(results[2])
        web_items, web_status = unpack(results[3])

        retrieved.ranking = {"ordinary": list(ordinary_ids), "hyde": list(hyde_ids), "graph": list(graph_ids)}
        retrieved.path_status.update(
            {"ordinary": ordinary_status, "hyde": hyde_status, "graph": graph_status, "web": web_status}
        )
        for path, ids in retrieved.ranking.items():
            await self._collect_candidates(retrieved, ids, path)
        if web_items:
            self._collect_web_candidates(retrieved, web_items)
        await self.store.query_cas(
            writer.query_id,
            lease_token=writer.lease_token,
            revision=writer.revision,
            patch={"path_status": retrieved.path_status},
        )
        writer.revision += 1
        return retrieved

    async def _collect_candidates(self, retrieved: Retrieved, chunk_ids: Sequence[str], path: str) -> None:
        chunks = await self.store.get_chunks(chunk_ids)
        by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            existing = retrieved.candidates.get(chunk_id)
            if existing is not None:
                if path not in existing.recall_sources:
                    existing.recall_sources.append(path)
                continue
            version = await self.store.get_version(chunk["version_id"])
            if version is None:
                continue
            doc = await self.store.get_document(version["document_id"])
            retrieved.document_by_source[chunk_id] = str(version["document_id"])
            retrieved.candidates[chunk_id] = Candidate(
                source_id=chunk_id,
                source_type=SourceType.ATTACHMENT if chunk.get("scope") == "session" else SourceType.LOCAL,
                chunk_id=chunk_id,
                version_id=chunk["version_id"],
                title=(doc or {}).get("title", "未命名资料"),
                text=chunk["content"],
                url=None,
                location=Location.model_validate(chunk["location"]),
                recall_sources=[path],
                rank_score=0.0,
                rerank_score=None,
                quality_flags=list(version.get("quality_flags") or []),
            )

    def _collect_web_candidates(self, retrieved: Retrieved, items: Sequence[dict[str, Any]]) -> None:
        for item in items:
            url = item.get("url") or ""
            text = item.get("content") or item.get("snippet") or ""
            if not url or not text:
                continue
            source_id = web_source_id(url, text)
            source_type = SourceType.WEB if item.get("content") else SourceType.WEB_SNIPPET
            retrieved.candidates[source_id] = Candidate(
                source_id=source_id,
                source_type=source_type,
                chunk_id=None,
                version_id=None,
                title=item.get("title") or url,
                text=text,
                url=url,
                location=None,
                recall_sources=["web"],
                rank_score=0.0,
                rerank_score=None,
                quality_flags=(
                    [QualityFlag.SOURCE_ACCESS_LIMITED.value] if source_type == SourceType.WEB_SNIPPET else []
                ),
            )
            retrieved.ranking.setdefault("web", []).append(source_id)

    def _fuse(self, retrieved: Retrieved) -> list[FusedItem]:
        weights = {
            "ordinary": float(self.config.retrieval.local_weights.get("ordinary", 1.0)),
            "hyde": float(self.config.retrieval.local_weights.get("hyde", 0.7)),
            "graph": float(self.config.retrieval.local_weights.get("graph", 0.8)),
        }
        fused = weighted_rrf(
            retrieved.ranking,
            weights,
            k=self.config.retrieval.rrf_k,
            rank_start=self.config.retrieval.rrf_rank_start,
        )
        for source_id in retrieved.ranking.get("web", []):
            if source_id not in {item.source_id for item in fused}:
                fused.append(FusedItem(source_id=source_id, score=0.0, recall_sources=["web"]))
        for item in fused:
            candidate = retrieved.candidates.get(item.source_id)
            if candidate:
                candidate.rank_score = item.score
                candidate.recall_sources = item.recall_sources
        return fused

    async def _rerank(
        self, question: str, fused: Sequence[FusedItem], retrieved: Retrieved, writer: EventWriter
    ) -> tuple[list[tuple[str, float]], str]:
        config = self.config
        candidates = [retrieved.candidates[item.source_id] for item in fused if item.source_id in retrieved.candidates]
        candidates = candidates[: config.retrieval.rerank_input_k]
        if not candidates:
            return [], PathState.SKIPPED.value
        fallback = [(item.source_id, 0.0) for item in fused[: config.retrieval.final_evidence_k]]
        if not self.reranker.configured:
            return fallback, PathState.UNAVAILABLE.value
        try:
            hits = await asyncio.wait_for(
                self.reranker.rerank(
                    question,
                    [item.text[:1500] for item in candidates],
                    top_n=len(candidates),
                    timeout=config.timeouts.rerank_seconds,
                ),
                timeout=config.timeouts.rerank_seconds,
            )
            ordered: list[tuple[str, float]] = []
            for hit in sorted(hits, key=lambda item: -item.score):
                if 0 <= hit.index < len(candidates):
                    candidate = candidates[hit.index]
                    candidate.rerank_score = hit.score
                    ordered.append((candidate.source_id, hit.score))
            if not ordered:
                return fallback, PathState.EMPTY.value
            trimmed = cliff_filter(
                [ScoredItem(source_id=sid, score=score) for sid, score in ordered],
                max_examine=config.retrieval.final_evidence_k,
                min_keep=config.retrieval.cliff_min_keep,
                abs_gap=config.retrieval.cliff_abs_gap,
                rel_gap=config.retrieval.cliff_rel_gap,
            )
            return [(item.source_id, item.score) for item in trimmed], PathState.SUCCEEDED.value
        except asyncio.TimeoutError:
            await writer.emit("warning", {"code": "RERANK_TIMEOUT", "message": "重排序超时，保留本地融合顺序"})
            return fallback, PathState.TIMEOUT.value
        except RerankUnavailable:
            await writer.emit(
                "warning", {"code": "RERANK_UNAVAILABLE", "message": "重排序不可用，保留本地融合顺序"}
            )
            return fallback, PathState.FAILED.value

    def _build_evidence(
        self, ranked: Sequence[tuple[str, float]], candidates: dict[str, Candidate], query_id: str
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen: set[str] = set()
        for source_id, _score in ranked:
            candidate = candidates.get(source_id)
            if candidate is None or source_id in seen:
                continue
            seen.add(source_id)
            evidence.append(
                Evidence(
                    evidence_id=f"E{len(evidence) + 1}",
                    query_id=uuid.UUID(query_id),
                    source_id=candidate.source_id,
                    source_type=candidate.source_type,
                    chunk_id=candidate.chunk_id,
                    version_id=candidate.version_id,
                    title=candidate.title,
                    text=candidate.text,
                    url=candidate.url,
                    location=candidate.location,
                    metadata=None,
                    quality_flags=candidate.quality_flags,
                )
            )
            if len(evidence) >= self.config.retrieval.final_evidence_k:
                break
        return evidence

    async def _enrich_metadata(self, evidence: Sequence[Evidence]) -> list[Evidence]:
        enriched: list[Evidence] = []
        for item in evidence:
            metadata = None
            if item.version_id:
                version = await self.store.get_version(item.version_id)
                if version and version.get("metadata"):
                    metadata = LegalMetadata.model_validate(version["metadata"])
            enriched.append(item.model_copy(update={"metadata": metadata}))
        return enriched

    # ---------- 生成 ----------
    async def _generate(
        self,
        writer: EventWriter,
        query: dict[str, Any],
        question: str,
        facts: Facts,
        evidence: Sequence[Evidence],
        decision,
        path_status: PathStatus,
    ) -> None:
        config = self.config
        session_id = uuid.UUID(query["session_id"])
        visible = await self.visibility.filter_visible_versions(
            [item.version_id for item in evidence if item.version_id], session_id=session_id
        )
        evidence = [item for item in evidence if item.version_id is None or item.version_id in visible]
        if not evidence:
            await self._short_circuit(
                writer,
                query,
                facts,
                Behavior.INSUFFICIENT,
                ReasoningCode.NO_RELEVANT_EVIDENCE,
                insufficient_text(ReasoningCode.NO_RELEVANT_EVIDENCE),
                path_status,
            )
            return

        # 冻结证据映射：在首个 delta 之前发送且只发送一次
        await writer.emit("evidence", {"items": [item.model_dump(mode="json") for item in evidence]})
        # 证据快照先落库再生成：生成失败时查询对象仍保留与 SSE 一致的证据，
        # 避免出现"事件里有证据、查询结果里没有"的不一致。
        await self.store.query_cas(
            writer.query_id,
            lease_token=writer.lease_token,
            revision=writer.revision,
            patch={
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "facts": facts.model_dump(mode="json"),
                "path_status": path_status.model_dump(mode="json"),
            },
        )
        writer.revision += 1
        await writer.emit("stage", {"name": "generate"})

        if not self.answer_model.configured:
            await self._fail(writer, "MODEL_SERVICE_UNAVAILABLE", "回答模型服务未配置（LLM_BASE_URL/LLM_MODEL）")
            return
        try:
            self.answer_model.verify_manifest()
        except Exception as exc:
            await self._fail(writer, "MODEL_DEPLOYMENT_UNVERIFIED", str(exc))
            return

        try:
            selected, budget_warnings = fit_context(
                evidence,
                counter=self.counter,
                max_input_tokens=config.models.online_max_input_tokens,
                max_evidence_chars=config.retrieval.evidence_max_chars,
                facts=facts,
                question=question,
            )
        except ContextOverflow as exc:
            await self._fail(writer, "INPUT_TOO_LONG", str(exc))
            return
        for warning in budget_warnings:
            await writer.emit("warning", {"code": warning.upper(), "message": "上下文已按预算裁剪冗余证据"})

        messages = build_messages(facts, selected, question)
        answer_parts: list[str] = []
        buffer: list[str] = []
        last_flush = time.monotonic()
        emitted_any = False
        terminal = False

        async def flush() -> None:
            nonlocal last_flush
            if not buffer:
                return
            text = "".join(buffer)
            buffer.clear()
            answer_parts.append(text)
            await writer.emit("delta", {"text": text}, patch={"answer": "".join(answer_parts)})
            last_flush = time.monotonic()

        try:
            async for delta in self.answer_model.stream(
                messages,
                max_tokens=config.models.online_max_output_tokens,
                temperature=config.models.online_temperature,
                timeout_seconds=config.timeouts.generation_seconds,
            ):
                emitted_any = True
                buffer.append(delta)
                if len("".join(buffer)) >= 50 or (time.monotonic() - last_flush) >= 0.1:
                    await flush()
                if await self._cancel_requested(writer.query_id):
                    await flush()
                    await writer.finish(
                        QueryState.CANCELLED,
                        {
                            "answer": "".join(answer_parts),
                            "behavior": decision.behavior.value,
                            "citation_validation": CitationValidation.PENDING.value,
                            "warnings": decision.warnings + ["CANCELLED"],
                            "evidence": [item.model_dump(mode="json") for item in evidence],
                            "reasoning_code": decision.reasoning_code.value,
                            "facts": query["facts"],
                            "path_status": path_status.model_dump(mode="json"),
                        },
                        {"status": QueryState.CANCELLED.value},
                    )
                    terminal = True
                    break
            if terminal:
                return
            await flush()
        except Exception as exc:
            await flush()
            if emitted_any:
                status = QueryState.INTERRUPTED
                await writer.finish(
                    status,
                    {
                        "answer": "".join(answer_parts),
                        "behavior": decision.behavior.value,
                        "citation_validation": CitationValidation.PENDING.value,
                        "warnings": decision.warnings + ["GENERATION_INTERRUPTED"],
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                        "reasoning_code": decision.reasoning_code.value,
                        "facts": query["facts"],
                        "path_status": path_status.model_dump(mode="json"),
                    },
                    {
                        "code": "GENERATION_INTERRUPTED",
                        "message": f"生成中断：{exc}",
                        "status": status.value,
                    },
                )
                await self._project_message(query, "".join(answer_parts), status, facts)
            else:
                await self._fail(writer, "MODEL_SERVICE_UNAVAILABLE", f"回答模型调用失败：{exc}")
            return

        answer = "".join(answer_parts)
        validation, invalid = validate_citations(answer, [item.evidence_id for item in evidence])
        warnings = list(decision.warnings)
        if validation == CitationValidation.FAILED:
            warnings.append("citation_validation_failed" if invalid else "citation_missing")
            if invalid:
                await writer.emit(
                    "warning",
                    {
                        "code": "CITATION_UNVERIFIED",
                        "message": "回答引用了不存在的证据：" + ", ".join(sorted(set(invalid))),
                    },
                )
        status = QueryState.COMPLETED_WITH_WARNINGS if warnings else QueryState.COMPLETED
        await self._finalize(
            writer,
            query,
            status=status,
            behavior=decision.behavior,
            reasoning_code=decision.reasoning_code,
            answer=answer,
            evidence=evidence,
            validation=validation,
            warnings=warnings,
            model_id=self.settings.env.llm_model,
            artifact_sha=self.answer_model.artifact_sha256(),
            path_status=path_status,
        )

    async def _cancel_requested(self, query_id: str) -> bool:
        doc = await self.store.get_query(query_id)
        return bool(doc and doc.get("cancel_requested"))

    # ---------- 收尾 ----------
    async def _short_circuit(
        self,
        writer: EventWriter,
        query: dict[str, Any],
        facts: Facts,
        behavior: Behavior,
        reasoning_code: ReasoningCode,
        text: str,
        path_status: PathStatus,
        warnings: Sequence[str] = (),
    ) -> None:
        await writer.emit("evidence", {"items": []})
        await writer.emit("delta", {"text": text}, patch={"answer": text})
        status = QueryState.COMPLETED_WITH_WARNINGS if warnings else QueryState.COMPLETED
        await self._finalize(
            writer,
            query,
            status=status,
            behavior=behavior,
            reasoning_code=reasoning_code,
            answer=text,
            evidence=[],
            validation=CitationValidation.NOT_APPLICABLE,
            warnings=list(warnings),
            model_id=None,
            artifact_sha=None,
            path_status=path_status,
        )

    async def _finalize(
        self,
        writer: EventWriter,
        query: dict[str, Any],
        *,
        status: QueryState,
        behavior: Behavior,
        reasoning_code: ReasoningCode,
        answer: str,
        evidence: Sequence[Evidence],
        validation: CitationValidation,
        warnings: Sequence[str],
        model_id: str | None,
        artifact_sha: str | None,
        path_status: PathStatus,
    ) -> None:
        visible = await self.visibility.filter_visible_versions(
            [item.version_id for item in evidence if item.version_id],
            session_id=uuid.UUID(query["session_id"]),
        )
        evidence_list = [item for item in evidence if item.version_id is None or item.version_id in visible]
        await writer.finish(
            status,
            {
                "behavior": behavior.value,
                "reasoning_code": reasoning_code.value,
                "answer": answer,
                "citation_validation": validation.value,
                "warnings": list(warnings),
                "evidence": [item.model_dump(mode="json") for item in evidence_list],
                "generation_model_id": model_id,
                "generation_artifact_sha256": artifact_sha,
                "facts": query["facts"],
                "path_status": path_status.model_dump(mode="json"),
            },
            {
                "status": status.value,
                "behavior": behavior.value,
                "citation_validation": validation.value,
                "last_event_seq": writer.last_seq + 1,
            },
        )
        await self._project_message(query, answer, status, Facts.model_validate(query["facts"]))

    async def _project_message(
        self, query: dict[str, Any], answer: str, status: QueryState, facts: Facts
    ) -> None:
        """assistant 消息投影按 query_id+role 幂等写入，不重复生成答案。"""
        await self.store.upsert_message(
            {
                "message_id": str(uuid.uuid4()),
                "query_id": query["query_id"],
                "session_id": query["session_id"],
                "role": "assistant",
                "content": answer,
                "status": "partial" if status in {QueryState.INTERRUPTED, QueryState.CANCELLED} else "complete",
                "facts": facts.model_dump(mode="json"),
                "created_at": utcnow(),
            }
        )
        await self.store.touch_session(query["session_id"], query["query_id"])

    async def _fail(self, writer: EventWriter, code: str, message: str) -> None:
        doc = await self.store.get_query(writer.query_id)
        if doc is None:
            return
        try:
            await writer.finish(
                QueryState.FAILED,
                {
                    "answer": doc.get("answer", ""),
                    "behavior": doc.get("behavior"),
                    "citation_validation": CitationValidation.PENDING.value,
                    "warnings": list(doc.get("warnings") or []) + [code],
                    "evidence": doc.get("evidence") or [],
                    "reasoning_code": doc.get("reasoning_code"),
                    "facts": doc.get("facts"),
                    "path_status": doc.get("path_status"),
                },
                {"code": code, "message": message, "status": QueryState.FAILED.value},
            )
        except LeaseLost:
            return
