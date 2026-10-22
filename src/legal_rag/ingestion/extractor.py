"""自动抽取：元数据、条款、引用与争议主题。

原则：
- 规则优先，缺失值保持 null，不靠模型置信度把“未知效力”变成“现行有效”；
- 引用只有在“标题可唯一解析 + 条号存在 + 原文有短引”时才建立 CITES，
  否则写入 graph_unresolved，绝不连到“看起来最新”的条款；
- 辅助模型仅在已配置时补充空字段，且补出的值必须能在原文中找到。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Sequence

from ..domain.ids import assertion_id, entity_id, sha256_hex

_DOC_NUMBER = re.compile(r"[\u4e00-\u9fa5]{2,12}〔\d{4}〕\d+号")
_CASE_NUMBER = re.compile(r"[（(]\d{4}[）)][\u4e00-\u9fa5\d]{2,12}?第?\d+号")
_ISSUED = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日(?:公布|通过|修订|修正)")
_EFFECTIVE = re.compile(r"自(\d{4})年(\d{1,2})月(\d{1,2})日起(?:施行|实施)")
_CITATION = re.compile(r"《([^》]{2,40})》\s*(?:第([一二三四五六七八九十百千零〇0-9]+)条)?")
_ISSUERS = (
    "全国人民代表大会常务委员会",
    "全国人民代表大会",
    "最高人民法院",
    "最高人民检察院",
    "人力资源社会保障部",
    "国务院",
)
_ISSUE_LABELS = {
    "wage_arrears": "工资拖欠",
    "termination": "劳动合同解除",
    "economic_compensation": "经济补偿",
    "other": "其他劳动用工",
}


def _safe_date(year: str, month: str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def provision_entity_id(version_id: str, article: str) -> str:
    return entity_id("Provision", f"{version_id}|{article}", version_id)


def case_entity_id(case_identity_key: str, version_id: str) -> str:
    return entity_id("Case", case_identity_key, version_id)


def issue_entity_id(scenario: str) -> str:
    return entity_id("Issue", scenario)


def organization_entity_id(name: str) -> str:
    return entity_id("Organization", name)


@dataclass(slots=True)
class ExtractedMetadata:
    jurisdiction: str = "CN-national"
    issued_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    effectivity: str = "unknown"
    effectivity_source_chunk_id: str | None = None
    case_number: str | None = None
    issuer: str | None = None
    document_number: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "issued_date": self.issued_date,
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "effectivity": self.effectivity,
            "effectivity_source_chunk_id": self.effectivity_source_chunk_id,
            "case_number": self.case_number,
            "issuer": self.issuer,
            "document_number": self.document_number,
        }


@dataclass(slots=True)
class CitationCandidate:
    title: str
    article: str | None
    raw_text: str
    chunk_id: str


@dataclass(slots=True)
class ExtractionResult:
    metadata: ExtractedMetadata
    provisions: list[dict[str, Any]] = field(default_factory=list)
    citations: list[CitationCandidate] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def extract_document(
    *,
    title: str,
    doc_type: str,
    text: str,
    chunks: Sequence[dict[str, Any]],
    version_id: str,
    document_id: str,
) -> ExtractionResult:
    head = text[:3000]
    metadata = ExtractedMetadata()

    document_number = _DOC_NUMBER.search(head)
    if document_number:
        metadata.document_number = document_number.group(0)
    case_number = _CASE_NUMBER.search(head)
    if case_number:
        metadata.case_number = case_number.group(0)
    issued = _ISSUED.search(head)
    if issued:
        metadata.issued_date = _safe_date(*issued.groups())
    effective = _EFFECTIVE.search(head)
    if effective:
        metadata.effective_from = _safe_date(*effective.groups())
    for issuer in _ISSUERS:
        if issuer in head:
            metadata.issuer = issuer
            break
    # 只有原文明确写“现行有效”才标记，且必须能定位到具体切片
    if "现行有效" in head:
        for chunk in chunks:
            if "现行有效" in chunk["content"]:
                metadata.effectivity = "known_current"
                metadata.effectivity_source_chunk_id = chunk["chunk_id"]
                break

    provisions: list[dict[str, Any]] = []
    citations: list[CitationCandidate] = []
    issues: set[str] = set()
    for chunk in chunks:
        issues.update(tag for tag in chunk.get("scenario_tags", []) if tag != "other")
        article = (chunk.get("location") or {}).get("article")
        if article and doc_type in {"law", "interpretation", "guidance"}:
            provisions.append(
                {
                    "entity_id": provision_entity_id(version_id, article),
                    "article": article,
                    "label": f"{title}{article}",
                    "chunk_id": chunk["chunk_id"],
                }
            )
        if doc_type == "case":
            for match in _CITATION.finditer(chunk["content"]):
                citations.append(
                    CitationCandidate(
                        title=match.group(1),
                        article=f"第{match.group(2)}条" if match.group(2) else None,
                        raw_text=match.group(0),
                        chunk_id=chunk["chunk_id"],
                    )
                )

    if not metadata.issued_date or not metadata.effective_from:
        result_warning = "metadata_missing"
    else:
        result_warning = ""
    return ExtractionResult(
        metadata=metadata,
        provisions=provisions,
        citations=citations,
        issues=sorted(issues),
        warnings=[result_warning] if result_warning else [],
    )


@dataclass(slots=True)
class Resolution:
    resolved: list[dict[str, Any]]
    unresolved: list[dict[str, Any]]


def resolve_citations(
    citations: Sequence[CitationCandidate],
    *,
    source_entity: str,
    version_id: str,
    known_documents: Sequence[dict[str, Any]],
) -> Resolution:
    """把“《法规名》第N条”解析为确定 CITES；无法唯一定位时保留待解析记录。"""
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for citation in citations:
        normalized = citation.title.replace("中华人民共和国", "").strip()
        matches = [
            doc
            for doc in known_documents
            if normalized and (normalized in doc["title"].replace("中华人民共和国", "") or doc["title"].replace("中华人民共和国", "") in normalized)
        ]
        if len(matches) != 1:
            unresolved.append(
                {
                    "raw_text": citation.raw_text,
                    "cited_title": citation.title,
                    "article": citation.article,
                    "reason": "ambiguous_document" if matches else "document_not_found",
                    "citation_hash": sha256_hex(citation.raw_text, citation.chunk_id),
                    "source_chunk_id": citation.chunk_id,
                }
            )
            continue
        doc = matches[0]
        if not citation.article:
            unresolved.append(
                {
                    "raw_text": citation.raw_text,
                    "cited_title": citation.title,
                    "article": None,
                    "reason": "article_missing",
                    "citation_hash": sha256_hex(citation.raw_text, citation.chunk_id),
                    "source_chunk_id": citation.chunk_id,
                }
            )
            continue
        target_version = doc.get("version_id")
        if not target_version:
            unresolved.append(
                {
                    "raw_text": citation.raw_text,
                    "cited_title": citation.title,
                    "article": citation.article,
                    "reason": "version_not_published",
                    "citation_hash": sha256_hex(citation.raw_text, citation.chunk_id),
                    "source_chunk_id": citation.chunk_id,
                }
            )
            continue
        resolved.append(
            {
                "source_id": source_entity,
                "target_id": provision_entity_id(target_version, citation.article),
                "support_chunk_id": citation.chunk_id,
                "raw_text": citation.raw_text,
                "status": "source_supported",
                "assertion_id": assertion_id(source_entity, "CITES", citation.article, citation.chunk_id),
            }
        )
    return Resolution(resolved=resolved, unresolved=unresolved)
