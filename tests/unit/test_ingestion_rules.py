"""切分与抽取单元测试（对应 T08、T19 的规则部分）。"""

from __future__ import annotations

from legal_rag.adapters.parser import ParsedDocument, PageText
from legal_rag.ingestion.extractor import (
    CitationCandidate,
    case_entity_id,
    extract_document,
    provision_entity_id,
    resolve_citations,
)
from legal_rag.ingestion.splitter import split_document, to_chunk_document

LAW_TEXT = """中华人民共和国劳动合同法

第一章 总则

第一条 为了完善劳动合同制度，明确劳动合同双方当事人的权利和义务，保护劳动者的合法权益，构建和发展和谐稳定的劳动关系，制定本法。

第八十五条 用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿；劳动报酬低于当地最低工资标准的，应当支付其差额部分；逾期不支付的，责令用人单位按应付金额百分之五十以上百分之一百以下的标准向劳动者加付赔偿金：（一）未按照劳动合同的约定或者国家规定及时足额支付劳动者劳动报酬的；（二）低于当地最低工资标准支付劳动者工资的；（三）安排加班不支付加班费的；（四）解除或者终止劳动合同，未依照本法规定给予经济补偿的。
"""


def test_split_keeps_article_and_is_deterministic():
    parsed = ParsedDocument(text=LAW_TEXT, pages=[PageText(page=1, text=LAW_TEXT)])
    drafts = split_document(parsed, version_id="v1", target_min=400, target_max=800, soft_max=1000, overlap=80)
    assert [draft.article for draft in drafts if draft.article] == ["第一条", "第八十五条"]
    article_draft = next(draft for draft in drafts if draft.article == "第一条")
    first = to_chunk_document(article_draft, version_id="v1", document_id="d1", embedding_revision="r1", quality_flags=[])
    again = to_chunk_document(article_draft, version_id="v1", document_id="d1", embedding_revision="r1", quality_flags=[])
    assert first["chunk_id"] == again["chunk_id"]
    assert article_draft.start_char <= article_draft.end_char
    assert first["location"]["article"] == "第一条"


def test_long_article_splits_with_overlap():
    long_article = "第九条 " + "工资应当按月支付，不得克扣或者无故拖欠。" * 80
    parsed = ParsedDocument(text=long_article)
    drafts = split_document(parsed, version_id="v2", target_min=400, target_max=800, soft_max=1000, overlap=80)
    assert len(drafts) > 1
    assert all(draft.article == "第九条" for draft in drafts)
    assert [draft.ordinal for draft in drafts] == list(range(len(drafts)))


def test_split_tags_scenarios():
    parsed = ParsedDocument(text="第一条 用人单位拖欠工资的，应当支付经济补偿。")
    drafts = split_document(parsed, version_id="v3", target_min=400, target_max=800, soft_max=1000, overlap=80)
    assert "wage_arrears" in drafts[0].scenario_tags


def test_extract_metadata_rules_and_provisions():
    text = "中华人民共和国劳动合同法\n（2012年12月28日修订）\n自2013年7月1日起施行\n劳部发〔1994〕489号\n第八十五条 用人单位拖欠劳动报酬的，应当支付赔偿金。"
    parsed = ParsedDocument(text=text)
    drafts = split_document(parsed, version_id="v4", target_min=400, target_max=800, soft_max=1000, overlap=80)
    chunks = [
        to_chunk_document(draft, version_id="v4", document_id="d4", embedding_revision="r", quality_flags=[])
        for draft in drafts
    ]
    result = extract_document(
        title="中华人民共和国劳动合同法",
        doc_type="law",
        text=text,
        chunks=chunks,
        version_id="v4",
        document_id="d4",
    )
    assert result.metadata.issued_date is not None
    assert result.metadata.effective_from is not None
    assert result.metadata.document_number == "劳部发〔1994〕489号"
    assert result.metadata.effectivity == "unknown"  # 未写“现行有效”不得推断
    assert result.provisions
    assert result.provisions[0]["entity_id"] == provision_entity_id("v4", "第八十五条")


def test_citation_resolution_unique_ambiguous_and_missing():
    known = [
        {"document_id": "d1", "title": "中华人民共和国劳动合同法", "version_id": "v-law"},
        {"document_id": "d2", "title": "中华人民共和国劳动法", "version_id": "v-law2"},
    ]
    citation = CitationCandidate(
        title="劳动合同法", article="第八十五条", raw_text="《劳动合同法》第八十五条", chunk_id="chunk-1"
    )
    resolution = resolve_citations([citation], source_entity="case-1", version_id="v-case", known_documents=known)
    assert len(resolution.resolved) == 1
    assert resolution.resolved[0]["target_id"] == provision_entity_id("v-law", "第八十五条")
    assert resolution.unresolved == []

    # 同名不同版本/修订版：无法唯一定位，必须保留待解析记录而不是自动连边
    ambiguous_known = [
        {"document_id": "d3", "title": "工资支付暂行规定", "version_id": "v-a"},
        {"document_id": "d4", "title": "工资支付暂行规定（2020修订）", "version_id": "v-b"},
    ]
    ambiguous = CitationCandidate(title="工资支付暂行规定", article="第七条", raw_text="《工资支付暂行规定》第七条", chunk_id="chunk-2")
    resolution = resolve_citations(
        [ambiguous], source_entity="case-1", version_id="v-case", known_documents=ambiguous_known
    )
    assert resolution.resolved == []
    assert resolution.unresolved[0]["reason"] == "ambiguous_document"

    missing = CitationCandidate(title="工资支付暂行规定", article="第七条", raw_text="《工资支付暂行规定》第七条", chunk_id="c3")
    resolution = resolve_citations([missing], source_entity="case-1", version_id="v-case", known_documents=known)
    assert resolution.unresolved[0]["reason"] == "document_not_found"
    assert case_entity_id("case-key", "v-case") != case_entity_id("case-key", "v-case-2")
