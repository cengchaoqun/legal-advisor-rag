"""事实抽取、澄清判定与引用校验单元测试（对应 T18、T27、T31 的策略部分）。"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from legal_rag.adapters.llm import TokenCounter, estimate_tokens
from legal_rag.domain.enums import Behavior, CitationValidation, QualityFlag, ReasoningCode, Scenario
from legal_rag.domain.models import Evidence, Facts, LegalMetadata, Location
from legal_rag.retrieval.policy import (
    ContextOverflow,
    build_messages,
    decide,
    extract_facts,
    fit_context,
    should_clarify,
    validate_citations,
)


def make_evidence(eid: str, text: str = "工资应当按月支付。", effectivity: str = "known_current") -> Evidence:
    return Evidence(
        evidence_id=eid,
        query_id=uuid.uuid4(),
        source_id="a" * 64,
        source_type="local",
        chunk_id="b" * 64,
        version_id="c" * 64,
        title="演示资料",
        text=text,
        url=None,
        location=Location(page=1, heading="第一章", article="第九条", start_char=0, end_char=len(text)),
        metadata=LegalMetadata(
            jurisdiction="CN-national",
            issued_date=date(2020, 1, 1),
            effective_from=date(2020, 1, 1),
            effective_to=None,
            effectivity=effectivity,
            effectivity_source_chunk_id="b" * 64,
            case_number=None,
            issuer=None,
            document_number=None,
        ),
        quality_flags=[],
    )


def test_extract_facts_detects_scenario_and_clarify():
    facts = extract_facts("公司拖欠工资怎么办？", scenario=None, location=None, relevant_date=None)
    assert facts.scenario == Scenario.WAGE_ARREARS
    assert facts.missing_fields
    assert should_clarify(facts) is True


def test_facts_with_concrete_details_do_not_clarify():
    facts = extract_facts(
        "公司拖欠我 2025 年 3 月工资 9000 元，月薪 9000 元，该怎么主张？",
        scenario=None,
        location="北京",
        relevant_date=date(2025, 3, 31),
    )
    assert facts.known_facts
    assert should_clarify(facts) is False
    assert facts.location == "北京"


def test_no_evidence_is_insufficient():
    facts = extract_facts("拖欠 3 个月工资共 27000 元怎么办？", scenario=None, location=None, relevant_date=None)
    decision = decide(facts=facts, evidence=[], question="拖欠 3 个月工资共 27000 元怎么办？")
    assert decision.behavior == Behavior.INSUFFICIENT
    assert decision.reasoning_code == ReasoningCode.NO_RELEVANT_EVIDENCE


def test_validity_question_with_unknown_effectivity_is_uncertain():
    facts = extract_facts("这份规定现在有效吗？拖欠 3 个月工资 27000 元。", scenario=None, location=None, relevant_date=None)
    evidence = [make_evidence("E1", effectivity="unknown")]
    decision = decide(facts=facts, evidence=evidence, question="这份规定现在有效吗？")
    assert decision.behavior == Behavior.INSUFFICIENT
    assert decision.reasoning_code == ReasoningCode.APPLICABILITY_UNCERTAIN


def test_multiple_documents_without_effectivity_is_not_version_conflict():
    """不同资料各有 version_id，不应被判为版本冲突（真实语料回归）。"""
    facts = extract_facts("拖欠 3 个月工资共 27000 元怎么办？", scenario=None, location=None, relevant_date=None)
    first = make_evidence("E1", effectivity="unknown").model_copy(update={"version_id": "c" * 64, "title": "资料甲"})
    second = make_evidence("E2", effectivity="unknown").model_copy(update={"version_id": "d" * 64, "title": "资料乙"})
    decision = decide(
        facts=facts,
        evidence=[first, second],
        question="拖欠 3 个月工资共 27000 元怎么办？",
        document_ids={"E1": "doc-1", "E2": "doc-2"},
    )
    assert decision.behavior == Behavior.ANSWER
    assert "version_conflict" not in decision.warnings


def test_same_document_two_versions_without_effectivity_is_conflict():
    """同一份资料出现两个版本且都无生效信息时，才判为版本冲突。"""
    facts = extract_facts("拖欠 3 个月工资共 27000 元怎么办？", scenario=None, location=None, relevant_date=None)
    first = make_evidence("E1", effectivity="unknown").model_copy(update={"version_id": "c" * 64})
    second = make_evidence("E2", effectivity="unknown").model_copy(update={"version_id": "d" * 64})
    decision = decide(
        facts=facts,
        evidence=[first, second],
        question="拖欠 3 个月工资共 27000 元怎么办？",
        document_ids={"E1": "doc-1", "E2": "doc-1"},
    )
    assert decision.behavior == Behavior.INSUFFICIENT
    assert decision.reasoning_code == ReasoningCode.EVIDENCE_CONFLICT


def test_normal_evidence_answers():
    facts = extract_facts("拖欠 3 个月工资共 27000 元怎么办？", scenario=None, location=None, relevant_date=None)
    decision = decide(facts=facts, evidence=[make_evidence("E1")], question="拖欠 3 个月工资共 27000 元怎么办？")
    assert decision.behavior == Behavior.ANSWER
    assert decision.reasoning_code == ReasoningCode.EVIDENCE_SUFFICIENT


def test_citation_validation_detects_missing_and_fake_ids():
    status, invalid = validate_citations("结论 [E1]，另见 [E999]。", ["E1"])
    assert status == CitationValidation.FAILED
    assert invalid == ["E999"]

    status, invalid = validate_citations("结论 [E1]。", ["E1"])
    assert status == CitationValidation.PASSED and invalid == []

    status, invalid = validate_citations("没有引用。", ["E1"])
    assert status == CitationValidation.FAILED and invalid == []


class FixedCostCounter:
    """固定成本的 tokenizer 替身：按字符数折算，便于精确断言裁剪行为。"""

    mode = "tokenizer"

    def count_messages(self, messages):
        return sum(len(message.get("content", "")) for message in messages) // 10


def test_fit_context_trims_low_rank_evidence_but_never_truncates_text():
    facts = extract_facts("拖欠 3 个月工资 27000 元怎么办？", scenario=None, location=None, relevant_date=None)
    evidence = [make_evidence(f"E{i}", text="证据正文" * 60) for i in range(1, 4)]
    counter = FixedCostCounter()
    base = counter.count_messages(build_messages(facts, [], "拖欠 3 个月工资 27000 元怎么办？"))
    selected, warnings = fit_context(
        evidence,
        counter=counter,
        max_input_tokens=base + 30,
        max_evidence_chars=100000,
        facts=facts,
        question="拖欠 3 个月工资 27000 元怎么办？",
    )
    assert len(selected) == 1
    assert selected[0].evidence_id == "E1"
    assert all(item.text.endswith("证据正文") for item in selected)
    assert "context_trimmed" in warnings
    assert "token_budget_estimated" not in warnings


def test_fit_context_raises_when_question_alone_overflows():
    facts = extract_facts("欠薪", scenario=None, location=None, relevant_date=None)
    with pytest.raises(ContextOverflow):
        fit_context(
            [],
            counter=FixedCostCounter(),
            max_input_tokens=5,
            max_evidence_chars=2400,
            facts=facts,
            question="欠薪" * 50,
        )


def test_messages_keep_four_sections_and_untrusted_evidence():
    facts = Facts(
        scenario=Scenario.WAGE_ARREARS,
        location=None,
        relevant_date=None,
        known_facts=["金额线索"],
        missing_fields=[],
    )
    messages = build_messages(facts, [make_evidence("E1")], "拖欠工资怎么办？")
    assert messages[0]["role"] == "system"
    for keyword in ("范围", "用户事实", "编号证据", "输出规则", "不可信"):
        assert keyword in messages[0]["content"]
    assert "[E1]" in messages[1]["content"]


def test_estimate_tokens_is_conservative_for_cjk():
    assert estimate_tokens("工资拖欠") >= 4
    assert estimate_tokens("abcdefghij") < 10
