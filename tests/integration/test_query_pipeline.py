"""咨询流水线集成测试（T06、T12、T14、T18、T26、T31、T33）。

生成端使用显式替身（StubAnswerModel），仅用于验证链路与失败语义；
真实微调模型接入见 docs/05 与 scripts/verify_model_connection.py。
"""

from __future__ import annotations

import uuid

import pytest

from legal_rag.domain.enums import Behavior, CitationValidation, QueryState, ReasoningCode

from .helpers import LAW_MD, claim_and_import, create_query, create_session, run_query, upload

pytestmark = pytest.mark.usefixtures("require_deps")

QUESTION_WITH_FACTS = "公司拖欠我 2025 年 3 月工资 9000 元，月薪 9000 元，我可以主张什么？"
QUESTION_VAGUE = "公司拖欠工资怎么办？"


class StubAnswerModel:
    """明确标识的生成替身：不连接任何外部模型，只验证流式与终态逻辑。"""

    def __init__(self, pieces, *, fail_after: int | None = None) -> None:
        self._pieces = pieces
        self._fail_after = fail_after
        self.calls = 0
        self.configured = True

    def verify_manifest(self):
        return object()

    def artifact_sha256(self):
        return "f" * 64

    async def stream(self, messages, **kwargs):
        self.calls += 1
        for index, piece in enumerate(self._pieces):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("注入的流式中断")
            yield piece
        if self._fail_after is not None and self._fail_after <= len(self._pieces):
            raise RuntimeError("注入的流式中断")


async def import_law(client, runtime) -> dict:
    created = (await upload(client, filename="law.md", content=LAW_MD, title="测试用工资支付规则")).json()
    task = await claim_and_import(runtime, created["task_id"])
    assert task["status"] in {"succeeded", "succeeded_with_warnings"}, task
    return created


async def test_vague_question_clarifies_without_model(client, runtime):
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_VAGUE)
    doc = await run_query(runtime, created["query_id"])
    assert doc["status"] == QueryState.COMPLETED.value
    assert doc["behavior"] == Behavior.CLARIFY.value
    assert doc["reasoning_code"] == ReasoningCode.FACTS_MISSING.value
    assert doc["generation_model_id"] is None
    assert doc["path_status"] == {
        "ordinary": "skipped",
        "hyde": "skipped",
        "graph": "skipped",
        "web": "skipped",
        "rerank": "skipped",
    }
    assert doc["facts"]["missing_fields"]
    events = [item["event"] for item in doc["event_log"]]
    assert events[:2] == ["started", "stage"]
    assert events[-1] == "done"
    assert events.count("evidence") == 1
    assert [item["seq"] for item in doc["event_log"]] == list(range(1, len(events) + 1))


async def test_empty_knowledge_is_insufficient(client, runtime):
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_WITH_FACTS)
    doc = await run_query(runtime, created["query_id"])
    assert doc["status"] == QueryState.COMPLETED.value
    assert doc["behavior"] == Behavior.INSUFFICIENT.value
    assert doc["reasoning_code"] == ReasoningCode.NO_RELEVANT_EVIDENCE.value
    assert doc["evidence"] == []
    assert doc["citation_validation"] == CitationValidation.NOT_APPLICABLE.value


async def test_answer_uses_evidence_and_records_model_identity(client, runtime):
    await import_law(client, runtime)
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_WITH_FACTS)
    stub = StubAnswerModel(["摘要：用人单位应", "当按月支付工资。[E1]", "\n\n依据与条件：见 [E1]。"])
    doc = await run_query(runtime, created["query_id"], answer_model=stub)
    assert doc["status"] in {QueryState.COMPLETED.value, QueryState.COMPLETED_WITH_WARNINGS.value}
    assert doc["behavior"] == Behavior.ANSWER.value
    assert doc["reasoning_code"] == ReasoningCode.EVIDENCE_SUFFICIENT.value
    assert doc["citation_validation"] == CitationValidation.PASSED.value
    assert doc["generation_model_id"]
    assert doc["generation_artifact_sha256"] == "f" * 64
    assert stub.calls == 1
    assert doc["evidence"]
    assert doc["evidence"][0]["query_id"] == created["query_id"]
    assert doc["evidence"][0]["evidence_id"] == "E1"
    assert doc["answer"].endswith("[E1]。")
    # 事件顺序：evidence 必须在首个 delta 之前
    names = [item["event"] for item in doc["event_log"]]
    assert names.index("evidence") < names.index("delta")
    assert names.count("evidence") == 1
    assert [item["seq"] for item in doc["event_log"]] == list(range(1, len(names) + 1))


async def test_fake_citation_is_marked_failed(client, runtime):
    await import_law(client, runtime)
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_WITH_FACTS)
    stub = StubAnswerModel(["摘要：依据不存在的来源 [E999] 得出结论。"])
    doc = await run_query(runtime, created["query_id"], answer_model=stub)
    assert doc["citation_validation"] == CitationValidation.FAILED.value
    assert "citation_validation_failed" in doc["warnings"]
    assert all(item["evidence_id"] != "E999" for item in doc["evidence"])
    assert doc["status"] == QueryState.COMPLETED_WITH_WARNINGS.value


async def test_stream_interruption_keeps_partial_and_marks_interrupted(client, runtime):
    await import_law(client, runtime)
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_WITH_FACTS)
    stub = StubAnswerModel(["摘要：已经开始生成"], fail_after=1)
    doc = await run_query(runtime, created["query_id"], answer_model=stub)
    assert doc["status"] == QueryState.INTERRUPTED.value
    assert doc["answer"] == "摘要：已经开始生成"
    assert "GENERATION_INTERRUPTED" in doc["warnings"]
    assert doc["event_log"][-1]["event"] == "failed"
    assert doc["event_log"][-1]["payload"]["status"] == QueryState.INTERRUPTED.value
    messages = await runtime.store.list_messages(session_id)
    assistant = [item for item in messages if item["role"] == "assistant"]
    assert assistant and assistant[-1]["status"] == "partial"


async def test_unconfigured_model_fails_instead_of_using_aux(client, runtime, monkeypatch):
    await import_law(client, runtime)
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_WITH_FACTS)
    # 场景一：回答模型完全未配置
    monkeypatch.setattr(runtime.settings.env, "llm_base_url", "")
    doc = await run_query(runtime, created["query_id"])
    assert doc["status"] == QueryState.FAILED.value
    assert "MODEL_SERVICE_UNAVAILABLE" in doc["warnings"]
    assert doc["generation_model_id"] is None
    assert doc["event_log"][-1]["event"] == "failed"

    # 场景二：已配置且清单通过，但推理服务不可用（流在首 token 前失败）
    from legal_rag.adapters.llm import ModelServiceUnavailable

    class DownModel:
        configured = True

        def verify_manifest(self):
            return object()

        def artifact_sha256(self):
            return "f" * 64

        async def stream(self, messages, **kwargs):
            raise ModelServiceUnavailable("连接被拒绝")
            yield ""  # pragma: no cover

    second = await create_query(client, session_id, QUESTION_WITH_FACTS)
    doc2 = await run_query(runtime, second["query_id"], answer_model=DownModel())
    assert doc2["status"] == QueryState.FAILED.value
    assert "MODEL_SERVICE_UNAVAILABLE" in doc2["warnings"]
    assert doc2["generation_model_id"] is None
    # 辅助模型即使可用也不得接管最终回答：事件里不出现其模型名
    assert all("qwen-flash" not in (item.get("payload", {}).get("message") or "") for item in doc2["event_log"])


async def test_evidence_isolated_between_queries_and_withdrawn_sources(client, runtime):
    law = await import_law(client, runtime)
    session_id = await create_session(client)
    first = await create_query(client, session_id, QUESTION_WITH_FACTS)
    doc1 = await run_query(runtime, first["query_id"], answer_model=StubAnswerModel(["结论 [E1]"]))
    second = await create_query(client, session_id, QUESTION_WITH_FACTS)
    doc2 = await run_query(runtime, second["query_id"], answer_model=StubAnswerModel(["结论 [E1]"]))
    assert doc1["evidence"][0]["query_id"] == first["query_id"]
    assert doc2["evidence"][0]["query_id"] == second["query_id"]
    # E1 只在各自 query 内有效：跨查询访问同一编号取到的是本轮快照
    ev1 = (await client.get(f"/api/queries/{first['query_id']}/evidence/E1")).json()
    ev2 = (await client.get(f"/api/queries/{second['query_id']}/evidence/E1")).json()
    assert ev1["query_id"] == first["query_id"]
    assert ev2["query_id"] == second["query_id"]
    missing = await client.get(f"/api/queries/{first['query_id']}/evidence/E9")
    assert missing.status_code == 404

    # 撤销资料后：新查询不再召回，旧证据详情返回 410
    await client.delete(f"/api/documents/{law['document_id']}")
    third = await create_query(client, session_id, QUESTION_WITH_FACTS)
    doc3 = await run_query(runtime, third["query_id"])
    assert doc3["status"] == QueryState.COMPLETED.value
    assert doc3["evidence"] == []
    withdrawn = await client.get(f"/api/queries/{first['query_id']}/evidence/E1")
    assert withdrawn.status_code == 410


async def test_facts_snapshot_is_reusable_and_carries_reasoning(client, runtime):
    session_id = await create_session(client)
    created = await create_query(client, session_id, QUESTION_VAGUE)
    doc = await run_query(runtime, created["query_id"])
    facts = doc["facts"]
    assert set(facts) == {"scenario", "location", "relevant_date", "known_facts", "missing_fields"}
    assert facts["scenario"] == "wage_arrears"
    assert facts["known_facts"] == []  # 未提供事实时不得把模型猜测写成事实
    messages = await runtime.store.list_messages(session_id)
    assert all(item["facts"] == facts for item in messages if item["query_id"] == created["query_id"])
