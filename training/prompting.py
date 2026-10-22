"""训练用提示词构造：与应用侧共享同一 system 提示与证据格式，避免训练/推理漂移。"""

from __future__ import annotations

from legal_rag.retrieval.policy import build_messages
from legal_rag.domain.models import Facts, SampleEvidence


def build_training_messages(sample: dict) -> list[dict[str, str]]:
    # 离线证据使用 SampleEvidence，不创建虚构的在线查询身份。
    facts = Facts(scenario=sample["scenario"], location=None, relevant_date=None, known_facts=[], missing_fields=[])
    evidence = [SampleEvidence.model_validate(item) for item in sample.get("evidence", [])]
    messages = build_messages(facts, evidence, sample["question"])
    return messages + [{"role": "assistant", "content": sample["target_answer"]}]
