"""无需模型权重的回归：监督边界、数据隔离、评估分母与部署产物身份。"""
from __future__ import annotations

import copy
import importlib
import json

import pytest


class TinyTokenizer:
    eos_token_id = 999
    pad_token_id = 0
    def __init__(self):
        self.calls = []
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]
    def __call__(self, text, **kwargs):
        return {"input_ids": [10 + ord(char) for char in text]}


def sample():
    return {"sample_id": "s1", "scenario": "wage_arrears", "question": "请摘录原文", "evidence": [], "target_answer": "摘要：依据不足。\n依据与条件：没有证据。\n需补充事项：提供原文。", "expected_behavior": "insufficient", "ability": "abstention", "origin_group_id": "g1", "split": "test"}


def test_encoding_masks_only_prompt_and_rejects_truncation():
    module = importlib.import_module("training.data")
    tokenizer = TinyTokenizer()
    result = module.encode_sample(sample(), tokenizer, max_seq_length=200, max_input_tokens=20, max_output_tokens=150)
    assert result["labels"][:3] == [-100] * 3
    assert result["labels"][3:] == result["input_ids"][3:]
    assert result["labels"][-1] == tokenizer.eos_token_id
    assert tokenizer.calls[0]["enable_thinking"] is False
    with pytest.raises(module.SampleTooLong):
        module.encode_sample(sample(), tokenizer, max_seq_length=5, max_input_tokens=20, max_output_tokens=150)


def test_duplicate_source_group_cannot_cross_splits():
    module = importlib.import_module("training.data")
    first = sample()
    second = copy.deepcopy(first)
    second.update(sample_id="s2", split="train", question="另一个不同问题")
    with pytest.raises(ValueError, match="来源组"):
        module.check_split_integrity({"test": [first], "train": [second], "validation": []})


def test_missing_predictions_stay_in_frozen_test_denominator():
    module = importlib.import_module("training.evaluate")
    rows = [sample(), {**sample(), "sample_id": "s2"}]
    result = module.score_predictions(rows, [{"sample_id": "s1", "output": sample()["target_answer"], "status": "ok"}])
    assert result["samples"] == 2
    assert result["failed_samples"] == 1
    assert result["behavior_accuracy_silver"] == 0.5
    with pytest.raises(ValueError, match="重复"):
        module.score_predictions(rows, [{"sample_id": "s1", "output": "x"}] * 2)


def test_three_sections_do_not_automatically_mean_clarify():
    from training.evaluate import infer_behavior
    assert infer_behavior("摘要：原文记载如下。[E1]\n依据与条件：仅解释片段。[E1]\n需补充事项：具体适用需核实日期。", 1) == "answer"
    assert infer_behavior("摘要：需澄清。\n依据与条件：缺少解除日期。\n需补充事项：请补充日期。", 0) == "clarify"


def test_manifest_file_does_not_change_model_identity(tmp_path):
    from legal_rag.adapters.llm import directory_digest
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
    before = directory_digest(tmp_path)
    (tmp_path / "deployment_manifest.json").write_text(json.dumps({"directory_sha256": before}), encoding="utf-8")
    assert directory_digest(tmp_path) == before
    (tmp_path / "config.json").write_text('{"model_type":"tampered"}', encoding="utf-8")
    assert directory_digest(tmp_path) != before


def test_bootstrap_targets_only_quote_source_text():
    module = importlib.import_module("training.bootstrap")
    rows = module.make_source_samples({
        "group_id": "g1", "scenario": "termination", "title": "真实来源测试", "url": "https://www.court.gov.cn/demo",
        "version_id": "a" * 64, "metadata": None, "quality_flags": ["effectivity_unknown"],
        "chunks": [{"chunk_id": "b" * 64, "content": "甲与乙就解除劳动合同发生争议。法院根据双方提交的材料作出判断。", "location": {"page": None, "heading": "裁判理由", "article": None, "start_char": 100, "end_char": 136}}],
    }, "test")
    assert {item["expected_behavior"] for item in rows} >= {"answer", "insufficient", "clarify"}
    assert all("query_id" not in ev for item in rows for ev in item["evidence"])
    assert all(item["origin_group_id"] == "g1" for item in rows)


def test_export_verifier_checks_actual_weight_files(tmp_path):
    from legal_rag.adapters.llm import directory_digest
    from training.data import file_sha256
    from training.serve_model import verify_export
    # 这些字节只用于哈希回归测试，不是可加载权重，也不计作训练产物。
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"hash-fixture-only")
    manifest = {"files": {"model.safetensors": file_sha256(weights)}, "directory_sha256": directory_digest(tmp_path)}
    (tmp_path / "deployment_manifest.json").write_text(json.dumps(manifest))
    assert verify_export(tmp_path) == manifest
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="摘要"):
        verify_export(tmp_path)


async def test_answer_request_disables_qwen3_thinking():
    from types import SimpleNamespace
    import httpx
    from legal_rag.adapters.llm import AnswerModelClient
    settings = SimpleNamespace(env=SimpleNamespace(llm_base_url="http://test/v1", llm_model="test-only", llm_api_key="", answer_model_configured=lambda: True))
    seen = []
    async def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"回答"}}]}\n\ndata: [DONE]\n\n')
    client = AnswerModelClient(settings)
    client.verify_manifest = lambda: None
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        text = await client.complete([{"role": "user", "content": "测试"}], max_tokens=10, temperature=0, timeout_seconds=1)
        assert text == "回答"
        assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        await client.close()


@pytest.mark.parametrize('body', [
    'data: {"error":{"message":"sampler failed"}}\n\ndata: [DONE]\n\n',
    'data: [DONE]\n\n',
])
async def test_answer_stream_error_or_empty_output_is_not_success(body):
    from types import SimpleNamespace
    import httpx
    from legal_rag.adapters.llm import AnswerModelClient, ModelServiceUnavailable
    settings = SimpleNamespace(env=SimpleNamespace(llm_base_url='http://test/v1', llm_model='test', llm_api_key='', answer_model_configured=lambda: True))
    client = AnswerModelClient(settings)
    client.verify_manifest = lambda: None
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body)))
    try:
        with pytest.raises(ModelServiceUnavailable):
            await client.complete([{'role': 'user', 'content': '测试'}], max_tokens=10, temperature=0.1, timeout_seconds=1)
    finally:
        await client.close()
