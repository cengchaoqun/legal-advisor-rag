"""训练数据的共同边界：冻结划分、真实模板计数、只监督回答，超长绝不截断。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from training.prompting import build_training_messages


class SampleTooLong(ValueError):
    pass


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_split_integrity(splits: dict[str, list[dict]]) -> None:
    seen_ids, groups, questions, sources = set(), {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            sid, group = row["sample_id"], row["origin_group_id"]
            if sid in seen_ids:
                raise ValueError(f"样本ID重复：{sid}")
            seen_ids.add(sid)
            if row["split"] != split:
                raise ValueError(f"样本split与文件不一致：{sid}")
            if group in groups and groups[group] != split:
                raise ValueError(f"来源组跨split：{group}")
            groups[group] = split
            question = re.sub(r"\s+", "", row["question"])
            if question in questions and questions[question] != split:
                raise ValueError(f"相同问题跨split：{sid}")
            questions[question] = split
            for evidence in row["evidence"]:
                if "query_id" in evidence:
                    raise ValueError(f"离线证据不得含query_id：{sid}")
                source = evidence["source_id"]
                if source in sources and sources[source] != split:
                    raise ValueError(f"同一证据跨split：{source}")
                sources[source] = split


def encode_sample(sample, tokenizer, *, max_seq_length, max_input_tokens, max_output_tokens):
    messages = build_training_messages(sample)
    prompt = tokenizer.apply_chat_template(messages[:2], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    answer = tokenizer(messages[2]["content"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    if len(prompt) > max_input_tokens or len(answer) > max_output_tokens or len(prompt) + len(answer) > max_seq_length:
        raise SampleTooLong(f"{sample['sample_id']}: input={len(prompt)}, answer={len(answer)}, total={len(prompt)+len(answer)}")
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer, "attention_mask": [1] * (len(prompt) + len(answer))}


def encode_rows(rows, tokenizer, *, max_seq_length, max_input_tokens, max_output_tokens):
    accepted, rejected = [], []
    for row in rows:
        try:
            accepted.append(encode_sample(row, tokenizer, max_seq_length=max_seq_length, max_input_tokens=max_input_tokens, max_output_tokens=max_output_tokens))
        except SampleTooLong as exc:
            rejected.append({"sample_id": row["sample_id"], "reason": str(exc)})
    return accepted, rejected
