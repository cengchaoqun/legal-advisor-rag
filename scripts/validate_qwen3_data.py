"""离线校验Schema、来源原文、分组隔离、冻结摘要；服务器可加真实tokenizer校验。"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from legal_rag.domain.models import DatasetSample, EvaluationLabel
from training.data import check_split_integrity, encode_rows, file_sha256, load_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/qwen3_sft_v1")
    parser.add_argument("--tokenizer-path", help="服务器已有Qwen3-8B目录，设置后local_files_only，不下载")
    parser.add_argument("--max-seq-length", type=int, default=1536)
    args = parser.parse_args()
    directory = ROOT / args.data_dir
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert file_sha256(directory / name) == expected, f"冻结摘要不一致：{name}"
    splits = {split: load_jsonl(directory / f"{split}.jsonl") for split in ("train", "validation", "test")}
    check_split_integrity(splits)
    sources = load_jsonl(directory / "sources_snapshot.jsonl")
    chunks = {chunk["chunk_id"]: chunk["content"] for source in sources for chunk in source["chunks"]}
    labels = {row["sample_id"]: EvaluationLabel.model_validate(row) for row in load_jsonl(directory / "labels.jsonl")}
    ids = set()
    for rows in splits.values():
        for row in rows:
            DatasetSample.model_validate(row)
            ids.add(row["sample_id"])
            assert row["sample_id"] in labels, "缺少评估标签"
            assert labels[row["sample_id"]].expected_behavior.value == row["expected_behavior"]
            assert {f"E{number}" for number in re.findall(r"\[E(\d+)\]", row["target_answer"])} <= {ev["evidence_id"] for ev in row["evidence"]}
            for ev in row["evidence"]:
                assert ev["text"] in chunks[ev["chunk_id"]], f"证据并非来源子串：{row['sample_id']}"
    assert ids == set(labels), "sidecar样本集合与数据集不一致"
    report = {"result": "passed", "by_split": {split: len(rows) for split, rows in splits.items()}, "source_groups": manifest["source_groups"], "source_text_match": True, "split_isolation": True, "frozen_hashes": True, "tokenizer": "not_run（未提供本地Qwen3 tokenizer）"}
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True, trust_remote_code=False)
        report["tokenizer"] = {}
        for split, rows in splits.items():
            accepted, rejected = encode_rows(rows, tokenizer, max_seq_length=args.max_seq_length, max_input_tokens=args.max_seq_length - 512, max_output_tokens=512)
            report["tokenizer"][split] = {"accepted": len(accepted), "rejected": rejected}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
