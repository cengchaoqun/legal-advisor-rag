"""导出真实资料并生成离线银标数据，无云模型、付费调用或权重下载。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.bootstrap import group_assignments, make_source_samples
from training.data import check_split_integrity, file_sha256, write_jsonl
from legal_rag.domain.models import DatasetSample
from legal_rag.settings import get_settings
from legal_rag.storage.mongo import MongoStore


async def export_sources():
    store = MongoStore(get_settings())
    await store.connect()
    rows = []
    assigned_scenarios = Counter()
    try:
        async for document in store.collection("documents").find({"scope": "knowledge", "tombstoned": False}):
            version = await store.get_version(document.get("current_version_id"))
            if not version or version["publication_state"] != "published":
                continue
            chunks = await store.collection("chunks").find({"version_id": version["version_id"]}, {"_id": 0}).sort("ordinal", 1).to_list(length=None)
            scenarios = Counter(tag for chunk in chunks for tag in chunk.get("scenario_tags", []) if tag != "other")
            if not scenarios:
                continue  # 不把范围外资料硬标为三个训练场景之一。
            scenario = min(scenarios, key=lambda tag: (assigned_scenarios[tag], tag))
            assigned_scenarios[scenario] += 1
            rows.append({"document_id": document["document_id"], "version_id": version["version_id"], "title": document["title"], "url": version.get("source_url"), "metadata": version.get("metadata"), "quality_flags": version.get("quality_flags", []), "scenario": scenario, "chunks": chunks})
    finally:
        await store.close()
    return sorted(rows, key=lambda item: item["version_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/qwen3_sft_v1")
    parser.add_argument("--snapshot", help="已有sources_snapshot.jsonl；指定后无需连接数据库")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "dataset_manifest.json").exists():
        raise SystemExit("数据集已冻结；请用新的 --output 目录，不覆盖固定测试集")
    if args.snapshot:
        from training.data import load_jsonl
        sources = load_jsonl(Path(args.snapshot))
    else:
        sources = asyncio.run(export_sources())
    sources = [source for source in sources if source.get("url")]
    assignment = group_assignments(sources, args.seed)
    splits = {"train": [], "validation": [], "test": []}
    for source in sources:
        split = assignment[source["version_id"]]
        for row in make_source_samples(source, split):
            DatasetSample.model_validate(row)
            splits[split].append(row)
    if not all(splits.values()):
        raise SystemExit("某个split为空；增加独立来源，不能用复制样本填补")
    check_split_integrity(splits)
    write_jsonl(out / "sources_snapshot.jsonl", sources)
    labels = []
    for split, rows in splits.items():
        write_jsonl(out / f"{split}.jsonl", rows)
        for row in rows:
            labels.append({"sample_id": row["sample_id"], "relevant_source_ids": [ev["source_id"] for ev in row["evidence"]], "required_points": [], "expected_behavior": row["expected_behavior"], "label_type": "synthetic_silver"})
    write_jsonl(out / "labels.jsonl", labels)
    report = {"seed": args.seed, "source_documents": len(sources), "source_groups": len({source["group_id"] for source in sources}), "by_split": {split: len(rows) for split, rows in splits.items()}, "by_scenario": {split: dict(Counter(row["scenario"] for row in rows)) for split, rows in splits.items()}, "by_ability": {split: dict(Counter(row["ability"] for row in rows)) for split, rows in splits.items()}, "files": {path.name: file_sha256(path) for path in sorted(out.glob("*.jsonl"))}, "scope": "基于真实来源的确定性阅读/引用/拒答工程银标，不是专家法律问答金标；模板相关性较高，不证明法律泛化或专业准确率", "target_counts_300_60_90_met": False, "legal_judge": "not_run", "tokenizer_validation": "pending_on_training_server"}
    (out / "dataset_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
