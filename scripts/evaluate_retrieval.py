"""固定测试集上的检索评估（轨道二：检索组件差异，非微调效果）。

用法：
    uv run python -X utf8 scripts/evaluate_retrieval.py --split test --web off

要求 data/datasets/test.jsonl 与 data/datasets/labels.jsonl 存在；缺失时明确退出，
不生成占位指标。输出写到 artifacts/，用于程序外报告。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.domain.models import DatasetSample, EvaluationLabel  # noqa: E402
from legal_rag.retrieval.fusion import weighted_rrf  # noqa: E402
from legal_rag.runtime import Runtime  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def evaluate(split: str, top_k: int) -> int:
    dataset_dir = ROOT / "data" / "datasets"
    samples = [DatasetSample.model_validate(item) for item in load_jsonl(dataset_dir / f"{split}.jsonl")]
    labels = {item["sample_id"]: EvaluationLabel.model_validate(item) for item in load_jsonl(dataset_dir / "labels.jsonl")}
    if not samples:
        print(f"缺少 {dataset_dir / f'{split}.jsonl'}：未生成数据集，评估不执行。", file=sys.stderr)
        return 2

    runtime = Runtime()
    await runtime.startup()
    try:
        versions = await runtime.visibility.knowledge_versions()
        version_ids = [item["version_id"] for item in versions]
        if not version_ids:
            print("知识库为空：请先导入资料再评估。", file=sys.stderr)
            return 2
        rows = []
        hits_total = 0
        recall_sum = 0.0
        mrr_sum = 0.0
        started = time.perf_counter()
        for sample in samples:
            label = labels.get(sample.sample_id)
            if label is None:
                continue
            dense, sparse = await runtime.embedder.encode([sample.question])
            hits_dense = await runtime.vectors.search_dense(dense[0], version_ids, "knowledge", None, 20)
            hits_sparse = await runtime.vectors.search_sparse(sparse[0], version_ids, "knowledge", None, 20)
            fused = weighted_rrf(
                {"dense": [hit.chunk_id for hit in hits_dense], "sparse": [hit.chunk_id for hit in hits_sparse]},
                {"dense": 0.7, "sparse": 0.3},
                k=60,
            )
            ranked = [item.source_id for item in fused[:top_k]]
            relevant = set(label.relevant_source_ids)
            hit_ids = [item for item in ranked if item in relevant]
            recall = len(hit_ids) / len(relevant) if relevant else 0.0
            reciprocal = 0.0
            for rank, item in enumerate(ranked, start=1):
                if item in relevant:
                    reciprocal = 1.0 / rank
                    break
            recall_sum += recall
            mrr_sum += reciprocal
            hits_total += len(hit_ids)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "scenario": sample.scenario.value,
                    "relevant": sorted(relevant),
                    "retrieved": ranked,
                    "recall": round(recall, 4),
                    "reciprocal_rank": round(reciprocal, 4),
                }
            )
        elapsed = time.perf_counter() - started
        metrics = {
            "split": split,
            "samples": len(rows),
            "top_k": top_k,
            "recall_at_k": round(recall_sum / len(rows), 4) if rows else None,
            "mrr": round(mrr_sum / len(rows), 4) if rows else None,
            "hit_chunks": hits_total,
            "seconds": round(elapsed, 2),
            "note": "银标相关集非穷尽；该指标只衡量检索覆盖，不代表法律正确率",
        }
        out_dir = ROOT / "artifacts" / f"retrieval-{split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "per_sample.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"结果目录：{out_dir}")
        return 0
    finally:
        await runtime.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--web", default="off", choices=["on", "off"])
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()
    if args.web == "on":
        print("主实验固定 web=off；联网条件请单独记录来源快照后再比较。", file=sys.stderr)
    return asyncio.run(evaluate(args.split, args.top_k))


if __name__ == "__main__":
    raise SystemExit(main())
