"""生成银标数据集：先按来源组划分 split，再调用辅助模型生成并自动质检。

用法：
    uv run python -X utf8 scripts/build_dataset.py --config docs/contracts/默认配置.json --pilot 30

说明：
- 缺少辅助模型配置或价格时明确阻塞（退出码 2），不生成占位样本；
- 生成结果固定 label_type=synthetic_silver，不称 gold；
- 测试集一经冻结不参与训练，评估标签写入 labels.jsonl。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.adapters.llm import AuxModelClient, AuxModelUnavailable  # noqa: E402
from legal_rag.domain.enums import Ability, Behavior, Scenario  # noqa: E402
from legal_rag.domain.ids import sha256_hex  # noqa: E402
from legal_rag.runtime import Runtime  # noqa: E402
from legal_rag.settings import get_settings  # noqa: E402

ABILITIES = [item.value for item in Ability]
STEP = {"train": 0.667, "validation": 0.133, "test": 0.2}
CITATION = re.compile(r"\[E(\d+)\]")


def trigrams(text: str) -> set[str]:
    clean = re.sub(r"\s+", "", text)
    return {clean[index : index + 3] for index in range(max(0, len(clean) - 2))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def split_groups(groups: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = groups[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = max(1, round(total * STEP["train"]))
    val_end = min(total, train_end + max(1, round(total * STEP["validation"])))
    assignment = {}
    for index, group in enumerate(shuffled):
        assignment[group] = "train" if index < train_end else ("validation" if index < val_end else "test")
    return assignment


async def generate_for_group(
    aux: AuxModelClient,
    *,
    group: str,
    scenario: str,
    evidence_rows: list[dict],
    wanted: int,
    generator_revision: str,
) -> list[dict]:
    evidence_text = "\n\n".join(
        f"[E{index + 1}] {row['title']}（{row.get('article') or row.get('heading') or '片段'}）\n{row['text'][:800]}"
        for index, row in enumerate(evidence_rows)
    )
    ability_list = "、".join(ABILITIES)
    prompt = (
        "你是劳动法数据标注助手。仅依据下列编号证据生成测试题，不得引入外部法条或编造数字。\n"
        f"证据：\n{evidence_text}\n\n"
        f"请生成 {wanted} 条中文问答，覆盖能力：{ability_list}，场景：{scenario}。\n"
        "严格输出 JSON 数组，每项字段：question、expected_behavior(answer/clarify/insufficient)、"
        "target_answer（结论必须带 [E#] 引用，编号不超过证据数量）、ability、required_points（字符串数组）。\n"
        "涉及未知效力或版本时必须写明限制；证据不足的题把 expected_behavior 设为 insufficient。"
    )
    try:
        raw = await aux.complete([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.2)
    except AuxModelUnavailable as exc:
        raise SystemExit(f"辅助模型不可用，数据集生成阻塞：{exc}")
    start = raw.find("[")
    end = raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []

    rows: list[dict] = []
    for index, item in enumerate(items[:wanted]):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("target_answer", "")).strip()
        ability = item.get("ability")
        behavior = item.get("expected_behavior")
        if not question or not answer or ability not in ABILITIES or behavior not in {b.value for b in Behavior}:
            continue
        cited = {int(number) for number in CITATION.findall(answer)}
        if any(number < 1 or number > len(evidence_rows) for number in cited):
            continue  # 引用不存在的证据：丢弃而不是修补
        rows.append(
            {
                "sample_id": f"{group}-{index:03d}",
                "split": None,
                "origin_group_id": group,
                "scenario": scenario if scenario in {s.value for s in Scenario} else "other",
                "ability": ability,
                "question": question,
                "evidence": [
                    {
                        "evidence_id": f"E{position + 1}",
                        "source_id": row["chunk_id"],
                        "source_type": "local",
                        "chunk_id": row["chunk_id"],
                        "version_id": row["version_id"],
                        "title": row["title"],
                        "text": row["text"][:1200],
                        "url": None,
                        "location": row.get("location"),
                        "metadata": row.get("metadata"),
                        "quality_flags": row.get("quality_flags", []),
                    }
                    for position, row in enumerate(evidence_rows)
                ],
                "expected_behavior": behavior,
                "target_answer": answer,
                "label_type": "synthetic_silver",
                "generator_revision": generator_revision,
                "source_manifest_revision": "sources-v1",
                "required_points": item.get("required_points") if isinstance(item.get("required_points"), list) else [],
            }
        )
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="docs/contracts/默认配置.json")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--group", action="append", default=[], help="只处理指定来源组（调试用）")
    args = parser.parse_args()

    settings = get_settings()
    config = settings.config
    runtime = Runtime()
    await runtime.startup()
    try:
        if not runtime.aux_model.configured:
            print("辅助模型未配置（AUX_LLM_*），无法生成银标样本；先完成配置与 30 条成本探针。", file=sys.stderr)
            return 2

        import_path = ROOT / "data" / "manifests" / "import_result.json"
        if not import_path.exists():
            print("缺少 data/manifests/import_result.json：请先运行 import_seed.py 导入真实资料。", file=sys.stderr)
            return 2
        imported = json.loads(import_path.read_text(encoding="utf-8")).get("accepted", [])
        if not imported:
            print("没有已导入资料，无法生成样本。", file=sys.stderr)
            return 2

        groups: dict[str, dict] = {}
        for entry in imported:
            version = await runtime.store.get_version(entry["version_id"])
            if not version:
                continue
            group = sha256_hex(version["sha256"])[:16]
            groups.setdefault(
                group,
                {"version_id": version["version_id"], "document_id": version["document_id"], "scenario": "other"},
            )

        assignment = split_groups(sorted(groups), config.dataset.seed)
        manifest = {
            "seed": config.dataset.seed,
            "targets": {
                "train": config.dataset.train_target,
                "validation": config.dataset.validation_target,
                "test": config.dataset.test_target,
            },
            "groups": {group: assignment[group] for group in sorted(groups)},
            "note": "先划分来源组再生成问题；同组不得跨 split",
        }
        out_dir = ROOT / "data" / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        dataset_dir = ROOT / "data" / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        per_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
        labels: list[dict] = []
        limit = args.pilot or None
        produced = 0
        seen: list[set[str]] = []

        for group, meta in groups.items():
            if args.group and group not in args.group:
                continue
            if limit and produced >= limit:
                break
            split = assignment[group]
            chunks = [
                chunk
                async for chunk in runtime.store.collection("chunks")
                .find({"version_id": meta["version_id"]})
                .sort("ordinal", 1)
                .limit(3)
            ]
            if not chunks:
                continue
            document = await runtime.store.get_document(meta["document_id"])
            scenario = (chunks[0].get("scenario_tags") or ["other"])[0]
            evidence_rows = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "version_id": chunk["version_id"],
                    "title": (document or {}).get("title", "资料"),
                    "text": chunk["content"],
                    "location": chunk["location"],
                    "article": (chunk.get("location") or {}).get("article"),
                    "heading": (chunk.get("location") or {}).get("heading"),
                    "metadata": None,
                    "quality_flags": chunk.get("quality_flags") or [],
                }
                for chunk in chunks
            ]
            rows = await generate_for_group(
                runtime.aux_model,
                group=group,
                scenario=scenario,
                evidence_rows=evidence_rows,
                wanted=3 if not limit else min(3, limit - produced),
                generator_revision=f"{settings.env.aux_llm_model}:prompt-v1",
            )
            for row in rows:
                fingerprint = trigrams(row["question"])
                if any(jaccard(fingerprint, other) >= config.dataset.near_duplicate_trigram_jaccard for other in seen):
                    continue
                seen.append(fingerprint)
                row["split"] = split
                labels.append(
                    {
                        "sample_id": row["sample_id"],
                        "relevant_source_ids": [item["source_id"] for item in row["evidence"]],
                        "required_points": row.pop("required_points", []),
                        "expected_behavior": row["expected_behavior"],
                        "label_type": "synthetic_silver",
                    }
                )
                per_split[split].append(row)
                produced += 1

        for split, rows in per_split.items():
            path = dataset_dir / f"{split}.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
        (dataset_dir / "labels.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in labels), encoding="utf-8"
        )
        stats = {
            "produced": produced,
            "by_split": {split: len(rows) for split, rows in per_split.items()},
            "targets": manifest["targets"],
            "label_type": "synthetic_silver",
            "note": "自动生成银标，未经人工法律验证；未达目标数量时如实记录",
        }
        (dataset_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    finally:
        await runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
