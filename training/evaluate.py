"""自动评估：基于固定测试集与银标，允许无提升，禁止虚构指标。

用法：uv run --group training python -m training.evaluate --run-id <ID>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
CITATION = re.compile(r"\[E(\d+)\]")
SECTION_MARKERS = ("摘要", "依据与条件", "需补充事项")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_behavior(text: str, evidence_count: int) -> str:
    # 只从摘要识别行为；“需补充事项”是所有答案的固定段落，不等于澄清。
    summary = re.search(r"摘要[：:]\s*([^\n]+)", text)
    lead = summary.group(1) if summary else text.split("\n", 1)[0]
    if any(word in lead for word in ("依据不足", "无法给出结论", "无法作出", "不能判断")):
        return "insufficient"
    if any(word in lead for word in ("需澄清", "请补充", "需要补充", "还需要补充", "事实不足")):
        return "clarify"
    return "answer"


def score_predictions(test_rows: list[dict], predictions: list[dict]) -> dict:
    samples = {row["sample_id"]: row for row in test_rows}
    if not samples or len(samples) != len(test_rows):
        raise ValueError("测试集为空或样本ID重复")
    by_id = {}
    for row in predictions:
        sid = row["sample_id"]
        if sid not in samples:
            raise ValueError(f"预测包含未知样本：{sid}")
        if sid in by_id:
            raise ValueError(f"预测ID重复：{sid}")
        by_id[sid] = row
    hits = formats = valid = citations = no_citation = over_refusal = failed = 0
    for sid, sample in samples.items():
        prediction = by_id.get(sid, {})
        text = prediction.get("output", "")
        if prediction.get("status", "ok") != "ok" or not text.strip():
            failed += 1
            continue
        behavior = infer_behavior(text, len(sample["evidence"]))
        hits += behavior == sample["expected_behavior"]
        formats += all(marker in text for marker in SECTION_MARKERS)
        ids = {item["evidence_id"] for item in sample["evidence"]}
        cited = CITATION.findall(text)
        citations += len(cited)
        valid += sum(f"E{number}" in ids for number in cited)
        no_citation += not cited
        over_refusal += sample["expected_behavior"] == "answer" and behavior != "answer"
    total = len(samples)
    return {"samples": total, "prediction_records": len(by_id), "failed_samples": failed, "run_failure_rate": round(failed / total, 4), "behavior_accuracy_silver": round(hits / total, 4), "format_compliance": round(formats / total, 4), "citation_valid_rate": round(valid / citations, 4) if citations else None, "citation_count": citations, "no_citation": no_citation, "over_refusal": over_refusal, "faithfulness_judge": "not_run"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", default="data/qwen3_sft_v1")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / args.run_id
    from training.data import file_sha256
    frozen = json.loads((run_dir / "data_manifest.json").read_text())["files"]["test.jsonl"]
    if file_sha256(ROOT / args.data_dir / "test.jsonl") != frozen:
        raise SystemExit("测试集发生变化，拒绝评估")
    prediction_configs = [json.loads((run_dir / f"{variant}_prediction_config.json").read_text()) for variant in ("base", "adapter")]
    if prediction_configs[0] != prediction_configs[1] or prediction_configs[0]["test_sha256"] != frozen:
        raise SystemExit("base/adapter预测的基础快照、测试集或解码配置不一致")
    test_rows = {
        row["sample_id"]: row
        for row in load_jsonl(ROOT / args.data_dir / "test.jsonl")
    }
    labels = {row["sample_id"]: row for row in load_jsonl(ROOT / args.data_dir / "labels.jsonl")}
    if not test_rows:
        raise SystemExit("测试集为空：无法评估")

    metrics: dict[str, dict] = {}
    raw: dict[str, list[dict]] = {}
    for variant in ("base", "adapter"):
        predictions = load_jsonl(run_dir / f"{variant}_predictions.jsonl")
        if not (run_dir / f"{variant}_predictions.jsonl").exists():
            raise SystemExit(f"缺少 {variant}_predictions.jsonl：先运行真实推理")
        raw[variant] = predictions
        metrics[variant] = score_predictions(list(test_rows.values()), predictions)
        metrics[variant]["by_scenario"] = {scenario: score_predictions([row for row in test_rows.values() if row["scenario"] == scenario], [row for row in predictions if test_rows[row["sample_id"]]["scenario"] == scenario]) for scenario in sorted({row["scenario"] for row in test_rows.values()})}

    report = {
        "run_id": args.run_id,
        "test_sha256": frozen,
        "evaluation_type": "automatic_silver",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "limitations": [
            "银标由自动生成与自动质检产生，未经人工法律验证",
            "行为准确率以银标 expected_behavior 为分母，标签本身可能错误",
            "行为识别为基于摘要的规则代理，不能当作专业法律评审；模板工程集不证明法律泛化",
            "缺失、超长和运行失败题保留在冻结测试集分母中",
            "引用有效率只表示引用编号存在，不代表引用支持结论",
            "未配置独立评审模型时忠实性评分不执行，不填占位分数",
        ],
        "improvement_required": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# 自动评估报告（{args.run_id}）",
        "",
        "> 自动生成银标，未经人工法律验证；结果允许无提升。",
        "",
        "| 指标 | base | adapter |",
        "| --- | --- | --- |",
    ]
    for key in ("samples", "failed_samples", "run_failure_rate", "behavior_accuracy_silver", "format_compliance", "citation_valid_rate", "no_citation", "over_refusal"):
        lines.append(f"| {key} | {metrics['base'].get(key)} | {metrics['adapter'].get(key)} |")
    lines += ["", "## 限制", ""] + [f"- {item}" for item in report["limitations"]]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
