"""资料治理：剔除已废止/草案文本，并把"典型案例合集"拆成单案文档。

为什么需要这一步：
- 站内检索抓到的《劳动争议司法解释（二）（三）（四）》是 2006/2010/2013 年的旧解释，
  已被法释〔2020〕26号（现行《解释（一）》）整合废止；征求意见稿也不是生效文本。
  把它们当现行规范导入会让检索命中失效条文，因此标记为不可导入（保留记录，不删除证据）。
- "最高法发布劳动争议典型案例"这类合集页一个页面含 5～12 个案例。按文档计数只有 1 份，
  按 docs/03 的要求应以案例为单位组织，否则检索时整篇合集作为一个证据块返回，引用定位不准。

处理规则（全部可逆，只改清单状态与派生文件，不改动 data/raw 下的原始抓取内容）：
- normative 且命中已废止文号 → access_status = "superseded"
- 标题含"征求意见稿"或正文声明为征求意见 → access_status = "draft"
- case 文档内含 ≥2 个"案例N"小节 → 拆成单案 md，父文档 access_status = "split_into_cases"
- 其余保持 ok

用法：
    uv run python -X utf8 scripts/curate_sources.py
    uv run python -X utf8 scripts/curate_sources.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 已被法释〔2020〕26号（《解释（一）》，2021-01-01 施行）整合废止的旧劳动争议司法解释
ABOLISHED_DOC_NUMBERS = {
    "法释〔2006〕6号": "旧《解释（二）》，已被法释〔2020〕26号整合废止",
    "法释〔2010〕12号": "旧《解释（三）》，已被法释〔2020〕26号整合废止",
    "法释〔2013〕4号": "旧《解释（四）》，已被法释〔2020〕26号整合废止",
}
DRAFT_RE = re.compile(r"征求意见稿|向社会公开征求意见|公开征求意见的通知")
CASE_HEAD_RE = re.compile(r"^\s*(案例[一二三四五六七八九十百\d]{1,3})[：:、．.]?\s*(.*)$")

SCENARIO_RULES = {
    "wage_arrears": re.compile(r"工资|欠薪|报酬|加班费|劳动报酬|支付工资|农民工"),
    "termination": re.compile(r"解除|辞退|开除|末位|终止|裁员|违法解除|继续履行|确认劳动关系"),
    "economic_compensation": re.compile(r"经济补偿|赔偿金|补偿金|二倍工资|双倍工资|代通知金|工伤保险待遇"),
}


SECTION_STOP_RE = re.compile(r"^\s*[【\[]?\s*(基本案情|裁判结果|裁判理由|典型意义|裁判要点|相关法条|关键词)")


def case_display_name(body: str) -> str:
    """取小节标题行之后的案名（通常跨 1～3 行，以“——某甲与某乙…案”结尾）。"""
    lines = [line.strip() for line in body.split("\n")]
    # 跳过首个“案例N”标题行
    if lines and CASE_HEAD_RE.match(lines[0]):
        lines = lines[1:]
    picked: list[str] = []
    for line in lines:
        if not line:
            continue
        if SECTION_STOP_RE.match(line):
            break
        picked.append(line.strip(" —-—"))
        if len(picked) >= 3 or "案" in line and ("诉" in line or "与" in line):
            break
    name = " ".join(part for part in picked if part).strip()
    name = re.sub(r"\s{2,}", " ", name)
    return name[:80] if name else "未命名案例"


def scenario_tags(text: str) -> list[str]:
    tags = [name for name, rule in SCENARIO_RULES.items() if rule.search(text)]
    return tags or ["other"]


def split_case_sections(body: str) -> list[tuple[str, str]]:
    """按"案例N"小节切分；同一编号出现两次（目录+正文）时保留内容更长的那一段。"""
    lines = body.split("\n")
    chunks: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is not None:
            chunks.setdefault(current, []).append("\n".join(buffer).strip())

    for line in lines:
        match = CASE_HEAD_RE.match(line)
        if match:
            flush()
            current = match.group(1)
            if current not in chunks:
                order.append(current)
            buffer = [line]
        elif current is not None:
            buffer.append(line)
    flush()

    result: list[tuple[str, str]] = []
    for label in order:
        # 同一编号会同时出现在目录与正文，取最长的一段作为该案例正文
        best = max(chunks.get(label, [""]), key=len)
        if len(best) >= 300:
            result.append((label, best))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/sources.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_dir = ROOT / "data" / "raw"

    # 幂等：先撤销上一轮拆分的结果（删除派生条目与其文件，父文档恢复为 ok），再重新拆分。
    rows = [row for row in rows if not row.get("parent_source_id")]
    for row in rows:
        if row.get("access_status") == "split_into_cases":
            row["access_status"] = "ok"
            row["usage_notes"] = None
    out: list[dict] = []
    stats = {"superseded": 0, "draft": 0, "split_parents": 0, "split_cases": 0}

    for row in rows:
        if row.get("access_status") != "ok" or not row.get("local_object"):
            out.append(row)
            continue
        path = ROOT / row["local_object"]
        if not path.exists():
            out.append(row)
            continue
        text = path.read_text(encoding="utf-8")

        if row.get("source_type") == "normative":
            abolished = [note for number, note in ABOLISHED_DOC_NUMBERS.items() if number in text]
            if abolished:
                row["access_status"] = "superseded"
                row["usage_notes"] = "；".join(abolished)
                stats["superseded"] += 1
                out.append(row)
                continue
            if DRAFT_RE.search(row.get("title", "")) or (
                "征求意见" in text[:400] and "施行" not in text[:400]
            ):
                row["access_status"] = "draft"
                row["usage_notes"] = "征求意见稿/征求意见报道，不是生效文本"
                stats["draft"] += 1
                out.append(row)
                continue

        if row.get("source_type") == "case":
            sections = split_case_sections(text)
            if len(sections) >= 2:
                row["access_status"] = "split_into_cases"
                row["usage_notes"] = f"合集页已拆分为 {len(sections)} 个单案文档"
                out.append(row)
                stats["split_parents"] += 1
                if args.dry_run:
                    continue
                for label, body in sections:
                    case_name = case_display_name(body)
                    case_title = f"{case_name}（{row['title'][:24]}）"[:200]
                    sha = __import__("hashlib").sha256(body.encode("utf-8")).hexdigest()
                    target = raw_dir / f"{sha}.md"
                    target.write_text(
                        f"# {case_title}\n\n来源：{row['url']}（{row['title']}）\n\n{body}\n", encoding="utf-8"
                    )
                    out.append(
                        {
                            "source_id": sha[:16],
                            "title": case_title,
                            "url": row["url"],
                            "publisher": row.get("publisher"),
                            "source_type": "case",
                            "source_subtype": "case_summary",
                            "scenario_tags": scenario_tags(body),
                            "retrieved_at": row.get("retrieved_at") or datetime.now(timezone.utc).isoformat(),
                            "content_sha256": sha,
                            "local_object": f"data/raw/{sha}.md",
                            "version_notes": "由官方典型案例合集页按案例小节拆分，正文未改写",
                            "access_status": "ok",
                            "usage_notes": "拆分自合集页，引用定位仍指向原始发布页",
                            "parent_source_id": row.get("source_id"),
                        }
                    )
                    stats["split_cases"] += 1
                continue

        out.append(row)

    if not args.dry_run:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for entry in out:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    by_status: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    for entry in out:
        by_status[entry["access_status"]] = by_status.get(entry["access_status"], 0) + 1
        if entry["access_status"] == "ok":
            for tag in entry.get("scenario_tags", []):
                by_scenario[tag] = by_scenario.get(tag, 0) + 1
    summary = {
        "total": len(out),
        "stats": stats,
        "by_status": by_status,
        "ok_by_type": {
            t: sum(1 for e in out if e["access_status"] == "ok" and e.get("source_type") == t)
            for t in {e.get("source_type") for e in out if e["access_status"] == "ok"}
        },
        "ok_scenario_tags": by_scenario,
        "dry_run": args.dry_run,
    }
    (manifest_path.parent / "curate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
