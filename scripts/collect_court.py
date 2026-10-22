"""采集最高人民法院站内公开的劳动用工资料（司法解释、指导性案例、典型案例）。

为什么单独写一个脚本：
- `collect_sources.py` 只接受人工给出的 URL 清单，无法发现条目；
- 本脚本先用站内检索（/search.html?content=&page=）发现候选，再逐条抓取详情页正文，
  并按标题自动判定资料类型与场景标签，产出与 `collect_sources.py` 相同结构的清单。

合规约束（与 docs/03 一致）：
- 只访问允许域名（court.gov.cn 及其子域）；单域名并发 1，请求间隔 ≥ min_request_interval_seconds；
- 站内检索批量上限受 sources.batch_page_limit 约束；
- 401/403/429/验证码/登录要求一律记为 access_blocked，**不重试绕行、不换 UA 破解**；
- flk.npc.gov.cn 的 robots.txt 明确禁止自动抓取，本脚本不访问该域名；
- 抓不到正文的条目不会写 local_object，也就不会进入可检索资料（import_seed.py 只导入 access_status=ok）。

用法：
    uv run python -X utf8 scripts/collect_court.py --pages 4
    uv run python -X utf8 scripts/collect_court.py --keywords 劳动争议 --pages 6 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urljoin

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.domain.ids import content_sha256, sha256_hex  # noqa: E402
from legal_rag.settings import get_settings  # noqa: E402

BASE = "https://www.court.gov.cn"
SEARCH = f"{BASE}/search.html"
USER_AGENT = "legal-rag-demo/0.1 (research; contact: local)"

# 详情页链接形如 /fabu/xiangqing/450751.html、/zixun/xiangqing/472691.html
DETAIL_RE = re.compile(r'href="(/(?:fabu|zixun|gongbao)/xiangqing/\d+\.html)"[^>]*>\s*([^<]{4,120})')

# 劳动用工相关性判定：标题或正文需命中，避免把无关新闻混入资料库
LABOR_HINT = re.compile(
    r"劳动|工资|欠薪|用工|劳动合同|劳务|工伤|经济补偿|赔偿金|竞业|社保|加班|裁员|解除|辞退|就业|用人单位|劳动者|新业态|劳务派遣"
)

# 场景标签（与 domain.enums.Scenario 对齐）
SCENARIO_RULES = {
    "wage_arrears": re.compile(r"工资|欠薪|报酬|加班费|劳动报酬|支付工资"),
    "termination": re.compile(r"解除|辞退|开除|末位|终止|裁员|违法解除|继续履行"),
    "economic_compensation": re.compile(r"经济补偿|赔偿金|补偿金|二倍工资|双倍工资|代通知金"),
}

# 资料类型判定：先判案例，再判规范，避免“典型案例”被当成规范性文件
CASE_RE = re.compile(r"案例|指导性案例|典型案例|参考案例")
NORMATIVE_RE = re.compile(r"解释|规定|批复|意见|通知|决定|办法|规则|答复|函")

# 强相关：必须是官方资料本体（指导性案例/参考案例/典型案例/司法解释/批复答复），
# 而不是“某地法院化解劳动争议”这类工作新闻。
STRONG_RE = re.compile(
    r"指导性案例|入库参考案例|参考案例|典型案例|司法解释|关于审理|关于适用|批复|答复|劳动争议.*解释"
)
# 发布主体：只保留最高人民法院层面的发布，或权威发布栏目（/fabu/）
PUBLISHER_RE = re.compile(r"最高法|最高人民法院|两高|指导性案例|入库参考案例|审判委员会")


def clean_text(html: str) -> str:
    """抽取详情页正文：优先取 <div class="txt..."> 与 <div class="txt_etr"> 之间的片段。"""
    start = re.search(r'<div class="txt(?: big| small)?">', html)
    if not start:
        return ""
    tail_match = re.search(r'<div class="txt_etr">', html[start.end() :])
    body = html[start.end() : start.end() + tail_match.start()] if tail_match else html[start.end() :]
    body = re.sub(r"(?is)<(script|style).*?</\1>", " ", body)
    body = re.sub(r"(?i)<br\s*/?>", "\n", body)
    body = re.sub(r"(?i)</(p|div|tr|li|h[1-6])>", "\n", body)
    body = re.sub(r"(?s)<[^>]+>", "", body)
    body = body.replace("&nbsp;", " ").replace("&amp;", "&").replace("&ldquo;", "“").replace("&rdquo;", "”")
    body = re.sub(r"[ \t　]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def classify(title: str) -> tuple[str, str | None]:
    """返回 (source_type, source_subtype)；无法判定时为 unknown。"""
    if CASE_RE.search(title):
        return "case", "case_summary"
    if NORMATIVE_RE.search(title):
        return "normative", "interpretation"
    return "unknown", None


def scenario_tags(title: str, body: str) -> list[str]:
    text = title + "\n" + body[:4000]
    tags = [name for name, rule in SCENARIO_RULES.items() if rule.search(text)]
    return tags or ["other"]


def published_at(html: str) -> str | None:
    match = re.search(r"发布时间：\s*(20\d{2}-\d{2}-\d{2})", html)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", nargs="*", default=["劳动争议", "指导性案例", "典型案例", "劳动合同", "欠薪"])
    parser.add_argument("--pages", type=int, default=4, help="每个关键词翻多少页检索结果")
    parser.add_argument("--manifest", default="data/manifests/sources.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="只列出候选，不抓正文")
    args = parser.parse_args()

    settings = get_settings()
    sources_cfg = settings.config.sources
    interval = max(2.0, sources_cfg.min_request_interval_seconds)
    manifest_path = ROOT / args.manifest
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if manifest_path.exists():
        existing = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {item["url"] for item in existing}

    candidates: dict[str, str] = {}
    requests_used = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True) as client:
        # 1) 站内检索发现候选
        for keyword in args.keywords:
            for page in range(1, args.pages + 1):
                if requests_used >= sources_cfg.batch_page_limit * 4:
                    break
                url = f"{SEARCH}?{urlencode({'content': keyword, 'page': page})}"
                try:
                    resp = client.get(url)
                except Exception as exc:
                    print(f"检索失败 {keyword} p{page}: {type(exc).__name__}")
                    break
                requests_used += 1
                if resp.status_code in {401, 403, 429}:
                    print(f"检索被拒绝 {keyword} p{page}: HTTP {resp.status_code}")
                    break
                found = DETAIL_RE.findall(resp.text)
                if not found:
                    break
                for path, title in found:
                    full = urljoin(BASE, path)
                    title = title.strip()
                    if full in candidates or full in known:
                        continue
                    if LABOR_HINT.search(title) and STRONG_RE.search(title) and (
                        "/fabu/" in path or PUBLISHER_RE.search(title)
                    ):
                        candidates[full] = title
                time.sleep(interval)
        print(f"检索请求 {requests_used} 次，劳动相关候选 {len(candidates)} 条")
        if args.dry_run:
            for url, title in sorted(candidates.items()):
                print(url, "|", title)
            return 0

        # 2) 逐条抓取正文
        entries = list(existing)
        fetched = blocked = skipped = 0
        for url, title in sorted(candidates.items()):
            if fetched + blocked + skipped >= sources_cfg.batch_page_limit:
                print("达到单次抓取上限，剩余条目留待下次执行。")
                break
            time.sleep(interval)
            status = "ok"
            body = ""
            try:
                resp = client.get(url)
                if resp.status_code in {401, 403, 429}:
                    status = "access_blocked"
                elif resp.status_code >= 400:
                    status = f"http_{resp.status_code}"
                else:
                    body = clean_text(resp.text)
                    if len(body) < 200:
                        status = "insufficient_text"
                    elif not LABOR_HINT.search(body[:3000]):
                        status = "not_labor_related"
            except Exception as exc:
                status = f"error:{type(exc).__name__}"

            source_type, subtype = classify(title)
            if status == "ok" and source_type == "unknown":
                # 类型无法判定的不做猜测，标记为待人工确认且不写入可导入清单
                status = "type_undetermined"

            sha = content_sha256(body.encode("utf-8")) if body else None
            local_object = None
            if status == "ok" and sha:
                local_object = f"data/raw/{sha}.md"
                text = f"# {title}\n\n来源：{url}\n发布时间：{published_at(resp.text) or '未标注'}\n\n{body}\n"
                (ROOT / local_object).write_text(text, encoding="utf-8")
                fetched += 1
            elif status == "ok":
                skipped += 1
            else:
                blocked += 1

            entries.append(
                {
                    "source_id": sha256_hex(url)[:16],
                    "title": title,
                    "url": url,
                    "publisher": "court.gov.cn",
                    "source_type": source_type,
                    "source_subtype": subtype,
                    "scenario_tags": scenario_tags(title, body) if status == "ok" else [],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": sha,
                    "local_object": local_object,
                    "version_notes": "最高人民法院站内公开页；发布/修订时间以页面标注为准",
                    "access_status": status,
                    "usage_notes": "站内检索自动发现；类型与场景标签由标题+正文规则判定，未人工复核" if status == "ok" else None,
                }
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for entry in entries:
        by_status[entry["access_status"]] = by_status.get(entry["access_status"], 0) + 1
        if entry["access_status"] == "ok":
            by_type[entry["source_type"]] = by_type.get(entry["source_type"], 0) + 1
    summary = {
        "total": len(entries),
        "fetched_this_run": fetched,
        "blocked_or_skipped_this_run": blocked + skipped,
        "by_status": by_status,
        "by_type_ok": by_type,
        "note": "仅统计实际访问结果；access_blocked / robots_disallow 不代表已获得资料",
    }
    (manifest_path.parent / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
