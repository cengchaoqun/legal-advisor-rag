"""采集公开资料：只访问允许域名，记录来源与访问状态，不做绕过限制的抓取。

用法：
    uv run python -X utf8 scripts/collect_sources.py --manifest data/manifests/sources.jsonl --limit 24

行为：
- 单域名并发 1、请求间隔 ≥2 秒、单次任务最多 30 页；
- 403/验证码/登录要求记录 access_blocked，不重试刷取；
- 只有拿到正文的条目才写 local_object 并计入可检索资料。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.domain.ids import content_sha256, sha256_hex  # noqa: E402
from legal_rag.settings import get_settings  # noqa: E402

USER_AGENT = "legal-rag-demo/0.1 (research; contact: local)"


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    query = "&".join(
        part for part in parsed.query.split("&") if part and not part.startswith(("utm_", "spm", "from="))
    )
    return parsed._replace(query=query, fragment="").geturl()


def html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def allowed(url: str, domains: list[str], allow_subdomains: bool) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or (allow_subdomains and host.endswith("." + domain)) for domain in domains)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/sources.jsonl")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--url", action="append", default=[], help="追加待采集 URL")
    args = parser.parse_args()

    settings = get_settings()
    config = settings.config
    manifest_path = ROOT / args.manifest
    entries = load_manifest(manifest_path)
    known = {entry["url"] for entry in entries}
    queue = [url for url in args.url if url not in known]
    if not queue:
        print("没有新的待采集 URL。请用 --url 提供官方发布页地址，或先维护 sources.jsonl 的待获取清单。")
        return 0
    if len(queue) > config.sources.batch_page_limit:
        print(f"单次最多采集 {config.sources.batch_page_limit} 页，已截断队列。")
        queue = queue[: config.sources.batch_page_limit]
    queue = queue[: args.limit]

    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    last_request: dict[str, float] = {}
    blocked = 0
    fetched = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True) as client:
        for url in queue:
            normalized = normalize_url(url)
            host = urlparse(normalized).hostname or ""
            if not allowed(normalized, config.sources.source_domains, config.sources.allow_subdomains):
                entries.append(
                    {
                        "source_id": sha256_hex("blocked", normalized)[:16],
                        "title": None,
                        "url": normalized,
                        "publisher": None,
                        "source_type": "unknown",
                        "scenario_tags": [],
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "content_sha256": None,
                        "local_object": None,
                        "version_notes": None,
                        "access_status": "domain_not_allowed",
                        "usage_notes": "域名不在允许清单，未抓取",
                    }
                )
                blocked += 1
                continue

            # 尊重 robots：明确禁止时记录并跳过；读取失败（无 robots.txt）按“未声明限制”处理。
            # 注意：必须区分“禁止”与“读取异常”，否则会把 Disallow 吞掉继续抓取。
            robots_disallow = False
            try:
                robots = urllib.robotparser.RobotFileParser(f"https://{host}/robots.txt")
                robots.read()
                robots_disallow = not robots.can_fetch(USER_AGENT, normalized)
            except Exception:
                robots_disallow = False
            if robots_disallow:
                entries.append(
                    {
                        "source_id": sha256_hex("robots", normalized)[:16],
                        "title": None,
                        "url": normalized,
                        "publisher": host,
                        "source_type": "unknown",
                        "scenario_tags": [],
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "content_sha256": None,
                        "local_object": None,
                        "version_notes": None,
                        "access_status": "robots_disallow",
                        "usage_notes": "robots.txt 明确禁止自动抓取，未抓取",
                    }
                )
                blocked += 1
                continue

            interval = config.sources.min_request_interval_seconds
            wait = interval - (time.monotonic() - last_request.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            last_request[host] = time.monotonic()

            status = "ok"
            body = ""
            try:
                response = client.get(normalized)
                if response.status_code in {401, 403, 429}:
                    status = "access_blocked"
                elif response.status_code >= 400:
                    status = f"http_{response.status_code}"
                else:
                    body = html_to_text(response.text)
                    if len(body) < 200:
                        status = "insufficient_text"
            except Exception as exc:
                status = f"error:{type(exc).__name__}"

            sha = content_sha256(body.encode("utf-8")) if body else None
            local_object = None
            if sha:
                local_object = f"data/raw/{sha}.txt"
                (ROOT / local_object).write_text(body, encoding="utf-8")
                fetched += 1
            if status != "ok":
                blocked += 1

            entries.append(
                {
                    "source_id": sha256_hex(normalized)[:16],
                    "title": None,
                    "url": normalized,
                    "publisher": host,
                    "source_type": "unknown",
                    "source_subtype": None,
                    "scenario_tags": [],
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": sha,
                    "local_object": local_object,
                    "version_notes": None,
                    "access_status": status,
                    "usage_notes": "自动采集，标题/类型需在清单中人工补全" if status == "ok" else None,
                }
            )

    save_manifest(manifest_path, entries)
    summary = {
        "total": len(entries),
        "fetched": fetched,
        "blocked_or_failed": blocked,
        "by_status": {status: sum(1 for item in entries if item["access_status"] == status) for status in {item["access_status"] for item in entries}},
        "note": "仅统计实际访问结果；access_blocked 不代表已获得资料",
    }
    (manifest_path.parent / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
