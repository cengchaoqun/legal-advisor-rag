"""按来源清单导入资料：走真实 HTTP API，等待任务终态并汇总结果。

用法：
    uv run python -X utf8 scripts/import_seed.py --manifest data/manifests/sources.jsonl
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.domain.ids import sha256_hex  # noqa: E402

DOC_TYPE_BY_SOURCE = {
    "law": "law",
    "interpretation": "interpretation",
    "case_summary": "case",
    "judgment": "case",
    "guidance": "guidance",
}


def deterministic_request_id(url: str, sha: str) -> uuid.UUID:
    # 同一来源重复导入保持幂等
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{url}|{sha}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/sources.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest
    if not manifest_path.exists():
        print(f"清单不存在：{manifest_path}", file=sys.stderr)
        return 1
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    candidates = [item for item in entries if item.get("access_status") == "ok" and item.get("local_object")]
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print("清单中没有可导入条目（需要 access_status=ok 且 local_object 存在）。")
        return 1

    accepted, failed = [], []
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        for entry in candidates:
            path = ROOT / entry["local_object"]
            if not path.exists():
                failed.append({"url": entry.get("url"), "reason": "local_object_missing"})
                continue
            data = path.read_bytes()
            sha = entry.get("content_sha256") or sha256_hex(data.hex())[:64]
            doc_type = DOC_TYPE_BY_SOURCE.get(entry.get("source_subtype") or entry.get("source_type"), "unknown")
            files = {"file": (path.name, data, mimetypes.guess_type(path.name)[0] or "text/plain")}
            form = {
                "request_id": str(deterministic_request_id(entry.get("url", path.name), sha)),
                "doc_type": doc_type,
                "title": (entry.get("title") or path.stem)[:200],
            }
            if entry.get("url"):
                form["source_url"] = entry["url"]
            response = client.post("/api/documents", data=form, files=files)
            if response.status_code != 202:
                failed.append({"url": entry.get("url"), "reason": response.text[:200]})
                continue
            payload = response.json()
            task_id = payload["task_id"]
            status = "queued"
            for _ in range(120):
                task = client.get(f"/api/import-tasks/{task_id}").json()
                status = task["status"]
                if status in {"succeeded", "succeeded_with_warnings", "failed", "cancelled"}:
                    break
                time.sleep(1)
            record = {
                "url": entry.get("url"),
                "document_id": payload["document_id"],
                "version_id": payload["version_id"],
                "task_id": task_id,
                "status": status,
                "duplicate": payload.get("duplicate"),
            }
            (accepted if status.startswith("succeeded") else failed).append(record)
            print(json.dumps(record, ensure_ascii=False))

    summary = {"imported": len(accepted), "failed": len(failed), "accepted": accepted, "failures": failed}
    out = ROOT / "data" / "manifests" / "import_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"imported": len(accepted), "failed": len(failed)}, ensure_ascii=False))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
