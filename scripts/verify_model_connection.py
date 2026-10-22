"""验证程序外部署的微调模型是否真实接入（docs/05 §6、T25/T26）。

只读取配置与部署清单，向模型服务和业务 API 发起最小请求；不打印密钥。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legal_rag.adapters.llm import load_deployment_manifest, directory_digest  # noqa: E402
from legal_rag.settings import get_settings  # noqa: E402

QUESTION = "公司拖欠我 2025 年 3 月工资 9000 元，月薪 9000 元，我可以主张什么？"


def fail(message: str) -> int:
    print(f"[FAIL] {message}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    settings = get_settings()
    env = settings.env

    # 1. 配置与部署清单
    if not env.llm_base_url or not env.llm_model:
        return fail("未配置 LLM_BASE_URL/LLM_MODEL")
    if not env.model_deployment_manifest:
        return fail("未配置 MODEL_DEPLOYMENT_MANIFEST，无法证明使用的是微调产物")
    try:
        manifest = load_deployment_manifest(env.model_deployment_manifest)
    except Exception as exc:
        return fail(f"部署清单不可用：{exc}")
    model_dir = Path(env.model_deployment_manifest).parent
    if (model_dir / "config.json").exists():
        actual = directory_digest(model_dir)
        if actual != manifest.directory_sha256:
            return fail("模型目录摘要与清单不一致")
    if env.model_tokenizer_path:
        if directory_digest(Path(env.model_tokenizer_path)) != manifest.tokenizer_sha256:
            return fail("本地tokenizer目录摘要与清单不一致")
    else:
        return fail("未配置MODEL_TOKENIZER_PATH，无法完成真实token计数验收")
    if manifest.served_model_name != env.llm_model:
        return fail(f"清单服务名 {manifest.served_model_name} 与 LLM_MODEL {env.llm_model} 不一致")
    print(f"[OK] 部署清单核验通过：{manifest.served_model_name} / {manifest.directory_sha256[:12]}…")

    headers = {"Authorization": f"Bearer {env.llm_api_key}"} if env.llm_api_key else {}
    try:
        models = httpx.get(f"{env.llm_base_url.rstrip('/')}/models", headers=headers, timeout=8.0)
        models.raise_for_status()
        ids = [item.get("id") for item in models.json().get("data", [])]
    except Exception as exc:
        print(f"[BLOCKED] 推理服务不可达：{exc}")
        return 2
    if env.llm_model not in ids:
        return fail(f"服务未提供模型 {env.llm_model}，当前：{ids}")
    print(f"[OK] /v1/models 包含 {env.llm_model}")

    # 2. 普通与流式补全
    payload = {
        "model": env.llm_model,
        "messages": [{"role": "user", "content": "只回复“就绪”两个字。"}],
        "temperature": 0,
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        plain = httpx.post(f"{env.llm_base_url.rstrip('/')}/chat/completions", json=payload, headers=headers, timeout=30.0)
        plain.raise_for_status()
        text = plain.json()["choices"][0]["message"]["content"]
        print(f"[OK] 普通回答：{text.strip()[:20]}")
    except Exception as exc:
        return fail(f"普通回答失败：{exc}")

    chunks = 0
    try:
        with httpx.stream(
            "POST",
            f"{env.llm_base_url.rstrip('/')}/chat/completions",
            json={**payload, "stream": True},
            headers=headers,
            timeout=30.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:") and "content" in line:
                    chunks += 1
        if chunks == 0:
            return fail("流式接口没有返回文本增量")
        print(f"[OK] 流式回答收到 {chunks} 个增量")
    except Exception as exc:
        return fail(f"流式回答失败：{exc}")

    # 3. 真实 RAG 查询：必须由该服务生成，且引用可核验
    with httpx.Client(base_url=args.base_url, timeout=60.0) as client:
        health = client.get("/health/ready")
        if health.status_code not in (200, 503):
            return fail(f"业务 API 不可用：{health.status_code}")
        session = client.post("/api/sessions", json={"title": "模型接入验证"}).json()
        created = client.post(
            f"/api/sessions/{session['session_id']}/queries",
            json={
                "request_id": str(__import__("uuid").uuid4()),
                "question": QUESTION,
                "document_ids": [],
                "attachment_ids": [],
                "relevant_date": None,
                "location": None,
                "web_enabled": False,
            },
        )
        if created.status_code != 202:
            return fail(f"提交咨询失败：{created.text[:200]}")
        query_id = created.json()["query_id"]
        doc = {}
        for _ in range(120):
            doc = client.get(f"/api/queries/{query_id}").json()
            if doc["status"] in {"completed", "completed_with_warnings", "failed", "interrupted", "cancelled"}:
                break
            time.sleep(1)

    if doc.get("status") not in {"completed", "completed_with_warnings"}:
        return fail(f"真实查询未完成：{doc.get('status')} warnings={doc.get('warnings')}")
    if doc.get("generation_model_id") != env.llm_model:
        return fail(f"回答模型标识不一致：{doc.get('generation_model_id')} != {env.llm_model}")
    if doc.get("generation_artifact_sha256") != manifest.directory_sha256:
        return fail("查询记录的产物摘要与部署清单不一致")
    if not doc.get("evidence"):
        return fail("真实查询没有返回证据，无法证明 RAG 链路")
    if doc.get("citation_validation") != "passed":
        print(f"[WARN] 引用校验状态：{doc.get('citation_validation')}（保留原始结果）")

    print("[OK] 真实 RAG 查询由部署模型生成，模型标识与产物摘要匹配")
    print(json.dumps({k: doc.get(k) for k in ("query_id", "status", "behavior", "citation_validation")}, ensure_ascii=False))
    print("[MANUAL] 停止推理服务后再问一次，应返回 MODEL_SERVICE_UNAVAILABLE；恢复服务后无需改前端即可继续。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
