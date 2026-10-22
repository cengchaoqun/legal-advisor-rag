"""环境与契约预检。

offline：不联网、不计费，验证默认配置、机器契约、领域样例与必需环境变量。
live：只做连通性检查（Mongo/Milvus/Neo4j/对象存储/模型服务 /v1/models），
      不发起付费生成，也不租用任何 GPU。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jsonschema import Draft202012Validator, FormatChecker  # noqa: E402

from legal_rag.domain.models import (  # noqa: E402
    DatasetSample,
    Document,
    Evidence,
    Facts,
    PathStatus,
    QueryStatus,
    TaskStatus,
)
from legal_rag.runtime import Runtime  # noqa: E402
from legal_rag.settings import ConfigError, EnvSettings, PROJECT_ROOT, get_settings, load_config  # noqa: E402

CONTRACTS = PROJECT_ROOT / "docs" / "contracts"


def check_contracts() -> dict:
    from referencing import Registry, Resource

    core = json.loads((CONTRACTS / "核心对象.schema.json").read_text(encoding="utf-8-sig"))
    examples = json.loads((CONTRACTS / "契约样例.json").read_text(encoding="utf-8-sig"))
    Draft202012Validator.check_schema(core)
    registry = Registry().with_resources([(core["$id"], Resource.from_contents(core))])
    validator_errors: list[str] = []
    for example in examples:
        schema = {"$ref": core["$id"] + "#/$defs/" + example["schema"]}
        validator = Draft202012Validator(schema, registry=registry, format_checker=FormatChecker())
        for error in validator.iter_errors(example["value"]):
            validator_errors.append(f"{example['schema']}: {error.message}")
    if validator_errors:
        raise SystemExit("契约样例校验失败：\n" + "\n".join(validator_errors))
    return {"definitions": len(core["$defs"]), "examples": len(examples)}


def check_models() -> None:
    """用领域模型自检关键对象，确保 Pydantic 与 Schema 未漂移。"""
    Facts(scenario="other", location=None, relevant_date=None, known_facts=[], missing_fields=[])
    PathStatus.all_pending()
    for model in (Document, Evidence, QueryStatus, TaskStatus, DatasetSample):
        assert model.model_json_schema()


def check_env(env: EnvSettings) -> dict:
    required = {
        "MONGO_URI": env.mongo_uri,
        "MILVUS_URI": env.milvus_uri,
        "NEO4J_URI": env.neo4j_uri,
        "MINIO_ENDPOINT": env.minio_endpoint,
        "LLM_BASE_URL": env.llm_base_url,
        "LLM_MODEL": env.llm_model,
    }
    missing = [name for name, value in required.items() if not value]
    secrets_missing = [
        name
        for name, value in {
            "MINIO_ACCESS_KEY": env.minio_access_key,
            "MINIO_SECRET_KEY": env.minio_secret_key,
            "EMBED_MODEL_REVISION": env.embed_model_revision,
        }.items()
        if not value
    ]
    return {"missing_required": missing, "missing_secrets": secrets_missing}


def run_offline() -> int:
    settings = get_settings()
    contracts = check_contracts()
    check_models()
    env_report = check_env(settings.env)
    report = {
        "mode": "offline",
        "config_revision": settings.config.revision,
        "budget": {
            "total": settings.config.budget.total_limit,
            "api": settings.config.budget.api_limit,
            "gpu": settings.config.budget.gpu_limit,
            "reserve": settings.config.budget.reserve_limit,
            "paid_execution_enabled": settings.config.budget.paid_execution_enabled,
        },
        "pricing": settings.pricing.revision if settings.pricing else "unconfigured（付费批任务将被拒绝）",
        "contracts": contracts,
        "env": env_report,
        "app_mode": settings.env.app_mode,
        "answer_model_configured": settings.env.answer_model_configured(),
        "aux_model_configured": settings.env.aux_model_configured(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if env_report["missing_required"]:
        print("\n缺少必需环境变量：" + ", ".join(env_report["missing_required"]), file=sys.stderr)
        return 1
    print("\noffline 预检通过：配置与契约一致，未联网、未产生费用。")
    return 0


async def run_live() -> int:
    runtime = Runtime()
    await runtime.startup()
    try:
        health = await runtime.health()
        answer = await runtime.answer_model.health()
        report = {
            "mode": "live",
            "ready": health["status"],
            "checks": health["checks"],
            "answer_model": answer,
            "note": "仅连通性检查；未发起生成请求，未租用 GPU。",
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        blocking = [
            name
            for name in ("mongo", "milvus", "neo4j", "object_store")
            if health["checks"].get(name) != "ok"
        ]
        if blocking:
            print("\nlive 预检未通过：必要依赖不可用 " + ", ".join(blocking), file=sys.stderr)
            return 1
        if not answer.get("ok"):
            print("\n提示：回答模型未就绪（" + str(answer.get("error")) + "），咨询将返回 MODEL_SERVICE_UNAVAILABLE。")
        return 0
    finally:
        await runtime.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="掌柜智库环境预检")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    args = parser.parse_args()
    try:
        return run_offline() if args.mode == "offline" else asyncio.run(run_live())
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
