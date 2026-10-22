"""合并 LoRA 并导出可部署模型目录 + deployment_manifest.json。

用法：
    uv run --group training python -m training.merge_export --run-id <ID> --output-dir artifacts/<ID>/merged_model

清单中的目录摘要按“相对路径排序 + 逐文件 SHA256”计算，供业务后端核对真实权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from legal_rag.adapters.llm import directory_digest  # noqa: E402
from training.configuration import saved_config, model_kwargs, model_location
from training.data import file_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-path", help="同一基础模型快照的本地目录，可在更换服务器时重定位")
    parser.add_argument("--served-name", default="")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = ROOT / "artifacts" / args.run_id
    config = saved_config(run_dir)
    if json.loads((run_dir / "run_summary.json").read_text())["smoke"]:
        raise SystemExit("smoke不可作为正式模型导出")
    if args.model_path:
        config["training"]["model_path"] = str(Path(args.model_path).resolve())
    train_cfg = config["training"]
    adapter_dir = run_dir / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise SystemExit("适配器不存在：不能把空目录当作已训练模型导出")
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("缺少自动评估结果 metrics.json：先完成 base/adapter 预测与评估")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("run_id") != args.run_id or set(metrics.get("metrics", {})) != {"base", "adapter"}:
        raise SystemExit("评估结果与当前run不匹配或未完成基座/LoRA对比")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "merged_model"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit("导出目录非空，不覆盖既有模型或混入陈旧文件")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_location(config), **model_kwargs(config)
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_location(config),
        torch_dtype=torch.bfloat16,
        **model_kwargs(config),
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output_dir)
    # 小体积副本只供业务电脑精确token计数，不包含任何8B模型权重。
    tokenizer_dir = run_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    adapter_sha = (adapter_dir / "adapter_model.safetensors")
    manifest = {
        "served_model_name": args.served_name or config["serving"]["served_model_name"],
        "base_model": train_cfg["base_model"],
        "base_revision": train_cfg.get("base_revision"),
        "adapter_sha256": __import__("hashlib").sha256(adapter_sha.read_bytes()).hexdigest() if adapter_sha.exists() else None,
        "directory_sha256": directory_digest(output_dir),
        "tokenizer_sha256": directory_digest(tokenizer_dir),
        "chat_template_sha256": __import__("hashlib").sha256(json.dumps(tokenizer.chat_template, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "files": {path.relative_to(output_dir).as_posix(): file_sha256(path) for path in sorted(output_dir.rglob("*")) if path.is_file()},
        "tokenizer_files": {path.relative_to(tokenizer_dir).as_posix(): file_sha256(path) for path in sorted(tokenizer_dir.rglob("*")) if path.is_file()},
        "enable_thinking": False,
        "max_model_len": config["serving"]["max_model_len"],
        "artifact_type": "merged_model",
        "engine": config["serving"]["engine"],
        "evaluation_report": str(metrics_path.relative_to(ROOT)),
        "run_id": args.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "目录摘要覆盖合并后全部文件；服务别名相同不代表权重相同",
    }
    manifest_path = output_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if directory_digest(output_dir) != manifest["directory_sha256"]:
        raise SystemExit("清单写入后目录摘要不一致")
    print(json.dumps({k: manifest[k] for k in ("served_model_name", "directory_sha256", "run_id")}, ensure_ascii=False))
    print(f"部署清单：{manifest_path}")
    print("下一步：vllm serve 该目录，并把 LLM_BASE_URL/LLM_MODEL/MODEL_DEPLOYMENT_MANIFEST 配置到后端。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
