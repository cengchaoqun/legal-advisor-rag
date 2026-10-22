"""固定完整测试集的基座/LoRA推理；失败和超长样本留在预测文件中。"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.configuration import model_kwargs, model_location, saved_config
from training.data import file_sha256, load_jsonl, write_jsonl
from training.prompting import build_training_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--variant", choices=["base", "adapter"], required=True)
    parser.add_argument("--data-dir", default="data/qwen3_sft_v1")
    parser.add_argument("--model-path", help="移动服务器后指定同一基础快照的新目录")
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    run_dir = ROOT / "artifacts" / args.run_id
    config = saved_config(run_dir)
    if json.loads((run_dir / "run_summary.json").read_text())["smoke"]:
        raise SystemExit("smoke不能充当正式模型对比")
    if args.model_path:
        config["training"]["model_path"] = str(Path(args.model_path).resolve())
    test_path = ROOT / args.data_dir / "test.jsonl"
    frozen = json.loads((run_dir / "data_manifest.json").read_text())["files"]["test.jsonl"]
    if file_sha256(test_path) != frozen:
        raise SystemExit("固定测试集摘要变化，拒绝与原run比较")
    test_rows = load_jsonl(test_path)
    if not test_rows:
        raise SystemExit("固定测试集为空")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("需要支持BF16的CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_location(config), **model_kwargs(config))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_location(config), torch_dtype=torch.bfloat16, attn_implementation="sdpa", **model_kwargs(config)).to("cuda")
    if args.variant == "adapter":
        model = PeftModel.from_pretrained(model, str(run_dir / "adapter"))
    model.eval()
    model.config.use_cache = True
    out_path = run_dir / f"{args.variant}_predictions.jsonl"
    if out_path.exists():
        raise SystemExit("预测文件已存在，不覆盖既有结果；请使用新的run或另存原结果")
    # 先写对比配置；即使中途崩溃，也能对部分预测按完整冻结分母评估。
    (run_dir / f"{args.variant}_prediction_config.json").write_text(json.dumps({"base_model": config["training"]["base_model"], "base_revision": config["training"]["base_revision"], "test_sha256": frozen, "do_sample": False, "enable_thinking": False, "max_input_tokens": config["models"]["online_max_input_tokens"], "max_new_tokens": config["training"]["inference_max_new_tokens"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_path, [])
    outputs = []
    for row in test_rows:
        item = {"sample_id": row["sample_id"], "variant": args.variant, "status": "ok", "output": ""}
        started = time.perf_counter()
        try:
            messages = build_training_messages(row)[:2]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            item["input_tokens"] = len(ids)
            if len(ids) > config["models"]["online_max_input_tokens"]:
                raise ValueError("INPUT_TOO_LONG；完整证据无法容纳，禁止截断")
            inputs = torch.tensor([ids], device="cuda")
            with torch.inference_mode():
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), do_sample=False, max_new_tokens=config["training"]["inference_max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
            item["output"] = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True).strip()
            item["output_tokens"] = int(generated.shape[1] - len(ids))
            item["hit_output_limit"] = item["output_tokens"] == config["training"]["inference_max_new_tokens"]
        except (ValueError, RuntimeError) as exc:
            item.update(status="failed", error=str(exc)[:500])
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        item["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        outputs.append(item)
        # 每题落盘，进程意外退出仍保留已完成记录；缺失题由评估器计作失败。
        write_jsonl(out_path, outputs)
    print(json.dumps({"variant": args.variant, "samples": len(outputs), "failed": sum(row["status"] == "failed" for row in outputs), "path": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
