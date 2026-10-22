"""外部单卡24GB：Qwen3-8B BF16 SFT + LoRA，不做量化或全参数训练。"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.configuration import load_training_config, model_kwargs, model_location
from training.data import check_split_integrity, encode_rows, file_sha256, load_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="docs/contracts/默认配置.json")
    parser.add_argument("--model-path", help="服务器已下载的完整Qwen3-8B目录；可防止重复下载")
    parser.add_argument("--base-revision", help="该目录对应的实际40位模型commit SHA")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--data-dir", default="data/qwen3_sft_v1")
    parser.add_argument("--max-seq-length", type=int, choices=[1024, 1536], default=None)
    args = parser.parse_args()
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    config = load_training_config(args.config, args.model_path, args.base_revision)
    train = config["training"]
    if args.max_seq_length:
        train["max_seq_length"] = args.max_seq_length
        config["models"]["online_max_input_tokens"] = args.max_seq_length - train["inference_max_new_tokens"]
        config["serving"]["max_model_len"] = args.max_seq_length
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("本配置需要支持BF16的CUDA GPU；不静默降级CPU/FP16/QLoRA")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram < 22:
        raise SystemExit(f"检测到{vram:.1f}GiB，低于本24GB普通LoRA配置的预检门槛")
    if args.smoke_steps < 0:
        raise SystemExit("smoke-steps不得为负")
    set_seed(train["seed"])
    run_id = args.run_id or f"qwen3-8b-{'smoke' if args.smoke_steps else 'sft'}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        raise SystemExit("run-id必须是目录名")
    run_dir = ROOT / "artifacts" / run_id
    if run_dir.exists():
        raise SystemExit("run目录已存在；使用新run-id以免覆盖实验")
    data_dir = ROOT / args.data_dir
    splits = {split: load_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "validation", "test")}
    check_split_integrity(splits)
    if not all(splits.values()):
        raise SystemExit("训练/验证/固定测试集均不可为空")
    # tokenizer先校验，模型权重后加载；数据有问题时避免占满GPU。
    tokenizer = AutoTokenizer.from_pretrained(model_location(config), **model_kwargs(config))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    limits = {"max_seq_length": train["max_seq_length"], "max_input_tokens": config["models"]["online_max_input_tokens"], "max_output_tokens": train["inference_max_new_tokens"]}
    encoded, rejected = {}, {}
    for split in ("train", "validation"):
        rows = splits[split][:train["smoke_samples"]] if args.smoke_steps and split == "train" else splits[split]
        encoded[split], rejected[split] = encode_rows(rows, tokenizer, **limits)
    if not all(encoded.values()):
        raise SystemExit("真实tokenizer校验后训练或验证集为空；请检查长度报告与证据长度")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "rejected_lengths.jsonl", [{"split": split, **item} for split, items in rejected.items() for item in items])
    manifest = {"data_dir": str(data_dir), "files": {path.name: file_sha256(path) for path in sorted(data_dir.glob("*.jsonl"))}, "frozen_counts": {split: len(rows) for split, rows in splits.items()}, "used_counts": {split: len(rows) for split, rows in encoded.items()}, "rejected_counts": {split: len(rows) for split, rows in rejected.items()}, "smoke": bool(args.smoke_steps), "note": "超长样本剔除并报告，不截断；固定测试集完整保留，训练不使用test"}
    (run_dir / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    model = AutoModelForCausalLM.from_pretrained(model_location(config), torch_dtype=torch.bfloat16, attn_implementation="sdpa", **model_kwargs(config))
    if model.config.model_type != "qwen3":
        raise SystemExit("模型类型不是qwen3")
    missing = [name for name in train["target_modules"] if not any(key.endswith('.' + name) for key, _ in model.named_modules())]
    if missing:
        raise SystemExit(f"LoRA目标模块不存在：{missing}")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=train["r"], lora_alpha=train["alpha"], lora_dropout=train["dropout"], target_modules=train["target_modules"], task_type="CAUSAL_LM", bias="none"))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    training_args = TrainingArguments(output_dir=str(run_dir / "checkpoints"), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=train["gradient_accumulation_steps"], num_train_epochs=train["epochs"], learning_rate=train["learning_rate"], warmup_ratio=train["warmup_ratio"], weight_decay=train["weight_decay"], max_grad_norm=train["max_grad_norm"], gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="epoch", eval_strategy="epoch", load_best_model_at_end=not bool(args.smoke_steps), metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, bf16=True, fp16=False, optim="adamw_torch", report_to=[], seed=train["seed"], max_steps=args.smoke_steps or -1, eval_accumulation_steps=1, prediction_loss_only=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(encoded["train"]), eval_dataset=Dataset.from_list(encoded["validation"]), data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        (run_dir / "failure.json").write_text(json.dumps({"error": "CUDA_OOM", "suggestion": "停止其他GPU进程；以新run-id和--max-seq-length 1024重做10步探针；不自动改QLoRA"}, ensure_ascii=False), encoding="utf-8")
        raise
    adapter_dir = run_dir / ("adapter_smoke" if args.smoke_steps else "adapter")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    environment = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "vram_gib": round(vram, 2), "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2)}
    from importlib.metadata import version
    environment["packages"] = {name: version(name) for name in ("transformers", "peft", "datasets", "accelerate")}
    (run_dir / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(run_dir / "train_log.jsonl", trainer.state.log_history)
    summary = {"run_id": run_id, "smoke": bool(args.smoke_steps), "global_step": trainer.state.global_step, "train_seconds": round(time.perf_counter() - started, 1), "adapter_dir": str(adapter_dir.relative_to(ROOT)), "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_loss": trainer.state.best_metric, "environment": environment}
    (run_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
