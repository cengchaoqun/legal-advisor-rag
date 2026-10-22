"""外部训练统一配置：同一run复用保存配置，不受后续业务默认值变化影响。"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_training_config(path, model_path=None, base_revision=None):
    config = json.loads((ROOT / path).read_text(encoding="utf-8-sig"))
    if model_path:
        location = Path(model_path).expanduser().resolve()
        if not (location / "config.json").exists():
            raise ValueError(f"本地模型目录无config.json：{location}")
        metadata = json.loads((location / "config.json").read_text())
        if metadata.get("model_type") != "qwen3":
            raise ValueError("本次配置仅接受Qwen3模型，不静默换成其他基础模型")
        config["training"]["model_path"] = str(location)
    if base_revision:
        config["training"]["base_revision"] = base_revision
    revision = config["training"].get("base_revision")
    if not revision or not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("必须以 --base-revision 指定实际下载快照的40位commit SHA；不能用main/占位标记")
    return config


def model_location(config):
    train = config["training"]
    return train.get("model_path") or train["base_model"]


def model_kwargs(config):
    return {"revision": config["training"]["base_revision"], "local_files_only": bool(config["training"].get("model_path")), "trust_remote_code": False}


def saved_config(run_dir):
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
