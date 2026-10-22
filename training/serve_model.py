"""服务器端先核验导出文件，再用同一目录启动vLLM，避免别名与权重错配。"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from legal_rag.adapters.llm import directory_digest
from training.data import file_sha256


def verify_export(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "deployment_manifest.json").read_text())
    files = manifest.get("files")
    if not files or not any(name.endswith(".safetensors") for name in files):
        raise ValueError("部署清单必须包含真实权重文件摘要")
    actual_names = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file() and path.name != "deployment_manifest.json"}
    if actual_names != set(files):
        raise ValueError("模型文件集合与清单不一致")
    for name, expected in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != expected:
            raise ValueError(f"模型文件摘要不一致或越界：{name}")
    if directory_digest(directory) != manifest["directory_sha256"]:
        raise ValueError("整体模型摘要不一致")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = verify_export(args.model_dir)
    print(json.dumps({"verified": True, "served_model_name": manifest["served_model_name"], "directory_sha256": manifest["directory_sha256"]}), flush=True)
    if not args.verify_only:
        key = os.environ.get("MODEL_SERVICE_API_KEY")
        if not key:
            raise SystemExit("请设置MODEL_SERVICE_API_KEY；密钥不写日志或清单")
        command = ["vllm", "serve", str(Path(args.model_dir).resolve()), "--host", "127.0.0.1", "--port", str(args.port), "--served-model-name", manifest["served_model_name"], "--dtype", "bfloat16", "--max-model-len", str(manifest["max_model_len"]), "--max-num-seqs", "1", "--gpu-memory-utilization", "0.85", "--enforce-eager", "--api-key", key]
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
