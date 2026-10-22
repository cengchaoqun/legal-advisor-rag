"""BGE-M3 嵌入适配：一次推理同时产出 1024 维稠密与稀疏（词权重）向量。

约束：
- 导入与查询必须使用同一模型快照（EMBED_MODEL_REVISION），否则拒绝建索引；
- 稀疏输出保持 token_id → 权重，不转成语义字符串；
- 模型不可用时明确抛错，不静默换用其他稠密模型（会破坏混合检索契约）。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Sequence

from ..settings import Settings


class EmbeddingUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_model = None
_model_key: str | None = None


class BGEM3Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._async_lock = asyncio.Lock()

    def _load(self, revision: str):
        global _model, _model_key
        with _lock:
            if _model is not None and _model_key == revision:
                return _model
            try:
                from FlagEmbedding import BGEM3FlagModel
            except Exception as exc:  # 依赖未安装
                raise EmbeddingUnavailable(
                    "未安装 FlagEmbedding/torch，无法生成 BGE-M3 向量；请执行 uv sync --extra embedding"
                ) from exc
            try:
                # 有 CUDA 时用 GPU（保持 fp32，避免半精度带来的向量数值差异）；否则回落 CPU。
                try:
                    import torch

                    devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
                except Exception:
                    devices = ["cpu"]
                _model = BGEM3FlagModel(
                    self._settings.config.models.embedding,
                    use_fp16=False,
                    devices=devices,
                    revision=revision or None,
                )
                _model_key = revision
                return _model
            except Exception as exc:
                raise EmbeddingUnavailable(f"BGE-M3 模型加载失败：{exc}") from exc

    def _encode_sync(self, texts: Sequence[str], revision: str, batch_size: int) -> tuple[list[list[float]], list[dict[int, float]]]:
        model = self._load(revision)
        output = model.encode(
            list(texts),
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_rows = output["dense_vecs"]
        sparse_rows = output["lexical_weights"]
        dense: list[list[float]] = []
        for row in dense_rows:
            values = [float(value) for value in row]
            norm = sum(value * value for value in values) ** 0.5 or 1.0
            dense.append([value / norm for value in values])
        sparse: list[dict[int, float]] = []
        for row in sparse_rows:
            sparse.append({int(token): float(weight) for token, weight in row.items() if float(weight) > 0})
        return dense, sparse

    async def encode(self, texts: Sequence[str], revision: str | None = None) -> tuple[list[list[float]], list[dict[int, float]]]:
        if not texts:
            return [], []
        resolved = revision or self._settings.require_embedding_revision()
        batch = self._settings.config.models.embedding_batch_size
        async with self._async_lock:  # embedding 互斥：导入/查询共享同一实例
            return await asyncio.to_thread(self._encode_sync, texts, resolved, batch)

    async def health(self) -> dict[str, object]:
        revision = self._settings.env.embedding_revision(self._settings.config)
        try:
            import FlagEmbedding  # noqa: F401
        except Exception:
            return {"ok": False, "error": "FlagEmbedding 未安装", "revision": revision}
        return {"ok": True, "model": self._settings.config.models.embedding, "revision": revision}
