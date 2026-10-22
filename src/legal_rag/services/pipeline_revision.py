"""处理指纹：解析器/切片/抽取/嵌入快照的确定性摘要。

相同文件在同一指纹下产生相同 version_id；指纹变化（例如切片策略或嵌入模型快照变更）
会产生新版本，便于用户在 /versions 追溯“为什么重新处理”。
"""

from __future__ import annotations

from ..domain.ids import stable_hash
from ..settings import Settings

PARSER_REVISION = "pypdf_text+markdown_v1"
EXTRACTOR_REVISION = "rule_extract_v1"


def pipeline_revision(settings: Settings, embedding_revision: str) -> str:
    config = settings.config.ingestion
    return stable_hash(
        {
            "parser": PARSER_REVISION,
            "extractor": EXTRACTOR_REVISION,
            "chunk": {
                "min": config.chunk_target_min_chars,
                "max": config.chunk_target_max_chars,
                "soft_max": config.chunk_soft_max_chars,
                "overlap": config.chunk_overlap_chars,
            },
            "embedding": embedding_revision,
            "embedding_dim": settings.config.models.embedding_dimension,
        }
    )
