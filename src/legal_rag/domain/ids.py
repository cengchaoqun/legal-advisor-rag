"""确定性标识：业务身份用 UUID，内容身份用摘要，保证重跑/重试得到同一标识。

这些函数是跨库（MongoDB/Milvus/Neo4j）共用 chunk_id 的唯一来源，
任何改动都会改变版本指纹，必须随 pipeline_revision 一起演进。
"""

from __future__ import annotations

import hashlib
import json
from typing import Iterable


def sha256_hex(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version_id(document_id: str, file_sha256: str, pipeline_revision: str) -> str:
    return sha256_hex(document_id, file_sha256, pipeline_revision)


def chunk_id(version_id_value: str, ordinal: int, content: str) -> str:
    return sha256_hex(version_id_value, str(ordinal), sha256_hex(content))


def entity_id(entity_type: str, identity_key: str, version_id_value: str | None = None) -> str:
    parts = [entity_type, identity_key]
    if version_id_value:
        parts.append(version_id_value)
    return sha256_hex(*parts)


def assertion_id(source_entity: str, relation: str, target_entity: str, support_chunk: str) -> str:
    return sha256_hex("assertion", source_entity, relation, target_entity, support_chunk)


def path_id(node_ids: Iterable[str], relation_types: Iterable[str]) -> str:
    return sha256_hex("path", *node_ids, "|", *relation_types)


def web_source_id(normalized_url: str, text_sha256: str) -> str:
    return "web:" + sha256_hex(normalized_url, text_sha256)


def stable_hash(payload: object) -> str:
    """对结构化参数做稳定摘要（键排序、UTF-8、紧凑分隔符）。"""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_hex(canonical)
