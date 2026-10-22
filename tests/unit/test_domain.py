"""领域契约与确定性标识单元测试（对应 T01、T29、T30 的身份规则）。"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from legal_rag.domain.ids import chunk_id, sha256_hex, stable_hash, version_id
from legal_rag.domain.models import Evidence, Path, SampleEvidence


def test_version_and_chunk_ids_are_deterministic():
    document_id = str(uuid.uuid4())
    v1 = version_id(document_id, "a" * 64, "rev-1")
    assert v1 == version_id(document_id, "a" * 64, "rev-1")
    assert v1 != version_id(document_id, "a" * 64, "rev-2")
    assert chunk_id(v1, 0, "正文") == chunk_id(v1, 0, "正文")
    assert chunk_id(v1, 1, "正文") != chunk_id(v1, 0, "正文")
    assert sha256_hex("x") != sha256_hex("y")


def test_sample_evidence_rejects_query_identity():
    base = {
        "evidence_id": "E1",
        "source_id": "a" * 64,
        "source_type": "local",
        "chunk_id": "b" * 64,
        "version_id": "c" * 64,
        "title": "离线片段",
        "text": "正文",
        "url": None,
        "location": None,
        "metadata": None,
        "quality_flags": [],
    }
    SampleEvidence.model_validate(base)
    with pytest.raises(ValidationError):
        SampleEvidence.model_validate({**base, "query_id": str(uuid.uuid4())})


def test_online_evidence_requires_query_id():
    payload = {
        "evidence_id": "E1",
        "source_id": "a" * 64,
        "source_type": "local",
        "chunk_id": "b" * 64,
        "version_id": "c" * 64,
        "title": "在线片段",
        "text": "正文",
        "url": None,
        "location": None,
        "metadata": None,
        "quality_flags": [],
    }
    with pytest.raises(ValidationError):
        Evidence.model_validate(payload)
    Evidence.model_validate({**payload, "query_id": str(uuid.uuid4())})


def test_path_requires_relation_count_matching_nodes():
    payload = {
        "path_id": "p1",
        "node_ids": ["a", "b", "c"],
        "relation_types": ["CITES"],
        "support_chunk_ids": ["chunk"],
        "status": "source_supported",
        "version_ids": ["v1"],
    }
    with pytest.raises(ValidationError):
        Path.model_validate(payload)
    Path.model_validate({**payload, "relation_types": ["CITES", "CONCERNS"]})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SampleEvidence.model_validate(
            {
                "evidence_id": "E1",
                "source_id": "a",
                "source_type": "local",
                "chunk_id": None,
                "version_id": None,
                "title": "x",
                "text": "y",
                "url": None,
                "location": None,
                "metadata": None,
                "quality_flags": [],
                "guess": 1,
            }
        )


def test_stable_hash_ignores_key_order():
    payload = {"parser": "p", "chunk": {"min": 400}, "embedding": "rev"}
    assert stable_hash(payload) == stable_hash(dict(reversed(list(payload.items()))))
