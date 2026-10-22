"""导入流水线集成测试（T03、T04、T05、T08、T28、T29、T32）。

使用确定性嵌入替身验证工程链路；真实 BGE-M3 语义效果不在本文件断言范围内。
"""

from __future__ import annotations

import pytest

from legal_rag.domain.enums import Availability, GraphState, ImportStatus, PublicationState, VectorState

from .helpers import CASE_MD, LAW_MD, claim_and_import, upload

pytestmark = pytest.mark.usefixtures("require_deps")


async def test_import_publishes_limited_when_effectivity_unknown(client, runtime):
    created = (await upload(client, filename="law.md", content=LAW_MD, title="测试用工资支付规则")).json()
    task = await claim_and_import(runtime, created["task_id"])
    assert task["status"] == ImportStatus.SUCCEEDED_WITH_WARNINGS.value
    assert task["stage"] == "done"

    version = await runtime.store.get_version(created["version_id"])
    assert version["publication_state"] == PublicationState.PUBLISHED.value
    assert version["availability"] == Availability.LIMITED.value
    assert version["vector_state"] == VectorState.READY.value
    assert version["published_at"] is not None
    assert "effectivity_unknown" in version["quality_flags"]

    document = await runtime.store.get_document(created["document_id"])
    assert document["current_version_id"] == created["version_id"]

    chunks = await runtime.collection_count("chunks", {"version_id": created["version_id"]})
    assert chunks > 2

    item = (await client.get(f"/api/documents/{created['document_id']}")).json()
    assert item["availability"] == "limited"
    assert item["version"]["pipeline_revision"] == version["pipeline_revision"]
    assert item["version"]["published_at"] is not None


async def test_metadata_missing_is_limited_not_blocked(client, runtime):
    text = "# 无日期测试资料\n\n第一条 未标注公布与施行日期的测试条款。\n\n第二条 另一条测试条款。"
    created = (await upload(client, filename="nodate.md", content=text, title="无日期资料")).json()
    task = await claim_and_import(runtime, created["task_id"])
    assert task["status"] == ImportStatus.SUCCEEDED_WITH_WARNINGS.value
    version = await runtime.store.get_version(created["version_id"])
    assert version["availability"] == Availability.LIMITED.value
    assert "metadata_missing" in version["quality_flags"]
    # 缺失元数据不进入人工审核，仍可被检索
    assert version["publication_state"] == "published"


async def test_graph_failure_publishes_limited_and_retry_only_graph(client, runtime, monkeypatch):
    created = (await upload(client, filename="law.md", content=LAW_MD, title="测试用工资支付规则")).json()

    async def boom(*args, **kwargs):
        raise RuntimeError("neo4j 注入失败")

    monkeypatch.setattr(runtime.graph, "write_version_graph", boom)
    task = await claim_and_import(runtime, created["task_id"])
    assert task["status"] == ImportStatus.SUCCEEDED_WITH_WARNINGS.value
    assert task["graph_state"] == GraphState.FAILED.value
    assert task["retryable"] is True
    assert "graph_unavailable" in task["warnings"]

    version = await runtime.store.get_version(created["version_id"])
    assert version["availability"] == Availability.LIMITED.value
    assert "write_graph" not in (task.get("stage_success") or [])
    chunks_before = await runtime.collection_count("chunks", {"version_id": created["version_id"]})

    # 恢复：重试只补图阶段，不重复切片
    monkeypatch.undo()
    requeued = await client.post(f"/api/import-tasks/{created['task_id']}/retry", json={"request_id": str(__import__("uuid").uuid4())})
    assert requeued.status_code == 202
    task2 = await claim_and_import(runtime, created["task_id"])
    assert task2["graph_state"] in {GraphState.READY.value, GraphState.EMPTY.value}
    chunks_after = await runtime.collection_count("chunks", {"version_id": created["version_id"]})
    assert chunks_after == chunks_before


async def test_vector_failure_keeps_old_version_visible(client, runtime, monkeypatch):
    first = (await upload(client, filename="law.md", content=LAW_MD, title="测试用工资支付规则")).json()
    await claim_and_import(runtime, first["task_id"])

    async def fail_encode(*args, **kwargs):
        raise RuntimeError("embedding 注入失败")

    monkeypatch.setattr(runtime.embedder, "encode", fail_encode)
    second = (
        await upload(
            client,
            filename="law.md",
            content=LAW_MD + "\n\n第八十六条 新增测试条款。",
            title="测试用工资支付规则",
            replace_document_id=first["document_id"],
        )
    ).json()
    assert second["version_id"] != first["version_id"]
    task = await claim_and_import(runtime, second["task_id"])
    assert task["status"] == ImportStatus.FAILED.value
    assert task["retryable"] is True

    new_version = await runtime.store.get_version(second["version_id"])
    assert new_version["vector_state"] == VectorState.FAILED.value
    assert new_version["availability"] == Availability.UNAVAILABLE.value
    assert new_version["publication_state"] == PublicationState.STAGED.value

    document = await runtime.store.get_document(first["document_id"])
    assert document["current_version_id"] == first["version_id"]
    item = (await client.get(f"/api/documents/{first['document_id']}")).json()
    assert item["availability"] == "limited"


async def test_case_citation_resolution_and_graph_browse(client, runtime):
    law = (await upload(client, filename="law.md", content=LAW_MD, title="测试用工资支付规则")).json()
    await claim_and_import(runtime, law["task_id"])
    case = (
        await upload(client, filename="case.md", content=CASE_MD, title="测试用案例摘要", doc_type="case")
    ).json()
    await claim_and_import(runtime, case["task_id"])

    # 引用能够唯一定位到已发布版本，形成有原文支持的 CITES 路径
    unresolved = await runtime.store.list_unresolved()
    assert all(item["cited_title"] != "测试用工资支付规则" for item in unresolved)

    entities = (await client.get("/api/graph/entities?q=测试用案例摘要")).json()
    assert entities["items"]
    entity_id = entities["items"][0]["entity_id"]
    relations = (await client.get(f"/api/graph/relations?entity_id={entity_id}&max_hops=2")).json()
    assert relations["paths"], "案例应能沿引用路径连到条款"
    node_ids = {node["entity_id"] for node in relations["nodes"]}
    chunk_ids = {chunk["chunk_id"] for chunk in relations["support_chunks"]}
    for path in relations["paths"]:
        assert set(path["node_ids"]) <= node_ids
        assert set(path["support_chunk_ids"]) <= chunk_ids
        assert len(path["relation_types"]) == len(path["node_ids"]) - 1

    # 撤销后实体不再可见，且不返回 503
    await client.delete(f"/api/documents/{case['document_id']}")
    after = await client.get(f"/api/graph/relations?entity_id={entity_id}&max_hops=2")
    assert after.status_code == 404


async def test_unknown_entity_returns_404(client):
    response = await client.get("/api/graph/relations?entity_id=not-exists&max_hops=2")
    assert response.status_code == 404
    invalid = await client.get("/api/graph/relations?entity_id=x&max_hops=3")
    assert invalid.status_code == 422
