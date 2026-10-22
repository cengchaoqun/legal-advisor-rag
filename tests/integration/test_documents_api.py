"""资料接口与身份规则集成测试（T01、T02、T22、T28、T29）。"""

from __future__ import annotations

import uuid

import pytest

from .helpers import CASE_MD, LAW_MD, upload

pytestmark = pytest.mark.usefixtures("require_deps")


async def test_same_request_same_file_is_idempotent(client):
    request_id = uuid.uuid4()
    first = (await upload(client, filename="a.md", content=LAW_MD, request_id=request_id)).json()
    second = (await upload(client, filename="a.md", content=LAW_MD, request_id=request_id)).json()
    assert first["document_id"] == second["document_id"]
    assert first["version_id"] == second["version_id"]
    assert first["task_id"] == second["task_id"]
    assert second["duplicate"] is True


async def test_same_request_different_file_conflicts(client):
    request_id = uuid.uuid4()
    assert (await upload(client, filename="a.md", content=LAW_MD, request_id=request_id)).status_code == 202
    conflict = await upload(client, filename="b.md", content=CASE_MD, request_id=request_id)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


async def test_same_content_different_request_reuses_registration(client):
    first = (await upload(client, filename="a.md", content=LAW_MD)).json()
    second = (await upload(client, filename="a.md", content=LAW_MD)).json()
    assert second["duplicate"] is True
    assert second["document_id"] == first["document_id"]


async def test_explicit_replace_keeps_target_identity(client):
    # B 先导入同一份文件
    doc_b = (await upload(client, filename="x.md", content=LAW_MD, title="B 资料")).json()
    # A 是另一份资料，显式用同一份文件替换
    doc_a = (await upload(client, filename="a.md", content=CASE_MD, doc_type="case", title="A 资料")).json()
    replaced = (
        await upload(
            client,
            filename="x.md",
            content=LAW_MD,
            title="A 资料",
            replace_document_id=doc_a["document_id"],
        )
    ).json()
    assert replaced["document_id"] == doc_a["document_id"]
    # 内容与 B 相同但身份必须属于 A：对象可复用，版本号不能借用 B
    assert replaced["version_id"] != doc_b["version_id"]
    assert replaced["duplicate"] is False

    versions = (await client.get(f"/api/documents/{doc_a['document_id']}/versions")).json()
    assert versions["items"]
    assert versions["items"][0]["version_id"] == replaced["version_id"]


async def test_list_item_shape_and_filters(client):
    created = (await upload(client, filename="工资规则.md", content=LAW_MD, title="测试用工资规则")).json()
    listing = (await client.get("/api/documents?limit=20")).json()
    item = next(row for row in listing["items"] if row["document"]["document_id"] == created["document_id"])
    assert set(item) == {"document", "version", "availability", "latest_task"}
    # 尚未发布：无当前版本，派生可用性为 unavailable，但最近任务可见
    assert item["version"] is None
    assert item["availability"] == "unavailable"
    assert item["latest_task"]["task_id"] == created["task_id"]

    filtered = (await client.get("/api/documents?availability=unavailable")).json()
    assert all(row["availability"] == "unavailable" for row in filtered["items"])

    by_title = (await client.get("/api/documents?q=工资规则")).json()
    assert any(row["document"]["document_id"] == created["document_id"] for row in by_title["items"])

    blank = await client.get("/api/documents?q=%20%20")
    assert blank.status_code == 422

    cursor_ok = (await client.get("/api/documents?limit=1")).json()
    if cursor_ok.get("next_cursor"):
        tampered = await client.get("/api/documents?limit=2&cursor=" + cursor_ok["next_cursor"])
        assert tampered.status_code in (200, 400)


async def test_version_traceability_fields(client):
    created = (await upload(client, filename="a.md", content=LAW_MD, title="版本溯源资料")).json()
    versions = (await client.get(f"/api/documents/{created['document_id']}/versions")).json()
    staged = versions["items"][0]
    assert len(staged["pipeline_revision"]) == 64
    assert staged["published_at"] is None
    assert staged["publication_state"] == "staged"
