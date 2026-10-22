"""租约、CAS 与崩溃恢复（T17、T24）。"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest

from legal_rag.domain.enums import ImportStatus, QueryState
from legal_rag.storage.mongo import utcnow

from .helpers import LAW_MD, create_query, create_session, upload

pytestmark = pytest.mark.usefixtures("require_deps")


async def test_stale_lease_cannot_overwrite(client, runtime):
    created = (await upload(client, filename="law.md", content=LAW_MD, title="租约测试资料")).json()
    claimed = await runtime.store.claim_task("worker-a", 60)
    assert claimed and claimed["task_id"] == created["task_id"]

    # 模拟租约过期后被 worker-b 接管
    await runtime.store.collection("import_tasks").update_one(
        {"task_id": created["task_id"]},
        {"$set": {"lease_until": utcnow() - timedelta(seconds=5)}},
    )
    claimed_b = await runtime.store.claim_task("worker-b", 60)
    assert claimed_b and claimed_b["lease_token"] == "worker-b"

    stale = await runtime.store.update_task_cas(
        created["task_id"], "worker-a", {"stage": "parse", "status": ImportStatus.RUNNING.value}
    )
    assert stale is False
    fresh = await runtime.store.update_task_cas(
        created["task_id"], "worker-b", {"stage": "parse", "status": ImportStatus.RUNNING.value}
    )
    assert fresh is True
    task = await runtime.store.get_task(created["task_id"])
    assert task["lease_token"] == "worker-b"


async def test_expired_running_query_recovers_as_interrupted(client, runtime):
    session_id = await create_session(client)
    created = await create_query(client, session_id, "公司拖欠工资怎么办？")
    query_id = created["query_id"]
    await runtime.store.collection("queries").update_one(
        {"query_id": query_id},
        {
            "$set": {
                "status": QueryState.RUNNING.value,
                "lease_token": "died-worker",
                "lease_until": utcnow() - timedelta(seconds=30),
                "answer": "已提交的部分答案",
                "last_event_seq": 2,
                "revision": 2,
                "event_log": [
                    {"seq": 1, "event": "started", "payload": {"status": "running"}, "created_at": utcnow()},
                    {"seq": 2, "event": "delta", "payload": {"text": "已提交的部分答案"}, "created_at": utcnow()},
                ],
            }
        },
    )
    recovered = await runtime.store.recover_expired_queries()
    assert query_id in recovered
    doc = await runtime.store.get_query(query_id)
    assert doc["status"] == QueryState.INTERRUPTED.value
    assert doc["answer"] == "已提交的部分答案"
    assert doc["event_log"][-1]["event"] == "failed"
    assert doc["event_log"][-1]["payload"]["status"] == QueryState.INTERRUPTED.value
    assert doc["last_event_seq"] == 3

    # 再次恢复不重复追加终止事件
    again = await runtime.store.recover_expired_queries()
    assert query_id not in again
    doc2 = await runtime.store.get_query(query_id)
    assert len(doc2["event_log"]) == 3


async def test_query_request_id_is_idempotent(client, runtime):
    session_id = await create_session(client)
    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "question": "公司拖欠工资怎么办？",
        "document_ids": [],
        "attachment_ids": [],
        "relevant_date": None,
        "location": None,
        "web_enabled": False,
    }
    first = await client.post(f"/api/sessions/{session_id}/queries", json=payload)
    second = await client.post(f"/api/sessions/{session_id}/queries", json=payload)
    assert first.status_code == 202 and second.status_code == 202
    assert first.json()["query_id"] == second.json()["query_id"]
    count = await runtime.store.collection_count("queries", {"session_id": session_id})
    assert count == 1


async def test_cancel_is_idempotent_and_does_not_touch_terminal(client, runtime):
    from .helpers import run_query

    session_id = await create_session(client)
    created = await create_query(client, session_id, "公司拖欠工资怎么办？")
    await run_query(runtime, created["query_id"])
    response = await client.post(f"/api/queries/{created['query_id']}/cancel", json={"request_id": str(uuid.uuid4())})
    assert response.status_code == 200
    body = response.json()
    assert body["cancel_requested"] is False
    assert body["status"] in {"completed", "completed_with_warnings"}
