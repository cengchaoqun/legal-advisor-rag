"""SSE 重放、游标校验与过期语义（T13、T33）。"""

from __future__ import annotations

import json

import pytest

from .helpers import create_query, create_session, run_query

pytestmark = pytest.mark.usefixtures("require_deps")


async def _clarify_query(client, runtime):
    session_id = await create_session(client)
    created = await create_query(client, session_id, "公司拖欠工资怎么办？")
    await run_query(runtime, created["query_id"])
    return created["query_id"]


async def _read_stream(client, url, headers=None):
    events = []
    async with client.stream("GET", url, headers=headers or {}) as response:
        assert response.status_code == 200, await response.aread()
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


async def test_replay_returns_same_events_without_duplicates(client, runtime):
    query_id = await _clarify_query(client, runtime)
    first = await _read_stream(client, f"/api/queries/{query_id}/events")
    assert [item["seq"] for item in first] == list(range(1, len(first) + 1))
    assert first[-1]["event"] == "done"
    terminal = next(item for item in first if item["event"] == "done")
    assert terminal["payload"]["last_event_seq"] == len(first)

    replay = await _read_stream(
        client,
        f"/api/queries/{query_id}/events",
        headers={"Last-Event-ID": f"{query_id}:{len(first) - 1}"},
    )
    assert [item["seq"] for item in replay] == [len(first)]
    assert replay[0]["event"] == "done"


async def test_cursor_errors_are_400(client, runtime):
    query_id = await _clarify_query(client, runtime)
    wrong_query = await client.get(
        f"/api/queries/{query_id}/events", headers={"Last-Event-ID": f"{'0' * 8}-0000-0000-0000-000000000000:1"}
    )
    assert wrong_query.status_code == 400

    future = await client.get(f"/api/queries/{query_id}/events?after_seq=999")
    assert future.status_code == 400

    mismatch = await client.get(
        f"/api/queries/{query_id}/events?after_seq=1", headers={"Last-Event-ID": f"{query_id}:2"}
    )
    assert mismatch.status_code == 400


async def test_expired_events_return_410(client, runtime):
    query_id = await _clarify_query(client, runtime)
    await runtime.store.collection("queries").update_one(
        {"query_id": query_id}, {"$set": {"events_expired": True}}
    )
    response = await client.get(f"/api/queries/{query_id}/events")
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "EVENT_HISTORY_EXPIRED"

    snapshot = (await client.get(f"/api/queries/{query_id}")).json()
    assert snapshot["status"] == "completed"
    assert snapshot["answer"]


async def test_unknown_query_is_404(client, runtime):
    response = await client.get("/api/queries/00000000-0000-0000-0000-000000000000/events")
    assert response.status_code == 404
