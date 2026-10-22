"""集成测试辅助：上传、直接驱动导入流水线、构造查询。"""

from __future__ import annotations

import uuid
from datetime import timedelta

from pymongo import ReturnDocument

from legal_rag.storage.mongo import utcnow
from legal_rag.worker.main import Worker

# 明确标注为测试夹具的合成文本，不含真实法律条文，仅用于验证工程链路
LAW_MD = """# 测试用工资支付规则（合成）

（2020年12月29日公布）
自2021年1月1日起施行

第一条 用人单位应当按照约定周期足额支付劳动报酬，不得克扣或者无故拖欠。

第八十五条 用人单位拖欠劳动报酬的，应当限期支付；逾期不支付的，应当加付赔偿金。
"""

CASE_MD = """# 测试用案例摘要（合成）

案号：（2021）测01民终1234号

基本事实：劳动者主张用人单位拖欠劳动报酬。

裁判理由：依照《测试用工资支付规则》第八十五条，用人单位应当限期支付劳动报酬。
"""


async def upload(
    client,
    *,
    filename: str,
    content: str,
    request_id: uuid.UUID | None = None,
    title: str | None = None,
    doc_type: str = "law",
    replace_document_id: str | None = None,
):
    data = {
        "request_id": str(request_id or uuid.uuid4()),
        "doc_type": doc_type,
    }
    if title:
        data["title"] = title
    if replace_document_id:
        data["replace_document_id"] = replace_document_id
    files = {"file": (filename, content.encode("utf-8"), "text/markdown")}
    return await client.post("/api/documents", data=data, files=files)


async def create_session(client) -> str:
    response = await client.post("/api/sessions", json={"title": "集成测试会话"})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


async def create_query(client, session_id: str, question: str, **overrides) -> dict:
    payload = {
        "request_id": str(uuid.uuid4()),
        "question": question,
        "document_ids": [],
        "attachment_ids": [],
        "relevant_date": None,
        "location": None,
        "web_enabled": False,
        **overrides,
    }
    response = await client.post(f"/api/sessions/{session_id}/queries", json=payload)
    assert response.status_code == 202, response.text
    return response.json()


async def run_query(runtime, query_id: str, *, answer_model=None) -> dict:
    """按 worker 路径领取并执行查询，便于断言事件与终态。"""
    from legal_rag.retrieval.query_pipeline import QueryPipeline

    claimed = await runtime.store.collection("queries").find_one_and_update(
        {"query_id": query_id, "status": "queued"},
        {
            "$set": {
                "status": "running",
                "lease_token": "test-worker",
                "lease_until": utcnow() + timedelta(seconds=300),
                "updated_at": utcnow(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    assert claimed is not None, "查询未被领取"
    pipeline = QueryPipeline(
        runtime.settings,
        runtime.store,
        runtime.vectors,
        runtime.graph,
        runtime.embedder,
        answer_model or runtime.answer_model,
        runtime.aux_model,
        runtime.reranker,
        runtime.web,
        runtime.visibility,
    )
    await pipeline.run(query_id, "test-worker")
    return await runtime.store.get_query(query_id) or {}


async def claim_and_import(runtime, task_id: str) -> dict:
    """按 worker 的真实路径领取并执行导入，验证租约、CAS 与阶段恢复。"""
    task = await runtime.store.collection("import_tasks").find_one_and_update(
        {"task_id": task_id},
        {
            "$set": {
                "status": "running",
                "lease_token": "test-worker",
                "lease_until": utcnow() + timedelta(seconds=600),
                "updated_at": utcnow(),
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    worker = Worker(runtime)
    await worker._handle_import(task)  # noqa: SLF001 - 复用真实执行路径
    return await runtime.store.get_task(task_id) or {}
