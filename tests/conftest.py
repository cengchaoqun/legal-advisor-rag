"""测试环境装配：隔离数据库、可替换嵌入器、异步 HTTP 客户端。

说明：仅在测试中使用确定性向量替身（TEST_DOUBLE），用于验证检索/发布/事件链路，
不代表真实 BGE-M3 语义效果；生产路径必须使用真实嵌入模型（见 adapters/embedding.py）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("APP_MODE", "test")
os.environ.setdefault("MONGO_DB", "legal_rag_test")
os.environ.setdefault("LEGAL_RAG_ROOT", str(ROOT))
os.environ.setdefault("EMBED_MODEL_REVISION", "test-revision-1")

from httpx import ASGITransport, AsyncClient  # noqa: E402

from legal_rag.api.main import create_app  # noqa: E402
from legal_rag.domain.ids import sha256_hex  # noqa: E402
from legal_rag.runtime import Runtime  # noqa: E402


class DeterministicEmbedder:
    """测试替身：同文本产生同向量，保证可重复；不做语义相似度评估。"""

    name = "TEST_DOUBLE"

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    async def encode(self, texts, revision=None):
        self.calls.append(list(texts))
        dense: list[list[float]] = []
        sparse: list[dict[int, float]] = []
        for text in texts:
            seed = sha256_hex(text)
            raw = [int(seed[index % 64 : index % 64 + 2], 16) / 255.0 for index in range(self.dimension)]
            norm = sum(value * value for value in raw) ** 0.5 or 1.0
            dense.append([value / norm for value in raw])
            tokens: dict[int, float] = {}
            for token in text.replace("\n", " ").split():
                key = int(sha256_hex(token)[:8], 16) % 100000
                tokens[key] = tokens.get(key, 0.0) + 1.0
            sparse.append(tokens or {1: 1.0})
        return dense, sparse

    async def health(self):
        return {"ok": True, "double": True}


@pytest.fixture()
async def runtime():
    """每个测试独立装配：避免跨事件循环复用异步客户端；依赖缺失时明确跳过。"""
    rt = Runtime()
    try:
        await rt.store.connect()
    except Exception as exc:  # 中间件未启动：集成测试标记阻塞而不是伪造通过
        pytest.skip(f"MongoDB 不可用，集成测试阻塞：{exc}")
    await rt.startup()
    await rt.store.ensure_indexes()
    try:
        yield rt
    finally:
        await rt.shutdown()


@pytest.fixture()
async def require_deps(runtime):
    """集成用例要求 Milvus/Neo4j/对象存储可用；任一缺失则明确跳过。"""
    health = await runtime.health()
    missing = [name for name in ("milvus", "neo4j", "object_store") if health["checks"].get(name) != "ok"]
    if missing:
        pytest.skip("中间件未就绪（" + "、".join(missing) + "），集成测试阻塞")


@pytest.fixture()
async def clean_db(runtime):
    db = runtime.store.db
    for name in await db.list_collection_names():
        if name.startswith("system."):
            continue
        await db[name].delete_many({})
    await runtime.budget.ensure_account()
    yield


@pytest.fixture()
def app(runtime, clean_db, monkeypatch):
    rt = runtime
    double = DeterministicEmbedder()
    app = create_app(rt)
    app.state.runtime = rt
    original = rt.embedder
    rt.embedder = double
    yield app
    rt.embedder = original


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
