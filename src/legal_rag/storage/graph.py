"""Neo4j 图存储：实体、关系与原文证据定位。

约束：
- 所有关系边必须携带 assertion_id，可反查 RelationEvidence 与支持切片；
- 版本化节点（Provision/Case/Version）带 version_ids，查询始终携带可见版本白名单；
- 只使用固定模板与参数，不拼接用户输入为 Cypher；
- SUPERSEDES 只接受原文明确替代，不由日期推断（抽取阶段控制）。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from neo4j import AsyncDriver, AsyncGraphDatabase

from ..settings import Settings

SEMANTIC_RELATIONS = (
    "HAS_VERSION",
    "HAS_PROVISION",
    "CITES",
    "CONCERNS",
    "ISSUED_BY",
    "SUPERSEDES",
)

TEMPLATES: dict[str, str] = {
    # 版本内条款回取：条款正文必须回取其自身 SUPPORTED_BY 切片
    "provision_lookup": """
        MATCH (v:DocumentVersion {version_id: $version_id})-[:HAS_PROVISION]->(p:Provision)
        WHERE p.article IN $articles
        MATCH (p)-[:SUPPORTED_BY]->(e:EvidenceChunk)
        WHERE e.version_id = $version_id
        RETURN p.entity_id AS provision_id, p.article AS article, e.chunk_id AS chunk_id
        ORDER BY p.article, e.chunk_id
        LIMIT $limit
    """,
    # 争议主题下的案例证据
    "issue_cases": """
        MATCH (i:Issue {entity_id: $issue_id})<-[:CONCERNS]-(c:Case)
        WHERE any(v IN c.version_ids WHERE v IN $visible_version_ids)
        MATCH (c)-[:SUPPORTED_BY]->(e:EvidenceChunk)
        WHERE e.version_id IN $visible_version_ids
        RETURN c.entity_id AS case_id, e.chunk_id AS chunk_id, c.label AS case_label
        ORDER BY c.entity_id, e.chunk_id
        LIMIT $limit
    """,
    # 案例引用条款及其来源
    "case_provisions": """
        MATCH (c:Case {entity_id: $case_id})-[r:CITES]->(p:Provision)
        MATCH (a:RelationEvidence {assertion_id: r.assertion_id})-[:SUPPORTED_BY]->(e:EvidenceChunk)
        WHERE a.status = 'source_supported'
          AND c.version_ids IN $visible_version_ids
          AND p.version_ids IN $visible_version_ids
          AND e.version_id IN $visible_version_ids
        RETURN c.entity_id AS case_id, p.entity_id AS provision_id,
               a.assertion_id AS assertion_id, e.chunk_id AS support_chunk_id
        ORDER BY assertion_id
        LIMIT $limit
    """,
    # 版本内条款列表
    "version_provisions": """
        MATCH (v:DocumentVersion {version_id: $version_id})-[:HAS_PROVISION]->(p:Provision)
        RETURN p.entity_id AS provision_id, p.article AS article
        ORDER BY p.article
        LIMIT $limit
    """,
}


@dataclass(slots=True)
class GraphPath:
    node_ids: list[str]
    relation_types: list[str]
    assertion_ids: list[str | None]
    labels: list[str]
    entity_types: list[str]
    support_chunk_ids: list[str] = field(default_factory=list)
    version_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Neighborhood:
    paths: list[GraphPath]
    nodes: list[dict[str, Any]]
    support_chunk_ids: list[str]
    truncated: bool


class GraphStore:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._driver: AsyncDriver | None = None

    def _driver_or_raise(self) -> AsyncDriver:
        if self._driver is None:
            env = self._settings.env
            self._driver = AsyncGraphDatabase.driver(
                env.neo4j_uri, auth=(env.neo4j_user, env.neo4j_password)
            )
        return self._driver

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def _run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        driver = self._driver_or_raise()
        async with driver.session() as session:
            result = await session.run(query, params or {})
            return [record.data() async for record in result]

    async def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT legal_document_id IF NOT EXISTS FOR (n:LegalDocument) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT document_version_id IF NOT EXISTS FOR (n:DocumentVersion) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT provision_id IF NOT EXISTS FOR (n:Provision) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (n:Case) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (n:Issue) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (n:Organization) REQUIRE n.entity_id IS UNIQUE",
            "CREATE CONSTRAINT evidence_chunk_id IF NOT EXISTS FOR (n:EvidenceChunk) REQUIRE n.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT relation_evidence_id IF NOT EXISTS FOR (n:RelationEvidence) REQUIRE n.assertion_id IS UNIQUE",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (n:Provision) ON (n.label)",
            "CREATE INDEX case_label IF NOT EXISTS FOR (n:Case) ON (n.label)",
        ]
        for statement in statements:
            await self._run(statement)

    # ---------- 写入 ----------
    async def write_version_graph(self, payload: dict[str, Any]) -> None:
        """幂等写入一个版本的图结构：节点 MERGE，边按 assertion_id 去重。"""
        document = payload["document"]
        version = payload["version"]
        provisions = payload.get("provisions", [])
        chunks = payload.get("chunks", [])
        case = payload.get("case")
        issues = payload.get("issues", [])
        organization = payload.get("organization")
        citations = payload.get("citations", [])

        await self.ensure_constraints()
        doc_entity = document["document_id"]
        version_entity = version["version_id"]

        await self._run(
            """
            MERGE (d:LegalDocument {entity_id: $doc_id})
            SET d.label = $title, d.entity_type = 'LegalDocument', d.doc_type = $doc_type,
                d.version_ids = CASE WHEN $vid IN coalesce(d.version_ids, []) THEN d.version_ids
                                     ELSE coalesce(d.version_ids, []) + $vid END
            MERGE (v:DocumentVersion {entity_id: $vid})
            SET v.version_id = $vid, v.document_id = $doc_id, v.label = $title,
                v.entity_type = 'DocumentVersion', v.availability = $availability,
                v.version_ids = [$vid]
            MERGE (d)-[:HAS_VERSION]->(v)
            """,
            {
                "doc_id": doc_entity,
                "title": document["title"],
                "doc_type": document["doc_type"],
                "vid": version_entity,
                "availability": version.get("availability", "limited"),
            },
        )

        # 证据切片节点：图浏览直接按 chunk_id 回取 MongoDB 正文
        if chunks:
            await self._run(
                """
                UNWIND $rows AS row
                MATCH (v:DocumentVersion {entity_id: $vid})
                MERGE (e:EvidenceChunk {chunk_id: row.chunk_id})
                SET e.entity_id = row.chunk_id, e.label = row.label, e.entity_type = 'EvidenceChunk',
                    e.version_id = $vid, e.ordinal = row.ordinal, e.version_ids = [$vid]
                MERGE (v)-[:SUPPORTED_BY]->(e)
                """,
                {"vid": version_entity, "rows": chunks},
            )

        if provisions:
            await self._run(
                """
                UNWIND $rows AS row
                MATCH (v:DocumentVersion {entity_id: $vid})
                MERGE (p:Provision {entity_id: row.entity_id})
                SET p.label = row.label, p.entity_type = 'Provision', p.article = row.article,
                    p.document_id = $doc_id, p.version_ids = [$vid], p.chunk_id = row.chunk_id
                MERGE (v)-[:HAS_PROVISION]->(p)
                WITH p, row
                MATCH (e:EvidenceChunk {chunk_id: row.chunk_id})
                MERGE (p)-[:SUPPORTED_BY]->(e)
                """,
                {"vid": version_entity, "doc_id": doc_entity, "rows": provisions},
            )

        if case:
            await self._run(
                """
                MATCH (v:DocumentVersion {entity_id: $vid})
                MERGE (c:Case {entity_id: $case_id})
                SET c.label = $label, c.entity_type = 'Case', c.case_identity_key = $identity_key,
                    c.document_id = $doc_id, c.version_ids = [$vid], c.chunk_id = $chunk_id
                MERGE (v)-[:HAS_VERSION]->(c)
                WITH c
                MATCH (e:EvidenceChunk {chunk_id: $chunk_id})
                MERGE (c)-[:SUPPORTED_BY]->(e)
                """,
                {
                    "vid": version_entity,
                    "doc_id": doc_entity,
                    "case_id": case["entity_id"],
                    "label": case["label"],
                    "identity_key": case["case_identity_key"],
                    "chunk_id": case["chunk_id"],
                },
            )

        for issue in issues:
            await self._run(
                """
                MERGE (i:Issue {entity_id: $entity_id})
                SET i.label = $label, i.entity_type = 'Issue', i.scenario = $scenario,
                    i.version_ids = CASE WHEN $vid IN coalesce(i.version_ids, []) THEN i.version_ids
                                         ELSE coalesce(i.version_ids, []) + $vid END
                """,
                {
                    "entity_id": issue["entity_id"],
                    "label": issue["label"],
                    "scenario": issue["scenario"],
                    "vid": version_entity,
                },
            )
            source_label = "Case" if case else "Provision"
            source_id = case["entity_id"] if case else (provisions[0]["entity_id"] if provisions else None)
            if source_id:
                await self._run(
                    f"""
                    MATCH (s:{source_label} {{entity_id: $source_id}})
                    MATCH (i:Issue {{entity_id: $issue_id}})
                    MERGE (s)-[r:CONCERNS]->(i)
                    ON CREATE SET r.assertion_id = $assertion_id
                    """,
                    {
                        "source_id": source_id,
                        "issue_id": issue["entity_id"],
                        "assertion_id": issue["assertion_id"],
                    },
                )

        if organization:
            await self._run(
                """
                MERGE (o:Organization {entity_id: $entity_id})
                SET o.label = $label, o.entity_type = 'Organization',
                    o.version_ids = CASE WHEN $vid IN coalesce(o.version_ids, []) THEN o.version_ids
                                         ELSE coalesce(o.version_ids, []) + $vid END
                WITH o
                MATCH (v:DocumentVersion {entity_id: $vid})
                MERGE (v)-[r:ISSUED_BY]->(o)
                ON CREATE SET r.assertion_id = $assertion_id
                """,
                {
                    "entity_id": organization["entity_id"],
                    "label": organization["label"],
                    "vid": version_entity,
                    "assertion_id": organization["assertion_id"],
                },
            )

        if citations:
            await self._run(
                """
                UNWIND $rows AS row
                MATCH (s {entity_id: row.source_id})
                MATCH (t:Provision {entity_id: row.target_id})
                MATCH (e:EvidenceChunk {chunk_id: row.support_chunk_id})
                MERGE (a:RelationEvidence {assertion_id: row.assertion_id})
                SET a.method = 'rule_citation', a.status = row.status,
                    a.raw_text = row.raw_text, a.version_ids = [$vid]
                MERGE (a)-[:SUPPORTED_BY]->(e)
                MERGE (s)-[r:CITES]->(t)
                SET r.assertion_id = row.assertion_id
                """,
                {"vid": version_entity, "rows": citations},
            )

    async def delete_version(self, version_id: str) -> None:
        """撤销清理：删除该版本的版本化节点与切片节点，保留共享 Issue/Organization。"""
        await self._run(
            """
            MATCH (v:DocumentVersion {entity_id: $vid})
            OPTIONAL MATCH (v)-[:HAS_PROVISION|HAS_VERSION]->(child)
            DETACH DELETE child
            WITH v
            DETACH DELETE v
            """,
            {"vid": version_id},
        )
        await self._run(
            """
            MATCH (e:EvidenceChunk)
            WHERE e.version_id = $vid
            DETACH DELETE e
            """,
            {"vid": version_id},
        )

    # ---------- 读取 ----------
    async def entities(
        self,
        q: str | None,
        visible_version_ids: Sequence[str],
        *,
        cursor: tuple[str, str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        if not visible_version_ids:
            return []
        params: dict[str, Any] = {"visible": list(visible_version_ids), "limit": limit, "q": q or ""}
        cursor_filter = ""
        if cursor:
            cursor_filter = (
                "AND (e.label > $cursor_label OR (e.label = $cursor_label AND e.entity_id > $cursor_id))"
            )
            params.update({"cursor_label": cursor[0], "cursor_id": cursor[1]})
        query = f"""
            MATCH (e)
            WHERE e.entity_type IN ['LegalDocument','DocumentVersion','Provision','Case','Issue','Organization']
              AND any(v IN coalesce(e.version_ids, []) WHERE v IN $visible)
              AND ($q = '' OR e.label CONTAINS $q)
              {cursor_filter}
            RETURN e.entity_id AS entity_id, e.label AS label, e.entity_type AS entity_type,
                   e.version_ids AS version_ids
            ORDER BY e.label, e.entity_id
            LIMIT $limit
        """
        return await self._run(query, params)

    async def entity(self, entity_id: str, visible_version_ids: Sequence[str]) -> dict[str, Any] | None:
        rows = await self._run(
            """
            MATCH (e {entity_id: $eid})
            WHERE any(v IN coalesce(e.version_ids, []) WHERE v IN $visible)
            RETURN e.entity_id AS entity_id, e.label AS label, e.entity_type AS entity_type,
                   e.version_ids AS version_ids
            LIMIT 1
            """,
            {"eid": entity_id, "visible": list(visible_version_ids)},
        )
        return rows[0] if rows else None

    async def neighborhood(
        self,
        entity_id: str,
        visible_version_ids: Sequence[str],
        *,
        max_hops: int = 2,
        max_paths: int = 50,
        max_nodes: int = 150,
        max_support_chunks: int = 100,
    ) -> Neighborhood:
        if not visible_version_ids:
            return Neighborhood([], [], [], False)
        params = {
            "eid": entity_id,
            "visible": list(visible_version_ids),
            "types": list(SEMANTIC_RELATIONS),
            "limit": max_paths + 1,
            "hops": max(1, min(max_hops, 2)),
        }
        forward = await self._run(
            f"""
            MATCH (s) WHERE s.entity_id = $eid
            MATCH p = (s)-[rels*1..{params['hops']}]->(n)
            WHERE all(r IN rels WHERE type(r) IN $types)
              AND all(x IN nodes(p) WHERE any(v IN coalesce(x.version_ids, []) WHERE v IN $visible))
            RETURN [x IN nodes(p) | x.entity_id] AS node_ids,
                   [r IN rels | type(r)] AS relation_types,
                   [r IN rels | coalesce(r.assertion_id, '')] AS assertion_ids,
                   [x IN nodes(p) | coalesce(x.label, '')] AS labels,
                   [x IN nodes(p) | coalesce(x.entity_type, '')] AS entity_types,
                   [x IN nodes(p) | coalesce(x.version_ids, [])] AS version_lists
            LIMIT $limit
            """,
            params,
        )
        backward = await self._run(
            f"""
            MATCH (s) WHERE s.entity_id = $eid
            MATCH p = (s)<-[rels*1..{params['hops']}]-(n)
            WHERE all(r IN rels WHERE type(r) IN $types)
              AND all(x IN nodes(p) WHERE any(v IN coalesce(x.version_ids, []) WHERE v IN $visible))
            RETURN [x IN nodes(p) | x.entity_id] AS node_ids,
                   [r IN rels | type(r)] AS relation_types,
                   [r IN rels | coalesce(r.assertion_id, '')] AS assertion_ids,
                   [x IN nodes(p) | coalesce(x.label, '')] AS labels,
                   [x IN nodes(p) | coalesce(x.entity_type, '')] AS entity_types,
                   [x IN nodes(p) | coalesce(x.version_ids, [])] AS version_lists
            LIMIT $limit
            """,
            params,
        )

        paths: list[GraphPath] = []
        for row in forward:
            paths.append(
                GraphPath(
                    node_ids=row["node_ids"],
                    relation_types=row["relation_types"],
                    assertion_ids=[item or None for item in row["assertion_ids"]],
                    labels=row["labels"],
                    entity_types=row["entity_types"],
                    version_ids=sorted({v for group in row["version_lists"] for v in group}),
                )
            )
        for row in backward:
            # 反向路径倒序后方向仍然正确：n -r2-> m -r1-> s
            paths.append(
                GraphPath(
                    node_ids=list(reversed(row["node_ids"])),
                    relation_types=list(reversed(row["relation_types"])),
                    assertion_ids=list(reversed([item or None for item in row["assertion_ids"]])),
                    labels=list(reversed(row["labels"])),
                    entity_types=list(reversed(row["entity_types"])),
                    version_ids=sorted({v for group in row["version_lists"] for v in group}),
                )
            )

        # 去重：同起点同终点同关系序列保留一条
        unique: dict[tuple, GraphPath] = {}
        for path in paths:
            key = (tuple(path.node_ids), tuple(path.relation_types))
            unique.setdefault(key, path)
        kept = sorted(unique.values(), key=lambda item: (len(item.node_ids), item.node_ids))[:max_paths]
        truncated = len(unique) > max_paths

        assertion_ids = sorted({aid for path in kept for aid in path.assertion_ids if aid})
        support_map: dict[str, list[str]] = {}
        if assertion_ids:
            rows = await self._run(
                """
                UNWIND $aids AS aid
                MATCH (a:RelationEvidence {assertion_id: aid})-[:SUPPORTED_BY]->(e:EvidenceChunk)
                WHERE e.version_id IN $visible
                RETURN aid AS assertion_id, e.chunk_id AS chunk_id
                """,
                {"aids": assertion_ids, "visible": list(visible_version_ids)},
            )
            for row in rows:
                support_map.setdefault(row["assertion_id"], []).append(row["chunk_id"])

        support_chunks: list[str] = []
        for path in kept:
            path_support: list[str] = []
            for aid in path.assertion_ids:
                if aid:
                    path_support.extend(support_map.get(aid, []))
            if not path_support:
                # 结构性路径（如 HAS_PROVISION）用端点条款自身的证据切片兜底
                extra = await self._run(
                    """
                    MATCH (n) WHERE n.entity_id IN $ids AND n.entity_type IN ['Provision','Case']
                    MATCH (n)-[:SUPPORTED_BY]->(e:EvidenceChunk)
                    WHERE e.version_id IN $visible
                    RETURN e.chunk_id AS chunk_id
                    LIMIT 3
                    """,
                    {"ids": path.node_ids, "visible": list(visible_version_ids)},
                )
                path_support.extend(row["chunk_id"] for row in extra)
            path.support_chunk_ids = list(dict.fromkeys(path_support))
            support_chunks.extend(path.support_chunk_ids)
        support_chunks = list(dict.fromkeys(support_chunks))
        if len(support_chunks) > max_support_chunks:
            # 超出上限时截断整条路径，避免返回“有路径但缺证据”的半条结果
            support_chunks = support_chunks[:max_support_chunks]
            allowed_chunks = set(support_chunks)
            kept = [path for path in kept if set(path.support_chunk_ids) <= allowed_chunks]
            truncated = True

        node_map: dict[str, dict[str, Any]] = {}
        for path in kept:
            for index, node_id in enumerate(path.node_ids):
                if node_id and node_id not in node_map:
                    node_map[node_id] = {
                        "entity_id": node_id,
                        "label": path.labels[index] or node_id,
                        "entity_type": path.entity_types[index] or "Provision",
                        "version_ids": path.version_ids,
                    }
        nodes = list(node_map.values())[:max_nodes]
        if len(node_map) > max_nodes:
            truncated = True
        allowed = {node["entity_id"] for node in nodes}
        kept = [path for path in kept if all(node_id in allowed for node_id in path.node_ids)]
        return Neighborhood(kept, nodes, support_chunks, truncated)

    async def provision_by_article(
        self, version_ids: Sequence[str], articles: Sequence[str], limit: int = 20
    ) -> list[dict[str, Any]]:
        if not version_ids or not articles:
            return []
        rows: list[dict[str, Any]] = []
        for version_id in version_ids:
            rows.extend(
                await self._run(
                    TEMPLATES["provision_lookup"],
                    {"version_id": version_id, "articles": list(articles), "limit": limit},
                )
            )
        return rows[:limit]

    async def scenario_chunks(
        self, scenario: str, version_ids: Sequence[str], limit: int = 20
    ) -> list[str]:
        """按争议主题取案例/条款证据，供图谱检索路径使用。"""
        if not version_ids or scenario == "other":
            return []
        rows = await self._run(
            """
            MATCH (i:Issue)
            WHERE i.label = $scenario OR i.scenario = $scenario
            MATCH (i)<-[:CONCERNS]-(s)
            WHERE any(v IN coalesce(s.version_ids, []) WHERE v IN $visible)
            MATCH (s)-[:SUPPORTED_BY]->(e:EvidenceChunk)
            WHERE e.version_id IN $visible
            RETURN e.chunk_id AS chunk_id, count(*) AS hits
            ORDER BY hits DESC, chunk_id ASC
            LIMIT $limit
            """,
            {"scenario": scenario, "visible": list(version_ids), "limit": limit},
        )
        return [row["chunk_id"] for row in rows]

    async def health(self) -> dict[str, Any]:
        try:
            await self._run("RETURN 1 AS ok")
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:160]}
