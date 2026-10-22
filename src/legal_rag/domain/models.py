"""领域模型：唯一契约实现，字段与 docs/contracts/核心对象.schema.json 一一对应。

约定：
- 全部对象禁止未知字段（对应 Schema 的 additionalProperties=false）；
- 时间统一 UTC，序列化为带 Z 的 ISO 8601；
- 可空字段必须显式给 null，不用空串替代；
- 内部字段（租约、预算预留、阶段产物）不进入这些模型，由仓储层以附加键形式保存。
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from .enums import (
    Ability,
    Availability,
    Behavior,
    CitationValidation,
    DocType,
    Effectivity,
    EntityType,
    GraphState,
    ImportStage,
    ImportStatus,
    MessageStatus,
    PathState,
    PathStatusValue,
    PublicationState,
    QualityFlag,
    QueryState,
    ReasoningCode,
    RelationType,
    Role,
    Scenario,
    Scope,
    SourceType,
    Split,
    VectorState,
)


def utcnow() -> datetime:
    """统一使用 UTC 且带时区，避免本地时区写入导致排序/比较错误。"""
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    @field_serializer("*", when_used="json", check_fields=False)
    def _serialize_datetime(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        if isinstance(value, date):
            return value.isoformat()
        return value


class LegalMetadata(ContractModel):
    jurisdiction: Literal["CN-national", "unknown"]
    issued_date: date | None
    effective_from: date | None
    effective_to: date | None
    effectivity: Effectivity
    effectivity_source_chunk_id: str | None
    case_number: str | None
    issuer: str | None
    document_number: str | None

    @field_validator("effective_to")
    @classmethod
    def _check_range(cls, value: date | None, info) -> date | None:
        start = info.data.get("effective_from")
        if value is not None and start is not None and not value > start:
            raise ValueError("effective_to 必须晚于 effective_from")
        return value


class Location(ContractModel):
    page: int | None = Field(default=None, ge=1)
    heading: str | None = None
    article: str | None = None
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @field_validator("end_char")
    @classmethod
    def _check_span(cls, value: int, info) -> int:
        start = info.data.get("start_char")
        if start is not None and value < start:
            raise ValueError("end_char 不得小于 start_char")
        return value


class Document(ContractModel):
    document_id: uuid.UUID
    title: str = Field(min_length=1, max_length=200)
    doc_type: DocType
    scope: Scope
    session_id: uuid.UUID | None
    current_version_id: str | None
    tombstoned: bool
    created_at: datetime
    updated_at: datetime


class Version(ContractModel):
    version_id: str
    document_id: uuid.UUID
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_url: str | None
    object_key: str
    metadata: LegalMetadata
    publication_state: PublicationState
    availability: Availability
    vector_state: VectorState
    graph_state: GraphState
    quality_flags: list[QualityFlag]
    created_at: datetime
    pipeline_revision: str
    published_at: datetime | None


class Chunk(ContractModel):
    chunk_id: str
    document_id: uuid.UUID
    version_id: str
    ordinal: int = Field(ge=0)
    content: str = Field(min_length=1)
    location: Location
    scenario_tags: list[Scenario]
    embedding_revision: str
    quality_flags: list[QualityFlag]


class Candidate(ContractModel):
    source_id: str
    source_type: SourceType
    chunk_id: str | None
    version_id: str | None
    title: str
    text: str
    url: str | None
    location: Location | None
    recall_sources: list[Literal["ordinary", "hyde", "graph", "web"]]
    rank_score: float
    rerank_score: float | None
    quality_flags: list[QualityFlag]


class Evidence(ContractModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    query_id: uuid.UUID
    source_id: str
    source_type: SourceType
    chunk_id: str | None
    version_id: str | None
    title: str
    text: str
    url: str | None
    location: Location | None
    metadata: LegalMetadata | None
    quality_flags: list[QualityFlag]


class SampleEvidence(ContractModel):
    """离线样本证据：与在线 Evidence 共享正文/来源规则，但没有 query_id。

    用 sample_id + evidence_id 定位，禁止为了通过校验填虚构查询 UUID。
    """

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    source_id: str
    source_type: SourceType
    chunk_id: str | None
    version_id: str | None
    title: str
    text: str
    url: str | None
    location: Location | None
    metadata: LegalMetadata | None
    quality_flags: list[QualityFlag]

    @field_validator("query_id", mode="before", check_fields=False)
    @classmethod
    def _reject_query_id(cls, value: Any) -> Any:  # pragma: no cover - 由 extra=forbid 兜底
        raise ValueError("SampleEvidence 不允许 query_id")


class Path(ContractModel):
    path_id: str
    node_ids: list[str] = Field(min_length=1)
    relation_types: list[RelationType]
    support_chunk_ids: list[str] = Field(min_length=1)
    status: PathStatusValue
    version_ids: list[str]

    @field_validator("relation_types")
    @classmethod
    def _check_relation_length(cls, value: list[RelationType], info) -> list[RelationType]:
        nodes = info.data.get("node_ids")
        if nodes is not None and len(value) != len(nodes) - 1:
            raise ValueError("relation_types 长度必须等于 node_ids 长度减一")
        return value


class Facts(ContractModel):
    scenario: Scenario
    location: str | None
    relevant_date: date | None
    known_facts: list[str]
    missing_fields: list[str]


class PathStatus(ContractModel):
    ordinary: PathState
    hyde: PathState
    graph: PathState
    web: PathState
    rerank: PathState

    @classmethod
    def all_pending(cls) -> "PathStatus":
        return cls(
            ordinary=PathState.PENDING,
            hyde=PathState.PENDING,
            graph=PathState.PENDING,
            web=PathState.PENDING,
            rerank=PathState.PENDING,
        )


class DocumentListItem(ContractModel):
    document: Document
    version: Version | None
    availability: Availability
    latest_task: "TaskStatus | None"


class Message(ContractModel):
    message_id: uuid.UUID
    query_id: uuid.UUID
    role: Role
    content: str
    status: MessageStatus
    created_at: datetime
    facts: Facts


class GraphEntity(ContractModel):
    entity_id: str
    label: str
    entity_type: EntityType
    version_ids: list[str]


class QueryRequest(ContractModel):
    request_id: uuid.UUID
    question: str = Field(min_length=1, max_length=2000)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)
    attachment_ids: list[uuid.UUID] = Field(default_factory=list, max_length=2)
    relevant_date: date | None = None
    location: str | None = Field(default=None, max_length=80)
    web_enabled: bool = False


class QueryStatus(ContractModel):
    query_id: uuid.UUID
    session_id: uuid.UUID
    status: QueryState
    behavior: Behavior | None
    answer: str
    citation_validation: CitationValidation
    warnings: list[str]
    evidence: list[Evidence]
    last_event_seq: int = Field(ge=0)
    created_at: datetime
    finished_at: datetime | None
    generation_model_id: str | None
    generation_artifact_sha256: str | None
    facts: Facts
    path_status: PathStatus
    reasoning_code: ReasoningCode | None


class TaskStatus(ContractModel):
    task_id: uuid.UUID
    document_id: uuid.UUID
    version_id: str
    status: ImportStatus
    stage: ImportStage
    availability: Availability
    vector_state: VectorState
    graph_state: GraphState
    warnings: list[str]
    retryable: bool
    attempt: int = Field(ge=0)
    error: "Error | None"
    updated_at: datetime


class Error(ContractModel):
    code: str
    message: str
    request_id: uuid.UUID
    retryable: bool
    details: dict[str, Any]


class DatasetSample(ContractModel):
    sample_id: str
    split: Split
    origin_group_id: str
    scenario: Scenario
    ability: Ability
    question: str
    evidence: list[SampleEvidence]
    expected_behavior: Behavior
    target_answer: str
    label_type: Literal["synthetic_silver"]
    generator_revision: str
    source_manifest_revision: str


class Session(ContractModel):
    session_id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    deleted: bool


class Cost(ContractModel):
    total_limit_yuan: Literal[300]
    settled_yuan: float = Field(ge=0)
    reserved_yuan: float = Field(ge=0)
    estimated_unsettled_yuan: float = Field(ge=0)
    pricing_revision: str
    last_reconciled_at: datetime | None


class EvaluationLabel(ContractModel):
    sample_id: str
    relevant_source_ids: list[str]
    required_points: list[str]
    expected_behavior: Behavior
    label_type: Literal["synthetic_silver"]


# 允许前向引用解析（DocumentListItem / TaskStatus 相互引用）
DocumentListItem.model_rebuild()
TaskStatus.model_rebuild()
