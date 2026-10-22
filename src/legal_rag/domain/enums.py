"""领域枚举：与 docs/contracts/核心对象.schema.json 的枚举逐项一致。

所有值使用小写 snake_case，中文只用于展示层映射，不进入持久化值。
"""

from __future__ import annotations

from enum import StrEnum


class QualityFlag(StrEnum):
    METADATA_MISSING = "metadata_missing"
    EFFECTIVITY_UNKNOWN = "effectivity_unknown"
    CITATION_UNRESOLVED = "citation_unresolved"
    GRAPH_UNAVAILABLE = "graph_unavailable"
    SOURCE_ACCESS_LIMITED = "source_access_limited"
    PARSER_PARTIAL = "parser_partial"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"


class DocType(StrEnum):
    LAW = "law"
    INTERPRETATION = "interpretation"
    CASE = "case"
    GUIDANCE = "guidance"
    UNKNOWN = "unknown"


class Scope(StrEnum):
    KNOWLEDGE = "knowledge"
    SESSION = "session"


class Availability(StrEnum):
    AVAILABLE = "available"
    LIMITED = "limited"
    UNAVAILABLE = "unavailable"


class PublicationState(StrEnum):
    STAGED = "staged"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"


class VectorState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"


class GraphState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    FAILED = "failed"
    EMPTY = "empty"


class ImportStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImportStage(StrEnum):
    VALIDATE = "validate"
    SAVE_ORIGINAL = "save_original"
    PARSE = "parse"
    SPLIT = "split"
    EXTRACT = "extract"
    WRITE_VECTORS = "write_vectors"
    WRITE_GRAPH = "write_graph"
    PUBLISH = "publish"
    DONE = "done"


IMPORT_STAGE_ORDER: tuple[ImportStage, ...] = (
    ImportStage.VALIDATE,
    ImportStage.SAVE_ORIGINAL,
    ImportStage.PARSE,
    ImportStage.SPLIT,
    ImportStage.EXTRACT,
    ImportStage.WRITE_VECTORS,
    ImportStage.WRITE_GRAPH,
    ImportStage.PUBLISH,
    ImportStage.DONE,
)


class QueryState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL_QUERY_STATES = frozenset(
    {
        QueryState.COMPLETED,
        QueryState.COMPLETED_WITH_WARNINGS,
        QueryState.FAILED,
        QueryState.INTERRUPTED,
        QueryState.CANCELLED,
    }
)


class Behavior(StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    INSUFFICIENT = "insufficient"


class ReasoningCode(StrEnum):
    EVIDENCE_SUFFICIENT = "EVIDENCE_SUFFICIENT"
    FACTS_MISSING = "FACTS_MISSING"
    NO_RELEVANT_EVIDENCE = "NO_RELEVANT_EVIDENCE"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    APPLICABILITY_UNCERTAIN = "APPLICABILITY_UNCERTAIN"


class CitationValidation(StrEnum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"


class PathState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    EMPTY = "empty"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class Scenario(StrEnum):
    WAGE_ARREARS = "wage_arrears"
    TERMINATION = "termination"
    ECONOMIC_COMPENSATION = "economic_compensation"
    OTHER = "other"


class SourceType(StrEnum):
    LOCAL = "local"
    WEB = "web"
    WEB_SNIPPET = "web_snippet"
    ATTACHMENT = "attachment"


class EntityType(StrEnum):
    LEGAL_DOCUMENT = "LegalDocument"
    DOCUMENT_VERSION = "DocumentVersion"
    PROVISION = "Provision"
    CASE = "Case"
    ISSUE = "Issue"
    ORGANIZATION = "Organization"


class RelationType(StrEnum):
    HAS_VERSION = "HAS_VERSION"
    HAS_PROVISION = "HAS_PROVISION"
    CITES = "CITES"
    CONCERNS = "CONCERNS"
    ISSUED_BY = "ISSUED_BY"
    SUPERSEDES = "SUPERSEDES"
    SUPPORTED_BY = "SUPPORTED_BY"


class PathStatusValue(StrEnum):
    SOURCE_SUPPORTED = "source_supported"
    UNRESOLVED = "unresolved"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Ability(StrEnum):
    TERMINOLOGY_FORMAT = "terminology_format"
    FAITHFULNESS = "faithfulness"
    INTEGRATION = "integration"
    CITATION = "citation"
    ABSTENTION = "abstention"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class MessageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class Effectivity(StrEnum):
    KNOWN_CURRENT = "known_current"
    KNOWN_HISTORICAL = "known_historical"
    UNKNOWN = "unknown"


# 中文展示映射：服务端固定，前端不得自行把 limited 翻译成“审核通过”
QUALITY_FLAG_LABELS: dict[str, str] = {
    "metadata_missing": "元数据缺失",
    "effectivity_unknown": "效力未知",
    "citation_unresolved": "引用版本未解析",
    "graph_unavailable": "图谱不可用",
    "source_access_limited": "来源访问受限",
    "parser_partial": "解析不完整",
    "citation_validation_failed": "引用校验未通过",
}

AVAILABILITY_LABELS: dict[str, str] = {
    "available": "可用",
    "limited": "受限可用",
    "unavailable": "不可用",
}
