"""运行配置：默认参数来自 docs/contracts/默认配置.json，密钥与地址来自环境变量。

设计原则：
- 非秘密参数只从默认配置读取，避免代码里出现第二套常量；
- 价格单独存放并带有效期，未验证或缺失时拒绝付费任务；
- 运行模式 live/test 决定是否允许测试替身（替身只能出现在 test 模式）。
"""

from __future__ import annotations

import json
import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

def _find_project_root() -> Path:
    """定位项目根目录：优先环境变量，其次从当前工作目录与包位置向上找配置标记。

    允许以非可编辑方式安装到 site-packages 后仍能找到 docs/contracts 与 web 目录。
    """
    override = os.environ.get("LEGAL_RAG_ROOT")
    if override:
        return Path(override).resolve()
    marker = Path("docs") / "contracts" / "默认配置.json"
    for base in (Path.cwd(), Path(__file__).resolve()):
        for candidate in (base, *base.parents):
            if (candidate / marker).exists():
                return candidate
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _find_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "docs" / "contracts" / "默认配置.json"


class ConfigError(RuntimeError):
    """配置缺失或自相矛盾时抛出，接口层转换为 PRICING_NOT_CONFIGURED 等稳定错误。"""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectSection(_Section):
    mode: str
    workspace: str
    jurisdiction: str
    scenarios: list[str]
    target_days: list[int]
    human_review: bool
    web_default: bool


class RuntimeSection(_Section):
    host: str
    port: int
    python: str
    query_concurrency: int
    import_concurrency: int
    embedding_concurrency: int
    lease_seconds: int
    heartbeat_seconds: int
    queue_wait_seconds: int


class IngestionSection(_Section):
    formats: list[str]
    parser: str
    mineru_optional: bool
    scan_without_ocr: str
    max_file_mib: int
    max_pdf_pages: int
    max_documents: int
    max_chunks: int
    chunk_target_min_chars: int
    chunk_target_max_chars: int
    chunk_soft_max_chars: int
    chunk_overlap_chars: int
    max_attempts: int
    auto_publish: bool
    vector_required: bool
    graph_failure_availability: str
    metadata_missing_availability: str


class ModelsSection(_Section):
    generation: str
    rerank: str
    judge: str
    embedding: str
    embedding_dimension: int
    embedding_revision: str | None
    require_revision_lock_before_index: bool
    embedding_batch_size: int
    online_temperature: float
    online_max_output_tokens: int
    aux_generation: str
    generation_kind: str
    online_max_input_tokens: int


class RetrievalSection(_Section):
    ordinary_dense_k: int
    ordinary_sparse_k: int
    ordinary_weights: list[float]
    ordinary_result_k: int
    hyde_max_chars: int
    hyde_result_k: int
    graph_max_hops: int
    graph_max_paths: int
    graph_result_k: int
    rrf_k: int
    rrf_rank_start: int
    local_weights: dict[str, float]
    web_max_results: int
    web_max_pages: int
    rerank_input_k: int
    final_evidence_k: int
    evidence_max_chars: int
    cliff_min_keep: int
    cliff_abs_gap: float
    cliff_rel_gap: float


class TimeoutsSection(_Section):
    query_total_seconds: int
    ordinary_seconds: int
    graph_seconds: int
    hyde_seconds: int
    web_seconds: int
    rerank_seconds: int
    generation_seconds: int
    import_stage_seconds: int
    transient_retries: int
    retry_wait_seconds: int


class SessionSection(_Section):
    history_messages: int
    history_max_chars: int
    max_question_chars: int
    max_attachments: int
    sse_heartbeat_seconds: int
    event_retention_days: int


class SourcesSection(_Section):
    target_normative_documents: int
    target_case_documents: int
    cases_per_scenario: int
    source_domains: list[str]
    allow_subdomains: bool
    per_domain_concurrency: int
    min_request_interval_seconds: int
    batch_page_limit: int


class DatasetSection(_Section):
    train_target: int
    validation_target: int
    test_target: int
    test_per_scenario: int
    min_test_per_ability: int
    label_type: str
    seed: int
    pilot_samples: int
    max_regenerations: int
    near_duplicate_trigram_jaccard: float
    split_before_generation: bool
    human_annotation: bool


class TrainingSection(_Section):
    base_model: str
    base_revision: str | None
    require_revision_lock: bool
    method: str
    target_modules: list[str]
    r: int
    alpha: int
    dropout: float
    max_seq_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    gradient_checkpointing: bool
    seed: int
    precision: str
    assistant_only_loss: bool
    smoke_samples: int
    smoke_steps: int
    minimum_candidate_vram_gib: int
    inference_do_sample: bool
    inference_max_new_tokens: int
    program_integration_required: bool


class BudgetSection(_Section):
    currency: str
    total_limit: float
    api_limit: float
    gpu_limit: float
    reserve_limit: float
    paid_execution_enabled: bool
    require_live_pricing: bool
    gpu_max_hours: int
    external_shutdown_required: bool


class ServingSection(_Section):
    outside_business_app: bool
    engine: str
    artifact_type: str
    served_model_name: str
    max_model_len: int
    answer_model_url_env: str
    deployment_manifest_env: str
    automatic_cloud_fallback: bool
    frontend_management: bool
    api_contract: str
    billing: str


class EvaluationSection(_Section):
    type: str
    online_retrieval_model_fixed: bool
    base_and_adapter_same_model: bool
    human_legal_validation: bool
    require_raw_predictions: bool
    require_improvement: bool


class AppConfig(_Section):
    revision: str
    project: ProjectSection
    runtime: RuntimeSection
    ingestion: IngestionSection
    models: ModelsSection
    retrieval: RetrievalSection
    timeouts: TimeoutsSection
    session: SessionSection
    sources: SourcesSection
    dataset: DatasetSection
    training: TrainingSection
    budget: BudgetSection
    evaluation: EvaluationSection
    serving: ServingSection

    def validate_invariants(self) -> None:
        budget = self.budget
        total = budget.api_limit + budget.gpu_limit + budget.reserve_limit
        if abs(total - budget.total_limit) > 1e-9:
            raise ConfigError(f"预算分配之和 {total} 与总上限 {budget.total_limit} 不一致")
        if self.models.embedding_dimension != 1024:
            raise ConfigError("嵌入维度必须为 1024（BGE-M3 契约）")
        if self.retrieval.graph_max_hops > 2:
            raise ConfigError("图检索最大跳数不得超过 2")
        if self.project.human_review or self.dataset.human_annotation:
            raise ConfigError("首期不设人工审核/标注门禁")
        if self.serving.automatic_cloud_fallback:
            raise ConfigError("禁止自动回退到辅助云模型接管最终回答")


class EnvSettings(BaseSettings):
    """环境变量；字段名大小写不敏感，密钥为空表示未配置。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_mode: Literal["live", "test"] = "live"
    mongo_uri: str = "mongodb://127.0.0.1:27017"
    mongo_db: str = "legal_rag"
    milvus_uri: str = "http://127.0.0.1:19530"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    minio_endpoint: str = "127.0.0.1:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_bucket: str = "legal-rag"
    minio_secure: bool = False

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "legal-labor-sft-v1"
    model_deployment_manifest: str = ""
    model_tokenizer_path: str = ""

    aux_llm_base_url: str = ""
    aux_llm_api_key: str = ""
    aux_llm_model: str = "qwen-flash"

    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"

    mcp_url: str = ""
    mcp_api_key: str = ""

    embed_model_revision: str = ""
    pricing_file: str = "deploy/pricing.json"

    def answer_model_configured(self) -> bool:
        return bool(self.llm_base_url and self.llm_model)

    def aux_model_configured(self) -> bool:
        return bool(self.aux_llm_base_url and self.aux_llm_api_key)

    def rerank_configured(self) -> bool:
        return bool(self.rerank_base_url and self.rerank_api_key)

    def web_configured(self) -> bool:
        return bool(self.mcp_url)

    def embedding_revision(self, config: AppConfig) -> str | None:
        return self.embed_model_revision or config.models.embedding_revision


class PricingEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input_per_1k_yuan: float = Field(ge=0)
    output_per_1k_yuan: float = Field(ge=0)


class PricingTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    vendor: str
    region: str
    verified: bool
    effective_date: date
    checked_at: date
    currency: str
    entries: list[PricingEntry]

    def price_for(self, model: str) -> PricingEntry | None:
        for entry in self.entries:
            if entry.model == model:
                return entry
        return None


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8-sig"))
        config = AppConfig.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ConfigError(f"默认配置不可用：{exc}") from exc
    config.validate_invariants()
    return config


def load_pricing(env: EnvSettings) -> PricingTable | None:
    """价格缺失或未验证时返回 None；调用方必须拒绝付费批任务。"""
    path = PROJECT_ROOT / env.pricing_file
    if not path.exists():
        return None
    try:
        table = PricingTable.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, json.JSONDecodeError, ValidationError):
        return None
    return table if table.verified else None


class Settings:
    """进程内单例，组合默认配置、环境变量与价格表。"""

    def __init__(self, config: AppConfig, env: EnvSettings, pricing: PricingTable | None) -> None:
        self.config = config
        self.env = env
        self.pricing = pricing

    @property
    def is_test_mode(self) -> bool:
        return self.env.app_mode == "test"

    @property
    def workspace_id(self) -> str:
        return self.config.project.workspace

    def require_embedding_revision(self) -> str:
        revision = self.env.embedding_revision(self.config)
        if not revision and self.config.models.require_revision_lock_before_index:
            raise ConfigError("嵌入模型快照未锁定（EMBED_MODEL_REVISION），拒绝建立索引")
        return revision or self.config.models.embedding

    def require_answer_model(self) -> None:
        if not self.env.answer_model_configured():
            raise ConfigError("回答模型服务未配置（LLM_BASE_URL/LLM_MODEL）")

    def require_pricing(self, model: str) -> PricingEntry:
        if self.pricing is None:
            raise ConfigError("价格配置缺失或未验证，拒绝付费调用")
        entry = self.pricing.price_for(model)
        if entry is None:
            raise ConfigError(f"模型 {model} 缺少有效单价")
        return entry

    def snapshot(self) -> dict[str, Any]:
        """就绪检查用的非秘密快照（不含任何密钥）。"""
        return {
            "app_mode": self.env.app_mode,
            "config_revision": self.config.revision,
            "paid_execution_enabled": self.config.budget.paid_execution_enabled,
            "pricing_revision": self.pricing.revision if self.pricing else "unconfigured",
            "answer_model": self.env.llm_model if self.env.answer_model_configured() else None,
            "aux_model": self.env.aux_llm_model if self.env.aux_model_configured() else None,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config = load_config()
    env = EnvSettings()
    return Settings(config, env, load_pricing(env))
