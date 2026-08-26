"""Validated environment-backed settings."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings shared by the API and future application adapters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    app_name: str = "customer-agent2"
    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_prefix: str = "/api/v1"

    dashscope_api_key: SecretStr = SecretStr("")
    dashscope_base_url: AnyHttpUrl = AnyHttpUrl("https://dashscope.aliyuncs.com/compatible-mode/v1")
    chat_model_final: str = "qwen3.7-max-preview"
    chat_model_fast: str = "qwen3.7-flash"
    llm_timeout_seconds: float = Field(default=100.0, gt=0)
    llm_first_packet_timeout_seconds: float = Field(default=30.0, gt=0)
    rag_global_timeout_seconds: float = Field(default=120.0, gt=0)

    embedding_provider: Literal["local"] = "local"
    local_embedding_model: str = "BAAI/bge-base-zh-v1.5"
    local_embedding_revision: str = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"
    local_embedding_dimension: int = Field(default=768, ge=1)
    local_embedding_max_tokens: int = Field(default=512, ge=1)
    local_embedding_device: str = "cpu"
    local_embedding_batch_size: int = Field(default=16, ge=1)
    embedding_normalize: bool = True

    rerank_enabled: bool = False
    rerank_provider: Literal["dashscope"] = "dashscope"
    rerank_model: str = "qwen3-rerank"
    dashscope_workspace_id: str | None = None

    database_url: PostgresDsn = PostgresDsn(
        "postgresql+asyncpg://customer_agent2:change_me@127.0.0.1:5432/customer_agent2"
    )
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=10, ge=0)
    database_pool_timeout_seconds: float = Field(default=30.0, gt=0)
    database_pool_recycle_seconds: int = Field(default=1800, ge=-1)

    redis_url: RedisDsn = RedisDsn("redis://127.0.0.1:6379/0")
    redis_key_prefix: str = "customer-agent2"
    redis_max_connections: int = Field(default=20, ge=1)
    redis_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    redis_socket_timeout_seconds: float = Field(default=5.0, gt=0)

    readiness_timeout_seconds: float = Field(default=3.0, gt=0)

    upload_max_file_mb: int = Field(default=50, ge=1)
    document_max_extracted_chars: int = Field(default=5_000_000, ge=1)
    document_max_pdf_pages: int = Field(default=1000, ge=1)
    document_max_docx_entries: int = Field(default=2000, ge=1)
    document_max_docx_uncompressed_mb: int = Field(default=200, ge=1)
    document_max_docx_expansion_ratio: int = Field(default=100, ge=1)
    document_max_csv_rows: int = Field(default=10_000, ge=1)
    document_max_csv_columns: int = Field(default=200, ge=1)
    chunk_target_tokens: int = Field(default=400, ge=1)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    retrieval_recall_budget: int = Field(default=20, ge=1)
    retrieval_hnsw_ef_search: int = Field(default=100, ge=1, le=1000)
    retrieval_rerank_candidate_limit: int = Field(default=40, ge=1)
    retrieval_context_top_k: int = Field(default=10, ge=1)

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Keep generated documentation URLs stable and unambiguous."""
        stripped = value.strip()
        if not stripped.startswith("/"):
            raise ValueError("API_PREFIX 必须以 / 开头")
        normalized = stripped.rstrip("/")
        if not normalized:
            raise ValueError("API_PREFIX 不能只包含 /")
        return normalized

    @field_validator("database_url")
    @classmethod
    def validate_database_driver(cls, value: PostgresDsn) -> PostgresDsn:
        """Require SQLAlchemy's asyncpg dialect for all application connections."""
        if value.scheme != "postgresql+asyncpg":
            raise ValueError("DATABASE_URL 必须使用 postgresql+asyncpg 驱动")
        return value

    @field_validator("redis_key_prefix")
    @classmethod
    def validate_redis_key_prefix(cls, value: str) -> str:
        """Prevent unscoped Redis keys caused by an empty prefix."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("REDIS_KEY_PREFIX 不能为空")
        return normalized

    @field_validator(
        "chat_model_final",
        "chat_model_fast",
        "local_embedding_model",
        "local_embedding_revision",
        "local_embedding_device",
        "rerank_model",
    )
    @classmethod
    def validate_model_text_settings(cls, value: str) -> str:
        """Reject model settings that would fail later with ambiguous provider errors."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("模型名称和设备配置不能为空")
        return normalized

    @field_validator("dashscope_workspace_id", mode="before")
    @classmethod
    def normalize_workspace_id(cls, value: object) -> object:
        """Treat an empty optional workspace ID as absent."""
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @model_validator(mode="after")
    def validate_retrieval_funnel(self) -> Self:
        """Reject a retrieval funnel that cannot produce the requested TopK."""
        top_k = self.retrieval_context_top_k
        if self.retrieval_recall_budget < top_k:
            raise ValueError("RETRIEVAL_RECALL_BUDGET 不能小于 RETRIEVAL_CONTEXT_TOP_K")
        if self.retrieval_rerank_candidate_limit < top_k:
            raise ValueError("RETRIEVAL_RERANK_CANDIDATE_LIMIT 不能小于 RETRIEVAL_CONTEXT_TOP_K")
        return self

    @model_validator(mode="after")
    def validate_model_runtime(self) -> Self:
        """Keep configured model capabilities consistent with the accepted baseline."""
        if self.llm_first_packet_timeout_seconds > self.llm_timeout_seconds:
            raise ValueError("LLM_FIRST_PACKET_TIMEOUT_SECONDS 不能大于 LLM_TIMEOUT_SECONDS")
        if self.rag_global_timeout_seconds < self.llm_timeout_seconds:
            raise ValueError("RAG_GLOBAL_TIMEOUT_SECONDS 不能小于 LLM_TIMEOUT_SECONDS")

        if self.local_embedding_model == "BAAI/bge-base-zh-v1.5":
            if self.local_embedding_dimension != 768:
                raise ValueError("bge-base-zh-v1.5 的 LOCAL_EMBEDDING_DIMENSION 必须为 768")
            if self.local_embedding_max_tokens != 512:
                raise ValueError("bge-base-zh-v1.5 的 LOCAL_EMBEDDING_MAX_TOKENS 必须为 512")

        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS 必须小于 CHUNK_TARGET_TOKENS")
        if self.chunk_target_tokens > self.local_embedding_max_tokens:
            raise ValueError("CHUNK_TARGET_TOKENS 不能超过 LOCAL_EMBEDDING_MAX_TOKENS")

        if self.rerank_enabled:
            if not self.dashscope_api_key.get_secret_value():
                raise ValueError("启用 Rerank 时必须配置 DASHSCOPE_API_KEY")
            if self.dashscope_workspace_id is None:
                raise ValueError("启用 Rerank 时必须配置 DASHSCOPE_WORKSPACE_ID")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for the process."""
    return Settings()
