"""Validated environment-backed settings."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, PostgresDsn, RedisDsn, field_validator, model_validator
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

    retrieval_recall_budget: int = Field(default=20, ge=1)
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

    @model_validator(mode="after")
    def validate_retrieval_funnel(self) -> Self:
        """Reject a retrieval funnel that cannot produce the requested TopK."""
        top_k = self.retrieval_context_top_k
        if self.retrieval_recall_budget < top_k:
            raise ValueError("RETRIEVAL_RECALL_BUDGET 不能小于 RETRIEVAL_CONTEXT_TOP_K")
        if self.retrieval_rerank_candidate_limit < top_k:
            raise ValueError("RETRIEVAL_RERANK_CANDIDATE_LIMIT 不能小于 RETRIEVAL_CONTEXT_TOP_K")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once for the process."""
    return Settings()
