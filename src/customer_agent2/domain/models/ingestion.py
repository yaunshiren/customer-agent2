"""Framework-independent contracts for transactional document ingestion."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models.chunk import ChunkingResult
from customer_agent2.domain.models.document import DocumentSource, ParsedDocument
from customer_agent2.domain.models.embedding import EmbeddingResult


class IngestionErrorCode(StrEnum):
    """Stable ingestion failures safe to persist or expose to later APIs."""

    KNOWLEDGE_BASE_NOT_FOUND = "knowledge_base_not_found"
    KNOWLEDGE_BASE_CONFLICT = "knowledge_base_conflict"
    DOCUMENT_NOT_FOUND = "document_not_found"
    INDEX_CONFIGURATION_MISMATCH = "index_configuration_mismatch"
    EMBEDDING_PROTOCOL = "embedding_protocol"
    VERSION_STATE_CONFLICT = "version_state_conflict"
    PERSISTENCE_FAILURE = "persistence_failure"
    FAILURE_RECORDING_FAILED = "failure_recording_failed"


class DocumentVersionState(StrEnum):
    """Persisted document-version states accepted by ADR-0002."""

    BUILDING = "building"
    ACTIVE = "active"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class IngestionError(RuntimeError):
    """Sanitized ingestion failure with a stable category and retry hint."""

    def __init__(
        self,
        code: IngestionErrorCode,
        public_message: str,
        *,
        retryable: bool,
    ) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class EmbeddingIndexConfiguration:
    """Embedding identity that must exactly match one knowledge-base index."""

    model_id: str
    model_revision: str
    dimension: int
    normalized: bool

    def __post_init__(self) -> None:
        normalized_model_id = self.model_id.strip()
        normalized_revision = self.model_revision.strip()
        if not normalized_model_id or not normalized_revision:
            raise ValueError("EmbeddingIndexConfiguration 模型身份不能为空")
        if self.dimension < 1:
            raise ValueError("EmbeddingIndexConfiguration.dimension 必须大于 0")
        object.__setattr__(self, "model_id", normalized_model_id)
        object.__setattr__(self, "model_revision", normalized_revision)


@dataclass(frozen=True, slots=True)
class DocumentIngestionRequest:
    """One in-memory document ingestion request before any database mutation."""

    knowledge_base_id: UUID
    source_key: str
    source: DocumentSource

    def __post_init__(self) -> None:
        normalized_source_key = self.source_key.strip()
        if not normalized_source_key:
            raise ValueError("DocumentIngestionRequest.source_key 不能为空")
        if len(normalized_source_key) > 1024:
            raise ValueError("DocumentIngestionRequest.source_key 不能超过 1024 个字符")
        if any(ord(character) < 32 for character in normalized_source_key):
            raise ValueError("DocumentIngestionRequest.source_key 不能包含控制字符")
        object.__setattr__(self, "source_key", normalized_source_key)


@dataclass(frozen=True, slots=True)
class KnowledgeBaseDraft:
    """Validated user-facing fields for one new knowledge base."""

    slug: str
    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        normalized_slug = self.slug.strip().lower()
        normalized_name = self.name.strip()
        normalized_description = self.description
        if normalized_description is not None:
            normalized_description = normalized_description.strip() or None
        if not normalized_slug or len(normalized_slug) > 100:
            raise ValueError("KnowledgeBaseDraft.slug 必须是不超过 100 个字符的非空值")
        if (
            normalized_slug.startswith("-")
            or normalized_slug.endswith("-")
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in normalized_slug
            )
        ):
            raise ValueError("KnowledgeBaseDraft.slug 只能包含小写字母、数字和中划线")
        if not normalized_name or len(normalized_name) > 200:
            raise ValueError("KnowledgeBaseDraft.name 必须是不超过 200 个字符的非空值")
        if normalized_description is not None and len(normalized_description) > 2000:
            raise ValueError("KnowledgeBaseDraft.description 不能超过 2000 个字符")
        object.__setattr__(self, "slug", normalized_slug)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "description", normalized_description)


@dataclass(frozen=True, slots=True)
class KnowledgeBase:
    """Created knowledge base with its immutable index identity."""

    id: UUID
    slug: str
    name: str
    description: str | None
    index_configuration: EmbeddingIndexConfiguration
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DocumentVersionSummary:
    """Public-safe status of one persisted ingestion attempt."""

    id: UUID
    version_number: int
    status: DocumentVersionState
    chunk_count: int
    content_sha256: str
    media_type: str | None
    parser_name: str | None
    parser_version: str | None
    error_code: str | None
    created_at: datetime
    activated_at: datetime | None

    def __post_init__(self) -> None:
        if self.version_number < 1 or self.chunk_count < 0:
            raise ValueError("DocumentVersionSummary 版本号或 Chunk 数量无效")


@dataclass(frozen=True, slots=True)
class DocumentStatus:
    """Logical document identity plus latest and currently active version state."""

    knowledge_base_id: UUID
    document_id: UUID
    source_key: str
    display_name: str
    latest_version: DocumentVersionSummary
    active_version_id: UUID | None


@dataclass(frozen=True, slots=True)
class IngestionAttempt:
    """Identity of one committed building version."""

    knowledge_base_id: UUID
    document_id: UUID
    version_id: UUID
    version_number: int

    def __post_init__(self) -> None:
        if self.version_number < 1:
            raise ValueError("IngestionAttempt.version_number 必须大于 0")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Successfully activated document version returned by the use case."""

    knowledge_base_id: UUID
    document_id: UUID
    version_id: UUID
    version_number: int
    chunk_count: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.version_number < 1 or self.chunk_count < 1:
            raise ValueError("IngestionResult 版本号和 Chunk 数量必须大于 0")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("IngestionResult.content_sha256 格式无效")


class IngestionRepository(Protocol):
    """Persistence port implementing the accepted building/active version semantics."""

    async def create_building_version(
        self,
        request: DocumentIngestionRequest,
        document: ParsedDocument,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> IngestionAttempt: ...

    async def activate_version(
        self,
        attempt: IngestionAttempt,
        chunking: ChunkingResult,
        embeddings: EmbeddingResult,
    ) -> None: ...

    async def mark_version_failed(
        self,
        attempt: IngestionAttempt,
        error_code: str,
    ) -> None: ...


class DocumentManagementRepository(Protocol):
    """Persistence port for the minimal knowledge-base and document API."""

    async def create_knowledge_base(
        self,
        draft: KnowledgeBaseDraft,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> KnowledgeBase: ...

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus | None: ...

    async def delete_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> bool: ...
