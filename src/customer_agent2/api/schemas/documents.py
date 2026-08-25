"""Typed schemas for the minimal M2-E ingestion HTTP contract."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from customer_agent2.domain.models import DocumentVersionState


class PublicErrorDetail(BaseModel):
    """Sanitized machine-readable application error."""

    code: str
    message: str
    retryable: bool


class PublicErrorResponse(BaseModel):
    """Stable wrapper matching FastAPI's detail response convention."""

    detail: PublicErrorDetail


class KnowledgeBaseCreateRequest(BaseModel):
    """Public fields accepted when creating a knowledge base."""

    slug: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)


class EmbeddingIndexResponse(BaseModel):
    """Immutable index identity returned with a knowledge base."""

    model_id: str
    model_revision: str
    dimension: int
    normalized: bool


class KnowledgeBaseResponse(BaseModel):
    """Created knowledge base representation."""

    id: UUID
    slug: str
    name: str
    description: str | None
    embedding: EmbeddingIndexResponse
    created_at: datetime


class DocumentUploadResponse(BaseModel):
    """Successful synchronous upload after the version is active."""

    knowledge_base_id: UUID
    document_id: UUID
    version_id: UUID
    version_number: int
    status: Literal["active"] = "active"
    chunk_count: int
    content_sha256: str


class DocumentVersionResponse(BaseModel):
    """Latest document-version state."""

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


class DocumentStatusResponse(BaseModel):
    """Logical document with latest attempt and current active version."""

    knowledge_base_id: UUID
    document_id: UUID
    source_key: str
    display_name: str
    latest_version: DocumentVersionResponse
    active_version_id: UUID | None
