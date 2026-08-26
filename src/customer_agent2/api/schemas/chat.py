"""Typed request and event schemas for the public streaming RAG API."""

from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from customer_agent2.domain.models import DocumentFormat, PipelineStage, VectorSearchScope


class ChatSearchScopeRequest(BaseModel):
    """Explicitly authorized database-side filters for one chat request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    knowledge_base_ids: tuple[UUID, ...] = Field(min_length=1, max_length=100)
    document_ids: tuple[UUID, ...] = Field(default=(), max_length=1000)
    document_formats: tuple[DocumentFormat, ...] = Field(default=(), max_length=5)
    parser_names: tuple[str, ...] = Field(default=(), max_length=100)
    sections: tuple[str, ...] = Field(default=(), max_length=100)
    page_numbers: tuple[int, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def validate_domain_scope(self) -> Self:
        """Reuse the domain contract so HTTP and internal callers share constraints."""
        self.to_domain()
        return self

    def to_domain(self) -> VectorSearchScope:
        """Convert this transport model to the framework-independent search scope."""
        return VectorSearchScope(
            knowledge_base_ids=self.knowledge_base_ids,
            document_ids=self.document_ids,
            document_formats=self.document_formats,
            parser_names=self.parser_names,
            sections=self.sections,
            page_numbers=self.page_numbers,
        )


class ChatStreamRequest(BaseModel):
    """One question and the only knowledge scope it may search."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(min_length=1, max_length=10_000)
    scope: ChatSearchScopeRequest

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        """Reject whitespace-only questions and keep the domain input canonical."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question 不能为空")
        return normalized


class SseEventData(BaseModel):
    """Fields present in every public SSE event payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: UUID
    sequence: int = Field(ge=1)


class SseStatusEventData(SseEventData):
    """Public pipeline progress event."""

    stage: PipelineStage


class SseContentEventData(SseEventData):
    """One non-empty answer-text delta."""

    delta: str = Field(min_length=1)


class SseSource(BaseModel):
    """Public citation coordinates without retrieved document content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_number: int = Field(ge=1)
    chunk_id: UUID
    knowledge_base_id: UUID
    document_id: UUID
    document_version_id: UUID
    source_key: str
    display_name: str
    document_format: DocumentFormat
    section: str | None
    page_number: int | None
    content_sha256: str
    similarity: float


class SseSourcesEventData(SseEventData):
    """Final citation mapping for an answer."""

    sources: tuple[SseSource, ...] = Field(min_length=1)


class SseErrorEventData(SseEventData):
    """Sanitized failure emitted after the HTTP stream has started."""

    code: str
    message: str
    retryable: bool


class SseTraceEntry(BaseModel):
    """Public-safe stage timing without prompts or document content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: PipelineStage
    duration_ms: float = Field(ge=0)
    candidate_count: int | None = Field(default=None, ge=0)


class SseTokenUsage(BaseModel):
    """Provider-reported token counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class SseDoneEventData(SseEventData):
    """Terminal outcome for every stream that remains connected."""

    outcome: Literal["completed", "no_context", "error"]
    trace: tuple[SseTraceEntry, ...] = ()
    model_id: str | None = None
    finish_reason: str | None = None
    usage: SseTokenUsage | None = None
