"""Typed application use-case bundle exposed to API adapters."""

from dataclasses import dataclass, field
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models import (
    DocumentIngestionRequest,
    DocumentStatus,
    IngestionResult,
    KnowledgeBase,
    KnowledgeBaseDraft,
    StreamingRagPipeline,
    VectorSearchRequest,
    VectorSearchResult,
)


class DocumentIngestionUseCase(Protocol):
    """API-facing surface of document ingestion orchestration."""

    async def ingest(self, request: DocumentIngestionRequest) -> IngestionResult: ...


class DocumentManagementUseCase(Protocol):
    """API-facing surface for minimal knowledge-base and document management."""

    async def create_knowledge_base(self, draft: KnowledgeBaseDraft) -> KnowledgeBase: ...

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus: ...

    async def delete_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> None: ...


class VectorRetrievalUseCase(Protocol):
    """Internal first-stage retrieval surface for later RAG orchestration."""

    async def search(self, request: VectorSearchRequest) -> VectorSearchResult: ...


class AsyncCloseable(Protocol):
    """One application-owned asynchronous resource."""

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Long-lived use cases plus resources owned by their shared service graph."""

    ingestion: DocumentIngestionUseCase
    documents: DocumentManagementUseCase
    retrieval: VectorRetrievalUseCase
    rag: StreamingRagPipeline
    closeables: tuple[AsyncCloseable, ...] = field(default=(), repr=False)

    async def aclose(self) -> None:
        """Close every owned resource in reverse construction order."""
        first_error: BaseException | None = None
        for resource in reversed(self.closeables):
            try:
                await resource.aclose()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
