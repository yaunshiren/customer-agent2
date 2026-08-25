"""Typed application use-case bundle exposed to API adapters."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models import (
    DocumentIngestionRequest,
    DocumentStatus,
    IngestionResult,
    KnowledgeBase,
    KnowledgeBaseDraft,
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


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Long-lived use cases sharing one model adapter and database pool."""

    ingestion: DocumentIngestionUseCase
    documents: DocumentManagementUseCase
