"""Application service for the minimal knowledge-base and document API."""

from uuid import UUID

from customer_agent2.domain.models import (
    DocumentManagementRepository,
    DocumentStatus,
    EmbeddingIndexConfiguration,
    IngestionError,
    IngestionErrorCode,
    KnowledgeBase,
    KnowledgeBaseDraft,
)


class DocumentManagementService:
    """Create index-compatible knowledge bases and manage scoped documents."""

    def __init__(
        self,
        repository: DocumentManagementRepository,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> None:
        self._repository = repository
        self._index_configuration = index_configuration

    async def create_knowledge_base(self, draft: KnowledgeBaseDraft) -> KnowledgeBase:
        """Create a knowledge base pinned to the active Embedding baseline."""
        return await self._repository.create_knowledge_base(
            draft,
            self._index_configuration,
        )

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus:
        """Return the latest attempt and current active version for one document."""
        document = await self._repository.get_document_status(
            knowledge_base_id,
            document_id,
        )
        if document is None:
            raise _document_not_found()
        return document

    async def delete_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> None:
        """Delete one scoped logical document and all dependent versions and chunks."""
        deleted = await self._repository.delete_document(knowledge_base_id, document_id)
        if not deleted:
            raise _document_not_found()


def _document_not_found() -> IngestionError:
    return IngestionError(
        IngestionErrorCode.DOCUMENT_NOT_FOUND,
        "文档不存在",
        retryable=False,
    )
