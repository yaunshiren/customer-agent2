"""Unit tests for minimal knowledge-base and document management use cases."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from customer_agent2.application import DocumentManagementService
from customer_agent2.domain.models import (
    DocumentStatus,
    DocumentVersionState,
    DocumentVersionSummary,
    EmbeddingIndexConfiguration,
    IngestionError,
    IngestionErrorCode,
    KnowledgeBase,
    KnowledgeBaseDraft,
)

INDEX_CONFIGURATION = EmbeddingIndexConfiguration("embedding", "revision", 3, True)


class RecordingManagementRepository:
    """In-memory management port fake."""

    def __init__(self) -> None:
        self.created: list[tuple[KnowledgeBaseDraft, EmbeddingIndexConfiguration]] = []
        self.status: DocumentStatus | None = None
        self.deleted = True
        self.delete_calls: list[tuple[UUID, UUID]] = []

    async def create_knowledge_base(
        self,
        draft: KnowledgeBaseDraft,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> KnowledgeBase:
        self.created.append((draft, index_configuration))
        return KnowledgeBase(
            id=uuid4(),
            slug=draft.slug,
            name=draft.name,
            description=draft.description,
            index_configuration=index_configuration,
            created_at=datetime.now(UTC),
        )

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus | None:
        return self.status

    async def delete_document(self, knowledge_base_id: UUID, document_id: UUID) -> bool:
        self.delete_calls.append((knowledge_base_id, document_id))
        return self.deleted


def document_status() -> DocumentStatus:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    return DocumentStatus(
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        source_key="guide.md",
        display_name="guide.md",
        latest_version=DocumentVersionSummary(
            id=version_id,
            version_number=1,
            status=DocumentVersionState.ACTIVE,
            chunk_count=2,
            content_sha256="a" * 64,
            media_type="text/markdown",
            parser_name="parser",
            parser_version="1",
            error_code=None,
            created_at=datetime.now(UTC),
            activated_at=datetime.now(UTC),
        ),
        active_version_id=version_id,
    )


@pytest.mark.asyncio
async def test_management_service_creates_index_compatible_knowledge_base() -> None:
    repository = RecordingManagementRepository()
    service = DocumentManagementService(repository, INDEX_CONFIGURATION)
    draft = KnowledgeBaseDraft(" Refund-Docs ", " 退款文档 ", " 测试 ")

    created = await service.create_knowledge_base(draft)

    assert created.slug == "refund-docs"
    assert repository.created == [(draft, INDEX_CONFIGURATION)]


@pytest.mark.asyncio
async def test_management_service_returns_status_and_scopes_delete() -> None:
    repository = RecordingManagementRepository()
    repository.status = document_status()
    service = DocumentManagementService(repository, INDEX_CONFIGURATION)

    loaded = await service.get_document_status(
        repository.status.knowledge_base_id,
        repository.status.document_id,
    )
    await service.delete_document(loaded.knowledge_base_id, loaded.document_id)

    assert loaded is repository.status
    assert repository.delete_calls == [(loaded.knowledge_base_id, loaded.document_id)]


@pytest.mark.parametrize("operation", ["status", "delete"])
@pytest.mark.asyncio
async def test_management_service_reports_missing_document(operation: str) -> None:
    repository = RecordingManagementRepository()
    repository.deleted = False
    service = DocumentManagementService(repository, INDEX_CONFIGURATION)

    with pytest.raises(IngestionError) as caught:
        if operation == "status":
            await service.get_document_status(uuid4(), uuid4())
        else:
            await service.delete_document(uuid4(), uuid4())

    assert caught.value.code is IngestionErrorCode.DOCUMENT_NOT_FOUND


def test_knowledge_base_draft_validates_and_normalizes_public_fields() -> None:
    assert KnowledgeBaseDraft(" docs-1 ", " 文档 ", "   ").description is None

    with pytest.raises(ValueError, match="slug"):
        KnowledgeBaseDraft("INVALID_SLUG", "文档")
