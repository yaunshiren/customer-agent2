"""SQLAlchemy adapter for minimal knowledge-base and document management."""

from typing import cast
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
from customer_agent2.infrastructure.persistence.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
)


class SQLAlchemyDocumentManagementRepository:
    """Persist knowledge bases and scoped document lifecycle operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_knowledge_base(
        self,
        draft: KnowledgeBaseDraft,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> KnowledgeBase:
        """Create one knowledge base or return a sanitized slug conflict."""
        try:
            async with self._session_factory.begin() as session:
                record = KnowledgeBaseRecord(
                    slug=draft.slug,
                    name=draft.name,
                    description=draft.description,
                    embedding_model_id=index_configuration.model_id,
                    embedding_model_revision=index_configuration.model_revision,
                    embedding_dimension=index_configuration.dimension,
                    embedding_normalized=index_configuration.normalized,
                )
                session.add(record)
                await session.flush()
                return KnowledgeBase(
                    id=record.id,
                    slug=record.slug,
                    name=record.name,
                    description=record.description,
                    index_configuration=index_configuration,
                    created_at=record.created_at,
                )
        except IntegrityError as error:
            sqlstate = cast(str | None, getattr(error.orig, "sqlstate", None))
            if sqlstate == "23505":
                raise IngestionError(
                    IngestionErrorCode.KNOWLEDGE_BASE_CONFLICT,
                    "知识库 slug 已存在",
                    retryable=False,
                ) from None
            raise _persistence_error("无法创建知识库") from None
        except SQLAlchemyError:
            raise _persistence_error("无法创建知识库") from None

    async def get_document_status(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> DocumentStatus | None:
        """Load one document's latest attempt and active version identity."""
        try:
            async with self._session_factory() as session:
                document = await session.scalar(
                    select(DocumentRecord).where(
                        DocumentRecord.id == document_id,
                        DocumentRecord.knowledge_base_id == knowledge_base_id,
                    )
                )
                if document is None:
                    return None

                chunk_count = (
                    select(func.count(ChunkRecord.id))
                    .where(ChunkRecord.document_version_id == DocumentVersionRecord.id)
                    .correlate(DocumentVersionRecord)
                    .scalar_subquery()
                )
                latest_row = (
                    await session.execute(
                        select(DocumentVersionRecord, chunk_count.label("chunk_count"))
                        .where(DocumentVersionRecord.document_id == document_id)
                        .order_by(DocumentVersionRecord.version_number.desc())
                        .limit(1)
                    )
                ).one_or_none()
                if latest_row is None:
                    return None
                latest_version = latest_row[0]
                latest_chunk_count = latest_row[1]
                active_version_id = await session.scalar(
                    select(DocumentVersionRecord.id).where(
                        DocumentVersionRecord.document_id == document_id,
                        DocumentVersionRecord.status == DocumentVersionState.ACTIVE.value,
                    )
                )

                return DocumentStatus(
                    knowledge_base_id=knowledge_base_id,
                    document_id=document.id,
                    source_key=document.source_key,
                    display_name=document.display_name,
                    latest_version=DocumentVersionSummary(
                        id=latest_version.id,
                        version_number=latest_version.version_number,
                        status=DocumentVersionState(latest_version.status),
                        chunk_count=latest_chunk_count,
                        content_sha256=latest_version.content_sha256,
                        media_type=latest_version.media_type,
                        parser_name=latest_version.parser_name,
                        parser_version=latest_version.parser_version,
                        error_code=latest_version.error_code,
                        created_at=latest_version.created_at,
                        activated_at=latest_version.activated_at,
                    ),
                    active_version_id=active_version_id,
                )
        except SQLAlchemyError:
            raise _persistence_error("无法查询文档状态") from None

    async def delete_document(
        self,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> bool:
        """Delete one scoped document and rely on accepted cascade semantics."""
        try:
            async with self._session_factory.begin() as session:
                deleted_id = await session.scalar(
                    delete(DocumentRecord)
                    .where(
                        DocumentRecord.id == document_id,
                        DocumentRecord.knowledge_base_id == knowledge_base_id,
                    )
                    .returning(DocumentRecord.id)
                )
                return deleted_id is not None
        except SQLAlchemyError:
            raise _persistence_error("无法删除文档") from None


def _persistence_error(message: str) -> IngestionError:
    return IngestionError(
        IngestionErrorCode.PERSISTENCE_FAILURE,
        message,
        retryable=True,
    )
