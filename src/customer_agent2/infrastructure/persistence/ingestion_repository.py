"""SQLAlchemy adapter for atomic document-version ingestion."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent2.domain.models import (
    ChunkingResult,
    DocumentIngestionRequest,
    EmbeddingIndexConfiguration,
    EmbeddingResult,
    IngestionAttempt,
    IngestionError,
    IngestionErrorCode,
    ParsedDocument,
)
from customer_agent2.infrastructure.persistence.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
)


class SQLAlchemyIngestionRepository:
    """Persist isolated building versions and activate them in one transaction."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def create_building_version(
        self,
        request: DocumentIngestionRequest,
        document: ParsedDocument,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> IngestionAttempt:
        """Commit one building version without holding a transaction during inference."""
        try:
            async with self._session_factory.begin() as session:
                knowledge_base = await _locked_knowledge_base(
                    session,
                    request.knowledge_base_id,
                )
                if knowledge_base is None:
                    raise IngestionError(
                        IngestionErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
                        "知识库不存在",
                        retryable=False,
                    )
                _ensure_index_configuration(knowledge_base, index_configuration)

                logical_document = await _find_document(
                    session,
                    request.knowledge_base_id,
                    request.source_key,
                )
                if logical_document is None:
                    logical_document = DocumentRecord(
                        knowledge_base_id=request.knowledge_base_id,
                        source_key=request.source_key,
                        display_name=request.source.filename,
                    )
                    session.add(logical_document)
                    await session.flush()
                else:
                    logical_document.display_name = request.source.filename

                maximum_version = cast(
                    int,
                    await session.scalar(
                        select(
                            func.coalesce(func.max(DocumentVersionRecord.version_number), 0)
                        ).where(DocumentVersionRecord.document_id == logical_document.id)
                    ),
                )
                version = DocumentVersionRecord(
                    document_id=logical_document.id,
                    knowledge_base_id=request.knowledge_base_id,
                    version_number=maximum_version + 1,
                    status="building",
                    content_sha256=document.source.content_sha256,
                    media_type=document.source.media_type,
                    parser_name=document.parser_name,
                    parser_version=document.parser_version,
                    source_metadata=_version_source_metadata(document, chunking=None),
                )
                session.add(version)
                await session.flush()
                return IngestionAttempt(
                    knowledge_base_id=request.knowledge_base_id,
                    document_id=logical_document.id,
                    version_id=version.id,
                    version_number=version.version_number,
                )
        except IngestionError:
            raise
        except SQLAlchemyError:
            raise _persistence_error("无法创建文档构建版本") from None

    async def activate_version(
        self,
        attempt: IngestionAttempt,
        chunking: ChunkingResult,
        embeddings: EmbeddingResult,
    ) -> None:
        """Insert every chunk and switch active versions within one transaction."""
        try:
            async with self._session_factory.begin() as session:
                version = _require_building_version(
                    await _locked_version(session, attempt),
                    attempt,
                )
                _ensure_activation_payload(version, chunking, embeddings)

                await session.execute(
                    update(DocumentVersionRecord)
                    .where(
                        DocumentVersionRecord.document_id == attempt.document_id,
                        DocumentVersionRecord.status == "active",
                    )
                    .values(status="superseded")
                )

                session.add_all(
                    [
                        ChunkRecord(
                            document_version_id=attempt.version_id,
                            knowledge_base_id=attempt.knowledge_base_id,
                            chunk_index=chunk.chunk_index,
                            content=chunk.content,
                            token_count=chunk.token_count,
                            content_sha256=chunk.content_sha256,
                            section=_section_value(chunk.section_path),
                            page_number=chunk.page_number,
                            source_metadata={
                                "block_start_ordinal": chunk.block_start_ordinal,
                                "block_end_ordinal": chunk.block_end_ordinal,
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                                "section_path": list(chunk.section_path),
                                "overlap_with_previous_tokens": (
                                    chunk.overlap_with_previous_tokens
                                ),
                            },
                            embedding=list(vector),
                        )
                        for chunk, vector in zip(
                            chunking.chunks,
                            embeddings.vectors,
                            strict=True,
                        )
                    ]
                )
                version.status = "active"
                version.error_code = None
                version.activated_at = datetime.now(UTC)
                version.source_metadata = _version_source_metadata(
                    chunking.source,
                    chunking=chunking,
                )
                await session.flush()
        except IngestionError:
            raise
        except SQLAlchemyError:
            raise _persistence_error("无法原子激活文档版本") from None

    async def mark_version_failed(
        self,
        attempt: IngestionAttempt,
        error_code: str,
    ) -> None:
        """Make a building version non-retrievable with a sanitized failure code."""
        normalized_error_code = error_code.strip()
        if not normalized_error_code or len(normalized_error_code) > 100:
            raise ValueError("error_code 必须是不超过 100 个字符的非空值")
        if any(ord(character) < 32 for character in normalized_error_code):
            raise ValueError("error_code 不能包含控制字符")

        try:
            async with self._session_factory.begin() as session:
                version = await _locked_version(session, attempt)
                if version is None:
                    raise _version_state_error()
                if version.status == "failed":
                    return
                if version.status != "building":
                    raise _version_state_error()
                version.status = "failed"
                version.error_code = normalized_error_code
                version.activated_at = None
                await session.flush()
        except IngestionError:
            raise
        except SQLAlchemyError:
            raise _persistence_error("无法记录文档入库失败状态") from None


async def _locked_knowledge_base(
    session: AsyncSession,
    knowledge_base_id: UUID,
) -> KnowledgeBaseRecord | None:
    result = await session.execute(
        select(KnowledgeBaseRecord)
        .where(KnowledgeBaseRecord.id == knowledge_base_id)
        .with_for_update()
    )
    return result.scalar_one_or_none()


async def _find_document(
    session: AsyncSession,
    knowledge_base_id: UUID,
    source_key: str,
) -> DocumentRecord | None:
    result = await session.execute(
        select(DocumentRecord).where(
            DocumentRecord.knowledge_base_id == knowledge_base_id,
            DocumentRecord.source_key == source_key,
        )
    )
    return result.scalar_one_or_none()


async def _locked_version(
    session: AsyncSession,
    attempt: IngestionAttempt,
) -> DocumentVersionRecord | None:
    result = await session.execute(
        select(DocumentVersionRecord)
        .where(
            DocumentVersionRecord.id == attempt.version_id,
            DocumentVersionRecord.document_id == attempt.document_id,
            DocumentVersionRecord.knowledge_base_id == attempt.knowledge_base_id,
            DocumentVersionRecord.version_number == attempt.version_number,
        )
        .with_for_update()
    )
    return result.scalar_one_or_none()


def _ensure_index_configuration(
    knowledge_base: KnowledgeBaseRecord,
    expected: EmbeddingIndexConfiguration,
) -> None:
    actual = EmbeddingIndexConfiguration(
        model_id=knowledge_base.embedding_model_id,
        model_revision=knowledge_base.embedding_model_revision,
        dimension=knowledge_base.embedding_dimension,
        normalized=knowledge_base.embedding_normalized,
    )
    if actual != expected:
        raise IngestionError(
            IngestionErrorCode.INDEX_CONFIGURATION_MISMATCH,
            "Embedding 模型与知识库索引配置不一致",
            retryable=False,
        )


def _require_building_version(
    version: DocumentVersionRecord | None,
    attempt: IngestionAttempt,
) -> DocumentVersionRecord:
    if (
        version is None
        or version.status != "building"
        or version.version_number != attempt.version_number
    ):
        raise _version_state_error()
    return version


def _ensure_activation_payload(
    version: DocumentVersionRecord,
    chunking: ChunkingResult,
    embeddings: EmbeddingResult,
) -> None:
    if version.content_sha256 != chunking.source.source.content_sha256 or len(
        chunking.chunks
    ) != len(embeddings.vectors):
        raise IngestionError(
            IngestionErrorCode.EMBEDDING_PROTOCOL,
            "待激活的 Chunk 与 Embedding 不一致",
            retryable=False,
        )


def _version_source_metadata(
    document: ParsedDocument,
    *,
    chunking: ChunkingResult | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "filename": document.source.filename,
        "document_format": document.source.document_format.value,
        "charset": document.source.charset,
        "byte_size": document.source.byte_size,
    }
    if chunking is not None:
        metadata.update(
            {
                "tokenizer_model_id": chunking.tokenizer_model_id,
                "tokenizer_revision": chunking.tokenizer_revision,
                "chunk_target_tokens": chunking.policy.target_tokens,
                "chunk_overlap_tokens": chunking.policy.overlap_tokens,
            }
        )
    return metadata


def _section_value(section_path: tuple[str, ...]) -> str | None:
    if not section_path:
        return None
    return " > ".join(section_path)[:500]


def _version_state_error() -> IngestionError:
    return IngestionError(
        IngestionErrorCode.VERSION_STATE_CONFLICT,
        "文档版本状态已变化。当前入库无法完成",
        retryable=True,
    )


def _persistence_error(message: str) -> IngestionError:
    return IngestionError(
        IngestionErrorCode.PERSISTENCE_FAILURE,
        message,
        retryable=True,
    )
