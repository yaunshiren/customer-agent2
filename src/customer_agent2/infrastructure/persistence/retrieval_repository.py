"""SQLAlchemy adapter for filtered active-version pgvector retrieval."""

import math
from collections.abc import Mapping
from typing import cast
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent2.domain.models import (
    DocumentFormat,
    EmbeddingIndexConfiguration,
    RetrievalError,
    RetrievalErrorCode,
    RetrievedChunkSource,
    VectorSearchCandidate,
    VectorSearchScope,
)
from customer_agent2.infrastructure.persistence.models import (
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
)


class SQLAlchemyVectorSearchRepository:
    """Search active compatible indexes globally or by intent-selected IDs."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def search(
        self,
        query_vector: tuple[float, ...],
        index_configuration: EmbeddingIndexConfiguration,
        scope: VectorSearchScope,
        *,
        limit: int,
        hnsw_ef_search: int,
    ) -> tuple[VectorSearchCandidate, ...]:
        """Return nearest active chunks from a directed or global scope."""
        _validate_search_parameters(
            query_vector,
            index_configuration,
            limit=limit,
            hnsw_ef_search=hnsw_ef_search,
        )
        try:
            async with self._session_factory.begin() as session:
                await _ensure_compatible_scope(session, scope, index_configuration)
                await _configure_hnsw(session, hnsw_ef_search)
                rows = await _search_rows(
                    session,
                    query_vector,
                    index_configuration,
                    scope,
                    limit=limit,
                )
                return tuple(
                    _candidate_from_row(rank, row) for rank, row in enumerate(rows, start=1)
                )
        except RetrievalError:
            raise
        except SQLAlchemyError:
            raise RetrievalError(
                RetrievalErrorCode.PERSISTENCE_FAILURE,
                "无法完成向量检索",
                retryable=True,
            ) from None
        except (TypeError, ValueError):
            raise RetrievalError(
                RetrievalErrorCode.PERSISTENCE_FAILURE,
                "检索数据格式无效",
                retryable=False,
            ) from None


class SQLAlchemyKnowledgeBaseScopeResolver:
    """Map configured Ragent-style collection slugs to knowledge-base UUIDs."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def resolve(self, slugs: tuple[str, ...]) -> tuple[UUID, ...]:
        """Resolve every configured slug while preserving configuration order."""
        if not slugs:
            return ()
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(KnowledgeBaseRecord.slug, KnowledgeBaseRecord.id).where(
                        KnowledgeBaseRecord.slug.in_(slugs)
                    )
                )
                resolved = {slug: knowledge_base_id for slug, knowledge_base_id in result.all()}
        except SQLAlchemyError:
            raise RetrievalError(
                RetrievalErrorCode.PERSISTENCE_FAILURE,
                "无法解析意图知识库范围",
                retryable=True,
            ) from None
        if set(resolved) != set(slugs):
            raise RetrievalError(
                RetrievalErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
                "意图绑定包含不存在的知识库",
                retryable=False,
            )
        return tuple(resolved[slug] for slug in slugs)


async def _ensure_compatible_scope(
    session: AsyncSession,
    scope: VectorSearchScope,
    expected: EmbeddingIndexConfiguration,
) -> None:
    if not scope.knowledge_base_ids:
        return
    result = await session.execute(
        select(KnowledgeBaseRecord).where(KnowledgeBaseRecord.id.in_(scope.knowledge_base_ids))
    )
    records = result.scalars().all()
    if len(records) != len(scope.knowledge_base_ids):
        raise RetrievalError(
            RetrievalErrorCode.KNOWLEDGE_BASE_NOT_FOUND,
            "检索范围包含不存在的知识库",
            retryable=False,
        )
    for record in records:
        actual = EmbeddingIndexConfiguration(
            model_id=record.embedding_model_id,
            model_revision=record.embedding_model_revision,
            dimension=record.embedding_dimension,
            normalized=record.embedding_normalized,
        )
        if actual != expected:
            raise RetrievalError(
                RetrievalErrorCode.INDEX_CONFIGURATION_MISMATCH,
                "Embedding 模型与检索范围的索引配置不一致",
                retryable=False,
            )


async def _configure_hnsw(session: AsyncSession, hnsw_ef_search: int) -> None:
    # SET LOCAL semantics keep these pgvector controls inside this transaction only.
    await session.execute(select(func.set_config("hnsw.iterative_scan", "strict_order", True)))
    await session.execute(select(func.set_config("hnsw.ef_search", str(hnsw_ef_search), True)))


async def _search_rows(
    session: AsyncSession,
    query_vector: tuple[float, ...],
    index_configuration: EmbeddingIndexConfiguration,
    scope: VectorSearchScope,
    *,
    limit: int,
) -> list[tuple[ChunkRecord, DocumentVersionRecord, DocumentRecord, float]]:
    distance = ChunkRecord.embedding.cosine_distance(list(query_vector)).label("cosine_distance")
    statement = (
        select(ChunkRecord, DocumentVersionRecord, DocumentRecord, distance)
        .join(
            DocumentVersionRecord,
            DocumentVersionRecord.id == ChunkRecord.document_version_id,
        )
        .join(DocumentRecord, DocumentRecord.id == DocumentVersionRecord.document_id)
        .join(
            KnowledgeBaseRecord,
            KnowledgeBaseRecord.id == ChunkRecord.knowledge_base_id,
        )
        .where(
            DocumentVersionRecord.status == "active",
            KnowledgeBaseRecord.embedding_model_id == index_configuration.model_id,
            (KnowledgeBaseRecord.embedding_model_revision == index_configuration.model_revision),
            KnowledgeBaseRecord.embedding_dimension == index_configuration.dimension,
            KnowledgeBaseRecord.embedding_normalized == index_configuration.normalized,
        )
    )
    if scope.knowledge_base_ids:
        statement = statement.where(ChunkRecord.knowledge_base_id.in_(scope.knowledge_base_ids))
    if scope.document_ids:
        statement = statement.where(DocumentVersionRecord.document_id.in_(scope.document_ids))
    if scope.document_formats:
        statement = statement.where(
            DocumentVersionRecord.source_metadata["document_format"].astext.in_(
                tuple(value.value for value in scope.document_formats)
            )
        )
    if scope.parser_names:
        statement = statement.where(DocumentVersionRecord.parser_name.in_(scope.parser_names))
    if scope.sections:
        statement = statement.where(ChunkRecord.section.in_(scope.sections))
    if scope.page_numbers:
        statement = statement.where(ChunkRecord.page_number.in_(scope.page_numbers))

    result = await session.execute(statement.order_by(distance).limit(limit))
    return [
        (
            cast(ChunkRecord, row[0]),
            cast(DocumentVersionRecord, row[1]),
            cast(DocumentRecord, row[2]),
            cast(float, row[3]),
        )
        for row in result.all()
    ]


def _candidate_from_row(
    rank: int,
    row: tuple[ChunkRecord, DocumentVersionRecord, DocumentRecord, float],
) -> VectorSearchCandidate:
    chunk, version, document, distance = row
    document_format = _document_format(version.source_metadata)
    source = _chunk_source(chunk.source_metadata)
    media_type = _required_text(version.media_type, "media_type")
    parser_name = _required_text(version.parser_name, "parser_name")
    parser_version = _required_text(version.parser_version, "parser_version")
    return VectorSearchCandidate(
        rank=rank,
        chunk_id=chunk.id,
        knowledge_base_id=chunk.knowledge_base_id,
        document_id=version.document_id,
        document_version_id=version.id,
        source_key=document.source_key,
        display_name=document.display_name,
        document_format=document_format,
        media_type=media_type,
        parser_name=parser_name,
        parser_version=parser_version,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        token_count=chunk.token_count,
        content_sha256=chunk.content_sha256,
        section=chunk.section,
        page_number=chunk.page_number,
        source=source,
        cosine_distance=distance,
        similarity=1.0 - distance,
    )


def _document_format(metadata: Mapping[str, object]) -> DocumentFormat:
    value = metadata.get("document_format")
    if not isinstance(value, str):
        raise ValueError("持久化文档格式无效")
    return DocumentFormat(value)


def _chunk_source(metadata: Mapping[str, object]) -> RetrievedChunkSource:
    raw_section_path = metadata.get("section_path")
    if not isinstance(raw_section_path, list):
        raise ValueError("持久化 Chunk 章节路径无效")
    section_path: list[str] = []
    for value in cast(list[object], raw_section_path):
        if not isinstance(value, str):
            raise ValueError("持久化 Chunk 章节路径无效")
        section_path.append(value)
    return RetrievedChunkSource(
        block_start_ordinal=_required_int(metadata, "block_start_ordinal"),
        block_end_ordinal=_required_int(metadata, "block_end_ordinal"),
        start_line=_required_int(metadata, "start_line"),
        end_line=_required_int(metadata, "end_line"),
        section_path=tuple(section_path),
        overlap_with_previous_tokens=_required_int(
            metadata,
            "overlap_with_previous_tokens",
        ),
    )


def _required_int(metadata: Mapping[str, object], key: str) -> int:
    value = metadata.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"持久化 Chunk 元数据 {key} 无效")
    return value


def _required_text(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"持久化文档字段 {field_name} 无效")
    return value


def _validate_search_parameters(
    query_vector: tuple[float, ...],
    index_configuration: EmbeddingIndexConfiguration,
    *,
    limit: int,
    hnsw_ef_search: int,
) -> None:
    if limit < 1 or hnsw_ef_search < 1:
        raise ValueError("检索 limit 和 hnsw_ef_search 必须大于 0")
    if len(query_vector) != index_configuration.dimension:
        raise ValueError("查询向量维度与索引配置不一致")
    if any(not math.isfinite(value) for value in query_vector):
        raise ValueError("查询向量不能包含 NaN 或无限值")
