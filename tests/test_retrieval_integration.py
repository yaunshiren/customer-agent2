"""Opt-in end-to-end PostgreSQL/pgvector tests for M2-G vector retrieval."""

import math
import os
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from customer_agent2.application import VectorRetrievalService
from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    DocumentFormat,
    EmbeddingRequest,
    RetrievalError,
    RetrievalErrorCode,
    VectorSearchRequest,
    VectorSearchScope,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.models import FakeEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyKnowledgeBaseScopeResolver,
    SQLAlchemyVectorSearchRepository,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]

MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"


@dataclass(frozen=True, slots=True)
class SeededRetrievalData:
    knowledge_base_id: UUID
    second_knowledge_base_id: UUID
    incompatible_knowledge_base_id: UUID
    markdown_document_id: UUID
    text_document_id: UUID
    active_markdown_version_id: UUID
    superseded_markdown_version_id: UUID

    @property
    def knowledge_base_ids(self) -> tuple[UUID, ...]:
        return (
            self.knowledge_base_id,
            self.second_knowledge_base_id,
            self.incompatible_knowledge_base_id,
        )


def embedding_model() -> FakeEmbeddingModel:
    return FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
    )


@pytest.mark.asyncio
async def test_real_vector_retrieval_ranks_active_chunks_and_applies_all_filters() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    model = embedding_model()
    query_vector = list((await model.embed(EmbeddingRequest(("如何退款",)))).vectors[0])
    seeded = await _seed_retrieval_data(manager, query_vector)
    service = VectorRetrievalService(
        model,
        SQLAlchemyVectorSearchRepository(manager.session_factory),
        recall_budget=20,
        hnsw_ef_search=100,
    )

    try:
        base_scope = VectorSearchScope((seeded.knowledge_base_id,))
        result = await service.search(VectorSearchRequest("如何退款", base_scope))

        assert [item.document_id for item in result.candidates] == [
            seeded.markdown_document_id,
            seeded.text_document_id,
        ]
        assert result.candidates[0].document_version_id == seeded.active_markdown_version_id
        assert all(
            item.document_version_id != seeded.superseded_markdown_version_id
            for item in result.candidates
        )
        assert math.isclose(result.candidates[0].similarity, 1.0, abs_tol=1e-6)
        assert result.candidates[0].source.section_path == ("退款流程",)

        global_result = await service.search(VectorSearchRequest("如何退款", VectorSearchScope()))
        global_knowledge_base_ids = {item.knowledge_base_id for item in global_result.candidates}
        assert {
            seeded.knowledge_base_id,
            seeded.second_knowledge_base_id,
        } <= global_knowledge_base_ids
        assert seeded.incompatible_knowledge_base_id not in global_knowledge_base_ids

        resolver = SQLAlchemyKnowledgeBaseScopeResolver(manager.session_factory)
        assert await resolver.resolve(
            (
                f"m2-g-second-{seeded.second_knowledge_base_id}",
                f"m2-g-primary-{seeded.knowledge_base_id}",
            )
        ) == (seeded.second_knowledge_base_id, seeded.knowledge_base_id)

        document_only = await service.search(
            VectorSearchRequest(
                "如何退款",
                VectorSearchScope(
                    (seeded.knowledge_base_id,),
                    document_ids=(seeded.text_document_id,),
                ),
            )
        )
        assert [item.document_id for item in document_only.candidates] == [seeded.text_document_id]

        metadata_only = await service.search(
            VectorSearchRequest(
                "如何退款",
                VectorSearchScope(
                    (seeded.knowledge_base_id,),
                    document_formats=(DocumentFormat.MARKDOWN,),
                    parser_names=("customer-agent2-markdown",),
                    sections=("退款流程",),
                    page_numbers=(2,),
                ),
            )
        )
        assert [item.document_id for item in metadata_only.candidates] == [
            seeded.markdown_document_id
        ]

        second_base = await service.search(
            VectorSearchRequest(
                "如何退款",
                VectorSearchScope((seeded.second_knowledge_base_id,)),
            )
        )
        assert {item.knowledge_base_id for item in second_base.candidates} == {
            seeded.second_knowledge_base_id
        }
    finally:
        await _remove_knowledge_bases(manager, seeded.knowledge_base_ids)
        await manager.close()


@pytest.mark.asyncio
async def test_real_vector_retrieval_rejects_missing_and_incompatible_scopes() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    model = embedding_model()
    query_vector = list((await model.embed(EmbeddingRequest(("退款",)))).vectors[0])
    seeded = await _seed_retrieval_data(manager, query_vector)
    service = VectorRetrievalService(
        model,
        SQLAlchemyVectorSearchRepository(manager.session_factory),
        recall_budget=20,
        hnsw_ef_search=100,
    )

    try:
        with pytest.raises(RetrievalError) as missing:
            await service.search(VectorSearchRequest("退款", VectorSearchScope((uuid4(),))))
        assert missing.value.code is RetrievalErrorCode.KNOWLEDGE_BASE_NOT_FOUND

        with pytest.raises(RetrievalError) as incompatible:
            await service.search(
                VectorSearchRequest(
                    "退款",
                    VectorSearchScope((seeded.incompatible_knowledge_base_id,)),
                )
            )
        assert incompatible.value.code is RetrievalErrorCode.INDEX_CONFIGURATION_MISMATCH
    finally:
        await _remove_knowledge_bases(manager, seeded.knowledge_base_ids)
        await manager.close()


async def _seed_retrieval_data(
    manager: DatabaseManager,
    query_vector: list[float],
) -> SeededRetrievalData:
    data = SeededRetrievalData(
        knowledge_base_id=uuid4(),
        second_knowledge_base_id=uuid4(),
        incompatible_knowledge_base_id=uuid4(),
        markdown_document_id=uuid4(),
        text_document_id=uuid4(),
        active_markdown_version_id=uuid4(),
        superseded_markdown_version_id=uuid4(),
    )
    second_document_id = uuid4()
    second_version_id = uuid4()
    text_version_id = uuid4()
    opposite_vector = [-value for value in query_vector]

    async with manager.session_factory.begin() as session:
        session.add_all(
            [
                _knowledge_base(data.knowledge_base_id, "primary"),
                _knowledge_base(data.second_knowledge_base_id, "second"),
                _knowledge_base(
                    data.incompatible_knowledge_base_id,
                    "incompatible",
                    revision="different-revision",
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                DocumentRecord(
                    id=data.markdown_document_id,
                    knowledge_base_id=data.knowledge_base_id,
                    source_key="guides/refund.md",
                    display_name="refund.md",
                ),
                DocumentRecord(
                    id=data.text_document_id,
                    knowledge_base_id=data.knowledge_base_id,
                    source_key="guides/shipping.txt",
                    display_name="shipping.txt",
                ),
                DocumentRecord(
                    id=second_document_id,
                    knowledge_base_id=data.second_knowledge_base_id,
                    source_key="shared/refund.md",
                    display_name="refund.md",
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                _version(
                    data.superseded_markdown_version_id,
                    data.markdown_document_id,
                    data.knowledge_base_id,
                    version_number=1,
                    status="superseded",
                    document_format=DocumentFormat.MARKDOWN,
                    parser_name="customer-agent2-markdown",
                    content_sha256="1" * 64,
                ),
                _version(
                    data.active_markdown_version_id,
                    data.markdown_document_id,
                    data.knowledge_base_id,
                    version_number=2,
                    status="active",
                    document_format=DocumentFormat.MARKDOWN,
                    parser_name="customer-agent2-markdown",
                    content_sha256="2" * 64,
                ),
                _version(
                    text_version_id,
                    data.text_document_id,
                    data.knowledge_base_id,
                    version_number=1,
                    status="active",
                    document_format=DocumentFormat.PLAIN_TEXT,
                    parser_name="customer-agent2-text",
                    content_sha256="3" * 64,
                ),
                _version(
                    second_version_id,
                    second_document_id,
                    data.second_knowledge_base_id,
                    version_number=1,
                    status="active",
                    document_format=DocumentFormat.MARKDOWN,
                    parser_name="customer-agent2-markdown",
                    content_sha256="4" * 64,
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                _chunk(
                    data.superseded_markdown_version_id,
                    data.knowledge_base_id,
                    "旧版退款流程",
                    query_vector,
                    section="退款流程",
                    page_number=1,
                    content_sha256="a" * 64,
                ),
                _chunk(
                    data.active_markdown_version_id,
                    data.knowledge_base_id,
                    "新版退款流程",
                    query_vector,
                    section="退款流程",
                    page_number=2,
                    content_sha256="b" * 64,
                ),
                _chunk(
                    text_version_id,
                    data.knowledge_base_id,
                    "配送时效",
                    opposite_vector,
                    section=None,
                    page_number=None,
                    content_sha256="c" * 64,
                ),
                _chunk(
                    second_version_id,
                    data.second_knowledge_base_id,
                    "第二知识库退款流程",
                    query_vector,
                    section="退款流程",
                    page_number=1,
                    content_sha256="d" * 64,
                ),
            ]
        )
    return data


def _knowledge_base(
    knowledge_base_id: UUID,
    suffix: str,
    *,
    revision: str = MODEL_REVISION,
) -> KnowledgeBaseRecord:
    return KnowledgeBaseRecord(
        id=knowledge_base_id,
        slug=f"m2-g-{suffix}-{knowledge_base_id}",
        name=f"M2-G {suffix}",
        embedding_model_id=MODEL_ID,
        embedding_model_revision=revision,
    )


def _version(
    version_id: UUID,
    document_id: UUID,
    knowledge_base_id: UUID,
    *,
    version_number: int,
    status: str,
    document_format: DocumentFormat,
    parser_name: str,
    content_sha256: str,
) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        id=version_id,
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        version_number=version_number,
        status=status,
        content_sha256=content_sha256,
        media_type="text/markdown" if document_format is DocumentFormat.MARKDOWN else "text/plain",
        parser_name=parser_name,
        parser_version="1",
        source_metadata={"document_format": document_format.value},
    )


def _chunk(
    version_id: UUID,
    knowledge_base_id: UUID,
    content: str,
    embedding: list[float],
    *,
    section: str | None,
    page_number: int | None,
    content_sha256: str,
) -> ChunkRecord:
    return ChunkRecord(
        document_version_id=version_id,
        knowledge_base_id=knowledge_base_id,
        chunk_index=0,
        content=content,
        token_count=len(content),
        content_sha256=content_sha256,
        section=section,
        page_number=page_number,
        source_metadata={
            "block_start_ordinal": 0,
            "block_end_ordinal": 0,
            "start_line": 1,
            "end_line": 2,
            "section_path": [section] if section is not None else [],
            "overlap_with_previous_tokens": 0,
        },
        embedding=embedding,
    )


async def _remove_knowledge_bases(
    manager: DatabaseManager,
    knowledge_base_ids: tuple[UUID, ...],
) -> None:
    async with manager.session_factory.begin() as session:
        await session.execute(
            delete(KnowledgeBaseRecord).where(KnowledgeBaseRecord.id.in_(knowledge_base_ids))
        )
