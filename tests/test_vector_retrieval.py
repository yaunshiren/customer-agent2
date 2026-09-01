"""Unit tests for typed, model-neutral vector retrieval orchestration."""

from uuid import uuid4

import pytest

from customer_agent2.application import VectorRetrievalService
from customer_agent2.domain.models import (
    DocumentFormat,
    EmbeddingIndexConfiguration,
    EmbeddingRequest,
    EmbeddingResult,
    RetrievalError,
    RetrievalErrorCode,
    RetrievedChunkSource,
    VectorSearchCandidate,
    VectorSearchRequest,
    VectorSearchScope,
)
from customer_agent2.infrastructure.models import FakeEmbeddingModel


class RecordingVectorSearchRepository:
    def __init__(self, candidates: tuple[VectorSearchCandidate, ...] = ()) -> None:
        self.candidates = candidates
        self.calls: list[
            tuple[
                tuple[float, ...],
                EmbeddingIndexConfiguration,
                VectorSearchScope,
                int,
                int,
            ]
        ] = []
        self.error: RetrievalError | None = None

    async def search(
        self,
        query_vector: tuple[float, ...],
        index_configuration: EmbeddingIndexConfiguration,
        scope: VectorSearchScope,
        *,
        limit: int,
        hnsw_ef_search: int,
    ) -> tuple[VectorSearchCandidate, ...]:
        self.calls.append((query_vector, index_configuration, scope, limit, hnsw_ef_search))
        if self.error is not None:
            raise self.error
        return self.candidates


class WrongIdentityEmbeddingModel(FakeEmbeddingModel):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        result = await super().embed(request)
        return EmbeddingResult(
            model_id="unexpected-model",
            model_revision=result.model_revision,
            vectors=result.vectors,
            dimension=result.dimension,
            normalized=result.normalized,
        )


def candidate() -> VectorSearchCandidate:
    return VectorSearchCandidate(
        rank=1,
        chunk_id=uuid4(),
        knowledge_base_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        source_key="refund.md",
        display_name="refund.md",
        document_format=DocumentFormat.MARKDOWN,
        media_type="text/markdown",
        parser_name="customer-agent2-markdown",
        parser_version="1",
        chunk_index=0,
        content="退款申请会在三个工作日内处理。",
        token_count=16,
        content_sha256="a" * 64,
        section="退款流程",
        page_number=None,
        source=RetrievedChunkSource(
            block_start_ordinal=0,
            block_end_ordinal=1,
            start_line=1,
            end_line=3,
            section_path=("退款流程",),
            overlap_with_previous_tokens=0,
        ),
        cosine_distance=0.2,
        similarity=0.8,
    )


def test_vector_search_scope_normalizes_and_deduplicates_filters() -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()

    scope = VectorSearchScope(
        knowledge_base_ids=(knowledge_base_id, knowledge_base_id),
        document_ids=(document_id, document_id),
        document_formats=(DocumentFormat.MARKDOWN, DocumentFormat.MARKDOWN),
        parser_names=(" parser-a ", "parser-a"),
        sections=(" 退款流程 ", "退款流程"),
        page_numbers=(1, 1, 2),
    )

    assert scope.knowledge_base_ids == (knowledge_base_id,)
    assert scope.document_ids == (document_id,)
    assert scope.document_formats == (DocumentFormat.MARKDOWN,)
    assert scope.parser_names == ("parser-a",)
    assert scope.sections == ("退款流程",)
    assert scope.page_numbers == (1, 2)


def test_vector_search_scope_empty_knowledge_bases_means_global() -> None:
    assert VectorSearchScope().knowledge_base_ids == ()


def test_vector_search_scope_rejects_invalid_metadata_filters() -> None:
    with pytest.raises(ValueError, match="page_numbers"):
        VectorSearchScope((uuid4(),), page_numbers=(0,))
    with pytest.raises(ValueError, match="parser_names"):
        VectorSearchScope((uuid4(),), parser_names=(" ",))


@pytest.mark.asyncio
async def test_vector_retrieval_embeds_once_and_forwards_typed_scope() -> None:
    model = FakeEmbeddingModel("embedding", revision="revision", dimension=8)
    repository = RecordingVectorSearchRepository((candidate(),))
    service = VectorRetrievalService(
        model,
        repository,
        recall_budget=20,
        hnsw_ef_search=100,
    )
    scope = VectorSearchScope((uuid4(),), document_formats=(DocumentFormat.MARKDOWN,))

    result = await service.search(VectorSearchRequest("  如何退款?  ", scope))

    assert model.requests == (EmbeddingRequest(("如何退款?",)),)
    assert result.candidates == repository.candidates
    assert result.index_configuration == EmbeddingIndexConfiguration(
        "embedding",
        "revision",
        8,
        True,
    )
    assert len(repository.calls) == 1
    query_vector, index_configuration, actual_scope, limit, ef_search = repository.calls[0]
    assert len(query_vector) == 8
    assert index_configuration == result.index_configuration
    assert actual_scope == scope
    assert limit == 20
    assert ef_search == 100


@pytest.mark.asyncio
async def test_vector_retrieval_rejects_embedding_identity_mismatch() -> None:
    repository = RecordingVectorSearchRepository()
    service = VectorRetrievalService(
        WrongIdentityEmbeddingModel("embedding", revision="revision", dimension=8),
        repository,
        recall_budget=20,
        hnsw_ef_search=100,
    )

    with pytest.raises(RetrievalError) as captured:
        await service.search(VectorSearchRequest("退款", VectorSearchScope((uuid4(),))))

    assert captured.value.code is RetrievalErrorCode.EMBEDDING_PROTOCOL
    assert captured.value.retryable is False
    assert repository.calls == []


@pytest.mark.asyncio
async def test_vector_retrieval_preserves_sanitized_repository_failure() -> None:
    repository = RecordingVectorSearchRepository()
    repository.error = RetrievalError(
        RetrievalErrorCode.INDEX_CONFIGURATION_MISMATCH,
        "索引配置不一致",
        retryable=False,
    )
    service = VectorRetrievalService(
        FakeEmbeddingModel("embedding", revision="revision", dimension=8),
        repository,
        recall_budget=20,
        hnsw_ef_search=100,
    )

    with pytest.raises(RetrievalError) as captured:
        await service.search(VectorSearchRequest("退款", VectorSearchScope((uuid4(),))))

    assert captured.value is repository.error
