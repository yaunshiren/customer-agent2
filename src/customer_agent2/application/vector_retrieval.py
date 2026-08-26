"""Embed one query and retrieve compatible active chunks through pgvector."""

import math

from customer_agent2.domain.models import (
    EmbeddingIndexConfiguration,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
    RetrievalError,
    RetrievalErrorCode,
    VectorSearchRepository,
    VectorSearchRequest,
    VectorSearchResult,
)


class VectorRetrievalService:
    """Provider-neutral orchestration for the first-stage vector recall channel."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        repository: VectorSearchRepository,
        *,
        recall_budget: int,
        hnsw_ef_search: int,
    ) -> None:
        if recall_budget < 1 or hnsw_ef_search < 1:
            raise ValueError("召回预算和 HNSW ef_search 必须大于 0")
        self._embedding_model = embedding_model
        self._repository = repository
        self._recall_budget = recall_budget
        self._hnsw_ef_search = hnsw_ef_search

    async def search(self, request: VectorSearchRequest) -> VectorSearchResult:
        """Embed a query once, then search only explicitly scoped compatible indexes."""
        index_configuration = self._index_configuration()
        embedding = await self._embedding_model.embed(EmbeddingRequest(texts=(request.query,)))
        query_vector = _validated_query_vector(embedding, index_configuration)
        candidates = await self._repository.search(
            query_vector,
            index_configuration,
            request.scope,
            limit=self._recall_budget,
            hnsw_ef_search=self._hnsw_ef_search,
        )
        return VectorSearchResult(
            index_configuration=index_configuration,
            candidates=candidates,
        )

    def _index_configuration(self) -> EmbeddingIndexConfiguration:
        return EmbeddingIndexConfiguration(
            model_id=self._embedding_model.model_id,
            model_revision=self._embedding_model.revision,
            dimension=self._embedding_model.dimension,
            normalized=self._embedding_model.normalized,
        )


def _validated_query_vector(
    result: EmbeddingResult,
    expected: EmbeddingIndexConfiguration,
) -> tuple[float, ...]:
    identity_matches = (
        result.model_id == expected.model_id
        and result.model_revision == expected.model_revision
        and result.dimension == expected.dimension
        and result.normalized == expected.normalized
    )
    if len(result.vectors) != 1 or not identity_matches:
        raise _embedding_protocol_error()

    vector = result.vectors[0]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if norm == 0 or (result.normalized and not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3)):
        raise _embedding_protocol_error()
    return vector


def _embedding_protocol_error() -> RetrievalError:
    return RetrievalError(
        RetrievalErrorCode.EMBEDDING_PROTOCOL,
        "Embedding 返回结果与检索索引配置不一致",
        retryable=False,
    )
