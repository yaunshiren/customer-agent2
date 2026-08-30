"""Deterministic weighted RRF, duplicate control, and bounded Rerank orchestration."""

import asyncio
import logging
import math
from dataclasses import dataclass, replace
from uuid import UUID

from customer_agent2.domain.models import (
    EmbeddingIndexConfiguration,
    ModelError,
    ModelErrorCode,
    RerankDegradationReason,
    RerankDocument,
    RerankModel,
    RerankRequest,
    RerankResult,
    RetrievalError,
    RetrievalErrorCode,
    VectorSearchCandidate,
    VectorSearchResult,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalFusionResult:
    """RRF output plus content-free counts needed by tests and observability."""

    index_configuration: EmbeddingIndexConfiguration
    candidates: tuple[VectorSearchCandidate, ...]
    rrf_scores: tuple[float, ...]
    query_hit_counts: tuple[int, ...]
    raw_candidate_count: int
    unique_chunk_count: int
    unique_content_count: int

    def __post_init__(self) -> None:
        candidate_count = len(self.candidates)
        if len(self.rrf_scores) != candidate_count or len(self.query_hit_counts) != candidate_count:
            raise ValueError("融合候选、RRF 分数和命中次数长度必须一致")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, candidate_count + 1)
        ):
            raise ValueError("融合候选必须使用连续排名")
        if any(not math.isfinite(score) or score <= 0 for score in self.rrf_scores):
            raise ValueError("RRF 分数必须是有限正数")
        if any(hit_count < 1 for hit_count in self.query_hit_counts):
            raise ValueError("查询命中次数必须大于 0")
        if not (
            self.raw_candidate_count >= self.unique_chunk_count >= self.unique_content_count >= 0
        ):
            raise ValueError("融合候选计数关系无效")
        if candidate_count > self.unique_content_count:
            raise ValueError("最终候选不能多于唯一内容数量")


@dataclass(frozen=True, slots=True)
class CandidateRerankResult:
    """Validated candidate order and explicit Rerank degradation metadata."""

    candidates: tuple[VectorSearchCandidate, ...]
    model_id: str
    degradation_reason: RerankDegradationReason | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        model_id = self.model_id.strip()
        if not self.candidates or not model_id:
            raise ValueError("Rerank 结果必须包含候选和 model_id")
        if tuple(candidate.rank for candidate in self.candidates) != tuple(
            range(1, len(self.candidates) + 1)
        ):
            raise ValueError("Rerank 候选必须使用连续排名")
        if self.total_tokens is not None and self.total_tokens < 0:
            raise ValueError("Rerank total_tokens 不能为负数")
        object.__setattr__(self, "model_id", model_id)


@dataclass(slots=True)
class _RrfAggregate:
    candidate: VectorSearchCandidate
    score: float
    query_hit_count: int
    best_rank: int
    best_similarity: float
    first_query_index: int


class RetrievalPostProcessor:
    """Apply the accepted M5-A fusion and Rerank baseline."""

    def __init__(
        self,
        rerank_model: RerankModel,
        *,
        rrf_k: int,
        rerank_candidate_limit: int,
        context_top_k: int,
        max_chunks_per_document: int,
        rerank_timeout_seconds: float,
    ) -> None:
        if rrf_k < 1:
            raise ValueError("rrf_k 必须大于 0")
        if context_top_k < 1 or rerank_candidate_limit < context_top_k:
            raise ValueError("Rerank 候选上限不能小于 context_top_k")
        if max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document 必须大于 0")
        if rerank_timeout_seconds <= 0:
            raise ValueError("rerank_timeout_seconds 必须大于 0")
        self._rerank_model = rerank_model
        self._rrf_k = rrf_k
        self._rerank_candidate_limit = rerank_candidate_limit
        self._context_top_k = context_top_k
        self._max_chunks_per_document = max_chunks_per_document
        self._rerank_timeout_seconds = rerank_timeout_seconds

    def fuse(
        self,
        results: tuple[VectorSearchResult, ...],
        *,
        query_weights: tuple[float, ...] | None = None,
    ) -> RetrievalFusionResult:
        """Fuse ranked lists, remove duplicate content, and enforce document diversity."""
        if not results:
            raise ValueError("results 不能为空")
        weights = (1.0,) * len(results) if query_weights is None else query_weights
        if len(weights) != len(results) or any(
            not math.isfinite(weight) or weight <= 0 for weight in weights
        ):
            raise ValueError("query_weights 必须与结果数量一致且全部为有限正数")
        first_configuration = results[0].index_configuration
        if any(result.index_configuration != first_configuration for result in results[1:]):
            raise RetrievalError(
                RetrievalErrorCode.INDEX_CONFIGURATION_MISMATCH,
                "多问题检索返回了不一致的索引配置",
                retryable=False,
            )

        aggregates: dict[UUID, _RrfAggregate] = {}
        raw_candidate_count = 0
        for query_index, (result, weight) in enumerate(zip(results, weights, strict=True)):
            seen_in_result: set[UUID] = set()
            raw_candidate_count += len(result.candidates)
            for candidate in result.candidates:
                if candidate.chunk_id in seen_in_result:
                    raise _retrieval_protocol_error("单个检索结果包含重复 Chunk")
                seen_in_result.add(candidate.chunk_id)
                contribution = weight / (self._rrf_k + candidate.rank)
                aggregate = aggregates.get(candidate.chunk_id)
                if aggregate is None:
                    aggregates[candidate.chunk_id] = _RrfAggregate(
                        candidate=candidate,
                        score=contribution,
                        query_hit_count=1,
                        best_rank=candidate.rank,
                        best_similarity=candidate.similarity,
                        first_query_index=query_index,
                    )
                    continue
                if _candidate_metadata(aggregate.candidate) != _candidate_metadata(candidate):
                    raise _retrieval_protocol_error("同一 Chunk 的检索元数据不一致")
                aggregate.score += contribution
                aggregate.query_hit_count += 1
                if _occurrence_key(query_index, candidate) < _occurrence_key(
                    aggregate.first_query_index,
                    aggregate.candidate,
                ):
                    aggregate.candidate = candidate
                    aggregate.first_query_index = query_index
                aggregate.best_rank = min(aggregate.best_rank, candidate.rank)
                aggregate.best_similarity = max(
                    aggregate.best_similarity,
                    candidate.similarity,
                )

        ordered = sorted(aggregates.values(), key=_aggregate_key)
        content_by_hash: dict[str, str] = {}
        unique_content: list[_RrfAggregate] = []
        for aggregate in ordered:
            content_hash = aggregate.candidate.content_sha256
            existing_content = content_by_hash.get(content_hash)
            if existing_content is not None:
                if existing_content != aggregate.candidate.content:
                    raise _retrieval_protocol_error("相同内容哈希对应了不同文本")
                continue
            content_by_hash[content_hash] = aggregate.candidate.content
            unique_content.append(aggregate)

        document_counts: dict[UUID, int] = {}
        selected: list[_RrfAggregate] = []
        for aggregate in unique_content:
            document_id = aggregate.candidate.document_id
            current_count = document_counts.get(document_id, 0)
            if current_count >= self._max_chunks_per_document:
                continue
            document_counts[document_id] = current_count + 1
            selected.append(aggregate)
            if len(selected) == self._rerank_candidate_limit:
                break

        return RetrievalFusionResult(
            index_configuration=first_configuration,
            candidates=tuple(
                replace(aggregate.candidate, rank=rank)
                for rank, aggregate in enumerate(selected, start=1)
            ),
            rrf_scores=tuple(aggregate.score for aggregate in selected),
            query_hit_counts=tuple(aggregate.query_hit_count for aggregate in selected),
            raw_candidate_count=raw_candidate_count,
            unique_chunk_count=len(aggregates),
            unique_content_count=len(unique_content),
        )

    async def rerank(
        self,
        request_id: UUID,
        query: str,
        candidates: tuple[VectorSearchCandidate, ...],
    ) -> CandidateRerankResult:
        """Rerank bounded candidates or preserve fusion order with an explicit reason."""
        if not candidates:
            raise ValueError("无候选内容时不能调用 Rerank")
        top_n = min(self._context_top_k, len(candidates))
        request = RerankRequest(
            query=query,
            documents=tuple(
                RerankDocument(str(candidate.chunk_id), candidate.content)
                for candidate in candidates
            ),
            top_n=top_n,
        )
        try:
            async with asyncio.timeout(self._rerank_timeout_seconds):
                result = await self._rerank_model.rerank(request)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return self._fallback(request_id, candidates, top_n, RerankDegradationReason.TIMEOUT)
        except ModelError as error:
            reason = (
                RerankDegradationReason.PROTOCOL
                if error.code is ModelErrorCode.PROTOCOL
                else RerankDegradationReason.PROVIDER_UNAVAILABLE
            )
            return self._fallback(
                request_id,
                candidates,
                top_n,
                reason,
                model_error_code=error.code.value,
            )

        try:
            reranked = _apply_rerank_result(candidates, request, result)
        except ValueError:
            return self._fallback(
                request_id,
                candidates,
                top_n,
                RerankDegradationReason.PROTOCOL,
            )
        if result.degradation_reason is not None:
            _log_degradation(
                request_id,
                result.model_id,
                result.degradation_reason,
            )
        return CandidateRerankResult(
            reranked,
            result.model_id,
            result.degradation_reason,
            result.total_tokens,
        )

    def _fallback(
        self,
        request_id: UUID,
        candidates: tuple[VectorSearchCandidate, ...],
        top_n: int,
        reason: RerankDegradationReason,
        *,
        model_error_code: str | None = None,
    ) -> CandidateRerankResult:
        _log_degradation(
            request_id,
            self._rerank_model.model_id,
            reason,
            model_error_code=model_error_code,
        )
        return CandidateRerankResult(
            _ranked(candidates[:top_n]),
            self._rerank_model.model_id,
            reason,
        )


def _apply_rerank_result(
    candidates: tuple[VectorSearchCandidate, ...],
    request: RerankRequest,
    result: RerankResult,
) -> tuple[VectorSearchCandidate, ...]:
    expected_count = request.result_limit
    if len(result.items) != expected_count:
        raise ValueError("Rerank 返回数量无效")
    indexes = tuple(item.original_index for item in result.items)
    if len(set(indexes)) != len(indexes):
        raise ValueError("Rerank 返回了重复候选")
    selected: list[VectorSearchCandidate] = []
    for item in result.items:
        if not 0 <= item.original_index < len(candidates):
            raise ValueError("Rerank original_index 越界")
        document = request.documents[item.original_index]
        if item.document_id != document.document_id:
            raise ValueError("Rerank 文档映射无效")
        selected.append(candidates[item.original_index])
    return _ranked(tuple(selected))


def _ranked(candidates: tuple[VectorSearchCandidate, ...]) -> tuple[VectorSearchCandidate, ...]:
    return tuple(
        replace(candidate, rank=rank) for rank, candidate in enumerate(candidates, start=1)
    )


def _aggregate_key(aggregate: _RrfAggregate) -> tuple[float, int, float, int, str]:
    return (
        -aggregate.score,
        aggregate.best_rank,
        -aggregate.best_similarity,
        aggregate.first_query_index,
        str(aggregate.candidate.chunk_id),
    )


def _candidate_metadata(candidate: VectorSearchCandidate) -> VectorSearchCandidate:
    return replace(
        candidate,
        rank=1,
        cosine_distance=0.0,
        similarity=1.0,
    )


def _occurrence_key(
    query_index: int,
    candidate: VectorSearchCandidate,
) -> tuple[int, float, int, str]:
    return (
        candidate.rank,
        -candidate.similarity,
        query_index,
        str(candidate.chunk_id),
    )


def _retrieval_protocol_error(message: str) -> RetrievalError:
    return RetrievalError(
        RetrievalErrorCode.RESULT_PROTOCOL,
        message,
        retryable=False,
    )


def _log_degradation(
    request_id: UUID,
    model_id: str,
    reason: RerankDegradationReason,
    *,
    model_error_code: str | None = None,
) -> None:
    logger.warning(
        "rerank_degraded",
        extra={
            "request_id": str(request_id),
            "model_id": model_id,
            "degradation_reason": reason.value,
            "model_error_code": model_error_code,
        },
    )
