"""Unit tests for the M5-A weighted RRF and Rerank baseline."""

import asyncio
import hashlib
import math
from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID, uuid4

import pytest

from customer_agent2.application import RetrievalPostProcessor
from customer_agent2.domain.models import (
    DocumentFormat,
    EmbeddingIndexConfiguration,
    ModelError,
    ModelErrorCode,
    RerankDegradationReason,
    RerankItem,
    RerankRequest,
    RerankResult,
    RetrievalError,
    RetrievalErrorCode,
    RetrievedChunkSource,
    VectorSearchCandidate,
    VectorSearchResult,
)
from customer_agent2.infrastructure.models import FakeRerankModel, NoOpRerankModel

INDEX = EmbeddingIndexConfiguration("embedding", "revision", 8, True)


class SlowRerankModel:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    @property
    def model_id(self) -> str:
        return "slow-rerank"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.started.set()
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError(f"超时前不应完成: {request}")


class MalformedRerankModel:
    @property
    def model_id(self) -> str:
        return "malformed-rerank"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        first = request.documents[0]
        return RerankResult(
            model_id=self.model_id,
            items=(
                RerankItem(0, first.document_id, 0.9),
                RerankItem(0, first.document_id, 0.8),
            ),
        )


def candidate(
    rank: int,
    label: str,
    *,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    content: str | None = None,
    content_sha256: str | None = None,
    similarity: float | None = None,
) -> VectorSearchCandidate:
    text = content or f"候选内容 {label}"
    score = similarity if similarity is not None else 1 - rank / 100
    return VectorSearchCandidate(
        rank=rank,
        chunk_id=chunk_id or uuid4(),
        knowledge_base_id=uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=uuid4(),
        source_key=f"manual/{label}.md",
        display_name=f"{label}.md",
        document_format=DocumentFormat.MARKDOWN,
        media_type="text/markdown",
        parser_name="markdown",
        parser_version="1",
        chunk_index=rank - 1,
        content=text,
        token_count=10,
        content_sha256=content_sha256 or hashlib.sha256(text.encode()).hexdigest(),
        section="测试",
        page_number=None,
        source=RetrievedChunkSource(
            block_start_ordinal=rank - 1,
            block_end_ordinal=rank - 1,
            start_line=rank,
            end_line=rank,
            section_path=("测试",),
            overlap_with_previous_tokens=0,
        ),
        cosine_distance=1 - score,
        similarity=score,
    )


def search_result(candidates: Sequence[VectorSearchCandidate]) -> VectorSearchResult:
    return VectorSearchResult(
        INDEX,
        tuple(replace(item, rank=rank) for rank, item in enumerate(candidates, start=1)),
    )


def postprocessor(
    model: FakeRerankModel | NoOpRerankModel | SlowRerankModel | MalformedRerankModel,
    *,
    candidate_limit: int = 40,
    top_k: int = 10,
    max_chunks_per_document: int = 2,
    timeout_seconds: float = 0.1,
) -> RetrievalPostProcessor:
    return RetrievalPostProcessor(
        model,
        rrf_k=60,
        rerank_candidate_limit=candidate_limit,
        context_top_k=top_k,
        max_chunks_per_document=max_chunks_per_document,
        rerank_timeout_seconds=timeout_seconds,
    )


def test_weighted_rrf_rewards_repeated_hits_and_keeps_stable_counts() -> None:
    repeated = candidate(1, "repeated")
    first_only = candidate(2, "first-only")
    second_only = candidate(1, "second-only")
    repeated_again = replace(
        repeated,
        rank=2,
        cosine_distance=0.2,
        similarity=0.8,
    )
    service = postprocessor(NoOpRerankModel())

    result = service.fuse(
        (
            search_result((repeated, first_only)),
            search_result((second_only, repeated_again)),
        )
    )

    assert [item.chunk_id for item in result.candidates] == [
        repeated.chunk_id,
        second_only.chunk_id,
        first_only.chunk_id,
    ]
    assert result.query_hit_counts == (2, 1, 1)
    assert result.raw_candidate_count == 4
    assert result.unique_chunk_count == 3
    assert result.unique_content_count == 3
    assert math.isclose(result.rrf_scores[0], 1 / 61 + 1 / 62)


def test_fusion_removes_duplicate_content_caps_each_document_and_truncates() -> None:
    first_document = uuid4()
    duplicate_text = "完全相同的退款说明"
    duplicate_hash = hashlib.sha256(duplicate_text.encode()).hexdigest()
    candidates = (
        candidate(1, "a", document_id=first_document),
        candidate(2, "b", document_id=first_document),
        candidate(3, "c", document_id=first_document),
        candidate(
            4,
            "duplicate-one",
            content=duplicate_text,
            content_sha256=duplicate_hash,
        ),
        candidate(
            5,
            "duplicate-two",
            content=duplicate_text,
            content_sha256=duplicate_hash,
        ),
        candidate(6, "tail"),
    )
    service = postprocessor(
        NoOpRerankModel(),
        candidate_limit=3,
        top_k=2,
        max_chunks_per_document=2,
    )

    result = service.fuse((search_result(candidates),))

    assert len(result.candidates) == 3
    assert [item.rank for item in result.candidates] == [1, 2, 3]
    assert sum(item.document_id == first_document for item in result.candidates) == 2
    assert result.raw_candidate_count == 6
    assert result.unique_chunk_count == 6
    assert result.unique_content_count == 5


def test_fusion_rejects_duplicate_chunk_metadata_or_hash_inconsistency() -> None:
    original = candidate(1, "original")
    changed_metadata = replace(original, source_key="different.md")
    service = postprocessor(NoOpRerankModel())

    with pytest.raises(RetrievalError) as duplicate_error:
        service.fuse((search_result((original, original)),))
    assert duplicate_error.value.code is RetrievalErrorCode.RESULT_PROTOCOL

    with pytest.raises(RetrievalError) as metadata_error:
        service.fuse(
            (
                search_result((original,)),
                search_result((changed_metadata,)),
            )
        )
    assert metadata_error.value.code is RetrievalErrorCode.RESULT_PROTOCOL

    conflicting_hash = candidate(
        1,
        "conflict",
        content="另一段内容",
        content_sha256=original.content_sha256,
    )
    with pytest.raises(RetrievalError) as hash_error:
        service.fuse((search_result((original, conflicting_hash)),))
    assert hash_error.value.code is RetrievalErrorCode.RESULT_PROTOCOL


@pytest.mark.asyncio
async def test_rerank_applies_provider_order_and_top_k_mapping() -> None:
    candidates = tuple(candidate(index, str(index)) for index in range(1, 4))
    model = FakeRerankModel(scores=(0.2, 0.9, 0.5))
    service = postprocessor(model, candidate_limit=3, top_k=2)

    result = await service.rerank(uuid4(), "退款问题", candidates)

    assert [item.chunk_id for item in result.candidates] == [
        candidates[1].chunk_id,
        candidates[2].chunk_id,
    ]
    assert [item.rank for item in result.candidates] == [1, 2]
    assert result.model_id == "fake-rerank"
    assert result.degradation_reason is None
    assert model.requests[0].top_n == 2
    assert [document.document_id for document in model.requests[0].documents] == [
        str(candidate.chunk_id) for candidate in candidates
    ]


@pytest.mark.asyncio
async def test_noop_and_known_provider_failure_preserve_fusion_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidates = tuple(candidate(index, str(index)) for index in range(1, 4))
    disabled = await postprocessor(
        NoOpRerankModel(),
        candidate_limit=3,
        top_k=2,
    ).rerank(uuid4(), "退款问题", candidates)
    provider_error = ModelError(ModelErrorCode.UNAVAILABLE, "暂时不可用", retryable=True)
    unavailable = await postprocessor(
        FakeRerankModel(error=provider_error),
        candidate_limit=3,
        top_k=2,
    ).rerank(uuid4(), "退款问题", candidates)

    expected = [candidates[0].chunk_id, candidates[1].chunk_id]
    assert [item.chunk_id for item in disabled.candidates] == expected
    assert disabled.degradation_reason is RerankDegradationReason.DISABLED
    assert [item.chunk_id for item in unavailable.candidates] == expected
    assert unavailable.degradation_reason is RerankDegradationReason.PROVIDER_UNAVAILABLE
    assert "退款问题" not in caplog.text
    assert "暂时不可用" not in caplog.text


@pytest.mark.asyncio
async def test_rerank_timeout_and_malformed_result_are_protocol_safe_fallbacks() -> None:
    candidates = tuple(candidate(index, str(index)) for index in range(1, 4))
    slow_model = SlowRerankModel()
    timed_out = await postprocessor(
        slow_model,
        candidate_limit=3,
        top_k=2,
        timeout_seconds=0.01,
    ).rerank(uuid4(), "退款问题", candidates)
    malformed = await postprocessor(
        MalformedRerankModel(),
        candidate_limit=3,
        top_k=2,
    ).rerank(uuid4(), "退款问题", candidates)

    assert timed_out.degradation_reason is RerankDegradationReason.TIMEOUT
    assert slow_model.cancelled is True
    assert malformed.degradation_reason is RerankDegradationReason.PROTOCOL
    assert [item.chunk_id for item in timed_out.candidates] == [
        candidates[0].chunk_id,
        candidates[1].chunk_id,
    ]
    assert [item.chunk_id for item in malformed.candidates] == [
        candidates[0].chunk_id,
        candidates[1].chunk_id,
    ]


@pytest.mark.asyncio
async def test_rerank_cancellation_propagates_and_cancels_model_call() -> None:
    candidates = (candidate(1, "one"),)
    slow_model = SlowRerankModel()
    task = asyncio.create_task(
        postprocessor(slow_model, timeout_seconds=1).rerank(
            uuid4(),
            "退款问题",
            candidates,
        )
    )
    await slow_model.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert slow_model.cancelled is True


def test_postprocessor_rejects_invalid_funnel_and_weights() -> None:
    with pytest.raises(ValueError, match="候选上限"):
        postprocessor(NoOpRerankModel(), candidate_limit=1, top_k=2)

    service = postprocessor(NoOpRerankModel())
    with pytest.raises(ValueError, match="query_weights"):
        service.fuse((search_result((candidate(1, "a"),)),), query_weights=())
    with pytest.raises(ValueError, match="query_weights"):
        service.fuse((search_result((candidate(1, "a"),)),), query_weights=(0.0,))
