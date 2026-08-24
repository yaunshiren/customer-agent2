"""Deterministic fake and No-op model adapter tests."""

import math

import pytest

from customer_agent2.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatRole,
    EmbeddingRequest,
    ModelError,
    ModelErrorCode,
    RerankDegradationReason,
    RerankDocument,
    RerankRequest,
    TokenUsage,
)
from customer_agent2.infrastructure.models import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeRerankModel,
    NoOpRerankModel,
)


def make_chat_request() -> ChatRequest:
    """Build one shared typed chat request."""
    return ChatRequest(messages=(ChatMessage(ChatRole.USER, "你好"),))


def make_rerank_request(*, top_n: int | None = None) -> RerankRequest:
    """Build ordered rerank candidates."""
    return RerankRequest(
        query="退款条件",
        documents=(
            RerankDocument("doc-a", "候选 A"),
            RerankDocument("doc-b", "候选 B"),
            RerankDocument("doc-c", "候选 C"),
        ),
        top_n=top_n,
    )


@pytest.mark.asyncio
async def test_fake_chat_supports_complete_and_streaming_results() -> None:
    request = make_chat_request()
    usage = TokenUsage(input_tokens=3, output_tokens=4)
    model = FakeChatModel(
        "chat-test",
        "你好, 世界",
        stream_chunks=("你好", ", 世界"),
        reasoning_content="简短推理",
        usage=usage,
    )

    result = await model.complete(request)
    chunks = [chunk async for chunk in model.stream(request)]

    assert result.content == "你好, 世界"
    assert result.reasoning_content == "简短推理"
    assert result.usage is usage
    assert "".join(chunk.delta for chunk in chunks) == result.content
    assert "".join(chunk.reasoning_delta for chunk in chunks) == "简短推理"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is usage
    assert model.completion_requests == (request,)
    assert model.stream_requests == (request,)


@pytest.mark.asyncio
async def test_fake_model_propagates_structured_failure() -> None:
    failure = ModelError(ModelErrorCode.TIMEOUT, "模型调用超时", retryable=True)
    model = FakeChatModel("chat-test", "", error=failure)

    with pytest.raises(ModelError) as captured:
        await model.complete(make_chat_request())

    assert captured.value.code is ModelErrorCode.TIMEOUT
    assert captured.value.retryable is True

    with pytest.raises(ModelError) as stream_captured:
        _ = [chunk async for chunk in model.stream(make_chat_request())]

    assert stream_captured.value.code is ModelErrorCode.TIMEOUT


@pytest.mark.asyncio
async def test_fake_embedding_is_deterministic_finite_and_normalized() -> None:
    request = EmbeddingRequest(texts=("同一文本", "另一文本"))
    model = FakeEmbeddingModel(dimension=8, normalized=True)

    first = await model.embed(request)
    second = await model.embed(request)

    assert first == second
    assert first.model_revision == model.revision
    assert len(first.vectors) == 2
    assert all(len(vector) == 8 for vector in first.vectors)
    assert all(math.isfinite(value) for vector in first.vectors for value in vector)
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0)
        for vector in first.vectors
    )
    assert model.requests == (request, request)


@pytest.mark.asyncio
async def test_fake_rerank_sorts_scores_stably_and_applies_top_n() -> None:
    model = FakeRerankModel(scores=(0.2, 0.9, 0.9))

    result = await model.rerank(make_rerank_request(top_n=2))

    assert [item.document_id for item in result.items] == ["doc-b", "doc-c"]
    assert [item.original_index for item in result.items] == [1, 2]
    assert result.degraded is False


@pytest.mark.asyncio
async def test_fake_rerank_reports_score_protocol_mismatch() -> None:
    model = FakeRerankModel(scores=(0.1,))

    with pytest.raises(ModelError) as captured:
        await model.rerank(make_rerank_request())

    assert captured.value.code is ModelErrorCode.PROTOCOL


@pytest.mark.asyncio
async def test_noop_rerank_preserves_order_and_marks_degradation() -> None:
    result = await NoOpRerankModel().rerank(make_rerank_request(top_n=2))

    assert [item.document_id for item in result.items] == ["doc-a", "doc-b"]
    assert all(item.score is None for item in result.items)
    assert result.degraded is True
    assert result.degradation_reason is RerankDegradationReason.DISABLED
