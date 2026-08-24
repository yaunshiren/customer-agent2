"""Provider-neutral model contract tests."""

import math

import pytest

from customer_agent2.application import ChatProfile, ModelGateway
from customer_agent2.domain.models import (
    ChatMessage,
    ChatRequest,
    ChatRole,
    EmbeddingRequest,
    EmbeddingResult,
    ModelError,
    ModelErrorCode,
    RerankDocument,
    RerankRequest,
)
from customer_agent2.infrastructure.models import (
    FakeChatModel,
    FakeEmbeddingModel,
    NoOpRerankModel,
)


def test_chat_and_embedding_requests_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="messages"):
        ChatRequest(messages=())

    with pytest.raises(ValueError, match="空文本"):
        EmbeddingRequest(texts=("有效文本", "  "))


def test_embedding_result_rejects_wrong_dimension_and_non_finite_values() -> None:
    with pytest.raises(ValueError, match="维度"):
        EmbeddingResult(
            model_id="embedding",
            vectors=((0.1, 0.2),),
            dimension=3,
            normalized=False,
        )

    with pytest.raises(ValueError, match="NaN"):
        EmbeddingResult(
            model_id="embedding",
            vectors=((math.nan,),),
            dimension=1,
            normalized=False,
        )


def test_rerank_request_validates_top_n_against_candidates() -> None:
    document = RerankDocument(document_id="doc-1", text="候选内容")

    with pytest.raises(ValueError, match="top_n"):
        RerankRequest(query="问题", documents=(document,), top_n=2)


def test_model_error_exposes_stable_category_and_safe_message() -> None:
    error = ModelError(
        ModelErrorCode.QUOTA_EXHAUSTED,
        "模型额度不足",
        retryable=False,
    )

    assert error.code is ModelErrorCode.QUOTA_EXHAUSTED
    assert error.public_message == "模型额度不足"
    assert str(error) == "模型额度不足"
    assert error.retryable is False


def test_gateway_selects_final_and_fast_chat_models_explicitly() -> None:
    final_chat = FakeChatModel("final-model", "最终回答")
    fast_chat = FakeChatModel("fast-model", "快速回答")
    gateway = ModelGateway(
        final_chat=final_chat,
        fast_chat=fast_chat,
        embedding=FakeEmbeddingModel(),
        rerank=NoOpRerankModel(),
    )

    assert gateway.chat(ChatProfile.FINAL) is final_chat
    assert gateway.chat(ChatProfile.FAST) is fast_chat


def test_chat_message_rejects_blank_content() -> None:
    with pytest.raises(ValueError, match="content"):
        ChatMessage(role=ChatRole.USER, content=" ")
