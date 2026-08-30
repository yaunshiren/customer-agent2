"""Concrete and controllable model adapters."""

from customer_agent2.infrastructure.models.dashscope_rerank import (
    DEFAULT_RERANK_INSTRUCTION,
    DashScopeRerankModel,
)
from customer_agent2.infrastructure.models.fakes import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeRerankModel,
)
from customer_agent2.infrastructure.models.noop_rerank import NoOpRerankModel
from customer_agent2.infrastructure.models.openai_chat import OpenAICompatibleChatModel
from customer_agent2.infrastructure.models.sentence_transformer_embedding import (
    SentenceTransformerEmbeddingModel,
)

__all__ = [
    "DEFAULT_RERANK_INSTRUCTION",
    "DashScopeRerankModel",
    "FakeChatModel",
    "FakeEmbeddingModel",
    "FakeRerankModel",
    "NoOpRerankModel",
    "OpenAICompatibleChatModel",
    "SentenceTransformerEmbeddingModel",
]
