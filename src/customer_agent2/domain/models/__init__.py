"""Public model contracts used by application and infrastructure layers."""

from customer_agent2.domain.models.chat import (
    ChatMessage,
    ChatModel,
    ChatRequest,
    ChatResult,
    ChatRole,
    ChatStreamChunk,
    TokenUsage,
)
from customer_agent2.domain.models.embedding import (
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
)
from customer_agent2.domain.models.errors import ModelError, ModelErrorCode
from customer_agent2.domain.models.rerank import (
    RerankDegradationReason,
    RerankDocument,
    RerankItem,
    RerankModel,
    RerankRequest,
    RerankResult,
)

__all__ = [
    "ChatMessage",
    "ChatModel",
    "ChatRequest",
    "ChatResult",
    "ChatRole",
    "ChatStreamChunk",
    "EmbeddingModel",
    "EmbeddingRequest",
    "EmbeddingResult",
    "ModelError",
    "ModelErrorCode",
    "RerankDegradationReason",
    "RerankDocument",
    "RerankItem",
    "RerankModel",
    "RerankRequest",
    "RerankResult",
    "TokenUsage",
]
