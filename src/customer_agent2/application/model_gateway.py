"""Typed access to final, fast, embedding, and rerank model capabilities."""

from dataclasses import dataclass
from enum import StrEnum

from customer_agent2.domain.models import ChatModel, EmbeddingModel, RerankModel


class ChatProfile(StrEnum):
    """Application-level reason for choosing a chat model."""

    FINAL = "final"
    FAST = "fast"


@dataclass(frozen=True, slots=True)
class ModelGateway:
    """Model capability graph injected into future application pipelines."""

    final_chat: ChatModel
    fast_chat: ChatModel
    embedding: EmbeddingModel
    rerank: RerankModel

    def chat(self, profile: ChatProfile) -> ChatModel:
        """Select final-answer or internal-task chat capability explicitly."""
        if profile is ChatProfile.FINAL:
            return self.final_chat
        return self.fast_chat
