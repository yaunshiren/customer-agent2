"""Concrete and controllable model adapters."""

from customer_agent2.infrastructure.models.fakes import (
    FakeChatModel,
    FakeEmbeddingModel,
    FakeRerankModel,
)
from customer_agent2.infrastructure.models.noop_rerank import NoOpRerankModel

__all__ = ["FakeChatModel", "FakeEmbeddingModel", "FakeRerankModel", "NoOpRerankModel"]
