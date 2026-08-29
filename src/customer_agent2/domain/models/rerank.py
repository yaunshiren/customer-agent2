"""Provider-neutral rerank model contracts."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class RerankDegradationReason(StrEnum):
    """Observable reasons for preserving retrieval order without reranking."""

    DISABLED = "disabled"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROTOCOL = "protocol"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class RerankDocument:
    """Minimal candidate content required by a rerank provider."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("RerankDocument.document_id 不能为空")
        if not self.text.strip():
            raise ValueError("RerankDocument.text 不能为空")


@dataclass(frozen=True, slots=True)
class RerankRequest:
    """A query, ordered candidates, and optional output limit."""

    query: str
    documents: tuple[RerankDocument, ...]
    top_n: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("RerankRequest.query 不能为空")
        if not self.documents:
            raise ValueError("RerankRequest.documents 不能为空")
        if self.top_n is not None and not 1 <= self.top_n <= len(self.documents):
            raise ValueError("RerankRequest.top_n 必须在候选数量范围内")

    @property
    def result_limit(self) -> int:
        """Return the requested result count or the complete candidate count."""
        return self.top_n if self.top_n is not None else len(self.documents)


@dataclass(frozen=True, slots=True)
class RerankItem:
    """One candidate's new position and optional relevance score."""

    original_index: int
    document_id: str
    score: float | None

    def __post_init__(self) -> None:
        if self.original_index < 0:
            raise ValueError("RerankItem.original_index 不能为负数")
        if not self.document_id.strip():
            raise ValueError("RerankItem.document_id 不能为空")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("RerankItem.score 必须是有限数值")


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Ordered rerank output with an explicit degradation signal."""

    model_id: str
    items: tuple[RerankItem, ...]
    degraded: bool = False
    degradation_reason: RerankDegradationReason | None = None

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("RerankResult.model_id 不能为空")
        if not self.items:
            raise ValueError("RerankResult.items 不能为空")
        if self.degraded != (self.degradation_reason is not None):
            raise ValueError("RerankResult 降级状态和原因必须一致")


class RerankModel(Protocol):
    """Rerank capability required by retrieval post-processing."""

    @property
    def model_id(self) -> str: ...

    async def rerank(self, request: RerankRequest) -> RerankResult: ...
