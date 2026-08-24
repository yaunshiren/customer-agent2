"""Provider-neutral embedding model contracts."""

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """A non-empty batch of texts to embed together."""

    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("EmbeddingRequest.texts 不能为空")
        if any(not text.strip() for text in self.texts):
            raise ValueError("EmbeddingRequest.texts 不能包含空文本")


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Validated embedding batch with explicit model capabilities."""

    model_id: str
    vectors: tuple[tuple[float, ...], ...]
    dimension: int
    normalized: bool

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("EmbeddingResult.model_id 不能为空")
        if self.dimension < 1:
            raise ValueError("EmbeddingResult.dimension 必须大于 0")
        if not self.vectors:
            raise ValueError("EmbeddingResult.vectors 不能为空")
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("EmbeddingResult 向量维度不一致")
        if any(not math.isfinite(value) for vector in self.vectors for value in vector):
            raise ValueError("EmbeddingResult 不能包含 NaN 或无限值")


class EmbeddingModel(Protocol):
    """Embedding capability required by ingestion and retrieval use cases."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int: ...

    @property
    def normalized(self) -> bool: ...

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult: ...
