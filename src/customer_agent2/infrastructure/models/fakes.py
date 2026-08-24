"""Deterministic model adapters for tests and offline smoke checks."""

import hashlib
import math
from collections.abc import AsyncIterator, Sequence

from customer_agent2.domain.models import (
    ChatRequest,
    ChatResult,
    ChatStreamChunk,
    EmbeddingRequest,
    EmbeddingResult,
    ModelError,
    ModelErrorCode,
    RerankItem,
    RerankRequest,
    RerankResult,
    TokenUsage,
)


class FakeChatModel:
    """Return configured complete and streaming chat results while recording requests."""

    def __init__(
        self,
        model_id: str,
        response: str,
        *,
        stream_chunks: Sequence[str] | None = None,
        reasoning_content: str | None = None,
        usage: TokenUsage | None = None,
        error: ModelError | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id 不能为空")
        self._model_id = model_id
        self._response = response
        self._stream_chunks = tuple(stream_chunks) if stream_chunks is not None else (response,)
        self._reasoning_content = reasoning_content
        self._usage = usage
        self._error = error
        self._completion_requests: list[ChatRequest] = []
        self._stream_requests: list[ChatRequest] = []

    @property
    def model_id(self) -> str:
        """Return the configured fake model identifier."""
        return self._model_id

    @property
    def completion_requests(self) -> tuple[ChatRequest, ...]:
        """Return immutable request history for assertions."""
        return tuple(self._completion_requests)

    @property
    def stream_requests(self) -> tuple[ChatRequest, ...]:
        """Return immutable streaming request history for assertions."""
        return tuple(self._stream_requests)

    async def complete(self, request: ChatRequest) -> ChatResult:
        """Return a configured complete response or failure."""
        self._completion_requests.append(request)
        error = self._error
        if error is not None:
            raise error
        return ChatResult(
            model_id=self.model_id,
            content=self._response,
            reasoning_content=self._reasoning_content,
            finish_reason="stop",
            usage=self._usage,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatStreamChunk]:
        """Yield configured chunks followed by one explicit completion event."""
        self._stream_requests.append(request)
        error = self._error
        if error is not None:
            raise error
        if self._reasoning_content:
            yield ChatStreamChunk(reasoning_delta=self._reasoning_content)
        for chunk in self._stream_chunks:
            if chunk:
                yield ChatStreamChunk(delta=chunk)
        yield ChatStreamChunk(finish_reason="stop", usage=self._usage)


class FakeEmbeddingModel:
    """Generate stable finite vectors without downloading model weights."""

    def __init__(
        self,
        model_id: str = "fake-embedding",
        *,
        revision: str = "fake-revision",
        dimension: int = 8,
        max_tokens: int = 512,
        normalized: bool = True,
        error: ModelError | None = None,
    ) -> None:
        if not model_id.strip() or not revision.strip():
            raise ValueError("model_id 和 revision 不能为空")
        if dimension < 1 or max_tokens < 1:
            raise ValueError("dimension 和 max_tokens 必须大于 0")
        self._model_id = model_id
        self._revision = revision
        self._dimension = dimension
        self._max_tokens = max_tokens
        self._normalized = normalized
        self._error = error
        self._requests: list[EmbeddingRequest] = []

    @property
    def model_id(self) -> str:
        """Return the configured fake model identifier."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the configured vector dimension."""
        return self._dimension

    @property
    def revision(self) -> str:
        """Return the configured fake model revision."""
        return self._revision

    @property
    def max_tokens(self) -> int:
        """Return the configured input token limit."""
        return self._max_tokens

    @property
    def normalized(self) -> bool:
        """Return whether generated vectors use unit L2 norm."""
        return self._normalized

    @property
    def requests(self) -> tuple[EmbeddingRequest, ...]:
        """Return immutable request history for assertions."""
        return tuple(self._requests)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Generate deterministic vectors or raise a configured model failure."""
        self._requests.append(request)
        error = self._error
        if error is not None:
            raise error
        vectors = tuple(self._vector_for(text) for text in request.texts)
        return EmbeddingResult(
            model_id=self.model_id,
            model_revision=self.revision,
            vectors=vectors,
            dimension=self.dimension,
            normalized=self.normalized,
        )

    def _vector_for(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = tuple(
            (digest[index % len(digest)] - 127.5) / 127.5 for index in range(self.dimension)
        )
        if not self.normalized:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)


class FakeRerankModel:
    """Sort candidates by configured scores while recording requests."""

    def __init__(
        self,
        model_id: str = "fake-rerank",
        *,
        scores: Sequence[float] | None = None,
        error: ModelError | None = None,
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id 不能为空")
        self._model_id = model_id
        self._scores = tuple(scores) if scores is not None else None
        self._error = error
        self._requests: list[RerankRequest] = []

    @property
    def model_id(self) -> str:
        """Return the configured fake model identifier."""
        return self._model_id

    @property
    def requests(self) -> tuple[RerankRequest, ...]:
        """Return immutable request history for assertions."""
        return tuple(self._requests)

    async def rerank(self, request: RerankRequest) -> RerankResult:
        """Return a stable score ordering or raise a configured model failure."""
        self._requests.append(request)
        error = self._error
        if error is not None:
            raise error

        scores = self._scores
        if scores is None:
            scores = tuple(
                float(len(request.documents) - index) for index in range(len(request.documents))
            )
        if len(scores) != len(request.documents):
            raise ModelError(
                ModelErrorCode.PROTOCOL,
                "Fake Rerank 分数数量与候选数量不一致",
                retryable=False,
            )

        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        items = tuple(
            RerankItem(
                original_index=index,
                document_id=request.documents[index].document_id,
                score=score,
            )
            for index, score in ranked[: request.result_limit]
        )
        return RerankResult(model_id=self.model_id, items=items)
