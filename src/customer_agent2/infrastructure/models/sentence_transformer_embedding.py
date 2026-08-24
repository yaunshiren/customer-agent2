"""Asynchronous adapter for local Sentence Transformers embeddings."""

import asyncio
import math
from collections.abc import Callable, Iterable
from typing import Protocol, SupportsFloat, cast

from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    EmbeddingRequest,
    EmbeddingResult,
    ModelError,
    ModelErrorCode,
)


class SentenceTransformerBackend(Protocol):
    """Narrow subset of SentenceTransformer used by this adapter."""

    max_seq_length: int

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        device: str,
    ) -> object: ...


BackendFactory = Callable[[str, str], SentenceTransformerBackend]


class SentenceTransformerEmbeddingModel:
    """Run local embedding inference without blocking the async event loop."""

    def __init__(
        self,
        *,
        model_id: str,
        dimension: int,
        max_tokens: int,
        device: str,
        batch_size: int,
        normalized: bool,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        normalized_model_id = model_id.strip()
        normalized_device = device.strip()
        if not normalized_model_id or not normalized_device:
            raise _configuration_error("本地 Embedding 模型或设备未配置")
        if dimension < 1 or max_tokens < 1 or batch_size < 1:
            raise _configuration_error("本地 Embedding 数值配置无效")

        self._model_id = normalized_model_id
        self._dimension = dimension
        self._max_tokens = max_tokens
        self._device = normalized_device
        self._batch_size = batch_size
        self._normalized = normalized
        self._backend_factory = backend_factory or _load_sentence_transformer
        self._backend: SentenceTransformerBackend | None = None
        self._inference_lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        backend_factory: BackendFactory | None = None,
    ) -> "SentenceTransformerEmbeddingModel":
        """Build the adapter entirely from validated application settings."""
        return cls(
            model_id=settings.local_embedding_model,
            dimension=settings.local_embedding_dimension,
            max_tokens=settings.local_embedding_max_tokens,
            device=settings.local_embedding_device,
            batch_size=settings.local_embedding_batch_size,
            normalized=settings.embedding_normalize,
            backend_factory=backend_factory,
        )

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model ID."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the vector dimension required by the index."""
        return self._dimension

    @property
    def max_tokens(self) -> int:
        """Return the enforced tokenizer sequence limit."""
        return self._max_tokens

    @property
    def normalized(self) -> bool:
        """Report whether output vectors use L2 normalization."""
        return self._normalized

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Embed one batch on a worker thread and serialize access to the model."""
        async with self._inference_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._embed_sync, request))
            try:
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                # The Python task cannot interrupt an in-flight native CPU kernel. Keep
                # ownership until it exits so another request cannot use the same model.
                await asyncio.gather(worker, return_exceptions=True)
                raise

    def _embed_sync(self, request: EmbeddingRequest) -> EmbeddingResult:
        backend = self._get_or_load_backend()
        try:
            encoded = backend.encode(
                list(request.texts),
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self._normalized,
                device=self._device,
            )
        except Exception:
            raise ModelError(
                ModelErrorCode.UNAVAILABLE,
                "本地 Embedding 推理失败",
                retryable=True,
            ) from None

        try:
            vectors = _coerce_vectors(encoded)
            if len(vectors) != len(request.texts):
                raise ValueError("batch shape mismatch")
            result = EmbeddingResult(
                model_id=self._model_id,
                vectors=vectors,
                dimension=self._dimension,
                normalized=self._normalized,
            )
            if self._normalized and any(
                not math.isclose(
                    math.sqrt(math.fsum(item * item for item in vector)),
                    1.0,
                    rel_tol=1e-3,
                    abs_tol=1e-3,
                )
                for vector in result.vectors
            ):
                raise ValueError("embedding vector is not normalized")
            return result
        except (TypeError, ValueError):
            raise ModelError(
                ModelErrorCode.PROTOCOL,
                "本地 Embedding 返回了无效向量",
                retryable=False,
            ) from None

    def _get_or_load_backend(self) -> SentenceTransformerBackend:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._backend_factory(self._model_id, self._device)
        except Exception:
            raise ModelError(
                ModelErrorCode.UNAVAILABLE,
                "本地 Embedding 模型无法加载",
                retryable=True,
            ) from None

        reported_max_tokens = backend.max_seq_length
        if reported_max_tokens < self._max_tokens:
            raise _configuration_error("本地 Embedding 模型不支持配置的最大 Token 数")
        backend.max_seq_length = self._max_tokens
        self._backend = backend
        return backend


def _load_sentence_transformer(model_id: str, device: str) -> SentenceTransformerBackend:
    from sentence_transformers import SentenceTransformer

    backend = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=False,
    )
    return cast(SentenceTransformerBackend, backend)


def _coerce_vectors(value: object) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("embedding output is not a matrix")
    rows = iter(cast(Iterable[object], value))
    vectors: list[tuple[float, ...]] = []
    for row in rows:
        if isinstance(row, (str, bytes)):
            raise TypeError("embedding row is not a vector")
        values = iter(cast(Iterable[object], row))
        vectors.append(tuple(float(cast(SupportsFloat | str, item)) for item in values))
    return tuple(vectors)


def _configuration_error(message: str) -> ModelError:
    return ModelError(ModelErrorCode.CONFIGURATION, message, retryable=False)
