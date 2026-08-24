"""Tests for the local Sentence Transformers embedding adapter."""

import asyncio
import math
import threading
import time
from collections.abc import Callable

import pytest

from customer_agent2.domain.models import EmbeddingRequest, ModelError, ModelErrorCode
from customer_agent2.infrastructure.models import SentenceTransformerEmbeddingModel
from tests.settings import IsolatedSettings


class FakeBackend:
    """Controllable in-memory substitute for SentenceTransformer."""

    def __init__(
        self,
        output: object,
        *,
        max_seq_length: int = 512,
        error: Exception | None = None,
    ) -> None:
        self.output = output
        self.max_seq_length = max_seq_length
        self.error = error
        self.calls: list[dict[str, object]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        device: str,
    ) -> object:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
                "device": device,
            }
        )
        if self.error is not None:
            raise self.error
        return self.output


def _model(
    backend: FakeBackend,
    *,
    normalized: bool = True,
    dimension: int = 3,
    max_tokens: int = 8,
    batch_size: int = 2,
) -> SentenceTransformerEmbeddingModel:
    return SentenceTransformerEmbeddingModel(
        model_id="test-embedding-model",
        revision="test-revision",
        dimension=dimension,
        max_tokens=max_tokens,
        device="cpu",
        batch_size=batch_size,
        normalized=normalized,
        backend_factory=lambda _model_id, _revision, _device: backend,
    )


@pytest.mark.asyncio
async def test_embed_translates_batch_settings_and_validates_result() -> None:
    backend = FakeBackend([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    model = _model(backend)

    result = await model.embed(EmbeddingRequest(texts=("第一段", "第二段")))

    assert model.model_id == "test-embedding-model"
    assert model.revision == "test-revision"
    assert model.dimension == 3
    assert model.max_tokens == 8
    assert model.normalized is True
    assert result.model_id == model.model_id
    assert result.model_revision == model.revision
    assert result.dimension == 3
    assert result.normalized is True
    assert result.vectors == ((1.0, 0.0, 0.0), (0.0, 0.6, 0.8))
    assert all(
        math.isclose(math.sqrt(sum(item * item for item in vector)), 1.0)
        for vector in result.vectors
    )
    assert backend.max_seq_length == 8
    assert backend.calls == [
        {
            "sentences": ["第一段", "第二段"],
            "batch_size": 2,
            "show_progress_bar": False,
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "device": "cpu",
        }
    ]


@pytest.mark.asyncio
async def test_from_settings_uses_all_local_embedding_configuration() -> None:
    backend = FakeBackend([[0.0, 2.0, 0.0]], max_seq_length=32)
    settings = IsolatedSettings(
        local_embedding_model="custom-model",
        local_embedding_revision="custom-revision",
        local_embedding_dimension=3,
        local_embedding_max_tokens=16,
        local_embedding_device="cpu",
        local_embedding_batch_size=7,
        embedding_normalize=False,
    )
    model = SentenceTransformerEmbeddingModel.from_settings(
        settings,
        backend_factory=lambda _model_id, _revision, _device: backend,
    )

    result = await model.embed(EmbeddingRequest(texts=("测试",)))

    assert model.model_id == "custom-model"
    assert model.revision == "custom-revision"
    assert model.dimension == 3
    assert model.max_tokens == 16
    assert model.normalized is False
    assert result.vectors == ((0.0, 2.0, 0.0),)
    assert backend.max_seq_length == 16
    assert backend.calls[0]["batch_size"] == 7
    assert backend.calls[0]["normalize_embeddings"] is False


@pytest.mark.parametrize(
    "output",
    [
        [[1.0, 0.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[math.nan, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ],
)
@pytest.mark.asyncio
async def test_embed_rejects_invalid_shape_values_and_normalization(output: object) -> None:
    model = _model(FakeBackend(output))

    with pytest.raises(ModelError) as caught:
        await model.embed(EmbeddingRequest(texts=("第一段", "第二段")))

    assert caught.value.code is ModelErrorCode.PROTOCOL
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_embed_maps_backend_failure_without_leaking_details() -> None:
    sensitive_detail = "C:/private/model/cache/internal-detail"
    model = _model(
        FakeBackend(
            [],
            error=RuntimeError(sensitive_detail),
        )
    )

    with pytest.raises(ModelError) as caught:
        await model.embed(EmbeddingRequest(texts=("测试",)))

    assert caught.value.code is ModelErrorCode.UNAVAILABLE
    assert caught.value.retryable is True
    assert sensitive_detail not in str(caught.value)


@pytest.mark.asyncio
async def test_embed_maps_model_load_failure_without_leaking_details() -> None:
    sensitive_detail = "C:/private/model/cache/load-detail"

    def failing_factory(model_id: str, revision: str, device: str) -> FakeBackend:
        raise OSError(f"{sensitive_detail}: {model_id}@{revision} on {device}")

    model = SentenceTransformerEmbeddingModel(
        model_id="test-embedding-model",
        revision="test-revision",
        dimension=3,
        max_tokens=8,
        device="cpu",
        batch_size=2,
        normalized=True,
        backend_factory=failing_factory,
    )

    with pytest.raises(ModelError) as caught:
        await model.embed(EmbeddingRequest(texts=("测试",)))

    assert caught.value.code is ModelErrorCode.UNAVAILABLE
    assert caught.value.retryable is True
    assert sensitive_detail not in str(caught.value)


@pytest.mark.asyncio
async def test_embed_rejects_model_with_insufficient_token_capacity() -> None:
    model = _model(FakeBackend([[1.0, 0.0, 0.0]], max_seq_length=4), max_tokens=8)

    with pytest.raises(ModelError) as caught:
        await model.embed(EmbeddingRequest(texts=("测试",)))

    assert caught.value.code is ModelErrorCode.CONFIGURATION
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_model_is_loaded_lazily_and_only_once() -> None:
    backend = FakeBackend([[1.0, 0.0, 0.0]])
    load_calls = 0

    def factory(model_id: str, revision: str, device: str) -> FakeBackend:
        nonlocal load_calls
        load_calls += 1
        assert model_id == "test-embedding-model"
        assert revision == "test-revision"
        assert device == "cpu"
        return backend

    model = SentenceTransformerEmbeddingModel(
        model_id="test-embedding-model",
        revision="test-revision",
        dimension=3,
        max_tokens=8,
        device="cpu",
        batch_size=2,
        normalized=True,
        backend_factory=factory,
    )
    assert load_calls == 0

    await model.embed(EmbeddingRequest(texts=("第一次",)))
    await model.embed(EmbeddingRequest(texts=("第二次",)))

    assert load_calls == 1
    assert len(backend.calls) == 2


class BlockingBackend(FakeBackend):
    """Backend that exposes thread and concurrency behavior."""

    def __init__(self, blocker: Callable[[], None]) -> None:
        super().__init__([[1.0, 0.0, 0.0]])
        self._blocker = blocker
        self._state_lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.thread_ids: list[int] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        device: str,
    ) -> object:
        with self._state_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.thread_ids.append(threading.get_ident())
        try:
            self._blocker()
            return super().encode(
                sentences,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
                normalize_embeddings=normalize_embeddings,
                device=device,
            )
        finally:
            with self._state_lock:
                self.active -= 1


@pytest.mark.asyncio
async def test_embed_runs_off_event_loop_and_serializes_model_access() -> None:
    backend = BlockingBackend(lambda: time.sleep(0.03))
    model = _model(backend)
    event_loop_thread_id = threading.get_ident()

    first, second = await asyncio.gather(
        model.embed(EmbeddingRequest(texts=("第一批",))),
        model.embed(EmbeddingRequest(texts=("第二批",))),
    )

    assert first.dimension == second.dimension == 3
    assert backend.max_active == 1
    assert backend.thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in backend.thread_ids)


@pytest.mark.asyncio
async def test_cancellation_waits_for_native_worker_before_releasing_model() -> None:
    started = threading.Event()
    release = threading.Event()

    def block() -> None:
        started.set()
        assert release.wait(timeout=1.0)

    backend = BlockingBackend(block)
    model = _model(backend)
    task = asyncio.create_task(model.embed(EmbeddingRequest(texts=("会被取消",))))
    assert await asyncio.to_thread(started.wait, 1.0)

    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    assert backend.active == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert backend.active == 0


def test_constructor_rejects_invalid_runtime_configuration() -> None:
    with pytest.raises(ModelError) as caught:
        SentenceTransformerEmbeddingModel(
            model_id=" ",
            revision=" ",
            dimension=0,
            max_tokens=0,
            device=" ",
            batch_size=0,
            normalized=True,
        )

    assert caught.value.code is ModelErrorCode.CONFIGURATION
