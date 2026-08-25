"""Unit tests for explicit transactional document-ingestion orchestration."""

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

import pytest

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
)
from customer_agent2.domain.models import (
    ChunkingPolicy,
    ChunkingResult,
    DocumentError,
    DocumentIngestionRequest,
    DocumentSource,
    EmbeddingIndexConfiguration,
    EmbeddingRequest,
    EmbeddingResult,
    IngestionAttempt,
    IngestionError,
    IngestionErrorCode,
    ModelError,
    ModelErrorCode,
    ParsedDocument,
)
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
)
from customer_agent2.infrastructure.models import FakeEmbeddingModel

MODEL_ID = "test-embedding"
MODEL_REVISION = "test-revision"


class CharacterTokenCodec:
    """Deterministic character codec sharing the fake Embedding identity."""

    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def revision(self) -> str:
        return MODEL_REVISION

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@dataclass(frozen=True, slots=True)
class CreatedVersion:
    request: DocumentIngestionRequest
    document: ParsedDocument
    index_configuration: EmbeddingIndexConfiguration


class RecordingIngestionRepository:
    """In-memory port fake exposing orchestration order and failure behavior."""

    def __init__(self) -> None:
        self.attempt = IngestionAttempt(uuid4(), uuid4(), uuid4(), 1)
        self.created: list[CreatedVersion] = []
        self.activations: list[tuple[IngestionAttempt, ChunkingResult, EmbeddingResult]] = []
        self.failures: list[tuple[IngestionAttempt, str]] = []
        self.activation_error: IngestionError | None = None
        self.failure_error: IngestionError | None = None

    async def create_building_version(
        self,
        request: DocumentIngestionRequest,
        document: ParsedDocument,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> IngestionAttempt:
        self.created.append(CreatedVersion(request, document, index_configuration))
        return self.attempt

    async def activate_version(
        self,
        attempt: IngestionAttempt,
        chunking: ChunkingResult,
        embeddings: EmbeddingResult,
    ) -> None:
        if self.activation_error is not None:
            raise self.activation_error
        self.activations.append((attempt, chunking, embeddings))

    async def mark_version_failed(self, attempt: IngestionAttempt, error_code: str) -> None:
        if self.failure_error is not None:
            raise self.failure_error
        self.failures.append((attempt, error_code))


class WrongBatchEmbeddingModel(FakeEmbeddingModel):
    """Return fewer vectors than requested to simulate a provider protocol bug."""

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        result = await super().embed(request)
        return EmbeddingResult(
            model_id=result.model_id,
            model_revision=result.model_revision,
            vectors=result.vectors[:1],
            dimension=result.dimension,
            normalized=result.normalized,
        )


class BlockingEmbeddingModel(FakeEmbeddingModel):
    """Wait until cancelled so the use case can prove failure-state cleanup."""

    def __init__(self) -> None:
        super().__init__(MODEL_ID, revision=MODEL_REVISION, dimension=3)
        self.started = asyncio.Event()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def build_service(
    repository: RecordingIngestionRepository,
    embedding_model: FakeEmbeddingModel | None = None,
    *,
    tokenizer_model_id: str = MODEL_ID,
) -> DocumentIngestionService:
    parser = DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=1024 * 1024,
            max_extracted_chars=1024 * 1024,
        ),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )

    class ConfiguredCharacterTokenCodec(CharacterTokenCodec):
        @property
        def model_id(self) -> str:
            return tokenizer_model_id

    return DocumentIngestionService(
        parser,
        StructureAwareDocumentChunker(
            ConfiguredCharacterTokenCodec(),
            ChunkingPolicy(target_tokens=100, overlap_tokens=10),
        ),
        embedding_model or FakeEmbeddingModel(MODEL_ID, revision=MODEL_REVISION, dimension=3),
        repository,
    )


def ingestion_request(
    content: str = "# 第一节\n\n第一段\n\n## 第二节\n\n第二段",
) -> DocumentIngestionRequest:
    return DocumentIngestionRequest(
        knowledge_base_id=uuid4(),
        source_key="manual/guide.md",
        source=DocumentSource("guide.md", content.encode(), "text/markdown"),
    )


@pytest.mark.asyncio
async def test_ingestion_builds_embeds_and_activates_one_complete_version() -> None:
    repository = RecordingIngestionRepository()
    model = FakeEmbeddingModel(MODEL_ID, revision=MODEL_REVISION, dimension=3)

    result = await build_service(repository, model).ingest(ingestion_request())

    assert result.version_id == repository.attempt.version_id
    assert result.chunk_count == 2
    assert len(repository.created) == 1
    created = repository.created[0]
    assert created.document.parser_name == "customer-agent2-markdown"
    assert created.document.parser_version == "1"
    assert created.index_configuration == EmbeddingIndexConfiguration(
        MODEL_ID,
        MODEL_REVISION,
        3,
        True,
    )
    assert [request.texts for request in model.requests] == [
        ("第一节\n\n第一段", "第二节\n\n第二段")
    ]
    assert len(repository.activations) == 1
    assert repository.failures == []


@pytest.mark.asyncio
async def test_parse_failure_does_not_create_a_building_version() -> None:
    repository = RecordingIngestionRepository()
    request = DocumentIngestionRequest(
        uuid4(),
        "empty.txt",
        DocumentSource("empty.txt", b"", "text/plain"),
    )

    with pytest.raises(DocumentError):
        await build_service(repository).ingest(request)

    assert repository.created == []
    assert repository.activations == []
    assert repository.failures == []


@pytest.mark.asyncio
async def test_tokenizer_identity_mismatch_fails_before_database_mutation() -> None:
    repository = RecordingIngestionRepository()

    with pytest.raises(IngestionError) as caught:
        await build_service(repository, tokenizer_model_id="wrong-tokenizer").ingest(
            ingestion_request()
        )

    assert caught.value.code is IngestionErrorCode.INDEX_CONFIGURATION_MISMATCH
    assert repository.created == []


@pytest.mark.asyncio
async def test_embedding_failure_marks_building_version_failed_without_activation() -> None:
    repository = RecordingIngestionRepository()
    model = FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=3,
        error=ModelError(ModelErrorCode.UNAVAILABLE, "Embedding 暂时不可用", retryable=True),
    )

    with pytest.raises(ModelError):
        await build_service(repository, model).ingest(ingestion_request())

    assert repository.activations == []
    assert repository.failures == [(repository.attempt, "embedding_unavailable")]


@pytest.mark.asyncio
async def test_invalid_embedding_batch_is_failed_as_a_protocol_error() -> None:
    repository = RecordingIngestionRepository()
    model = WrongBatchEmbeddingModel(MODEL_ID, revision=MODEL_REVISION, dimension=3)

    with pytest.raises(IngestionError) as caught:
        await build_service(repository, model).ingest(ingestion_request())

    assert caught.value.code is IngestionErrorCode.EMBEDDING_PROTOCOL
    assert repository.failures == [(repository.attempt, "embedding_protocol")]


@pytest.mark.asyncio
async def test_activation_failure_is_recorded_and_original_error_is_preserved() -> None:
    repository = RecordingIngestionRepository()
    repository.activation_error = IngestionError(
        IngestionErrorCode.PERSISTENCE_FAILURE,
        "无法原子激活文档版本",
        retryable=True,
    )

    with pytest.raises(IngestionError) as caught:
        await build_service(repository).ingest(ingestion_request())

    assert caught.value is repository.activation_error
    assert repository.failures == [(repository.attempt, "persistence_failure")]


@pytest.mark.asyncio
async def test_failure_recording_error_is_sanitized_and_not_silently_ignored() -> None:
    repository = RecordingIngestionRepository()
    repository.activation_error = IngestionError(
        IngestionErrorCode.PERSISTENCE_FAILURE,
        "无法原子激活文档版本",
        retryable=True,
    )
    repository.failure_error = IngestionError(
        IngestionErrorCode.PERSISTENCE_FAILURE,
        "无法记录失败状态",
        retryable=True,
    )

    with pytest.raises(IngestionError) as caught:
        await build_service(repository).ingest(ingestion_request())

    assert caught.value.code is IngestionErrorCode.FAILURE_RECORDING_FAILED
    assert "无法安全记录" in caught.value.public_message


@pytest.mark.asyncio
async def test_cancellation_marks_committed_building_version_failed() -> None:
    repository = RecordingIngestionRepository()
    model = BlockingEmbeddingModel()
    task = asyncio.create_task(build_service(repository, model).ingest(ingestion_request()))
    await asyncio.wait_for(model.started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert repository.activations == []
    assert repository.failures == [(repository.attempt, "cancelled")]


def test_ingestion_request_validates_source_key() -> None:
    source = DocumentSource("guide.txt", b"content")
    with pytest.raises(ValueError, match="source_key"):
        DocumentIngestionRequest(UUID(int=0), "  ", source)
