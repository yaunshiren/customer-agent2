"""Explicit Parse -> Chunk -> Embed -> Index document ingestion use case."""

import asyncio
import math
from collections.abc import Callable
from typing import TypeVar

from customer_agent2.application.document_chunking import StructureAwareDocumentChunker
from customer_agent2.application.document_parsing import DocumentParsingService
from customer_agent2.domain.models import (
    DocumentIngestionRequest,
    EmbeddingIndexConfiguration,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
    IngestionAttempt,
    IngestionError,
    IngestionErrorCode,
    IngestionRepository,
    IngestionResult,
    ModelError,
)

_ResultT = TypeVar("_ResultT")


class DocumentIngestionService:
    """Build a new isolated version and activate it only after every chunk is ready."""

    def __init__(
        self,
        parser: DocumentParsingService,
        chunker: StructureAwareDocumentChunker,
        embedding_model: EmbeddingModel,
        repository: IngestionRepository,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_model = embedding_model
        self._repository = repository

    async def ingest(self, request: DocumentIngestionRequest) -> IngestionResult:
        """Parse and chunk off-loop, then atomically replace the active index version."""
        document = await _run_owned_worker(lambda: self._parser.parse(request.source))
        chunking = await _run_owned_worker(lambda: self._chunker.chunk(document))
        index_configuration = self._index_configuration()
        if (
            chunking.tokenizer_model_id != index_configuration.model_id
            or chunking.tokenizer_revision != index_configuration.model_revision
        ):
            raise IngestionError(
                IngestionErrorCode.INDEX_CONFIGURATION_MISMATCH,
                "分词器与 Embedding 模型版本不一致",
                retryable=False,
            )

        attempt = await self._repository.create_building_version(
            request,
            document,
            index_configuration,
        )
        try:
            embeddings = await self._embedding_model.embed(
                EmbeddingRequest(texts=tuple(chunk.content for chunk in chunking.chunks))
            )
            _validate_embedding_batch(embeddings, index_configuration, len(chunking.chunks))
            await self._repository.activate_version(attempt, chunking, embeddings)
        except asyncio.CancelledError:
            await self._record_failure(attempt, "cancelled")
            raise
        except Exception as error:
            await self._record_failure(attempt, _failure_code(error))
            raise

        return IngestionResult(
            knowledge_base_id=attempt.knowledge_base_id,
            document_id=attempt.document_id,
            version_id=attempt.version_id,
            version_number=attempt.version_number,
            chunk_count=len(chunking.chunks),
            content_sha256=document.source.content_sha256,
        )

    def _index_configuration(self) -> EmbeddingIndexConfiguration:
        return EmbeddingIndexConfiguration(
            model_id=self._embedding_model.model_id,
            model_revision=self._embedding_model.revision,
            dimension=self._embedding_model.dimension,
            normalized=self._embedding_model.normalized,
        )

    async def _record_failure(
        self,
        attempt: IngestionAttempt,
        error_code: str,
    ) -> None:
        try:
            await self._repository.mark_version_failed(attempt, error_code)
        except Exception as persistence_error:
            raise IngestionError(
                IngestionErrorCode.FAILURE_RECORDING_FAILED,
                "入库失败且无法安全记录失败状态",
                retryable=True,
            ) from persistence_error


async def _run_owned_worker(operation: Callable[[], _ResultT]) -> _ResultT:
    worker = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await asyncio.gather(worker, return_exceptions=True)
        raise


def _validate_embedding_batch(
    result: EmbeddingResult,
    expected: EmbeddingIndexConfiguration,
    expected_count: int,
) -> None:
    identity_matches = (
        result.model_id == expected.model_id
        and result.model_revision == expected.model_revision
        and result.dimension == expected.dimension
        and result.normalized == expected.normalized
    )
    vectors_are_normalized = not result.normalized or all(
        math.isclose(
            math.sqrt(math.fsum(value * value for value in vector)),
            1.0,
            rel_tol=1e-3,
            abs_tol=1e-3,
        )
        for vector in result.vectors
    )
    if len(result.vectors) != expected_count or not identity_matches or not vectors_are_normalized:
        raise IngestionError(
            IngestionErrorCode.EMBEDDING_PROTOCOL,
            "Embedding 返回结果与入库索引配置不一致",
            retryable=False,
        )


def _failure_code(error: Exception) -> str:
    if isinstance(error, ModelError):
        return f"embedding_{error.code.value}"
    if isinstance(error, IngestionError):
        return error.code.value
    return "ingestion_unexpected"
