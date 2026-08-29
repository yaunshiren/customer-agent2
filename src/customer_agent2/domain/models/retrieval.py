"""Framework-independent contracts for scoped pgvector retrieval."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar
from uuid import UUID

from customer_agent2.domain.models.document import DocumentFormat
from customer_agent2.domain.models.ingestion import EmbeddingIndexConfiguration

_ValueT = TypeVar("_ValueT")


class RetrievalErrorCode(StrEnum):
    """Stable retrieval failures safe to expose through later API adapters."""

    KNOWLEDGE_BASE_NOT_FOUND = "knowledge_base_not_found"
    INDEX_CONFIGURATION_MISMATCH = "index_configuration_mismatch"
    EMBEDDING_PROTOCOL = "embedding_protocol"
    RESULT_PROTOCOL = "retrieval_result_protocol"
    PERSISTENCE_FAILURE = "persistence_failure"


class RetrievalError(RuntimeError):
    """A sanitized retrieval failure with a stable category and retry hint."""

    def __init__(
        self,
        code: RetrievalErrorCode,
        public_message: str,
        *,
        retryable: bool,
    ) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class VectorSearchScope:
    """Authorized database-side filters for one vector search."""

    knowledge_base_ids: tuple[UUID, ...]
    document_ids: tuple[UUID, ...] = ()
    document_formats: tuple[DocumentFormat, ...] = ()
    parser_names: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    page_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        knowledge_base_ids = _unique(self.knowledge_base_ids)
        document_ids = _unique(self.document_ids)
        document_formats = _unique(self.document_formats)
        parser_names = _normalized_text_filters(self.parser_names, "parser_names", 100)
        sections = _normalized_text_filters(self.sections, "sections", 500)
        page_numbers = _unique(self.page_numbers)

        if not knowledge_base_ids:
            raise ValueError("VectorSearchScope.knowledge_base_ids 不能为空")
        if len(knowledge_base_ids) > 100:
            raise ValueError("一次检索最多指定 100 个知识库")
        if len(document_ids) > 1000:
            raise ValueError("一次检索最多指定 1000 个文档")
        if len(document_formats) > len(DocumentFormat):
            raise ValueError("VectorSearchScope.document_formats 包含无效值")
        if len(parser_names) > 100 or len(sections) > 100:
            raise ValueError("一次检索最多指定 100 个解析器或章节过滤值")
        if len(page_numbers) > 1000 or any(value < 1 for value in page_numbers):
            raise ValueError("VectorSearchScope.page_numbers 必须是最多 1000 个正整数")

        object.__setattr__(self, "knowledge_base_ids", knowledge_base_ids)
        object.__setattr__(self, "document_ids", document_ids)
        object.__setattr__(self, "document_formats", document_formats)
        object.__setattr__(self, "parser_names", parser_names)
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "page_numbers", page_numbers)


@dataclass(frozen=True, slots=True)
class VectorSearchRequest:
    """One non-empty query plus an explicit authorized search scope."""

    query: str
    scope: VectorSearchScope

    def __post_init__(self) -> None:
        normalized_query = self.query.strip()
        if not normalized_query:
            raise ValueError("VectorSearchRequest.query 不能为空")
        if len(normalized_query) > 10_000:
            raise ValueError("VectorSearchRequest.query 不能超过 10000 个字符")
        object.__setattr__(self, "query", normalized_query)


@dataclass(frozen=True, slots=True)
class RetrievedChunkSource:
    """Typed source coordinates reconstructed from persisted chunk metadata."""

    block_start_ordinal: int
    block_end_ordinal: int
    start_line: int
    end_line: int
    section_path: tuple[str, ...]
    overlap_with_previous_tokens: int

    def __post_init__(self) -> None:
        if self.block_start_ordinal < 0 or self.block_end_ordinal < self.block_start_ordinal:
            raise ValueError("RetrievedChunkSource Block 范围无效")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("RetrievedChunkSource 行号范围无效")
        if self.overlap_with_previous_tokens < 0:
            raise ValueError("RetrievedChunkSource 重叠 Token 数不能小于 0")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("RetrievedChunkSource.section_path 不能包含空标题")


@dataclass(frozen=True, slots=True)
class VectorSearchCandidate:
    """One active-version chunk ranked by cosine distance."""

    rank: int
    chunk_id: UUID
    knowledge_base_id: UUID
    document_id: UUID
    document_version_id: UUID
    source_key: str
    display_name: str
    document_format: DocumentFormat
    media_type: str
    parser_name: str
    parser_version: str
    chunk_index: int
    content: str
    token_count: int
    content_sha256: str
    section: str | None
    page_number: int | None
    source: RetrievedChunkSource
    cosine_distance: float
    similarity: float

    def __post_init__(self) -> None:
        if self.rank < 1 or self.chunk_index < 0 or self.token_count < 1:
            raise ValueError("VectorSearchCandidate 排名或 Chunk 数值无效")
        required_text = (
            self.source_key,
            self.display_name,
            self.media_type,
            self.parser_name,
            self.parser_version,
            self.content,
        )
        if any(not value.strip() for value in required_text):
            raise ValueError("VectorSearchCandidate 必填文本不能为空")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("VectorSearchCandidate.content_sha256 格式无效")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("VectorSearchCandidate.page_number 必须大于 0")
        if not math.isfinite(self.cosine_distance) or not math.isfinite(self.similarity):
            raise ValueError("VectorSearchCandidate 距离和相似度必须是有限值")
        if not -1e-6 <= self.cosine_distance <= 2.000001:
            raise ValueError("VectorSearchCandidate.cosine_distance 超出归一化向量范围")
        if not math.isclose(
            self.similarity,
            1.0 - self.cosine_distance,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("VectorSearchCandidate 距离和相似度不一致")


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """Ranked vector candidates without retaining the potentially sensitive query."""

    index_configuration: EmbeddingIndexConfiguration
    candidates: tuple[VectorSearchCandidate, ...]

    def __post_init__(self) -> None:
        expected_ranks = tuple(range(1, len(self.candidates) + 1))
        if tuple(candidate.rank for candidate in self.candidates) != expected_ranks:
            raise ValueError("VectorSearchResult.candidates 必须按连续 rank 排序")


class VectorSearchRepository(Protocol):
    """Persistence port for compatible, active-only vector retrieval."""

    async def search(
        self,
        query_vector: tuple[float, ...],
        index_configuration: EmbeddingIndexConfiguration,
        scope: VectorSearchScope,
        *,
        limit: int,
        hnsw_ef_search: int,
    ) -> tuple[VectorSearchCandidate, ...]: ...


def _unique(values: tuple[_ValueT, ...]) -> tuple[_ValueT, ...]:
    return tuple(dict.fromkeys(values))


def _normalized_text_filters(
    values: tuple[str, ...],
    field_name: str,
    maximum_length: int,
) -> tuple[str, ...]:
    normalized = tuple(value.strip() for value in values)
    if any(not value or len(value) > maximum_length for value in normalized):
        raise ValueError(f"VectorSearchScope.{field_name} 包含空值或超长值")
    if any(any(ord(character) < 32 for character in value) for value in normalized):
        raise ValueError(f"VectorSearchScope.{field_name} 不能包含控制字符")
    return _unique(normalized)
