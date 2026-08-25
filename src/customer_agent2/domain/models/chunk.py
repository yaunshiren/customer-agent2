"""Provider-neutral tokenization and structure-aware chunking contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from customer_agent2.domain.models.document import ParsedDocument


class ChunkingErrorCode(StrEnum):
    """Stable chunking failures that do not expose tokenizer internals."""

    TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"
    TOKENIZER_PROTOCOL = "tokenizer_protocol"


class ChunkingError(RuntimeError):
    """A sanitized chunking failure with a stable machine-readable category."""

    def __init__(self, code: ChunkingErrorCode, public_message: str) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message


class TextTokenCodec(Protocol):
    """Exact token encoding needed for budget checks and overlapping windows."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def encode(self, text: str) -> tuple[int, ...]: ...

    def decode(self, token_ids: tuple[int, ...]) -> str: ...


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    """Confirmed M2 token budget and secondary-split overlap policy."""

    target_tokens: int = 400
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        if self.target_tokens < 1:
            raise ValueError("ChunkingPolicy.target_tokens 必须大于 0")
        if self.overlap_tokens < 0:
            raise ValueError("ChunkingPolicy.overlap_tokens 不能小于 0")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("ChunkingPolicy.overlap_tokens 必须小于 target_tokens")


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """One validated, source-aware chunk before embedding and persistence."""

    chunk_index: int
    content: str
    token_count: int
    content_sha256: str
    block_start_ordinal: int
    block_end_ordinal: int
    start_line: int
    end_line: int
    section_path: tuple[str, ...]
    overlap_with_previous_tokens: int = 0

    def __post_init__(self) -> None:
        normalized_content = self.content.strip()
        if self.chunk_index < 0:
            raise ValueError("ChunkDraft.chunk_index 不能小于 0")
        if not normalized_content:
            raise ValueError("ChunkDraft.content 不能为空")
        if self.token_count < 1:
            raise ValueError("ChunkDraft.token_count 必须大于 0")
        if self.block_start_ordinal < 0 or self.block_end_ordinal < self.block_start_ordinal:
            raise ValueError("ChunkDraft Block 范围无效")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("ChunkDraft 行号范围无效")
        if self.overlap_with_previous_tokens < 0:
            raise ValueError("ChunkDraft overlap 不能小于 0")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("ChunkDraft.content_sha256 格式无效")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("ChunkDraft.section_path 不能包含空标题")
        object.__setattr__(self, "content", normalized_content)


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    """Reproducible chunk output tied to its source, tokenizer, and policy."""

    source: ParsedDocument
    chunks: tuple[ChunkDraft, ...]
    tokenizer_model_id: str
    tokenizer_revision: str
    policy: ChunkingPolicy

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("ChunkingResult.chunks 不能为空")
        if not self.tokenizer_model_id.strip() or not self.tokenizer_revision.strip():
            raise ValueError("ChunkingResult 分词器身份不能为空")
        if tuple(chunk.chunk_index for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("ChunkingResult.chunks 必须按连续 chunk_index 排序")
        if any(chunk.token_count > self.policy.target_tokens for chunk in self.chunks):
            raise ValueError("ChunkingResult 包含超过 Token 预算的 Chunk")
