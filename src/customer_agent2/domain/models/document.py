"""Framework-independent document identification and parsing contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class DocumentFormat(StrEnum):
    """Document formats implemented by the current parser registry."""

    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"


class ParsedBlockKind(StrEnum):
    """Structural block categories preserved for later chunking."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE_BLOCK = "code_block"


class DocumentErrorCode(StrEnum):
    """Stable document failure categories exposed to later application use cases."""

    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    TYPE_MISMATCH = "type_mismatch"
    INVALID_ENCODING = "invalid_encoding"
    BINARY_CONTENT = "binary_content"
    EMPTY_CONTENT = "empty_content"
    PARSER_NOT_CONFIGURED = "parser_not_configured"


class DocumentError(RuntimeError):
    """A sanitized document failure with a stable machine-readable category."""

    def __init__(self, code: DocumentErrorCode, public_message: str) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message


@dataclass(frozen=True, slots=True)
class DocumentSource:
    """Untrusted in-memory upload input before type and size validation."""

    filename: str
    content: bytes
    declared_media_type: str | None = None

    def __post_init__(self) -> None:
        raw_filename, _, raw_media_type = _validate_source_values(
            self.filename,
            self.content,
            self.declared_media_type,
        )

        normalized_filename = raw_filename.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if not normalized_filename or normalized_filename in {".", ".."}:
            raise ValueError("DocumentSource.filename 不能为空")
        if len(normalized_filename) > 500:
            raise ValueError("DocumentSource.filename 不能超过 500 个字符")
        normalized_media_type = raw_media_type
        if normalized_media_type is not None:
            normalized_media_type = normalized_media_type.strip().lower() or None

        object.__setattr__(self, "filename", normalized_filename)
        object.__setattr__(self, "declared_media_type", normalized_media_type)


def _validate_source_values(
    filename: object,
    content: object,
    declared_media_type: object,
) -> tuple[str, bytes, str | None]:
    if not isinstance(filename, str):
        raise TypeError("DocumentSource.filename 必须是 str")
    if not isinstance(content, bytes):
        raise TypeError("DocumentSource.content 必须是 bytes")
    if declared_media_type is not None and not isinstance(declared_media_type, str):
        raise TypeError("DocumentSource.declared_media_type 必须是 str 或 None")
    return filename, content, declared_media_type


@dataclass(frozen=True, slots=True)
class IdentifiedDocument:
    """Decoded text after allowlist, size, MIME, binary, and encoding checks."""

    filename: str
    document_format: DocumentFormat
    media_type: str
    charset: str
    text: str
    byte_size: int
    content_sha256: str

    def __post_init__(self) -> None:
        if not self.filename.strip():
            raise ValueError("IdentifiedDocument.filename 不能为空")
        if not self.media_type.strip() or not self.charset.strip():
            raise ValueError("IdentifiedDocument 媒体类型和字符集不能为空")
        if self.byte_size < 1:
            raise ValueError("IdentifiedDocument.byte_size 必须大于 0")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("IdentifiedDocument.content_sha256 格式无效")


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """One ordered source-aware block ready for later structure-aware chunking."""

    kind: ParsedBlockKind
    text: str
    ordinal: int
    start_line: int
    end_line: int
    section_path: tuple[str, ...] = ()
    heading_level: int | None = None
    code_language: str | None = None

    def __post_init__(self) -> None:
        normalized_text = self.text.strip()
        if not normalized_text:
            raise ValueError("ParsedBlock.text 不能为空")
        if self.ordinal < 0:
            raise ValueError("ParsedBlock.ordinal 不能小于 0")
        if self.start_line < 1 or self.end_line < self.start_line:
            raise ValueError("ParsedBlock 行号范围无效")
        if any(not section.strip() for section in self.section_path):
            raise ValueError("ParsedBlock.section_path 不能包含空标题")
        if self.kind is ParsedBlockKind.HEADING:
            if self.heading_level is None or not 1 <= self.heading_level <= 6:
                raise ValueError("标题 Block 必须包含 1 到 6 级 heading_level")
        elif self.heading_level is not None:
            raise ValueError("非标题 Block 不能包含 heading_level")
        if self.code_language is not None and self.kind is not ParsedBlockKind.CODE_BLOCK:
            raise ValueError("只有代码 Block 可以包含 code_language")

        normalized_language = self.code_language
        if normalized_language is not None:
            normalized_language = normalized_language.strip() or None

        object.__setattr__(self, "text", normalized_text)
        object.__setattr__(self, "code_language", normalized_language)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Complete structured parser output retaining the identified source identity."""

    source: IdentifiedDocument
    blocks: tuple[ParsedBlock, ...]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("ParsedDocument.blocks 不能为空")
        if tuple(block.ordinal for block in self.blocks) != tuple(range(len(self.blocks))):
            raise ValueError("ParsedDocument.blocks 必须按连续 ordinal 排序")

    @property
    def text(self) -> str:
        """Return a stable plain-text view without discarding block boundaries."""
        return "\n\n".join(block.text for block in self.blocks)


class DocumentIdentifier(Protocol):
    """Validate and identify an untrusted in-memory document source."""

    def identify(self, source: DocumentSource) -> IdentifiedDocument: ...


class DocumentParser(Protocol):
    """Parse one identified document format into source-aware structural blocks."""

    @property
    def document_format(self) -> DocumentFormat: ...

    def parse(self, document: IdentifiedDocument) -> ParsedDocument: ...
