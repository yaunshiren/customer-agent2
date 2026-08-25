"""Strict allowlist identification for the currently supported text formats."""

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Final

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    DocumentSource,
    IdentifiedDocument,
)

_EXTENSION_FORMATS: Final = {
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.PLAIN_TEXT,
}
_MEDIA_TYPE_FORMATS: Final = {
    "text/markdown": DocumentFormat.MARKDOWN,
    "text/x-markdown": DocumentFormat.MARKDOWN,
    "text/plain": DocumentFormat.PLAIN_TEXT,
}
_CANONICAL_MEDIA_TYPES: Final = {
    DocumentFormat.MARKDOWN: "text/markdown",
    DocumentFormat.PLAIN_TEXT: "text/plain",
}
_GENERIC_MEDIA_TYPES: Final = frozenset({"application/octet-stream", "binary/octet-stream"})


class SafeTextDocumentIdentifier:
    """Validate size, extension, MIME, UTF-8, and text safety before parsing."""

    def __init__(self, *, max_file_size_bytes: int) -> None:
        if max_file_size_bytes < 1:
            raise ValueError("max_file_size_bytes 必须大于 0")
        self._max_file_size_bytes = max_file_size_bytes

    def identify(self, source: DocumentSource) -> IdentifiedDocument:
        """Return decoded text only after all untrusted-input checks pass."""
        byte_size = len(source.content)
        if byte_size == 0:
            raise DocumentError(DocumentErrorCode.EMPTY_FILE, "文档不能为空")
        if byte_size > self._max_file_size_bytes:
            raise DocumentError(DocumentErrorCode.FILE_TOO_LARGE, "文档超过允许的大小")

        document_format = self._identify_format(source)
        if b"\x00" in source.content:
            raise DocumentError(DocumentErrorCode.BINARY_CONTENT, "文档包含二进制内容")

        try:
            decoded_text = source.content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise DocumentError(
                DocumentErrorCode.INVALID_ENCODING,
                "当前仅支持 UTF-8 编码的文本文件",
            ) from None

        normalized_text = decoded_text.replace("\r\n", "\n").replace("\r", "\n")
        if _contains_disallowed_control_character(normalized_text):
            raise DocumentError(DocumentErrorCode.BINARY_CONTENT, "文档包含二进制内容")

        return IdentifiedDocument(
            filename=source.filename,
            document_format=document_format,
            media_type=_CANONICAL_MEDIA_TYPES[document_format],
            charset="utf-8",
            text=normalized_text,
            byte_size=byte_size,
            content_sha256=sha256(source.content).hexdigest(),
        )

    def _identify_format(self, source: DocumentSource) -> DocumentFormat:
        extension = PurePosixPath(source.filename).suffix.lower()
        extension_format = _EXTENSION_FORMATS.get(extension)
        declared_media_type = _base_media_type(source.declared_media_type)
        media_type_format = (
            _MEDIA_TYPE_FORMATS.get(declared_media_type)
            if declared_media_type is not None
            else None
        )

        if extension and extension_format is None:
            raise DocumentError(DocumentErrorCode.UNSUPPORTED_TYPE, "不支持该文档类型")
        if (
            extension_format is not None
            and declared_media_type is not None
            and declared_media_type not in _GENERIC_MEDIA_TYPES
            and media_type_format is None
        ):
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "声明的媒体类型与文件扩展名不一致",
            )
        if (
            extension_format is not None
            and media_type_format is not None
            and extension_format is not media_type_format
        ):
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "声明的媒体类型与文件扩展名不一致",
            )

        identified_format = extension_format or media_type_format
        if identified_format is None:
            raise DocumentError(DocumentErrorCode.UNSUPPORTED_TYPE, "不支持该文档类型")
        return identified_format


def _base_media_type(declared_media_type: str | None) -> str | None:
    if declared_media_type is None:
        return None
    return declared_media_type.partition(";")[0].strip() or None


def _contains_disallowed_control_character(text: str) -> bool:
    allowed = {"\n", "\t"}
    return any(
        character not in allowed and (ord(character) < 32 or 127 <= ord(character) <= 159)
        for character in text
    )
