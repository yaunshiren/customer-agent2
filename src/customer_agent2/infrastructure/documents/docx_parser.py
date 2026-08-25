"""python-docx adapter with ZIP bomb and structural limits."""

import re
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    IdentifiedDocument,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
)

_REQUIRED_MEMBERS = frozenset({"[Content_Types].xml", "word/document.xml"})
_HEADING_STYLE = re.compile(r"(?:heading|标题)\s*([1-6])", re.IGNORECASE)


class DocxDocumentParser:
    """Preserve Word headings, paragraphs, list styles, and table records."""

    def __init__(
        self,
        *,
        max_archive_entries: int,
        max_uncompressed_bytes: int,
        max_expansion_ratio: int,
        max_extracted_chars: int,
    ) -> None:
        limits = (
            max_archive_entries,
            max_uncompressed_bytes,
            max_expansion_ratio,
            max_extracted_chars,
        )
        if any(value < 1 for value in limits):
            raise ValueError("DOCX 解析限制必须大于 0")
        self._max_archive_entries = max_archive_entries
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._max_expansion_ratio = max_expansion_ratio
        self._max_extracted_chars = max_extracted_chars

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.DOCX

    @property
    def parser_name(self) -> str:
        return "customer-agent2-python-docx"

    @property
    def parser_version(self) -> str:
        return "1"

    def parse(self, document: IdentifiedDocument) -> ParsedDocument:
        """Parse a validated Office Open XML package in document order."""
        if document.document_format is not self.document_format:
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "文档类型与 DOCX 解析器不匹配",
            )

        try:
            _validate_archive(
                document.content,
                max_entries=self._max_archive_entries,
                max_uncompressed_bytes=self._max_uncompressed_bytes,
                max_expansion_ratio=self._max_expansion_ratio,
            )
            word_document = Document(BytesIO(document.content))
            blocks = self._parse_body(word_document)
        except DocumentError:
            raise
        except Exception:
            raise DocumentError(
                DocumentErrorCode.MALFORMED_DOCUMENT,
                "DOCX 结构无法安全解析",
            ) from None

        if not blocks:
            raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "DOCX 没有可解析的文本内容")
        return ParsedDocument(
            source=document,
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
        )

    def _parse_body(self, word_document: DocumentObject) -> list[ParsedBlock]:
        blocks: list[ParsedBlock] = []
        headings: list[tuple[int, str]] = []
        extracted_chars = 0
        table_number = 0

        for position, item in enumerate(word_document.iter_inner_content(), start=1):
            if isinstance(item, Paragraph):
                text = _normalize_text(item.text)
                if not text:
                    continue
                heading_level = _heading_level(item)
                kind = ParsedBlockKind.PARAGRAPH
                if heading_level is not None:
                    kind = ParsedBlockKind.HEADING
                    headings = [entry for entry in headings if entry[0] < heading_level]
                    headings.append((heading_level, text))
                elif _is_list_paragraph(item):
                    kind = ParsedBlockKind.LIST_ITEM
                blocks.append(
                    ParsedBlock(
                        kind=kind,
                        text=text,
                        ordinal=len(blocks),
                        start_line=position,
                        end_line=position,
                        section_path=tuple(value for _, value in headings),
                        heading_level=heading_level,
                    )
                )
                extracted_chars = _checked_extracted_chars(
                    extracted_chars,
                    text,
                    self._max_extracted_chars,
                )
            else:
                table_number += 1
                table_blocks = _table_blocks(
                    item,
                    position=position,
                    table_number=table_number,
                    headings=tuple(value for _, value in headings),
                    ordinal_start=len(blocks),
                )
                for block in table_blocks:
                    extracted_chars = _checked_extracted_chars(
                        extracted_chars,
                        block.text,
                        self._max_extracted_chars,
                    )
                blocks.extend(table_blocks)
        return blocks


def _validate_archive(
    content: bytes,
    *,
    max_entries: int,
    max_uncompressed_bytes: int,
    max_expansion_ratio: int,
) -> None:
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
    except BadZipFile:
        raise DocumentError(
            DocumentErrorCode.INVALID_FILE_SIGNATURE,
            "DOCX 不是有效的 Office Open XML 包",
        ) from None

    if len(entries) > max_entries:
        raise _resource_limit_error("DOCX 压缩包条目数超过允许上限")
    names = {entry.filename for entry in entries}
    contains_macro = any(name.lower().endswith("vbaproject.bin") for name in names)
    if not _REQUIRED_MEMBERS.issubset(names) or contains_macro:
        raise DocumentError(
            DocumentErrorCode.MALFORMED_DOCUMENT,
            "DOCX 缺少必要结构或包含不支持的宏内容",
        )

    total_uncompressed = 0
    total_compressed = 0
    for entry in entries:
        path = PurePosixPath(entry.filename.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts or entry.flag_bits & 0x1:
            raise DocumentError(
                DocumentErrorCode.MALFORMED_DOCUMENT,
                "DOCX 压缩包包含不安全条目",
            )
        total_uncompressed += entry.file_size
        total_compressed += entry.compress_size
        if total_uncompressed > max_uncompressed_bytes:
            raise _resource_limit_error("DOCX 解压后大小超过允许上限")

    expansion_ratio = total_uncompressed / max(total_compressed, 1)
    if expansion_ratio > max_expansion_ratio:
        raise _resource_limit_error("DOCX 压缩膨胀比例超过允许上限")


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    if style is None:
        return None
    for value in (style.style_id, style.name):
        match = _HEADING_STYLE.search(value or "")
        if match is not None:
            return int(match.group(1))
    return None


def _is_list_paragraph(paragraph: Paragraph) -> bool:
    style = paragraph.style
    if style is None:
        return False
    values = (style.style_id or "", style.name or "")
    return any(value.lower().startswith("list") or value.startswith("列表") for value in values)


def _table_blocks(
    table: Table,
    *,
    position: int,
    table_number: int,
    headings: tuple[str, ...],
    ordinal_start: int,
) -> list[ParsedBlock]:
    rows = [tuple(_normalize_text(cell.text) for cell in row.cells) for row in table.rows]
    rows = [row for row in rows if any(row)]
    if not rows:
        return []
    if len(rows) == 1:
        texts = [" | ".join(value for value in rows[0] if value)]
    else:
        headers = _unique_headers(rows[0])
        texts = [
            "\n".join(f"{header}: {value}" for header, value in zip(headers, row, strict=False))
            for row in rows[1:]
            if any(row)
        ]
    section_path = (*headings, f"表格 {table_number}")
    return [
        ParsedBlock(
            kind=ParsedBlockKind.PARAGRAPH,
            text=text,
            ordinal=ordinal_start + offset,
            start_line=position,
            end_line=position,
            section_path=section_path,
        )
        for offset, text in enumerate(texts)
        if text.strip()
    ]


def _unique_headers(values: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    headers: list[str] = []
    for index, value in enumerate(values, start=1):
        base = value or f"列 {index}"
        counts[base] = counts.get(base, 0) + 1
        suffix = f" ({counts[base]})" if counts[base] > 1 else ""
        headers.append(f"{base}{suffix}")
    return tuple(headers)


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in normalized.split("\n")).strip()


def _checked_extracted_chars(current: int, text: str, maximum: int) -> int:
    updated = current + len(text)
    if updated > maximum:
        raise _resource_limit_error("DOCX 提取文本超过允许上限")
    return updated


def _resource_limit_error(message: str) -> DocumentError:
    return DocumentError(DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED, message)
