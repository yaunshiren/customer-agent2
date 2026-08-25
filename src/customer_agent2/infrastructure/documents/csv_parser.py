"""Strict UTF-8 CSV parser producing self-contained record blocks."""

import csv
from collections.abc import Iterator
from io import StringIO

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    IdentifiedDocument,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
)


class CsvDocumentParser:
    """Repeat normalized headers in every record block for independent retrieval."""

    def __init__(
        self,
        *,
        max_rows: int,
        max_columns: int,
        max_extracted_chars: int,
    ) -> None:
        if min(max_rows, max_columns, max_extracted_chars) < 1:
            raise ValueError("CSV 解析限制必须大于 0")
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_extracted_chars = max_extracted_chars

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.CSV

    @property
    def parser_name(self) -> str:
        return "customer-agent2-csv"

    @property
    def parser_version(self) -> str:
        return "1"

    def parse(self, document: IdentifiedDocument) -> ParsedDocument:
        """Parse a comma-delimited table and reject ambiguous row shapes."""
        if document.document_format is not self.document_format:
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "文档类型与 CSV 解析器不匹配",
            )
        text = document.text
        if text is None:
            raise DocumentError(DocumentErrorCode.TYPE_MISMATCH, "CSV 文档缺少文本内容")

        try:
            blocks = self._parse_rows(text)
        except DocumentError:
            raise
        except csv.Error:
            raise DocumentError(
                DocumentErrorCode.MALFORMED_DOCUMENT,
                "CSV 引号或记录结构无效",
            ) from None
        if not blocks:
            raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "CSV 没有数据记录")
        return ParsedDocument(
            source=document,
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
        )

    def _parse_rows(self, text: str) -> list[ParsedBlock]:
        reader = csv.reader(StringIO(text, newline=""), strict=True)
        header_row = _next_nonempty_row(reader)
        if header_row is None:
            return []
        headers = _unique_headers(header_row)
        if len(headers) > self._max_columns:
            raise _resource_limit_error("CSV 列数超过允许上限")

        blocks: list[ParsedBlock] = []
        previous_line = reader.line_num
        extracted_chars = 0
        for row in reader:
            row_start = previous_line + 1
            previous_line = reader.line_num
            values = tuple(value.strip() for value in row)
            if not any(values):
                continue
            if len(values) != len(headers):
                raise DocumentError(
                    DocumentErrorCode.MALFORMED_DOCUMENT,
                    "CSV 数据列数与表头不一致",
                )
            if len(blocks) >= self._max_rows:
                raise _resource_limit_error("CSV 记录数超过允许上限")
            block_text = "\n".join(
                f"{header}: {value}" for header, value in zip(headers, values, strict=True)
            )
            extracted_chars += len(block_text)
            if extracted_chars > self._max_extracted_chars:
                raise _resource_limit_error("CSV 提取文本超过允许上限")
            blocks.append(
                ParsedBlock(
                    kind=ParsedBlockKind.PARAGRAPH,
                    text=block_text,
                    ordinal=len(blocks),
                    start_line=row_start,
                    end_line=reader.line_num,
                    section_path=("CSV 记录",),
                )
            )
        return blocks


def _next_nonempty_row(reader: Iterator[list[str]]) -> tuple[str, ...] | None:
    for row in reader:
        values = tuple(value.strip() for value in row)
        if any(values):
            return values
    return None


def _unique_headers(values: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    headers: list[str] = []
    for index, value in enumerate(values, start=1):
        base = value or f"列 {index}"
        counts[base] = counts.get(base, 0) + 1
        suffix = f" ({counts[base]})" if counts[base] > 1 else ""
        headers.append(f"{base}{suffix}")
    return tuple(headers)


def _resource_limit_error(message: str) -> DocumentError:
    return DocumentError(DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED, message)
