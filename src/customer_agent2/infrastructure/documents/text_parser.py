"""Plain UTF-8 text parser preserving paragraph source lines."""

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    IdentifiedDocument,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
)


class PlainTextDocumentParser:
    """Split plain text into non-empty paragraphs without guessing headings."""

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.PLAIN_TEXT

    def parse(self, document: IdentifiedDocument) -> ParsedDocument:
        """Return ordered paragraphs with one-based inclusive line ranges."""
        if document.document_format is not self.document_format:
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "文档类型与 TXT 解析器不匹配",
            )

        blocks: list[ParsedBlock] = []
        paragraph_lines: list[str] = []
        paragraph_start = 0
        lines = document.text.split("\n")

        for line_number, line in enumerate(lines, start=1):
            if line.strip():
                if not paragraph_lines:
                    paragraph_start = line_number
                paragraph_lines.append(line.rstrip())
                continue
            _append_paragraph(blocks, paragraph_lines, paragraph_start, line_number - 1)
            paragraph_lines = []

        _append_paragraph(blocks, paragraph_lines, paragraph_start, len(lines))
        if not blocks:
            raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "文档没有可解析的文本内容")
        return ParsedDocument(source=document, blocks=tuple(blocks))


def _append_paragraph(
    blocks: list[ParsedBlock],
    lines: list[str],
    start_line: int,
    end_line: int,
) -> None:
    if not lines:
        return
    blocks.append(
        ParsedBlock(
            kind=ParsedBlockKind.PARAGRAPH,
            text="\n".join(lines),
            ordinal=len(blocks),
            start_line=start_line,
            end_line=end_line,
        )
    )
