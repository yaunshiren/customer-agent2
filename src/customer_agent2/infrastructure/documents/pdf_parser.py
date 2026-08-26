"""pypdf adapter with explicit page and extracted-text limits."""

from io import BytesIO

from pypdf import PdfReader

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    IdentifiedDocument,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
)


class PdfDocumentParser:
    """Extract page-scoped PDF text without attempting OCR or decryption."""

    def __init__(self, *, max_pages: int, max_extracted_chars: int) -> None:
        if max_pages < 1 or max_extracted_chars < 1:
            raise ValueError("PDF 解析限制必须大于 0")
        self._max_pages = max_pages
        self._max_extracted_chars = max_extracted_chars

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.PDF

    @property
    def parser_name(self) -> str:
        return "customer-agent2-pypdf"

    @property
    def parser_version(self) -> str:
        return "1"

    def parse(self, document: IdentifiedDocument) -> ParsedDocument:
        """Return paragraph blocks whose section path records the source page."""
        if document.document_format is not self.document_format:
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "文档类型与 PDF 解析器不匹配",
            )

        try:
            reader = PdfReader(
                BytesIO(document.content),
                strict=True,
                root_object_recovery_limit=1000,
            )
            if reader.is_encrypted:
                raise DocumentError(
                    DocumentErrorCode.ENCRYPTED_DOCUMENT,
                    "当前不支持加密 PDF",
                )
            page_count = len(reader.pages)
            if page_count > self._max_pages:
                raise _resource_limit_error("PDF 页数超过允许上限")

            blocks: list[ParsedBlock] = []
            extracted_chars = 0
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = _normalize_extracted_text(page.extract_text() or "")
                extracted_chars += len(page_text)
                if extracted_chars > self._max_extracted_chars:
                    raise _resource_limit_error("PDF 提取文本超过允许上限")
                _append_page_paragraphs(blocks, page_text, page_number)
        except DocumentError:
            raise
        except Exception:
            raise DocumentError(
                DocumentErrorCode.MALFORMED_DOCUMENT,
                "PDF 结构无法安全解析",
            ) from None

        if not blocks:
            raise DocumentError(
                DocumentErrorCode.EMPTY_CONTENT,
                "PDF 没有可提取的文本内容。扫描件 OCR 尚未支持",
            )
        return ParsedDocument(
            source=document,
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
        )


def _append_page_paragraphs(
    blocks: list[ParsedBlock],
    text: str,
    page_number: int,
) -> None:
    paragraph_lines: list[str] = []
    paragraph_start = 0
    lines = text.split("\n")
    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if not paragraph_lines:
                paragraph_start = line_number
            paragraph_lines.append(line.rstrip())
            continue
        _append_pdf_paragraph(
            blocks,
            paragraph_lines,
            paragraph_start,
            line_number - 1,
            page_number,
        )
        paragraph_lines = []
    _append_pdf_paragraph(
        blocks,
        paragraph_lines,
        paragraph_start,
        len(lines),
        page_number,
    )


def _append_pdf_paragraph(
    blocks: list[ParsedBlock],
    lines: list[str],
    start_line: int,
    end_line: int,
    page_number: int,
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
            section_path=(f"第 {page_number} 页",),
            page_number=page_number,
        )
    )


def _normalize_extracted_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if any(
        character not in {"\n", "\t"} and (ord(character) < 32 or 127 <= ord(character) <= 159)
        for character in normalized
    ):
        raise DocumentError(DocumentErrorCode.BINARY_CONTENT, "PDF 提取文本包含控制字符")
    return normalized


def _resource_limit_error(message: str) -> DocumentError:
    return DocumentError(DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED, message)
