"""Document identification, parser selection, and structural parsing tests."""

from hashlib import sha256

import pytest

from customer_agent2.application import DocumentParsingService
from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    DocumentSource,
    ParsedBlockKind,
)
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeTextDocumentIdentifier,
)

MARKDOWN_SAMPLE = """# 退款政策

客户可在 **七天内** 申请退款。

- 保留订单号
- 商品未经使用

## 代码示例

```python
print("ok")
```
"""


def parsing_service(*, max_file_size_bytes: int = 1024 * 1024) -> DocumentParsingService:
    """Build the M2-B parser composition without framework or filesystem state."""
    return DocumentParsingService(
        SafeTextDocumentIdentifier(max_file_size_bytes=max_file_size_bytes),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )


def test_markdown_parser_preserves_structure_sections_and_source_lines() -> None:
    source = DocumentSource(
        filename="docs/refund.MD",
        content=MARKDOWN_SAMPLE.encode(),
        declared_media_type="text/markdown; charset=utf-8",
    )

    parsed = parsing_service().parse(source)

    assert parsed.parser_name == "customer-agent2-markdown"
    assert parsed.parser_version == "1"
    assert parsed.source.filename == "refund.MD"
    assert parsed.source.document_format is DocumentFormat.MARKDOWN
    assert parsed.source.media_type == "text/markdown"
    assert parsed.source.charset == "utf-8"
    assert parsed.source.byte_size == len(source.content)
    assert parsed.source.content_sha256 == sha256(source.content).hexdigest()
    assert [block.kind for block in parsed.blocks] == [
        ParsedBlockKind.HEADING,
        ParsedBlockKind.PARAGRAPH,
        ParsedBlockKind.LIST_ITEM,
        ParsedBlockKind.LIST_ITEM,
        ParsedBlockKind.HEADING,
        ParsedBlockKind.CODE_BLOCK,
    ]
    assert parsed.blocks[0].heading_level == 1
    assert parsed.blocks[0].section_path == ("退款政策",)
    assert parsed.blocks[1].text == "客户可在 七天内 申请退款。"
    assert parsed.blocks[1].start_line == 3
    assert parsed.blocks[2].section_path == ("退款政策",)
    assert parsed.blocks[4].section_path == ("退款政策", "代码示例")
    assert parsed.blocks[5].code_language == "python"
    assert parsed.blocks[5].text == 'print("ok")'
    assert parsed.blocks[5].start_line == 10
    assert parsed.blocks[5].end_line == 12


def test_plain_text_parser_accepts_utf8_bom_and_normalizes_newlines() -> None:
    content = "\ufeff第一段第一行\r\n第一段第二行\r\n\r\n第二段".encode()
    parsed = parsing_service().parse(
        DocumentSource(
            filename="notes.txt",
            content=content,
            declared_media_type="text/plain",
        )
    )

    assert parsed.parser_name == "customer-agent2-plain-text"
    assert parsed.parser_version == "1"
    assert parsed.source.document_format is DocumentFormat.PLAIN_TEXT
    assert [block.text for block in parsed.blocks] == [
        "第一段第一行\n第一段第二行",
        "第二段",
    ]
    assert [(block.start_line, block.end_line) for block in parsed.blocks] == [(1, 2), (4, 4)]
    assert parsed.text == "第一段第一行\n第一段第二行\n\n第二段"


def test_identifier_can_use_mime_without_an_extension() -> None:
    identified = SafeTextDocumentIdentifier(max_file_size_bytes=100).identify(
        DocumentSource(filename="README", content=b"hello", declared_media_type="text/markdown")
    )

    assert identified.document_format is DocumentFormat.MARKDOWN


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (DocumentSource("empty.txt", b""), DocumentErrorCode.EMPTY_FILE),
        (DocumentSource("large.txt", b"1234"), DocumentErrorCode.FILE_TOO_LARGE),
        (DocumentSource("manual.pdf", b"x"), DocumentErrorCode.UNSUPPORTED_TYPE),
        (
            DocumentSource("notes.txt", b"x", "text/markdown"),
            DocumentErrorCode.TYPE_MISMATCH,
        ),
        (
            DocumentSource("notes.txt", b"x", "application/pdf"),
            DocumentErrorCode.TYPE_MISMATCH,
        ),
        (DocumentSource("binary.txt", b"a\x00b"), DocumentErrorCode.BINARY_CONTENT),
        (DocumentSource("control.txt", b"a\x1fb"), DocumentErrorCode.BINARY_CONTENT),
        (DocumentSource("legacy.txt", b"\xff\xfe"), DocumentErrorCode.INVALID_ENCODING),
    ],
)
def test_identifier_rejects_invalid_or_unsafe_sources(
    source: DocumentSource,
    expected_code: DocumentErrorCode,
) -> None:
    identifier = SafeTextDocumentIdentifier(max_file_size_bytes=3)

    with pytest.raises(DocumentError) as captured:
        identifier.identify(source)

    assert captured.value.code is expected_code
    assert captured.value.public_message == str(captured.value)
    assert source.filename not in captured.value.public_message


@pytest.mark.parametrize(
    "source",
    [
        DocumentSource("blank.md", b"  \n\t"),
        DocumentSource("blank.txt", b"  \n\t"),
    ],
)
def test_parsers_reject_documents_without_meaningful_text(source: DocumentSource) -> None:
    with pytest.raises(DocumentError) as captured:
        parsing_service().parse(source)

    assert captured.value.code is DocumentErrorCode.EMPTY_CONTENT


def test_parser_service_reports_missing_and_duplicate_parser_configuration() -> None:
    identifier = SafeTextDocumentIdentifier(max_file_size_bytes=100)
    markdown_parser = MarkdownDocumentParser()
    markdown_only = DocumentParsingService(identifier, (markdown_parser,))

    with pytest.raises(DocumentError) as missing:
        markdown_only.parse(DocumentSource("notes.txt", b"hello"))
    assert missing.value.code is DocumentErrorCode.PARSER_NOT_CONFIGURED

    with pytest.raises(ValueError, match="重复注册"):
        DocumentParsingService(identifier, (markdown_parser, MarkdownDocumentParser()))


def test_parser_service_exposes_supported_formats_deterministically() -> None:
    assert parsing_service().supported_formats == (
        DocumentFormat.MARKDOWN,
        DocumentFormat.PLAIN_TEXT,
    )
