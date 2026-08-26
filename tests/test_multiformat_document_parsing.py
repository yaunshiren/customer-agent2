"""PDF, DOCX, and CSV parsing and hostile-input boundary tests."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from pypdf import PdfWriter

from customer_agent2.application import DocumentParsingService
from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    DocumentSource,
    ParsedBlockKind,
)
from customer_agent2.infrastructure.documents import (
    CsvDocumentParser,
    DocxDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
)
from tests.document_samples import build_docx_bytes, build_pdf_bytes


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


def multiformat_service(
    *,
    max_pdf_pages: int = 10,
    max_docx_entries: int = 100,
    max_docx_ratio: int = 100,
    max_csv_rows: int = 100,
    max_csv_columns: int = 20,
    max_extracted_chars: int = 100_000,
) -> DocumentParsingService:
    """Build all P0 parsers with deliberately small test limits."""
    return DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=1024 * 1024,
            max_extracted_chars=max_extracted_chars,
        ),
        (
            CsvDocumentParser(
                max_rows=max_csv_rows,
                max_columns=max_csv_columns,
                max_extracted_chars=max_extracted_chars,
            ),
            DocxDocumentParser(
                max_archive_entries=max_docx_entries,
                max_uncompressed_bytes=2 * 1024 * 1024,
                max_expansion_ratio=max_docx_ratio,
                max_extracted_chars=max_extracted_chars,
            ),
            MarkdownDocumentParser(),
            PdfDocumentParser(
                max_pages=max_pdf_pages,
                max_extracted_chars=max_extracted_chars,
            ),
            PlainTextDocumentParser(),
        ),
    )


def test_pdf_parser_preserves_page_source_and_binary_identity() -> None:
    content = build_pdf_bytes("Refund policy")
    parsed = multiformat_service().parse(DocumentSource("refund.pdf", content, "application/pdf"))

    assert parsed.parser_name == "customer-agent2-pypdf"
    assert parsed.source.document_format is DocumentFormat.PDF
    assert parsed.source.charset is None
    assert parsed.source.text is None
    assert parsed.source.content == content
    assert parsed.blocks[0].text == "Refund policy"
    assert parsed.blocks[0].section_path == ("第 1 页",)
    assert parsed.blocks[0].page_number == 1


def test_docx_parser_preserves_headings_lists_and_table_records() -> None:
    parsed = multiformat_service().parse(
        DocumentSource(
            "refund.docx",
            build_docx_bytes(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    )

    assert parsed.parser_name == "customer-agent2-python-docx"
    assert parsed.source.document_format is DocumentFormat.DOCX
    assert [block.kind for block in parsed.blocks[:3]] == [
        ParsedBlockKind.HEADING,
        ParsedBlockKind.PARAGRAPH,
        ParsedBlockKind.LIST_ITEM,
    ]
    assert parsed.blocks[1].section_path == ("Refund policy",)
    assert parsed.blocks[-1].section_path == ("Refund policy", "表格 1")
    assert parsed.blocks[-1].text == "Region: CN\nWindow: 7 days"


def test_csv_parser_repeats_headers_and_tracks_multiline_record_lines() -> None:
    content = '\ufeffname,policy\r\nAlice,"first line\r\nsecond line"\r\n'.encode()
    parsed = multiformat_service().parse(DocumentSource("refund.csv", content, "text/csv"))

    assert parsed.parser_name == "customer-agent2-csv"
    assert parsed.source.document_format is DocumentFormat.CSV
    assert parsed.blocks[0].text == "name: Alice\npolicy: first line\nsecond line"
    assert (parsed.blocks[0].start_line, parsed.blocks[0].end_line) == (2, 3)
    assert parsed.blocks[0].section_path == ("CSV 记录",)


def test_identifier_accepts_binary_mime_without_extension_and_rejects_mismatch() -> None:
    identifier = SafeDocumentIdentifier(
        max_file_size_bytes=1024 * 1024,
        max_extracted_chars=100_000,
    )
    identified = identifier.identify(
        DocumentSource("download", build_pdf_bytes(), "application/pdf")
    )
    assert identified.document_format is DocumentFormat.PDF

    with pytest.raises(DocumentError) as mismatch:
        identifier.identify(
            DocumentSource(
                "refund.pdf",
                build_pdf_bytes(),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )
    assert mismatch.value.code is DocumentErrorCode.TYPE_MISMATCH


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            DocumentSource("fake.pdf", b"not a pdf", "application/pdf"),
            DocumentErrorCode.INVALID_FILE_SIGNATURE,
        ),
        (
            DocumentSource("fake.docx", b"not a zip"),
            DocumentErrorCode.INVALID_FILE_SIGNATURE,
        ),
        (
            DocumentSource("broken.pdf", b"%PDF-1.7\nbroken"),
            DocumentErrorCode.MALFORMED_DOCUMENT,
        ),
        (
            DocumentSource("broken.docx", _zip_bytes({"only.txt": b"x"})),
            DocumentErrorCode.MALFORMED_DOCUMENT,
        ),
        (
            DocumentSource(
                "unsafe.docx",
                _zip_bytes(
                    {
                        "[Content_Types].xml": b"x",
                        "word/document.xml": b"y",
                        "..\\outside.xml": b"z",
                    }
                ),
            ),
            DocumentErrorCode.MALFORMED_DOCUMENT,
        ),
        (
            DocumentSource(
                "macro.docx",
                _zip_bytes(
                    {
                        "[Content_Types].xml": b"x",
                        "word/document.xml": b"y",
                        "word/VBAProject.bin": b"z",
                    }
                ),
            ),
            DocumentErrorCode.MALFORMED_DOCUMENT,
        ),
    ],
)
def test_binary_parsers_reject_spoofed_or_malformed_files(
    source: DocumentSource,
    expected_code: DocumentErrorCode,
) -> None:
    with pytest.raises(DocumentError) as captured:
        multiformat_service().parse(source)
    assert captured.value.code is expected_code


def test_pdf_parser_rejects_encryption_and_page_limit() -> None:
    encrypted = PdfWriter()
    encrypted.add_blank_page(width=100, height=100)
    encrypted.encrypt("password")
    encrypted_output = BytesIO()
    encrypted.write(encrypted_output)

    with pytest.raises(DocumentError) as encrypted_error:
        multiformat_service().parse(DocumentSource("encrypted.pdf", encrypted_output.getvalue()))
    assert encrypted_error.value.code is DocumentErrorCode.ENCRYPTED_DOCUMENT

    two_pages = PdfWriter()
    two_pages.add_blank_page(width=100, height=100)
    two_pages.add_blank_page(width=100, height=100)
    page_output = BytesIO()
    two_pages.write(page_output)
    with pytest.raises(DocumentError) as page_error:
        multiformat_service(max_pdf_pages=1).parse(
            DocumentSource("too-many-pages.pdf", page_output.getvalue())
        )
    assert page_error.value.code is DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED


def test_docx_parser_rejects_archive_expansion_limit() -> None:
    compressed = _zip_bytes(
        {
            "[Content_Types].xml": b"x" * 20_000,
            "word/document.xml": b"y" * 20_000,
        }
    )

    with pytest.raises(DocumentError) as captured:
        multiformat_service(max_docx_ratio=2).parse(DocumentSource("expanded.docx", compressed))
    assert captured.value.code is DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED


@pytest.mark.parametrize(
    ("content", "expected_code", "row_limit"),
    [
        (b"a,b\n1\n", DocumentErrorCode.MALFORMED_DOCUMENT, 100),
        (b"a,b\n1,2\n3,4\n", DocumentErrorCode.RESOURCE_LIMIT_EXCEEDED, 1),
        (b"a,b\n", DocumentErrorCode.EMPTY_CONTENT, 100),
    ],
)
def test_csv_parser_rejects_bad_shape_limits_and_header_only(
    content: bytes,
    expected_code: DocumentErrorCode,
    row_limit: int,
) -> None:
    with pytest.raises(DocumentError) as captured:
        multiformat_service(max_csv_rows=row_limit).parse(DocumentSource("records.csv", content))
    assert captured.value.code is expected_code


def test_multiformat_service_exposes_all_p0_formats() -> None:
    assert multiformat_service().supported_formats == (
        DocumentFormat.CSV,
        DocumentFormat.DOCX,
        DocumentFormat.MARKDOWN,
        DocumentFormat.PDF,
        DocumentFormat.PLAIN_TEXT,
    )
