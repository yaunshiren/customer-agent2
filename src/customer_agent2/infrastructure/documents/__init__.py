"""Concrete document identification and parsing adapters."""

from customer_agent2.infrastructure.documents.csv_parser import CsvDocumentParser
from customer_agent2.infrastructure.documents.docx_parser import DocxDocumentParser
from customer_agent2.infrastructure.documents.identifier import SafeDocumentIdentifier
from customer_agent2.infrastructure.documents.markdown_parser import MarkdownDocumentParser
from customer_agent2.infrastructure.documents.pdf_parser import PdfDocumentParser
from customer_agent2.infrastructure.documents.text_parser import PlainTextDocumentParser
from customer_agent2.infrastructure.documents.transformers_tokenizer import (
    TransformersTextTokenCodec,
)

__all__ = [
    "CsvDocumentParser",
    "DocxDocumentParser",
    "MarkdownDocumentParser",
    "PdfDocumentParser",
    "PlainTextDocumentParser",
    "SafeDocumentIdentifier",
    "TransformersTextTokenCodec",
]
