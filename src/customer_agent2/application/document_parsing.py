"""Application service that identifies documents and selects a parser explicitly."""

from collections.abc import Iterable

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    DocumentIdentifier,
    DocumentParser,
    DocumentSource,
    ParsedDocument,
)


class DocumentParsingService:
    """Compose type identification with a closed, injected parser registry."""

    def __init__(
        self,
        identifier: DocumentIdentifier,
        parsers: Iterable[DocumentParser],
    ) -> None:
        parser_registry: dict[DocumentFormat, DocumentParser] = {}
        for parser in parsers:
            if parser.document_format in parser_registry:
                raise ValueError(f"文档解析器重复注册: {parser.document_format.value}")
            parser_registry[parser.document_format] = parser
        if not parser_registry:
            raise ValueError("至少需要注册一个文档解析器")

        self._identifier = identifier
        self._parsers = parser_registry

    @property
    def supported_formats(self) -> tuple[DocumentFormat, ...]:
        """Return registered formats in deterministic value order."""
        return tuple(sorted(self._parsers, key=lambda item: item.value))

    def parse(self, source: DocumentSource) -> ParsedDocument:
        """Identify the source and dispatch it to exactly one configured parser."""
        identified = self._identifier.identify(source)
        parser = self._parsers.get(identified.document_format)
        if parser is None:
            raise DocumentError(
                DocumentErrorCode.PARSER_NOT_CONFIGURED,
                "该文档类型尚未配置解析器",
            )
        return parser.parse(identified)
