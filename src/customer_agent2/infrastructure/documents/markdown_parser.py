"""CommonMark adapter that preserves source-aware structural blocks."""

from markdown_it import MarkdownIt
from markdown_it.token import Token

from customer_agent2.domain.models import (
    DocumentError,
    DocumentErrorCode,
    DocumentFormat,
    IdentifiedDocument,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
)


class MarkdownDocumentParser:
    """Parse CommonMark into headings, paragraphs, list items, and code blocks."""

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark", {"html": False})

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.MARKDOWN

    @property
    def parser_name(self) -> str:
        """Return the stable parser identity persisted with document versions."""
        return "customer-agent2-markdown"

    @property
    def parser_version(self) -> str:
        """Return the application parser contract version."""
        return "1"

    def parse(self, document: IdentifiedDocument) -> ParsedDocument:
        """Convert a validated Markdown document into ordered structural blocks."""
        if document.document_format is not self.document_format:
            raise DocumentError(
                DocumentErrorCode.TYPE_MISMATCH,
                "文档类型与 Markdown 解析器不匹配",
            )

        text = document.text
        if text is None:
            raise DocumentError(DocumentErrorCode.TYPE_MISMATCH, "Markdown 文档缺少文本内容")
        tokens = self._markdown.parse(text)
        blocks: list[ParsedBlock] = []
        headings: list[tuple[int, str]] = []
        list_item_depth = 0

        for index, token in enumerate(tokens):
            if token.type == "list_item_open":
                list_item_depth += 1
                continue
            if token.type == "list_item_close":
                list_item_depth = max(0, list_item_depth - 1)
                continue
            if token.type == "heading_open":
                inline = _following_inline(tokens, index)
                text = _render_inline(inline)
                if not text:
                    continue
                heading_level = int(token.tag.removeprefix("h"))
                headings = [item for item in headings if item[0] < heading_level]
                headings.append((heading_level, text))
                start_line, end_line = _line_range(token)
                blocks.append(
                    ParsedBlock(
                        kind=ParsedBlockKind.HEADING,
                        text=text,
                        ordinal=len(blocks),
                        start_line=start_line,
                        end_line=end_line,
                        section_path=_section_path(headings),
                        heading_level=heading_level,
                    )
                )
                continue
            if token.type == "paragraph_open":
                inline = _following_inline(tokens, index)
                text = _render_inline(inline)
                if not text:
                    continue
                start_line, end_line = _line_range(token)
                kind = (
                    ParsedBlockKind.LIST_ITEM if list_item_depth > 0 else ParsedBlockKind.PARAGRAPH
                )
                blocks.append(
                    ParsedBlock(
                        kind=kind,
                        text=text,
                        ordinal=len(blocks),
                        start_line=start_line,
                        end_line=end_line,
                        section_path=_section_path(headings),
                    )
                )
                continue
            if token.type in {"fence", "code_block"}:
                text = token.content.strip()
                if not text:
                    continue
                start_line, end_line = _line_range(token)
                language = token.info.split(maxsplit=1)[0] if token.info.strip() else None
                blocks.append(
                    ParsedBlock(
                        kind=ParsedBlockKind.CODE_BLOCK,
                        text=text,
                        ordinal=len(blocks),
                        start_line=start_line,
                        end_line=end_line,
                        section_path=_section_path(headings),
                        code_language=language,
                    )
                )

        if not blocks:
            raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "文档没有可解析的文本内容")
        return ParsedDocument(
            source=document,
            blocks=tuple(blocks),
            parser_name=self.parser_name,
            parser_version=self.parser_version,
        )


def _following_inline(tokens: list[Token], index: int) -> Token:
    next_index = index + 1
    if next_index >= len(tokens) or tokens[next_index].type != "inline":
        raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "文档结构无法解析")
    return tokens[next_index]


def _render_inline(token: Token) -> str:
    if token.children is None:
        return token.content.strip()

    parts: list[str] = []
    for child in token.children:
        if child.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif child.content:
            parts.append(child.content)
    return "".join(parts).strip()


def _line_range(token: Token) -> tuple[int, int]:
    if token.map is None:
        raise DocumentError(DocumentErrorCode.EMPTY_CONTENT, "文档缺少来源行号")
    return token.map[0] + 1, token.map[1]


def _section_path(headings: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(text for _, text in headings)
