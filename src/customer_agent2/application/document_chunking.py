"""Structure-aware document chunking with exact token-budget fallback windows."""

from hashlib import sha256

from customer_agent2.domain.models import (
    ChunkDraft,
    ChunkingError,
    ChunkingErrorCode,
    ChunkingPolicy,
    ChunkingResult,
    ParsedBlock,
    ParsedBlockKind,
    ParsedDocument,
    TextTokenCodec,
)


class StructureAwareDocumentChunker:
    """Prefer source blocks and headings, then split oversized text by tokens."""

    def __init__(self, tokenizer: TextTokenCodec, policy: ChunkingPolicy) -> None:
        if not tokenizer.model_id.strip() or not tokenizer.revision.strip():
            raise ValueError("分词器模型 ID 和 revision 不能为空")
        self._tokenizer = tokenizer
        self._policy = policy

    def chunk(self, document: ParsedDocument) -> ChunkingResult:
        """Create deterministic source-aware drafts under the configured budget."""
        chunks: list[ChunkDraft] = []
        pending: list[ParsedBlock] = []

        for block in document.blocks:
            starts_new_structure = block.kind is ParsedBlockKind.HEADING or (
                bool(pending) and block.section_path != pending[-1].section_path
            )
            if starts_new_structure:
                self._emit_group(tuple(pending), chunks)
                pending = []

            if not pending:
                if self._count_tokens(block.text) <= self._policy.target_tokens:
                    pending = [block]
                else:
                    self._emit_group((block,), chunks)
                continue

            candidate = (*pending, block)
            if self._count_tokens(_group_text(candidate)) <= self._policy.target_tokens:
                pending.append(block)
                continue

            if len(pending) == 1 and pending[0].kind is ParsedBlockKind.HEADING:
                self._emit_group(candidate, chunks)
                pending = []
                continue

            self._emit_group(tuple(pending), chunks)
            if self._count_tokens(block.text) <= self._policy.target_tokens:
                pending = [block]
            else:
                self._emit_group((block,), chunks)
                pending = []

        self._emit_group(tuple(pending), chunks)
        return ChunkingResult(
            source=document,
            chunks=tuple(chunks),
            tokenizer_model_id=self._tokenizer.model_id,
            tokenizer_revision=self._tokenizer.revision,
            policy=self._policy,
        )

    def _emit_group(
        self,
        blocks: tuple[ParsedBlock, ...],
        chunks: list[ChunkDraft],
    ) -> None:
        if not blocks:
            return
        text = _group_text(blocks)
        token_ids = self._encode(text)
        if len(token_ids) <= self._policy.target_tokens:
            chunks.append(
                _chunk_draft(
                    blocks,
                    content=text,
                    token_count=len(token_ids),
                    chunk_index=len(chunks),
                    overlap_tokens=0,
                )
            )
            return
        self._emit_token_windows(blocks, token_ids, chunks)

    def _emit_token_windows(
        self,
        blocks: tuple[ParsedBlock, ...],
        token_ids: tuple[int, ...],
        chunks: list[ChunkDraft],
    ) -> None:
        start = 0
        previous_end = 0
        token_total = len(token_ids)

        while start < token_total:
            end = min(start + self._policy.target_tokens, token_total)
            content, actual_token_count, end = self._decode_budgeted_window(token_ids, start, end)
            overlap_tokens = 0 if previous_end == 0 else max(0, previous_end - start)
            chunks.append(
                _chunk_draft(
                    blocks,
                    content=content,
                    token_count=actual_token_count,
                    chunk_index=len(chunks),
                    overlap_tokens=overlap_tokens,
                )
            )
            if end >= token_total:
                break

            previous_end = end
            start = max(start + 1, end - self._policy.overlap_tokens)

    def _decode_budgeted_window(
        self,
        token_ids: tuple[int, ...],
        start: int,
        initial_end: int,
    ) -> tuple[str, int, int]:
        end = initial_end
        while end > start:
            content = self._tokenizer.decode(token_ids[start:end]).strip()
            if content:
                actual_token_count = len(self._encode(content))
                if actual_token_count <= self._policy.target_tokens:
                    return content, actual_token_count, end
                end -= max(1, actual_token_count - self._policy.target_tokens)
                continue
            end -= 1
        raise ChunkingError(
            ChunkingErrorCode.TOKENIZER_PROTOCOL,
            "分词器未能生成有效的 Chunk 文本",
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._encode(text))

    def _encode(self, text: str) -> tuple[int, ...]:
        token_ids = self._tokenizer.encode(text)
        if not token_ids:
            raise ChunkingError(
                ChunkingErrorCode.TOKENIZER_PROTOCOL,
                "分词器未能生成有效 Token",
            )
        return token_ids


def _group_text(blocks: tuple[ParsedBlock, ...]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _chunk_draft(
    blocks: tuple[ParsedBlock, ...],
    *,
    content: str,
    token_count: int,
    chunk_index: int,
    overlap_tokens: int,
) -> ChunkDraft:
    section_path = blocks[0].section_path
    if any(block.section_path != section_path for block in blocks):
        raise ValueError("一个结构分组不能跨越不同章节")
    return ChunkDraft(
        chunk_index=chunk_index,
        content=content,
        token_count=token_count,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
        block_start_ordinal=blocks[0].ordinal,
        block_end_ordinal=blocks[-1].ordinal,
        start_line=min(block.start_line for block in blocks),
        end_line=max(block.end_line for block in blocks),
        section_path=section_path,
        overlap_with_previous_tokens=overlap_tokens,
    )
