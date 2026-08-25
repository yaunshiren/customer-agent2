"""Structure-aware chunking and tokenizer adapter tests."""

from collections.abc import Sequence
from hashlib import sha256

import pytest

from customer_agent2.application import DocumentParsingService, StructureAwareDocumentChunker
from customer_agent2.domain.models import (
    ChunkingError,
    ChunkingErrorCode,
    ChunkingPolicy,
    DocumentSource,
    ParsedDocument,
)
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeTextDocumentIdentifier,
    TransformersTextTokenCodec,
)
from tests.settings import IsolatedSettings


class CharacterTokenCodec:
    """Deterministic reversible codec that treats each Unicode character as one token."""

    @property
    def model_id(self) -> str:
        return "character-tokenizer"

    @property
    def revision(self) -> str:
        return "test-revision"

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class EmptyTokenCodec(CharacterTokenCodec):
    """Broken codec used to verify stable protocol failures."""

    def encode(self, text: str) -> tuple[int, ...]:
        return ()


class FakeTransformersBackend:
    """Record the exact provider options used by the concrete adapter."""

    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool]] = []
        self.decode_calls: list[tuple[tuple[int, ...], bool, bool]] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.encode_calls.append((text, add_special_tokens))
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        captured_ids = tuple(token_ids)
        self.decode_calls.append((captured_ids, skip_special_tokens, clean_up_tokenization_spaces))
        return "".join(chr(token_id) for token_id in captured_ids)


def parse_document(filename: str, content: str) -> ParsedDocument:
    """Parse one supported text document for chunking tests."""
    service = DocumentParsingService(
        SafeTextDocumentIdentifier(max_file_size_bytes=1024 * 1024),
        (MarkdownDocumentParser(), PlainTextDocumentParser()),
    )
    return service.parse(DocumentSource(filename, content.encode()))


def test_chunker_keeps_heading_sections_separate_even_when_they_fit() -> None:
    parsed = parse_document(
        "sections.md",
        "# 第一节\n\n第一段。\n\n## 第二节\n\n第二段。",
    )
    chunker = StructureAwareDocumentChunker(
        CharacterTokenCodec(),
        ChunkingPolicy(target_tokens=100, overlap_tokens=10),
    )

    result = chunker.chunk(parsed)

    assert [chunk.content for chunk in result.chunks] == [
        "第一节\n\n第一段。",
        "第二节\n\n第二段。",
    ]
    assert result.chunks[0].section_path == ("第一节",)
    assert result.chunks[1].section_path == ("第一节", "第二节")
    assert [(chunk.block_start_ordinal, chunk.block_end_ordinal) for chunk in result.chunks] == [
        (0, 1),
        (2, 3),
    ]
    assert all(chunk.overlap_with_previous_tokens == 0 for chunk in result.chunks)


def test_chunker_merges_adjacent_plain_text_blocks_within_budget() -> None:
    parsed = parse_document("notes.txt", "第一段\n\n第二段")
    result = StructureAwareDocumentChunker(
        CharacterTokenCodec(),
        ChunkingPolicy(target_tokens=20, overlap_tokens=3),
    ).chunk(parsed)

    assert len(result.chunks) == 1
    assert result.chunks[0].content == "第一段\n\n第二段"
    assert result.chunks[0].start_line == 1
    assert result.chunks[0].end_line == 3


def test_oversized_block_uses_exact_overlapping_token_windows() -> None:
    parsed = parse_document("long.txt", "abcdefghijklmnopqrstuv")
    result = StructureAwareDocumentChunker(
        CharacterTokenCodec(),
        ChunkingPolicy(target_tokens=10, overlap_tokens=3),
    ).chunk(parsed)

    assert [chunk.content for chunk in result.chunks] == [
        "abcdefghij",
        "hijklmnopq",
        "opqrstuv",
    ]
    assert [chunk.token_count for chunk in result.chunks] == [10, 10, 8]
    assert [chunk.overlap_with_previous_tokens for chunk in result.chunks] == [0, 3, 3]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1, 2]
    assert all(chunk.block_start_ordinal == 0 for chunk in result.chunks)
    assert all(chunk.block_end_ordinal == 0 for chunk in result.chunks)
    assert result.chunks[1].content_sha256 == sha256(b"hijklmnopq").hexdigest()
    assert result.tokenizer_model_id == "character-tokenizer"
    assert result.tokenizer_revision == "test-revision"


def test_oversized_first_section_does_not_emit_a_heading_only_chunk() -> None:
    parsed = parse_document("heading.md", "# H\n\nabcdefghijklmnopqrstuv")
    result = StructureAwareDocumentChunker(
        CharacterTokenCodec(),
        ChunkingPolicy(target_tokens=10, overlap_tokens=3),
    ).chunk(parsed)

    assert len(result.chunks) > 1
    assert result.chunks[0].content == "H\n\nabcdefg"
    assert all(chunk.content != "H" for chunk in result.chunks)
    assert all(chunk.section_path == ("H",) for chunk in result.chunks)


def test_chunking_policy_and_broken_codec_fail_fast() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        ChunkingPolicy(target_tokens=10, overlap_tokens=10)

    parsed = parse_document("notes.txt", "有效内容")
    chunker = StructureAwareDocumentChunker(
        EmptyTokenCodec(),
        ChunkingPolicy(target_tokens=10, overlap_tokens=2),
    )
    with pytest.raises(ChunkingError) as captured:
        chunker.chunk(parsed)
    assert captured.value.code is ChunkingErrorCode.TOKENIZER_PROTOCOL


def test_transformers_codec_is_lazy_pinned_and_disables_special_tokens() -> None:
    backend = FakeTransformersBackend()
    load_calls: list[tuple[str, str]] = []

    def factory(model_id: str, revision: str) -> FakeTransformersBackend:
        load_calls.append((model_id, revision))
        return backend

    settings = IsolatedSettings(
        local_embedding_model="test-model",
        local_embedding_revision="test-revision",
        local_embedding_dimension=3,
        local_embedding_max_tokens=20,
        chunk_target_tokens=10,
        chunk_overlap_tokens=2,
    )
    codec = TransformersTextTokenCodec.from_settings(settings, backend_factory=factory)

    assert load_calls == []
    encoded = codec.encode("测试")
    decoded = codec.decode(encoded)

    assert codec.model_id == "test-model"
    assert codec.revision == "test-revision"
    assert load_calls == [("test-model", "test-revision")]
    assert backend.encode_calls == [("测试", False)]
    assert backend.decode_calls == [(encoded, True, False)]
    assert decoded == "测试"


def test_transformers_codec_sanitizes_model_load_failure() -> None:
    sensitive_detail = "C:/private/model/cache/tokenizer.json"

    def failing_factory(model_id: str, revision: str) -> FakeTransformersBackend:
        raise OSError(f"{sensitive_detail}: {model_id}@{revision}")

    codec = TransformersTextTokenCodec(
        model_id="test-model",
        revision="test-revision",
        backend_factory=failing_factory,
    )

    with pytest.raises(ChunkingError) as captured:
        codec.encode("测试")

    assert captured.value.code is ChunkingErrorCode.TOKENIZER_UNAVAILABLE
    assert sensitive_detail not in captured.value.public_message
    assert sensitive_detail not in str(captured.value)
