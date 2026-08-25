"""Opt-in offline smoke test for the pinned real BGE tokenizer and chunker."""

import os

import pytest

from customer_agent2.application import DocumentParsingService, StructureAwareDocumentChunker
from customer_agent2.domain.models import ChunkingPolicy, DocumentSource
from customer_agent2.infrastructure.documents import (
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
    TransformersTextTokenCodec,
)
from tests.settings import IsolatedSettings


@pytest.mark.model_smoke
@pytest.mark.skipif(
    os.getenv("RUN_LOCAL_MODEL_SMOKE") != "1",
    reason="set RUN_LOCAL_MODEL_SMOKE=1 to run cached tokenizer inference",
)
def test_real_bge_tokenizer_enforces_confirmed_chunk_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    settings = IsolatedSettings()
    parser = DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024,
            max_extracted_chars=settings.document_max_extracted_chars,
        ),
        (PlainTextDocumentParser(),),
    )
    parsed = parser.parse(DocumentSource("long.txt", ("客户退款条件与处理流程。" * 200).encode()))
    tokenizer = TransformersTextTokenCodec.from_settings(settings)
    result = StructureAwareDocumentChunker(
        tokenizer,
        ChunkingPolicy(settings.chunk_target_tokens, settings.chunk_overlap_tokens),
    ).chunk(parsed)

    assert result.tokenizer_model_id == settings.local_embedding_model
    assert result.tokenizer_revision == settings.local_embedding_revision
    assert len(result.chunks) > 1
    assert all(0 < chunk.token_count <= 400 for chunk in result.chunks)
    assert result.chunks[0].overlap_with_previous_tokens == 0
    assert all(chunk.overlap_with_previous_tokens == 64 for chunk in result.chunks[1:])
