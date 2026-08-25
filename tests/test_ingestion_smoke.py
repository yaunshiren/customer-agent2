"""Opt-in end-to-end smoke using cached BGE weights and real pgvector."""

import math
import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from customer_agent2.application import (
    DocumentIngestionService,
    DocumentParsingService,
    StructureAwareDocumentChunker,
)
from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingPolicy, DocumentIngestionRequest, DocumentSource
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    MarkdownDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
    TransformersTextTokenCodec,
)
from customer_agent2.infrastructure.models import SentenceTransformerEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    ChunkRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyIngestionRepository,
)


@pytest.mark.ingestion_smoke
@pytest.mark.skipif(
    os.getenv("RUN_INGESTION_SMOKE") != "1",
    reason="set RUN_INGESTION_SMOKE=1 to use cached BGE weights and local PostgreSQL",
)
@pytest.mark.asyncio
async def test_real_bge_ingestion_activates_normalized_pgvector_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    settings = Settings()
    manager = DatabaseManager(settings)
    await manager.open()
    knowledge_base_id = uuid4()

    try:
        async with manager.session_factory.begin() as session:
            session.add(
                KnowledgeBaseRecord(
                    id=knowledge_base_id,
                    slug=f"real-ingestion-{knowledge_base_id}",
                    name="Real M2-D ingestion smoke",
                    embedding_model_id=settings.local_embedding_model,
                    embedding_model_revision=settings.local_embedding_revision,
                    embedding_dimension=settings.local_embedding_dimension,
                    embedding_normalized=settings.embedding_normalize,
                )
            )

        parser = DocumentParsingService(
            SafeDocumentIdentifier(
                max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            (MarkdownDocumentParser(), PlainTextDocumentParser()),
        )
        chunker = StructureAwareDocumentChunker(
            TransformersTextTokenCodec.from_settings(settings),
            ChunkingPolicy(settings.chunk_target_tokens, settings.chunk_overlap_tokens),
        )
        service = DocumentIngestionService(
            parser,
            chunker,
            SentenceTransformerEmbeddingModel.from_settings(settings),
            SQLAlchemyIngestionRepository(manager.session_factory),
        )

        result = await service.ingest(
            DocumentIngestionRequest(
                knowledge_base_id,
                "smoke/refund.md",
                DocumentSource(
                    "refund.md",
                    "# 退款条件\n\n客户可在七天内申请退款。\n\n## 材料\n\n请保留订单号。".encode(),
                    "text/markdown",
                ),
            )
        )

        async with manager.session_factory() as session:
            version = await session.get(DocumentVersionRecord, result.version_id)
            chunks = list(
                await session.scalars(
                    select(ChunkRecord)
                    .where(ChunkRecord.document_version_id == result.version_id)
                    .order_by(ChunkRecord.chunk_index)
                )
            )

        assert version is not None
        assert version.status == "active"
        assert version.parser_name == "customer-agent2-markdown"
        assert len(chunks) == result.chunk_count == 2
        assert all(len(chunk.embedding) == settings.local_embedding_dimension for chunk in chunks)
        assert all(
            math.isclose(
                math.sqrt(math.fsum(value * value for value in chunk.embedding)),
                1.0,
                rel_tol=1e-3,
                abs_tol=1e-3,
            )
            for chunk in chunks
        )
    finally:
        try:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(KnowledgeBaseRecord).where(KnowledgeBaseRecord.id == knowledge_base_id)
                )
        finally:
            await manager.close()
