"""Opt-in real PostgreSQL/Redis HTTP integration for public APIs."""

import json
import os
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import delete, func, select

from customer_agent2.application import (
    BasicRagPromptBuilder,
    BasicStreamingRagPipeline,
    ConversationSummaryService,
    DocumentIngestionService,
    DocumentManagementService,
    DocumentParsingService,
    FastModelIntentClassifier,
    FastModelQueryRewriter,
    MemoryAwareStreamingRagPipeline,
    PersistentStreamingRagPipeline,
    RetrievalPostProcessor,
    StructureAwareDocumentChunker,
    SummarizingStreamingRagPipeline,
    VectorRetrievalService,
)
from customer_agent2.application.services import ApplicationServices
from customer_agent2.config import Settings
from customer_agent2.domain.models import ChunkingPolicy, EmbeddingIndexConfiguration
from customer_agent2.infrastructure import ApplicationResources
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.documents import (
    CsvDocumentParser,
    DocxDocumentParser,
    MarkdownDocumentParser,
    PdfDocumentParser,
    PlainTextDocumentParser,
    SafeDocumentIdentifier,
)
from customer_agent2.infrastructure.intents import load_default_intent_tree
from customer_agent2.infrastructure.models import (
    FakeChatModel,
    FakeEmbeddingModel,
    NoOpRerankModel,
)
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    ConversationRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    MessageRecord,
    RagRunRecord,
    SQLAlchemyConversationMemoryRepository,
    SQLAlchemyDocumentManagementRepository,
    SQLAlchemyIngestionRepository,
    SQLAlchemyRagRunRepository,
    SQLAlchemyVectorSearchRepository,
)
from customer_agent2.main import create_app
from tests.document_samples import CSV_SAMPLE, build_docx_bytes, build_pdf_bytes

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use PostgreSQL and Redis containers",
    ),
]

MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"


class CharacterTokenCodec:
    @property
    def model_id(self) -> str:
        return MODEL_ID

    @property
    def revision(self) -> str:
        return MODEL_REVISION

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(ord(character) for character in text)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def fake_model_services(
    settings: Settings,
    resources: ApplicationResources,
) -> ApplicationServices:
    database = resources.database
    assert isinstance(database, DatabaseManager)
    embedding = FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
    )
    parser = DocumentParsingService(
        SafeDocumentIdentifier(
            max_file_size_bytes=settings.upload_max_file_mb * 1024 * 1024,
            max_extracted_chars=settings.document_max_extracted_chars,
        ),
        (
            CsvDocumentParser(
                max_rows=settings.document_max_csv_rows,
                max_columns=settings.document_max_csv_columns,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            DocxDocumentParser(
                max_archive_entries=settings.document_max_docx_entries,
                max_uncompressed_bytes=(settings.document_max_docx_uncompressed_mb * 1024 * 1024),
                max_expansion_ratio=settings.document_max_docx_expansion_ratio,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            MarkdownDocumentParser(),
            PdfDocumentParser(
                max_pages=settings.document_max_pdf_pages,
                max_extracted_chars=settings.document_max_extracted_chars,
            ),
            PlainTextDocumentParser(),
        ),
    )
    ingestion_repository = SQLAlchemyIngestionRepository(database.session_factory)
    management_repository = SQLAlchemyDocumentManagementRepository(database.session_factory)
    retrieval_repository = SQLAlchemyVectorSearchRepository(database.session_factory)
    retrieval = VectorRetrievalService(
        embedding,
        retrieval_repository,
        recall_budget=settings.retrieval_recall_budget,
        hnsw_ef_search=settings.retrieval_hnsw_ef_search,
    )
    memory_repository = SQLAlchemyConversationMemoryRepository(database.session_factory)
    rag = SummarizingStreamingRagPipeline(
        PersistentStreamingRagPipeline(
            MemoryAwareStreamingRagPipeline(
                BasicStreamingRagPipeline(
                    retrieval,
                    BasicRagPromptBuilder(context_top_k=settings.retrieval_context_top_k),
                    FakeChatModel(
                        "fake-final",
                        "请按退款说明提交订单号 [1]。",
                        stream_chunks=("请按退款说明", "提交订单号 [1]。"),
                    ),
                    FastModelQueryRewriter(
                        FakeChatModel(
                            "fake-fast-rewrite",
                            (
                                '{"rewritten_question":"退款要求是什么?",'
                                '"sub_questions":["退款要求是什么?"]}'
                            ),
                        ),
                        timeout_seconds=settings.query_rewrite_timeout_seconds,
                        max_output_tokens=settings.query_rewrite_max_output_tokens,
                        max_sub_questions=settings.query_rewrite_max_sub_questions,
                    ),
                    FastModelIntentClassifier(
                        FakeChatModel(
                            "fake-fast-intent",
                            (
                                '{"scores":{"system_direct":0.02,'
                                '"knowledge_base":0.96,"clarification":0.02},'
                                '"clarification_question":null}'
                            ),
                        ),
                        load_default_intent_tree(),
                        high_confidence_threshold=(settings.intent_high_confidence_threshold),
                        ambiguity_margin=settings.intent_ambiguity_margin,
                        timeout_seconds=settings.intent_timeout_seconds,
                        max_output_tokens=settings.intent_max_output_tokens,
                    ),
                    RetrievalPostProcessor(
                        NoOpRerankModel(),
                        rrf_k=settings.retrieval_rrf_k,
                        rerank_candidate_limit=settings.retrieval_rerank_candidate_limit,
                        context_top_k=settings.retrieval_context_top_k,
                        max_chunks_per_document=(settings.retrieval_max_chunks_per_document),
                        rerank_timeout_seconds=settings.rerank_timeout_seconds,
                    ),
                    global_timeout_seconds=settings.rag_global_timeout_seconds,
                ),
                memory_repository,
                recent_turns=settings.memory_recent_turns,
            ),
            SQLAlchemyRagRunRepository(database.session_factory),
        ),
        ConversationSummaryService(
            memory_repository,
            FakeChatModel("fake-fast", "对话摘要"),
            trigger_turns=settings.memory_summary_trigger_turns,
            retain_recent_turns=settings.memory_recent_turns,
            timeout_seconds=settings.memory_summary_timeout_seconds,
            max_output_tokens=settings.memory_summary_max_output_tokens,
        ),
    )
    return ApplicationServices(
        ingestion=DocumentIngestionService(
            parser,
            StructureAwareDocumentChunker(
                CharacterTokenCodec(),
                ChunkingPolicy(target_tokens=100, overlap_tokens=10),
            ),
            embedding,
            ingestion_repository,
        ),
        documents=DocumentManagementService(
            management_repository,
            EmbeddingIndexConfiguration(
                MODEL_ID,
                MODEL_REVISION,
                EMBEDDING_DIMENSION,
                True,
            ),
        ),
        retrieval=retrieval,
        rag=rag,
    )


@pytest.mark.asyncio
async def test_real_http_api_creates_rebuilds_reports_and_deletes_document() -> None:
    settings = Settings()
    app = create_app(settings, service_factory=fake_model_services)
    slug = f"api-integration-{uuid4()}"
    knowledge_base_id: UUID | None = None

    async with app.router.lifespan_context(app):
        resources = app.state.resources
        assert isinstance(resources, ApplicationResources)
        database = resources.database
        assert isinstance(database, DatabaseManager)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                readiness = await client.get("/ready")
                assert health.status_code == 200
                assert readiness.status_code == 200
                assert readiness.json() == {
                    "status": "ready",
                    "checks": {
                        "postgresql": {"status": "ok", "version": None},
                        "pgvector": {"status": "ok", "version": "0.8.6"},
                        "redis": {"status": "ok", "version": None},
                    },
                }
                created = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "API integration"},
                )
                assert created.status_code == 201
                knowledge_base_id = UUID(created.json()["id"])
                duplicate = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "Duplicate"},
                )
                assert duplicate.status_code == 409
                assert duplicate.json()["detail"]["code"] == "knowledge_base_conflict"

                unsupported = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    files={
                        "file": (
                            "guide.xlsx",
                            b"not-an-xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    },
                )
                assert unsupported.status_code == 422
                assert unsupported.json()["detail"]["code"] == "unsupported_type"

                first = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    data={"source_key": "manual/guide.md"},
                    files={"file": ("guide.md", b"# old\n\nold content", "text/markdown")},
                )
                second = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    data={"source_key": "manual/guide.md"},
                    files={"file": ("guide.md", b"# new\n\nnew content", "text/markdown")},
                )
                assert first.status_code == second.status_code == 201
                assert first.json()["document_id"] == second.json()["document_id"]
                assert second.json()["version_number"] == 2

                document_id = UUID(second.json()["document_id"])
                loaded = await client.get(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                assert loaded.status_code == 200
                assert loaded.json()["latest_version"]["version_number"] == 2
                assert loaded.json()["latest_version"]["status"] == "active"
                assert loaded.json()["active_version_id"] == second.json()["version_id"]

                async with database.session_factory() as session:
                    statuses = list(
                        await session.scalars(
                            select(DocumentVersionRecord.status)
                            .where(DocumentVersionRecord.document_id == document_id)
                            .order_by(DocumentVersionRecord.version_number)
                        )
                    )
                assert statuses == ["superseded", "active"]

                deleted = await client.delete(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                missing = await client.get(
                    f"{settings.api_prefix}/knowledge-bases/"
                    f"{knowledge_base_id}/documents/{document_id}"
                )
                assert deleted.status_code == 204
                assert missing.status_code == 404

                async with database.session_factory() as session:
                    remaining_chunks = await session.scalar(
                        select(func.count(ChunkRecord.id)).where(
                            ChunkRecord.knowledge_base_id == knowledge_base_id
                        )
                    )
                assert remaining_chunks == 0
        finally:
            if knowledge_base_id is not None:
                async with database.session_factory.begin() as session:
                    await session.execute(
                        delete(KnowledgeBaseRecord).where(
                            KnowledgeBaseRecord.id == knowledge_base_id
                        )
                    )


@pytest.mark.asyncio
async def test_real_http_api_ingests_every_p0_document_format() -> None:
    settings = Settings()
    app = create_app(settings, service_factory=fake_model_services)
    slug = f"format-integration-{uuid4()}"
    knowledge_base_id: UUID | None = None
    samples = (
        ("guide.md", b"# Markdown\n\nRefund policy", "text/markdown", "customer-agent2-markdown"),
        ("guide.txt", b"Plain text refund policy", "text/plain", "customer-agent2-plain-text"),
        ("guide.pdf", build_pdf_bytes(), "application/pdf", "customer-agent2-pypdf"),
        (
            "guide.docx",
            build_docx_bytes(),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "customer-agent2-python-docx",
        ),
        ("guide.csv", CSV_SAMPLE, "text/csv", "customer-agent2-csv"),
    )

    async with app.router.lifespan_context(app):
        resources = app.state.resources
        assert isinstance(resources, ApplicationResources)
        database = resources.database
        assert isinstance(database, DatabaseManager)
        transport = httpx.ASGITransport(app=app)
        version_ids: list[UUID] = []
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "Format integration"},
                )
                assert created.status_code == 201
                knowledge_base_id = UUID(created.json()["id"])

                for filename, content, media_type, expected_parser in samples:
                    uploaded = await client.post(
                        f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                        files={"file": (filename, content, media_type)},
                    )
                    assert uploaded.status_code == 201
                    assert uploaded.json()["chunk_count"] >= 1
                    version_ids.append(UUID(uploaded.json()["version_id"]))
                    loaded = await client.get(
                        f"{settings.api_prefix}/knowledge-bases/"
                        f"{knowledge_base_id}/documents/{uploaded.json()['document_id']}"
                    )
                    assert loaded.status_code == 200
                    assert loaded.json()["latest_version"]["parser_name"] == expected_parser

                async with database.session_factory() as session:
                    parser_names = set(
                        await session.scalars(
                            select(DocumentVersionRecord.parser_name).where(
                                DocumentVersionRecord.id.in_(version_ids)
                            )
                        )
                    )
                    chunk_count = await session.scalar(
                        select(func.count(ChunkRecord.id)).where(
                            ChunkRecord.document_version_id.in_(version_ids)
                        )
                    )
                    pdf_page_numbers = set(
                        await session.scalars(
                            select(ChunkRecord.page_number).where(
                                ChunkRecord.document_version_id == version_ids[2]
                            )
                        )
                    )
                assert parser_names == {sample[3] for sample in samples}
                assert chunk_count is not None and chunk_count >= len(samples)
                assert pdf_page_numbers == {1}
        finally:
            if knowledge_base_id is not None:
                async with database.session_factory.begin() as session:
                    await session.execute(
                        delete(KnowledgeBaseRecord).where(
                            KnowledgeBaseRecord.id == knowledge_base_id
                        )
                    )


@pytest.mark.asyncio
async def test_real_http_api_streams_answer_and_sources_from_active_document() -> None:
    settings = Settings()
    app = create_app(settings, service_factory=fake_model_services)
    slug = f"chat-integration-{uuid4()}"
    knowledge_base_id: UUID | None = None
    conversation_id: UUID | None = None

    async with app.router.lifespan_context(app):
        resources = app.state.resources
        assert isinstance(resources, ApplicationResources)
        database = resources.database
        assert isinstance(database, DatabaseManager)
        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                created = await client.post(
                    f"{settings.api_prefix}/knowledge-bases",
                    json={"slug": slug, "name": "Chat integration"},
                )
                assert created.status_code == 201
                knowledge_base_id = UUID(created.json()["id"])
                uploaded = await client.post(
                    f"{settings.api_prefix}/knowledge-bases/{knowledge_base_id}/documents",
                    files={
                        "file": (
                            "refund.md",
                            "# 退款说明\n\n退款请提交订单号。".encode(),
                            "text/markdown",
                        )
                    },
                )
                assert uploaded.status_code == 201

                streamed = await client.post(
                    f"{settings.api_prefix}/chat/stream",
                    json={
                        "question": "如何申请退款?",
                        "scope": {"knowledge_base_ids": [str(knowledge_base_id)]},
                    },
                )

                assert streamed.status_code == 200
                assert "event: reply_to" in streamed.text
                assert "event: content" in streamed.text
                assert "event: sources" in streamed.text
                assert '"outcome":"completed"' in streamed.text
                assert str(knowledge_base_id) in streamed.text
                assert "退款请提交订单号。" not in streamed.text
                reply = _sse_payload(streamed.text, "reply_to")
                conversation_id = UUID(cast(str, reply["conversation_id"]))

                continued = await client.post(
                    f"{settings.api_prefix}/chat/stream",
                    json={
                        "question": "再说一次",
                        "scope": {"knowledge_base_ids": [str(knowledge_base_id)]},
                        "conversation_id": str(conversation_id),
                    },
                )
                continued_reply = _sse_payload(continued.text, "reply_to")
                assert continued.status_code == 200
                assert continued_reply["conversation_id"] == str(conversation_id)

                async with database.session_factory() as session:
                    messages = list(
                        await session.scalars(
                            select(MessageRecord)
                            .where(MessageRecord.conversation_id == conversation_id)
                            .order_by(MessageRecord.ordinal)
                        )
                    )
                    runs = list(
                        await session.scalars(
                            select(RagRunRecord)
                            .where(RagRunRecord.conversation_id == conversation_id)
                            .order_by(RagRunRecord.started_at)
                        )
                    )
                assert [message.role for message in messages] == [
                    "user",
                    "assistant",
                    "user",
                    "assistant",
                ]
                assert [run.status for run in runs] == ["completed", "completed"]
                assert all(run.source_chunk_ids for run in runs)
        finally:
            if conversation_id is not None:
                async with database.session_factory.begin() as session:
                    await session.execute(
                        delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                    )
            if knowledge_base_id is not None:
                async with database.session_factory.begin() as session:
                    await session.execute(
                        delete(KnowledgeBaseRecord).where(
                            KnowledgeBaseRecord.id == knowledge_base_id
                        )
                    )


def _sse_payload(body: str, event_name: str) -> dict[str, object]:
    for frame in body.strip().split("\n\n"):
        lines = frame.splitlines()
        if f"event: {event_name}" in lines:
            data_line = next(line for line in lines if line.startswith("data: "))
            return cast(dict[str, object], json.loads(data_line.removeprefix("data: ")))
    raise AssertionError(f"SSE event not found: {event_name}")
