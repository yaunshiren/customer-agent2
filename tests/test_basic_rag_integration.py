"""Opt-in real pgvector integration for the M3-A streaming RAG pipeline."""

import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from customer_agent2.application import (
    BasicRagPromptBuilder,
    BasicStreamingRagPipeline,
    FastModelIntentClassifier,
    FastModelQueryRewriter,
    VectorRetrievalService,
)
from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    EmbeddingRequest,
    PipelineContentEvent,
    PipelineDoneEvent,
    PipelineOutcome,
    PipelineSourcesEvent,
    RagPipelineRequest,
    VectorSearchScope,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.intents import load_default_intent_tree
from customer_agent2.infrastructure.models import FakeChatModel, FakeEmbeddingModel
from customer_agent2.infrastructure.persistence import (
    EMBEDDING_DIMENSION,
    ChunkRecord,
    DocumentRecord,
    DocumentVersionRecord,
    KnowledgeBaseRecord,
    SQLAlchemyVectorSearchRepository,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]

MODEL_ID = "BAAI/bge-base-zh-v1.5"
MODEL_REVISION = "f03589ceff5aac7111bd60cfc7d497ca17ecac65"


@pytest.mark.asyncio
async def test_real_pgvector_pipeline_retrieves_prompts_and_streams_with_sources() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    chunk_id = uuid4()
    question = "退款流程"
    embedding = FakeEmbeddingModel(
        MODEL_ID,
        revision=MODEL_REVISION,
        dimension=EMBEDDING_DIMENSION,
    )
    query_vector = list((await embedding.embed(EmbeddingRequest((question,)))).vectors[0])
    await _seed_active_chunk(
        manager,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        version_id=version_id,
        chunk_id=chunk_id,
        embedding=query_vector,
    )
    chat = FakeChatModel(
        "fake-final-chat",
        "请在三个工作日内提交申请[1]。",
        stream_chunks=("请在三个工作日内", "提交申请[1]。"),
    )
    pipeline = BasicStreamingRagPipeline(
        VectorRetrievalService(
            embedding,
            SQLAlchemyVectorSearchRepository(manager.session_factory),
            recall_budget=20,
            hnsw_ef_search=100,
        ),
        BasicRagPromptBuilder(context_top_k=10),
        chat,
        FastModelQueryRewriter(
            FakeChatModel(
                "fake-fast-chat",
                '{"rewritten_question":"退款流程","sub_questions":["退款流程"]}',
            ),
            timeout_seconds=1,
            max_output_tokens=512,
            max_sub_questions=3,
        ),
        FastModelIntentClassifier(
            FakeChatModel(
                "fake-fast-intent",
                (
                    '{"scores":{"system_direct":0.02,"knowledge_base":0.96,'
                    '"clarification":0.02},"clarification_question":null}'
                ),
            ),
            load_default_intent_tree(),
            high_confidence_threshold=0.75,
            ambiguity_margin=0.10,
            timeout_seconds=1,
            max_output_tokens=256,
        ),
        global_timeout_seconds=5,
    )

    try:
        events = [
            event
            async for event in pipeline.stream(
                RagPipelineRequest(
                    uuid4(),
                    question,
                    VectorSearchScope((knowledge_base_id,)),
                )
            )
        ]

        answer = "".join(event.delta for event in events if isinstance(event, PipelineContentEvent))
        assert answer == "请在三个工作日内提交申请[1]。"
        sources = next(event.sources for event in events if isinstance(event, PipelineSourcesEvent))
        assert len(sources) == 1
        assert sources[0].chunk_id == chunk_id
        assert sources[0].document_id == document_id
        done = next(event for event in events if isinstance(event, PipelineDoneEvent))
        assert done.outcome is PipelineOutcome.COMPLETED
        assert len(chat.stream_requests) == 1
        assert "退款申请应在三个工作日内提交" in chat.stream_requests[0].messages[1].content
    finally:
        await _remove_knowledge_base(manager, knowledge_base_id)
        await manager.close()


async def _seed_active_chunk(
    manager: DatabaseManager,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    chunk_id: UUID,
    embedding: list[float],
) -> None:
    async with manager.session_factory.begin() as session:
        session.add(
            KnowledgeBaseRecord(
                id=knowledge_base_id,
                slug=f"m3-a-{knowledge_base_id}",
                name="M3-A integration",
                embedding_model_id=MODEL_ID,
                embedding_model_revision=MODEL_REVISION,
            )
        )
        await session.flush()
        session.add(
            DocumentRecord(
                id=document_id,
                knowledge_base_id=knowledge_base_id,
                source_key="guides/refund.md",
                display_name="refund.md",
            )
        )
        await session.flush()
        session.add(
            DocumentVersionRecord(
                id=version_id,
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                version_number=1,
                status="active",
                content_sha256="a" * 64,
                media_type="text/markdown",
                parser_name="customer-agent2-markdown",
                parser_version="1",
                source_metadata={"document_format": "markdown"},
            )
        )
        await session.flush()
        session.add(
            ChunkRecord(
                id=chunk_id,
                document_version_id=version_id,
                knowledge_base_id=knowledge_base_id,
                chunk_index=0,
                content="退款申请应在三个工作日内提交。",
                token_count=16,
                content_sha256="b" * 64,
                section="退款流程",
                source_metadata={
                    "block_start_ordinal": 0,
                    "block_end_ordinal": 0,
                    "start_line": 1,
                    "end_line": 1,
                    "section_path": ["退款流程"],
                    "overlap_with_previous_tokens": 0,
                },
                embedding=embedding,
            )
        )


async def _remove_knowledge_base(
    manager: DatabaseManager,
    knowledge_base_id: UUID,
) -> None:
    async with manager.session_factory.begin() as session:
        await session.execute(
            delete(KnowledgeBaseRecord).where(KnowledgeBaseRecord.id == knowledge_base_id)
        )
