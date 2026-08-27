"""Opt-in PostgreSQL integration tests for recent memory and summaries."""

import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete

from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    ConversationSummaryUpdate,
    DocumentFormat,
    PipelineOutcome,
    RagPersistenceError,
    RagPersistenceErrorCode,
    RagRunBeginRequest,
    RagRunCompletion,
    RagSource,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.persistence import (
    ConversationRecord,
    SQLAlchemyConversationMemoryRepository,
    SQLAlchemyRagRunRepository,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]


def _source() -> RagSource:
    return RagSource(
        citation_number=1,
        chunk_id=uuid4(),
        knowledge_base_id=uuid4(),
        document_id=uuid4(),
        document_version_id=uuid4(),
        source_key="refund.md",
        display_name="refund.md",
        document_format=DocumentFormat.MARKDOWN,
        section="退款",
        page_number=None,
        content_sha256="a" * 64,
        similarity=0.9,
    )


@pytest.mark.asyncio
async def test_repository_loads_six_turns_and_optimistically_saves_summary() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    runs = SQLAlchemyRagRunRepository(manager.session_factory)
    memory = SQLAlchemyConversationMemoryRepository(manager.session_factory)
    conversation_id: UUID | None = None
    try:
        for turn in range(1, 14):
            started = await runs.begin_run(
                RagRunBeginRequest(
                    request_id=uuid4(),
                    conversation_id=conversation_id,
                    question=f"问题 {turn}",
                    knowledge_base_ids=(uuid4(),),
                )
            )
            conversation_id = started.conversation_id
            await runs.complete_run(
                RagRunCompletion(
                    rag_run_id=started.rag_run_id,
                    outcome=PipelineOutcome.COMPLETED,
                    answer=f"回答 {turn}",
                    sources=(_source(),),
                    trace=(),
                    model_id="fake-final",
                    finish_reason="stop",
                )
            )

        assert conversation_id is not None
        recent = await memory.load_memory(conversation_id, recent_turns=6)
        assert len(recent.messages) == 12
        assert recent.messages[0].content == "问题 8"
        assert recent.messages[-1].content == "回答 13"
        assert recent.summary is None

        candidate = await memory.prepare_summary(
            conversation_id,
            trigger_turns=12,
            retain_recent_turns=6,
        )
        assert candidate is not None
        assert len(candidate.messages) == 14
        assert candidate.messages[0].message.content == "问题 1"
        assert candidate.messages[-1].message.content == "回答 7"
        assert candidate.summarized_through_ordinal == 14
        assert candidate.source_message_count == 14

        update = ConversationSummaryUpdate(
            conversation_id=conversation_id,
            expected_summarized_through_ordinal=None,
            summarized_through_ordinal=14,
            source_message_count=14,
            content="用户连续询问退款问题, 已回答前七轮。",
            model_id="fake-fast",
        )
        assert await memory.save_summary(update) is True
        assert await memory.save_summary(update) is False

        started = await runs.begin_run(
            RagRunBeginRequest(
                request_id=uuid4(),
                conversation_id=conversation_id,
                question="问题 14",
                knowledge_base_ids=(uuid4(),),
            )
        )
        await runs.complete_run(
            RagRunCompletion(
                rag_run_id=started.rag_run_id,
                outcome=PipelineOutcome.COMPLETED,
                answer="回答 14",
                sources=(_source(),),
                trace=(),
                model_id="fake-final",
                finish_reason="stop",
            )
        )
        incremental = await memory.prepare_summary(
            conversation_id,
            trigger_turns=12,
            retain_recent_turns=6,
        )
        assert incremental is not None
        assert [item.message.content for item in incremental.messages] == [
            "问题 8",
            "回答 8",
        ]
        assert incremental.expected_summarized_through_ordinal == 14
        assert incremental.summarized_through_ordinal == 16
        assert incremental.source_message_count == 16
        assert (
            await memory.save_summary(
                ConversationSummaryUpdate(
                    conversation_id=conversation_id,
                    expected_summarized_through_ordinal=14,
                    summarized_through_ordinal=16,
                    source_message_count=16,
                    content="用户连续询问退款问题, 已回答前八轮。",
                    model_id="fake-fast",
                )
            )
            is True
        )

        loaded = await memory.load_memory(conversation_id, recent_turns=6)
        assert loaded.summary == "用户连续询问退款问题, 已回答前八轮。"
        assert loaded.messages[0].content == "问题 9"
        assert loaded.messages[-1].content == "回答 14"
    finally:
        if conversation_id is not None:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                )
        await manager.close()


@pytest.mark.asyncio
async def test_memory_repository_rejects_unknown_conversation() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    try:
        repository = SQLAlchemyConversationMemoryRepository(manager.session_factory)

        with pytest.raises(RagPersistenceError) as captured:
            await repository.load_memory(uuid4(), recent_turns=6)

        assert captured.value.code is RagPersistenceErrorCode.CONVERSATION_NOT_FOUND
    finally:
        await manager.close()
