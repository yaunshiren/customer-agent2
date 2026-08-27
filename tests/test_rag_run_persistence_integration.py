"""Opt-in PostgreSQL integration tests for conversation and RAG Run storage."""

import os
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select

from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    DocumentFormat,
    IntentRoute,
    PipelineOutcome,
    PipelineStage,
    PipelineTraceEntry,
    RagPersistenceError,
    RagPersistenceErrorCode,
    RagRunBeginRequest,
    RagRunCompletion,
    RagRunStatus,
    RagSource,
    TokenUsage,
)
from customer_agent2.infrastructure.database import DatabaseManager
from customer_agent2.infrastructure.persistence import (
    ConversationRecord,
    MessageRecord,
    RagRunRecord,
    SQLAlchemyRagRunRepository,
)

pytestmark = [
    pytest.mark.database_integration,
    pytest.mark.skipif(
        os.getenv("RUN_DATABASE_INTEGRATION") != "1",
        reason="set RUN_DATABASE_INTEGRATION=1 to use the migrated local PostgreSQL",
    ),
]


def _begin_request(conversation_id: UUID | None = None) -> RagRunBeginRequest:
    return RagRunBeginRequest(
        request_id=uuid4(),
        conversation_id=conversation_id,
        question="如何申请退款?",
        knowledge_base_ids=(uuid4(),),
    )


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
        similarity=0.95,
    )


@pytest.mark.asyncio
async def test_repository_persists_completed_run_and_continues_ordered_conversation() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    repository = SQLAlchemyRagRunRepository(manager.session_factory)
    conversation_id: UUID | None = None
    try:
        begin_request = _begin_request()
        started = await repository.begin_run(begin_request)
        conversation_id = started.conversation_id

        with pytest.raises(RagPersistenceError) as busy:
            await repository.begin_run(_begin_request(started.conversation_id))
        assert busy.value.code is RagPersistenceErrorCode.CONVERSATION_BUSY

        source = _source()
        assistant_message_id = await repository.complete_run(
            RagRunCompletion(
                rag_run_id=started.rag_run_id,
                outcome=PipelineOutcome.COMPLETED,
                answer="请提交订单号 [1]。",
                sources=(source,),
                trace=(PipelineTraceEntry(PipelineStage.RETRIEVING, 3.5, 1),),
                model_id="fake-final",
                finish_reason="stop",
                usage=TokenUsage(10, 6),
            )
        )
        assert assistant_message_id is not None

        second = await repository.begin_run(_begin_request(started.conversation_id))
        await repository.cancel_run(second.rag_run_id)

        async with manager.session_factory() as session:
            messages = list(
                await session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == started.conversation_id)
                    .order_by(MessageRecord.ordinal)
                )
            )
            first_run = await session.get(RagRunRecord, started.rag_run_id)
            second_run = await session.get(RagRunRecord, second.rag_run_id)

        assert [(message.ordinal, message.role) for message in messages] == [
            (1, "user"),
            (2, "assistant"),
            (3, "user"),
        ]
        assert messages[0].content == begin_request.question
        assert messages[1].id == assistant_message_id
        assert messages[1].content == "请提交订单号 [1]。"
        assert first_run is not None
        assert first_run.status == RagRunStatus.COMPLETED.value
        assert first_run.model_id == "fake-final"
        assert first_run.input_tokens == 10
        assert first_run.output_tokens == 6
        assert first_run.source_chunk_ids == [source.chunk_id]
        assert first_run.trace == [
            {
                "stage": "retrieving",
                "duration_ms": 3.5,
                "candidate_count": 1,
                "degradation_reason": None,
                "decision": None,
            }
        ]
        assert first_run.intent_route == IntentRoute.KNOWLEDGE_BASE.value
        assert first_run.finished_at is not None
        assert second_run is not None
        assert second_run.status == RagRunStatus.CANCELLED.value
        assert second_run.error_code == "client_cancelled"
    finally:
        if conversation_id is not None:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                )
        await manager.close()


@pytest.mark.asyncio
async def test_repository_persists_no_context_and_failure_without_assistant_message() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    repository = SQLAlchemyRagRunRepository(manager.session_factory)
    conversation_ids: list[UUID] = []
    try:
        no_context = await repository.begin_run(_begin_request())
        conversation_ids.append(no_context.conversation_id)
        assistant_id = await repository.complete_run(
            RagRunCompletion(
                rag_run_id=no_context.rag_run_id,
                outcome=PipelineOutcome.NO_CONTEXT,
                answer=None,
                sources=(),
                trace=(PipelineTraceEntry(PipelineStage.RETRIEVING, 1.0, 0),),
            )
        )
        assert assistant_id is None

        failed = await repository.begin_run(_begin_request())
        conversation_ids.append(failed.conversation_id)
        await repository.fail_run(failed.rag_run_id, "unavailable")

        async with manager.session_factory() as session:
            no_context_messages = list(
                await session.scalars(
                    select(MessageRecord).where(
                        MessageRecord.conversation_id == no_context.conversation_id
                    )
                )
            )
            failed_messages = list(
                await session.scalars(
                    select(MessageRecord).where(
                        MessageRecord.conversation_id == failed.conversation_id
                    )
                )
            )
            no_context_run = await session.get(RagRunRecord, no_context.rag_run_id)
            failed_run = await session.get(RagRunRecord, failed.rag_run_id)

        assert [message.role for message in no_context_messages] == ["user"]
        assert [message.role for message in failed_messages] == ["user"]
        assert no_context_run is not None
        assert no_context_run.status == RagRunStatus.NO_CONTEXT.value
        assert no_context_run.error_code is None
        assert failed_run is not None
        assert failed_run.status == RagRunStatus.FAILED.value
        assert failed_run.error_code == "unavailable"
    finally:
        if conversation_ids:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(ConversationRecord).where(ConversationRecord.id.in_(conversation_ids))
                )
        await manager.close()


@pytest.mark.asyncio
async def test_repository_persists_clarification_as_a_complete_memory_turn() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    repository = SQLAlchemyRagRunRepository(manager.session_factory)
    conversation_id: UUID | None = None
    try:
        started = await repository.begin_run(_begin_request())
        conversation_id = started.conversation_id
        assistant_id = await repository.complete_run(
            RagRunCompletion(
                rag_run_id=started.rag_run_id,
                outcome=PipelineOutcome.CLARIFICATION,
                answer="请问您想了解哪一种商品?",
                sources=(),
                trace=(
                    PipelineTraceEntry(
                        PipelineStage.INTENT,
                        2.0,
                        3,
                        decision="clarification",
                    ),
                ),
                intent_route=IntentRoute.CLARIFICATION,
                model_id="fake-fast",
                finish_reason="stop",
            )
        )

        async with manager.session_factory() as session:
            messages = list(
                await session.scalars(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == conversation_id)
                    .order_by(MessageRecord.ordinal)
                )
            )
            run = await session.get(RagRunRecord, started.rag_run_id)

        assert assistant_id is not None
        assert [(message.role, message.content) for message in messages] == [
            ("user", "如何申请退款?"),
            ("assistant", "请问您想了解哪一种商品?"),
        ]
        assert run is not None
        assert run.status == RagRunStatus.CLARIFICATION.value
        assert run.intent_route == IntentRoute.CLARIFICATION.value
        assert run.source_chunk_ids == []
    finally:
        if conversation_id is not None:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                )
        await manager.close()


@pytest.mark.asyncio
async def test_repository_rejects_unknown_conversation_and_terminal_rewrite() -> None:
    manager = DatabaseManager(Settings())
    await manager.open()
    repository = SQLAlchemyRagRunRepository(manager.session_factory)
    conversation_id: UUID | None = None
    try:
        with pytest.raises(RagPersistenceError) as missing:
            await repository.begin_run(_begin_request(uuid4()))
        assert missing.value.code is RagPersistenceErrorCode.CONVERSATION_NOT_FOUND

        started = await repository.begin_run(_begin_request())
        conversation_id = started.conversation_id
        await repository.cancel_run(started.rag_run_id)
        with pytest.raises(RagPersistenceError) as conflict:
            await repository.fail_run(started.rag_run_id, "late_failure")
        assert conflict.value.code is RagPersistenceErrorCode.RUN_STATE_CONFLICT
    finally:
        if conversation_id is not None:
            async with manager.session_factory.begin() as session:
                await session.execute(
                    delete(ConversationRecord).where(ConversationRecord.id == conversation_id)
                )
        await manager.close()
