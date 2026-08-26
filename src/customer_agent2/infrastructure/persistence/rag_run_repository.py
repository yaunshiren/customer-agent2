"""SQLAlchemy adapter for minimal conversation messages and RAG Run state."""

from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent2.domain.models import (
    PipelineOutcome,
    RagPersistenceError,
    RagPersistenceErrorCode,
    RagRunBeginRequest,
    RagRunCompletion,
    RagRunStart,
    RagRunStatus,
)
from customer_agent2.infrastructure.persistence.models import (
    ConversationRecord,
    MessageRecord,
    RagRunRecord,
)


class SQLAlchemyRagRunRepository:
    """Persist request start and exactly one terminal state in short transactions."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def begin_run(self, request: RagRunBeginRequest) -> RagRunStart:
        """Create or lock a conversation, then atomically save the user message and Run."""
        conversation_id = request.conversation_id or uuid4()
        rag_run_id = uuid4()
        user_message_id = uuid4()
        try:
            async with self._session_factory.begin() as session:
                if request.conversation_id is None:
                    session.add(ConversationRecord(id=conversation_id))
                    await session.flush()
                else:
                    conversation = await session.scalar(
                        select(ConversationRecord)
                        .where(ConversationRecord.id == conversation_id)
                        .with_for_update()
                    )
                    if conversation is None:
                        raise RagPersistenceError(
                            RagPersistenceErrorCode.CONVERSATION_NOT_FOUND,
                            "会话不存在",
                            retryable=False,
                        )
                    running_id = await session.scalar(
                        select(RagRunRecord.id).where(
                            RagRunRecord.conversation_id == conversation_id,
                            RagRunRecord.status == RagRunStatus.RUNNING.value,
                        )
                    )
                    if running_id is not None:
                        raise _conversation_busy_error()

                ordinal = await _next_message_ordinal(session, conversation_id)
                run = RagRunRecord(
                    id=rag_run_id,
                    request_id=request.request_id,
                    conversation_id=conversation_id,
                    knowledge_base_ids=list(request.knowledge_base_ids),
                    status=RagRunStatus.RUNNING.value,
                )
                session.add(run)
                await session.flush()
                session.add(
                    MessageRecord(
                        id=user_message_id,
                        conversation_id=conversation_id,
                        rag_run_id=rag_run_id,
                        ordinal=ordinal,
                        role="user",
                        content=request.question,
                    )
                )
                await _touch_conversation(session, conversation_id)
                await session.flush()
            return RagRunStart(
                conversation_id=conversation_id,
                user_message_id=user_message_id,
                rag_run_id=rag_run_id,
            )
        except RagPersistenceError:
            raise
        except IntegrityError as error:
            constraint_name = cast(str | None, getattr(error.orig, "constraint_name", None))
            if constraint_name == "ux_rag_runs_one_running_per_conversation":
                raise _conversation_busy_error() from None
            raise _persistence_error("无法开始 RAG Run") from None
        except SQLAlchemyError:
            raise _persistence_error("无法开始 RAG Run") from None

    async def complete_run(self, completion: RagRunCompletion) -> UUID | None:
        """Commit answer/message/trace before the public success done event is emitted."""
        assistant_message_id: UUID | None = None
        try:
            async with self._session_factory.begin() as session:
                run = await _lock_running_run(session, completion.rag_run_id)
                conversation = await session.scalar(
                    select(ConversationRecord)
                    .where(ConversationRecord.id == run.conversation_id)
                    .with_for_update()
                )
                if conversation is None:
                    raise _persistence_error("RAG Run 的会话状态无效")

                if completion.outcome is PipelineOutcome.COMPLETED:
                    assert completion.answer is not None
                    assistant_message_id = uuid4()
                    session.add(
                        MessageRecord(
                            id=assistant_message_id,
                            conversation_id=run.conversation_id,
                            rag_run_id=run.id,
                            ordinal=await _next_message_ordinal(session, run.conversation_id),
                            role="assistant",
                            content=completion.answer,
                        )
                    )

                usage = completion.usage
                run.status = completion.outcome.value
                run.model_id = completion.model_id
                run.finish_reason = completion.finish_reason
                run.input_tokens = usage.input_tokens if usage is not None else None
                run.output_tokens = usage.output_tokens if usage is not None else None
                run.trace = [
                    {
                        "stage": entry.stage.value,
                        "duration_ms": entry.duration_ms,
                        "candidate_count": entry.candidate_count,
                    }
                    for entry in completion.trace
                ]
                run.source_chunk_ids = [source.chunk_id for source in completion.sources]
                run.error_code = None
                run.finished_at = datetime.now(UTC)
                await _touch_conversation(session, run.conversation_id)
                await session.flush()
            return assistant_message_id
        except RagPersistenceError:
            raise
        except SQLAlchemyError:
            raise _persistence_error("无法完成 RAG Run 持久化") from None

    async def fail_run(self, rag_run_id: UUID, error_code: str) -> None:
        """Persist a stable failure code without partial answer content."""
        await self._mark_terminal(
            rag_run_id,
            RagRunStatus.FAILED,
            _validated_error_code(error_code),
        )

    async def cancel_run(self, rag_run_id: UUID) -> None:
        """Record caller cancellation or early stream closure."""
        await self._mark_terminal(
            rag_run_id,
            RagRunStatus.CANCELLED,
            "client_cancelled",
        )

    async def _mark_terminal(
        self,
        rag_run_id: UUID,
        status: RagRunStatus,
        error_code: str,
    ) -> None:
        try:
            async with self._session_factory.begin() as session:
                run = await _lock_running_run(session, rag_run_id)
                run.status = status.value
                run.error_code = error_code
                run.finished_at = datetime.now(UTC)
                await _touch_conversation(session, run.conversation_id)
                await session.flush()
        except RagPersistenceError:
            raise
        except SQLAlchemyError:
            raise _persistence_error("无法保存 RAG Run 终止状态") from None


async def _lock_running_run(session: AsyncSession, rag_run_id: UUID) -> RagRunRecord:
    run = await session.scalar(
        select(RagRunRecord).where(RagRunRecord.id == rag_run_id).with_for_update()
    )
    if run is None or run.status != RagRunStatus.RUNNING.value:
        raise RagPersistenceError(
            RagPersistenceErrorCode.RUN_STATE_CONFLICT,
            "RAG Run 状态不允许当前操作",
            retryable=False,
        )
    return run


async def _next_message_ordinal(session: AsyncSession, conversation_id: UUID) -> int:
    maximum = await session.scalar(
        select(func.coalesce(func.max(MessageRecord.ordinal), 0)).where(
            MessageRecord.conversation_id == conversation_id
        )
    )
    return cast(int, maximum) + 1


async def _touch_conversation(session: AsyncSession, conversation_id: UUID) -> None:
    await session.execute(
        update(ConversationRecord)
        .where(ConversationRecord.id == conversation_id)
        .values(updated_at=func.now())
    )


def _validated_error_code(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 100:
        return "internal_error"
    return normalized


def _conversation_busy_error() -> RagPersistenceError:
    return RagPersistenceError(
        RagPersistenceErrorCode.CONVERSATION_BUSY,
        "会话正在处理另一个请求",
        retryable=True,
    )


def _persistence_error(message: str) -> RagPersistenceError:
    return RagPersistenceError(
        RagPersistenceErrorCode.PERSISTENCE_FAILURE,
        message,
        retryable=True,
    )
