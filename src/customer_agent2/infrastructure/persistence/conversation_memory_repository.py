"""SQLAlchemy adapter for recent completed turns and durable summaries."""

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from customer_agent2.domain.models import (
    ChatMessage,
    ChatRole,
    ConversationMemory,
    ConversationSummaryCandidate,
    ConversationSummaryUpdate,
    RagPersistenceError,
    RagPersistenceErrorCode,
    RagRunStatus,
    StoredConversationMessage,
)
from customer_agent2.infrastructure.persistence.models import (
    ConversationRecord,
    ConversationSummaryRecord,
    MessageRecord,
    RagRunRecord,
)


class SQLAlchemyConversationMemoryRepository:
    """Read complete turns and update one summary row with optimistic ordering."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def load_memory(
        self,
        conversation_id: UUID,
        *,
        recent_turns: int,
    ) -> ConversationMemory:
        """Load the latest complete pairs; running and failed Runs are excluded."""
        if recent_turns < 1:
            raise ValueError("recent_turns 必须大于 0")
        try:
            async with self._session_factory() as session:
                await _ensure_conversation_exists(session, conversation_id)
                summary = await session.get(ConversationSummaryRecord, conversation_id)
                records = list(
                    await session.scalars(
                        _completed_messages(conversation_id)
                        .order_by(MessageRecord.ordinal.desc())
                        .limit(recent_turns * 2)
                    )
                )
            records.reverse()
            return ConversationMemory(
                messages=tuple(_chat_message(record) for record in records),
                summary=summary.content if summary is not None else None,
            )
        except RagPersistenceError:
            raise
        except (SQLAlchemyError, ValueError):
            raise _memory_persistence_error("无法加载会话记忆") from None

    async def prepare_summary(
        self,
        conversation_id: UUID,
        *,
        trigger_turns: int,
        retain_recent_turns: int,
    ) -> ConversationSummaryCandidate | None:
        """Select completed turns that crossed the recent window after the total trigger."""
        if retain_recent_turns < 1 or trigger_turns <= retain_recent_turns:
            raise ValueError("摘要触发轮数必须大于保留轮数")
        try:
            async with self._session_factory() as session:
                await _ensure_conversation_exists(session, conversation_id)
                summary = await session.get(ConversationSummaryRecord, conversation_id)
                previous_boundary = (
                    summary.summarized_through_ordinal if summary is not None else None
                )
                statement = _completed_messages(conversation_id).order_by(MessageRecord.ordinal)
                if previous_boundary is not None:
                    statement = statement.where(MessageRecord.ordinal > previous_boundary)
                records = list(await session.scalars(statement))

            if len(records) % 2 != 0:
                raise ValueError("completed 会话消息不是完整轮次")
            prior_count = summary.source_message_count if summary is not None else 0
            if (prior_count + len(records)) // 2 <= trigger_turns:
                return None

            retained_message_count = retain_recent_turns * 2
            selected = records[:-retained_message_count]
            if not selected:
                return None
            return ConversationSummaryCandidate(
                conversation_id=conversation_id,
                expected_summarized_through_ordinal=previous_boundary,
                previous_summary=summary.content if summary is not None else None,
                messages=tuple(
                    StoredConversationMessage(record.ordinal, _chat_message(record))
                    for record in selected
                ),
                summarized_through_ordinal=selected[-1].ordinal,
                source_message_count=prior_count + len(selected),
            )
        except RagPersistenceError:
            raise
        except (SQLAlchemyError, ValueError):
            raise _memory_persistence_error("无法准备会话摘要") from None

    async def save_summary(self, update: ConversationSummaryUpdate) -> bool:
        """Persist only if no newer summary has changed the candidate's base boundary."""
        try:
            async with self._session_factory.begin() as session:
                conversation = await session.scalar(
                    select(ConversationRecord)
                    .where(ConversationRecord.id == update.conversation_id)
                    .with_for_update()
                )
                if conversation is None:
                    raise _conversation_not_found_error()

                current = await session.scalar(
                    select(ConversationSummaryRecord)
                    .where(ConversationSummaryRecord.conversation_id == update.conversation_id)
                    .with_for_update()
                )
                current_boundary = (
                    current.summarized_through_ordinal if current is not None else None
                )
                if current_boundary != update.expected_summarized_through_ordinal:
                    return False

                if current is None:
                    session.add(
                        ConversationSummaryRecord(
                            conversation_id=update.conversation_id,
                            summarized_through_ordinal=(update.summarized_through_ordinal),
                            source_message_count=update.source_message_count,
                            content=update.content,
                            model_id=update.model_id,
                        )
                    )
                else:
                    current.summarized_through_ordinal = update.summarized_through_ordinal
                    current.source_message_count = update.source_message_count
                    current.content = update.content
                    current.model_id = update.model_id
                await session.flush()
            return True
        except RagPersistenceError:
            raise
        except SQLAlchemyError:
            raise _memory_persistence_error("无法保存会话摘要") from None


def _completed_messages(conversation_id: UUID) -> Select[tuple[MessageRecord]]:
    return (
        select(MessageRecord)
        .join(RagRunRecord, MessageRecord.rag_run_id == RagRunRecord.id)
        .where(
            MessageRecord.conversation_id == conversation_id,
            RagRunRecord.status == RagRunStatus.COMPLETED.value,
        )
    )


async def _ensure_conversation_exists(
    session: AsyncSession,
    conversation_id: UUID,
) -> None:
    existing_id = await session.scalar(
        select(ConversationRecord.id).where(ConversationRecord.id == conversation_id)
    )
    if existing_id is None:
        raise _conversation_not_found_error()


def _chat_message(record: MessageRecord) -> ChatMessage:
    try:
        role = ChatRole(record.role)
    except ValueError:
        raise ValueError("数据库消息角色无效") from None
    return ChatMessage(role, record.content)


def _conversation_not_found_error() -> RagPersistenceError:
    return RagPersistenceError(
        RagPersistenceErrorCode.CONVERSATION_NOT_FOUND,
        "会话不存在",
        retryable=False,
    )


def _memory_persistence_error(message: str) -> RagPersistenceError:
    return RagPersistenceError(
        RagPersistenceErrorCode.PERSISTENCE_FAILURE,
        message,
        retryable=True,
    )
