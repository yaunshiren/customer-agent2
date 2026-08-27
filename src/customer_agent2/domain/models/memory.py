"""Framework-independent conversation memory and summary contracts."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models.chat import ChatMessage, ChatRole


@dataclass(frozen=True, slots=True)
class ConversationMemory:
    """A persisted summary plus recent complete conversation turns."""

    messages: tuple[ChatMessage, ...] = ()
    summary: str | None = None

    def __post_init__(self) -> None:
        normalized_summary = self.summary.strip() if self.summary is not None else None
        if normalized_summary == "":
            raise ValueError("ConversationMemory.summary 不能为空")
        _validate_complete_turns(self.messages)
        object.__setattr__(self, "summary", normalized_summary)


@dataclass(frozen=True, slots=True)
class StoredConversationMessage:
    """One ordered persisted message selected for summary generation."""

    ordinal: int
    message: ChatMessage

    def __post_init__(self) -> None:
        if self.ordinal < 1:
            raise ValueError("StoredConversationMessage.ordinal 必须大于 0")


@dataclass(frozen=True, slots=True)
class ConversationSummaryCandidate:
    """Optimistic snapshot of old completed turns that should be summarized."""

    conversation_id: UUID
    expected_summarized_through_ordinal: int | None
    previous_summary: str | None
    messages: tuple[StoredConversationMessage, ...]
    summarized_through_ordinal: int
    source_message_count: int

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("ConversationSummaryCandidate.messages 不能为空")
        normalized_previous_summary = (
            self.previous_summary.strip() if self.previous_summary is not None else None
        )
        if normalized_previous_summary == "":
            raise ValueError("ConversationSummaryCandidate.previous_summary 不能为空")
        ordinals = tuple(item.ordinal for item in self.messages)
        if any(current >= following for current, following in pairwise(ordinals)):
            raise ValueError("摘要来源消息 ordinal 必须严格递增")
        if self.expected_summarized_through_ordinal is not None:
            if self.expected_summarized_through_ordinal < 1:
                raise ValueError("摘要旧边界必须大于 0")
            if self.summarized_through_ordinal <= self.expected_summarized_through_ordinal:
                raise ValueError("摘要新边界必须晚于旧边界")
        if self.messages[-1].ordinal != self.summarized_through_ordinal:
            raise ValueError("摘要边界必须等于最后一条来源消息 ordinal")
        if self.source_message_count < len(self.messages) or self.source_message_count % 2:
            raise ValueError("摘要累计来源消息数无效")
        _validate_complete_turns(tuple(item.message for item in self.messages))
        object.__setattr__(self, "previous_summary", normalized_previous_summary)


@dataclass(frozen=True, slots=True)
class ConversationSummaryUpdate:
    """Validated model result ready for optimistic persistence."""

    conversation_id: UUID
    expected_summarized_through_ordinal: int | None
    summarized_through_ordinal: int
    source_message_count: int
    content: str
    model_id: str

    def __post_init__(self) -> None:
        normalized_content = self.content.strip()
        normalized_model_id = self.model_id.strip()
        if not normalized_content or not normalized_model_id:
            raise ValueError("摘要正文和 model_id 不能为空")
        if self.summarized_through_ordinal < 1 or self.source_message_count < 1:
            raise ValueError("摘要边界和来源消息数必须大于 0")
        if self.source_message_count % 2:
            raise ValueError("摘要来源消息数必须包含完整轮次")
        if (
            self.expected_summarized_through_ordinal is not None
            and self.summarized_through_ordinal <= self.expected_summarized_through_ordinal
        ):
            raise ValueError("摘要新边界必须晚于旧边界")
        object.__setattr__(self, "content", normalized_content)
        object.__setattr__(self, "model_id", normalized_model_id)


class ConversationMemoryRepository(Protocol):
    """Persistence port for recent completed turns and durable summaries."""

    async def load_memory(
        self,
        conversation_id: UUID,
        *,
        recent_turns: int,
    ) -> ConversationMemory: ...

    async def prepare_summary(
        self,
        conversation_id: UUID,
        *,
        trigger_turns: int,
        retain_recent_turns: int,
    ) -> ConversationSummaryCandidate | None: ...

    async def save_summary(self, update: ConversationSummaryUpdate) -> bool: ...


def _validate_complete_turns(messages: tuple[ChatMessage, ...]) -> None:
    if len(messages) % 2 != 0:
        raise ValueError("会话记忆必须包含完整 user/assistant 消息对")
    for index, message in enumerate(messages):
        expected_role = ChatRole.USER if index % 2 == 0 else ChatRole.ASSISTANT
        if message.role is not expected_role:
            raise ValueError("会话记忆必须按 user/assistant 顺序排列")
