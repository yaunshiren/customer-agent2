"""Framework-independent contracts for minimal conversation and RAG Run storage."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models.chat import TokenUsage
from customer_agent2.domain.models.intent import IntentRoute
from customer_agent2.domain.models.pipeline import (
    PipelineOutcome,
    PipelineTraceEntry,
    RagSource,
)


class RagRunStatus(StrEnum):
    """Persisted lifecycle states for one started RAG request."""

    RUNNING = "running"
    COMPLETED = "completed"
    NO_CONTEXT = "no_context"
    CLARIFICATION = "clarification"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RagPersistenceErrorCode(StrEnum):
    """Stable persistence failures safe to expose through SSE."""

    CONVERSATION_NOT_FOUND = "conversation_not_found"
    CONVERSATION_BUSY = "conversation_busy"
    RUN_STATE_CONFLICT = "run_state_conflict"
    PERSISTENCE_FAILURE = "persistence_failure"


class RagPersistenceError(RuntimeError):
    """A sanitized conversation/RAG Run persistence failure."""

    def __init__(
        self,
        code: RagPersistenceErrorCode,
        public_message: str,
        *,
        retryable: bool,
    ) -> None:
        normalized_message = public_message.strip()
        if not normalized_message:
            raise ValueError("public_message 不能为空")
        super().__init__(normalized_message)
        self.code = code
        self.public_message = normalized_message
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class RagRunBeginRequest:
    """Data committed before retrieval; empty knowledge IDs mean global scope."""

    request_id: UUID
    conversation_id: UUID | None
    question: str
    knowledge_base_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        normalized_question = self.question.strip()
        if not normalized_question:
            raise ValueError("RagRunBeginRequest.question 不能为空")
        if len(set(self.knowledge_base_ids)) != len(self.knowledge_base_ids):
            raise ValueError("RagRunBeginRequest.knowledge_base_ids 不能重复")
        object.__setattr__(self, "question", normalized_question)


@dataclass(frozen=True, slots=True)
class RagRunStart:
    """Stable identities allocated by the begin transaction."""

    conversation_id: UUID
    user_message_id: UUID
    rag_run_id: UUID


@dataclass(frozen=True, slots=True)
class RagRunCompletion:
    """Successful or empty-context terminal data saved before public done."""

    rag_run_id: UUID
    outcome: PipelineOutcome
    answer: str | None
    sources: tuple[RagSource, ...]
    trace: tuple[PipelineTraceEntry, ...]
    intent_route: IntentRoute = IntentRoute.KNOWLEDGE_BASE
    model_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if self.outcome is PipelineOutcome.COMPLETED:
            if self.answer is None or not self.answer.strip():
                raise ValueError("completed RAG Run 必须包含非空答案")
            if self.model_id is None or not self.model_id.strip():
                raise ValueError("completed RAG Run 必须包含 model_id")
            if self.finish_reason is None or not self.finish_reason.strip():
                raise ValueError("completed RAG Run 必须包含 finish_reason")
            if self.intent_route is IntentRoute.KNOWLEDGE_BASE and not self.sources:
                raise ValueError("知识库 completed RAG Run 必须包含引用来源")
            if self.intent_route is IntentRoute.SYSTEM_DIRECT and self.sources:
                raise ValueError("系统直答 completed RAG Run 不能包含引用来源")
            if self.intent_route is IntentRoute.CLARIFICATION:
                raise ValueError("completed RAG Run 不能使用 clarification 路由")
        elif self.outcome is PipelineOutcome.CLARIFICATION:
            if (
                self.intent_route is not IntentRoute.CLARIFICATION
                or self.answer is None
                or not self.answer.strip()
                or self.sources
                or self.model_id is None
                or not self.model_id.strip()
                or self.finish_reason is None
                or not self.finish_reason.strip()
            ):
                raise ValueError("clarification RAG Run 终局字段无效")
        elif (
            self.intent_route is not IntentRoute.KNOWLEDGE_BASE
            or self.answer is not None
            or self.sources
            or self.model_id is not None
            or self.finish_reason is not None
            or self.usage is not None
        ):
            raise ValueError("no_context RAG Run 只能是无模型结果的知识库路由")


class RagRunRepository(Protocol):
    """Persistence port used by the streaming decorator."""

    async def begin_run(self, request: RagRunBeginRequest) -> RagRunStart: ...

    async def complete_run(self, completion: RagRunCompletion) -> UUID | None: ...

    async def fail_run(self, rag_run_id: UUID, error_code: str) -> None: ...

    async def cancel_run(self, rag_run_id: UUID) -> None: ...
