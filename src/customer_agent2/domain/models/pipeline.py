"""Typed contracts for the explicit streaming RAG pipeline."""

import math
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeAlias
from uuid import UUID

from customer_agent2.domain.models.chat import ChatMessage, ChatRole, TokenUsage
from customer_agent2.domain.models.document import DocumentFormat
from customer_agent2.domain.models.intent import (
    GuidanceReason,
    IntentDecision,
    IntentRoute,
)
from customer_agent2.domain.models.retrieval import (
    VectorSearchCandidate,
    VectorSearchResult,
    VectorSearchScope,
)


class RagPipelineErrorCode(StrEnum):
    """Stable failures produced by application-level RAG orchestration."""

    GLOBAL_TIMEOUT = "global_timeout"
    MODEL_STREAM_PROTOCOL = "model_stream_protocol"
    PIPELINE_PROTOCOL = "pipeline_protocol"


class RagPipelineError(RuntimeError):
    """A sanitized pipeline failure with a retry hint."""

    def __init__(
        self,
        code: RagPipelineErrorCode,
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


class PipelineStage(StrEnum):
    """Observable stages implemented by the streaming RAG pipeline."""

    REWRITING = "rewriting"
    INTENT = "intent"
    RETRIEVING = "retrieving"
    FUSING = "fusing"
    RERANKING = "reranking"
    PROMPTING = "prompting"
    GENERATING = "generating"
    COMPLETED = "completed"
    NO_CONTEXT = "no_context"
    CLARIFICATION = "clarification"


class PipelineOutcome(StrEnum):
    """Terminal outcomes represented without treating empty retrieval as an exception."""

    COMPLETED = "completed"
    NO_CONTEXT = "no_context"
    CLARIFICATION = "clarification"


@dataclass(frozen=True, slots=True)
class RagPipelineRequest:
    """One question and its explicitly authorized vector-search scope."""

    request_id: UUID
    question: str
    search_scope: VectorSearchScope
    conversation_id: UUID | None = None
    user_id: str | None = None
    memory_messages: tuple[ChatMessage, ...] = ()
    memory_summary: str | None = None

    def __post_init__(self) -> None:
        normalized_question = self.question.strip()
        if not normalized_question:
            raise ValueError("RagPipelineRequest.question 不能为空")
        if len(normalized_question) > 10_000:
            raise ValueError("RagPipelineRequest.question 不能超过 10000 个字符")
        normalized_user_id = self.user_id
        if normalized_user_id is not None:
            normalized_user_id = normalized_user_id.strip()
            if not normalized_user_id or len(normalized_user_id) > 200:
                raise ValueError("RagPipelineRequest.user_id 必须是不超过 200 个字符的非空值")
        normalized_summary = (
            self.memory_summary.strip() if self.memory_summary is not None else None
        )
        if normalized_summary == "":
            raise ValueError("RagPipelineRequest.memory_summary 不能为空")
        if (
            self.memory_messages or normalized_summary is not None
        ) and self.conversation_id is None:
            raise ValueError("会话记忆必须绑定 conversation_id")
        if len(self.memory_messages) % 2 != 0:
            raise ValueError("RagPipelineRequest.memory_messages 必须包含完整轮次")
        for index, message in enumerate(self.memory_messages):
            expected_role = ChatRole.USER if index % 2 == 0 else ChatRole.ASSISTANT
            if message.role is not expected_role:
                raise ValueError("RagPipelineRequest.memory_messages 顺序无效")
        object.__setattr__(self, "question", normalized_question)
        object.__setattr__(self, "user_id", normalized_user_id)
        object.__setattr__(self, "memory_summary", normalized_summary)


@dataclass(frozen=True, slots=True)
class RagSource:
    """Public-safe citation metadata for one retrieved chunk."""

    citation_number: int
    chunk_id: UUID
    knowledge_base_id: UUID
    document_id: UUID
    document_version_id: UUID
    source_key: str
    display_name: str
    document_format: DocumentFormat
    section: str | None
    page_number: int | None
    content_sha256: str
    similarity: float

    def __post_init__(self) -> None:
        if self.citation_number < 1:
            raise ValueError("RagSource.citation_number 必须大于 0")
        if not self.source_key.strip() or not self.display_name.strip():
            raise ValueError("RagSource 来源标识不能为空")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("RagSource.page_number 必须大于 0")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("RagSource.content_sha256 格式无效")
        if not math.isfinite(self.similarity):
            raise ValueError("RagSource.similarity 必须是有限值")


@dataclass(frozen=True, slots=True)
class PipelineTraceEntry:
    """Small per-stage trace safe to persist later without document contents."""

    stage: PipelineStage
    duration_ms: float
    candidate_count: int | None = None
    degradation_reason: str | None = None
    decision: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("PipelineTraceEntry.duration_ms 不能小于 0")
        if self.candidate_count is not None and self.candidate_count < 0:
            raise ValueError("PipelineTraceEntry.candidate_count 不能小于 0")
        degradation_reason = (
            self.degradation_reason.strip() if self.degradation_reason is not None else None
        )
        if degradation_reason is not None and (
            not degradation_reason
            or len(degradation_reason) > 100
            or any(ord(character) < 32 for character in degradation_reason)
        ):
            raise ValueError("PipelineTraceEntry.degradation_reason 格式无效")
        object.__setattr__(self, "degradation_reason", degradation_reason)
        decision = self.decision.strip() if self.decision is not None else None
        if decision is not None and (
            not decision
            or len(decision) > 100
            or any(ord(character) < 32 for character in decision)
        ):
            raise ValueError("PipelineTraceEntry.decision 格式无效")
        object.__setattr__(self, "decision", decision)


@dataclass(frozen=True, slots=True)
class PromptAssembly:
    """Provider-neutral messages and their stable citation mapping."""

    messages: tuple[ChatMessage, ...]
    sources: tuple[RagSource, ...]

    def __post_init__(self) -> None:
        if not self.messages or not self.sources:
            raise ValueError("PromptAssembly 必须包含消息和来源")
        if tuple(source.citation_number for source in self.sources) != tuple(
            range(1, len(self.sources) + 1)
        ):
            raise ValueError("PromptAssembly.sources 必须使用连续引用编号")


@dataclass(frozen=True, slots=True)
class ChatPipelineContext:
    """Request-local state passed explicitly between implemented M3-A stages."""

    request: RagPipelineRequest
    rewritten_question: str
    sub_questions: tuple[str, ...]
    memory_messages: tuple[ChatMessage, ...]
    summary: str | None
    intent_decision: IntentDecision | None
    retrieval_result: VectorSearchResult | None
    ranked_chunks: tuple[VectorSearchCandidate, ...]
    prompt_messages: tuple[ChatMessage, ...]
    sources: tuple[RagSource, ...]
    trace: tuple[PipelineTraceEntry, ...]

    def __post_init__(self) -> None:
        if not self.rewritten_question.strip() or not self.sub_questions:
            raise ValueError("ChatPipelineContext 问题状态不能为空")
        if any(not question.strip() for question in self.sub_questions):
            raise ValueError("ChatPipelineContext.sub_questions 不能包含空问题")
        if tuple(source.citation_number for source in self.sources) != tuple(
            range(1, len(self.sources) + 1)
        ):
            raise ValueError("ChatPipelineContext.sources 引用编号必须连续")

    @classmethod
    def start(cls, request: RagPipelineRequest) -> "ChatPipelineContext":
        """Create request context with memory supplied by the M4-A decorator."""
        return cls(
            request=request,
            rewritten_question=request.question,
            sub_questions=(request.question,),
            memory_messages=request.memory_messages,
            summary=request.memory_summary,
            intent_decision=None,
            retrieval_result=None,
            ranked_chunks=(),
            prompt_messages=(),
            sources=(),
            trace=(),
        )


@dataclass(frozen=True, slots=True)
class PipelineStatusEvent:
    """Internal progress event later mapped to the versioned SSE contract."""

    request_id: UUID
    stage: PipelineStage


@dataclass(frozen=True, slots=True)
class PipelineContentEvent:
    """One answer-text delta; provider reasoning is intentionally excluded."""

    request_id: UUID
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            raise ValueError("PipelineContentEvent.delta 不能为空")


@dataclass(frozen=True, slots=True)
class PipelineReplyToEvent:
    """Persistent identities for the user message answered by this stream."""

    request_id: UUID
    conversation_id: UUID
    user_message_id: UUID
    rag_run_id: UUID


@dataclass(frozen=True, slots=True)
class PipelineSourcesEvent:
    """Final stable citation mapping for emitted answer content."""

    request_id: UUID
    sources: tuple[RagSource, ...]

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("PipelineSourcesEvent.sources 不能为空")


@dataclass(frozen=True, slots=True)
class PipelineGuidanceEvent:
    """One public-safe clarification question retained as conversation memory."""

    request_id: UUID
    message: str
    reason: GuidanceReason

    def __post_init__(self) -> None:
        message = self.message.strip()
        if not message or len(message) > 1000:
            raise ValueError("PipelineGuidanceEvent.message 长度无效")
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class PipelineDoneEvent:
    """Terminal event for a generated answer or an empty-retrieval short circuit."""

    request_id: UUID
    outcome: PipelineOutcome
    trace: tuple[PipelineTraceEntry, ...]
    intent_route: IntentRoute = IntentRoute.KNOWLEDGE_BASE
    model_id: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage | None = None

    def __post_init__(self) -> None:
        if self.outcome is PipelineOutcome.COMPLETED:
            if self.intent_route is IntentRoute.CLARIFICATION:
                raise ValueError("completed 事件不能使用 clarification 路由")
            if self.model_id is None or not self.model_id.strip():
                raise ValueError("完成事件必须包含 model_id")
            if self.finish_reason is None or not self.finish_reason.strip():
                raise ValueError("完成事件必须包含 finish_reason")
        elif self.outcome is PipelineOutcome.CLARIFICATION:
            if self.intent_route is not IntentRoute.CLARIFICATION:
                raise ValueError("clarification 事件必须使用 clarification 路由")
            if self.model_id is None or not self.model_id.strip():
                raise ValueError("clarification 事件必须包含 model_id")
            if self.finish_reason is None or not self.finish_reason.strip():
                raise ValueError("clarification 事件必须包含 finish_reason")
        elif (
            self.intent_route is not IntentRoute.KNOWLEDGE_BASE
            or self.model_id is not None
            or self.finish_reason is not None
            or self.usage is not None
        ):
            raise ValueError("空检索事件只能是无模型结果的 knowledge_base 路由")


PipelineEvent: TypeAlias = (
    PipelineReplyToEvent
    | PipelineStatusEvent
    | PipelineContentEvent
    | PipelineGuidanceEvent
    | PipelineSourcesEvent
    | PipelineDoneEvent
)


class StreamingRagPipeline(Protocol):
    """Application use-case contract for later HTTP/SSE adaptation."""

    def stream(self, request: RagPipelineRequest) -> AsyncGenerator[PipelineEvent, None]: ...
