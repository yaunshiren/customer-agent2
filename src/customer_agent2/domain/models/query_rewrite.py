"""Framework-independent query rewrite contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models.chat import ChatMessage


class QueryRewriteDegradationReason(StrEnum):
    """Stable, content-free reasons for falling back to the original question."""

    MODEL_FAILURE = "query_rewrite_model_failure"
    PROTOCOL = "query_rewrite_protocol"
    TIMEOUT = "query_rewrite_timeout"


@dataclass(frozen=True, slots=True)
class QueryRewriteRequest:
    """One question plus bounded conversation context supplied by M4-A."""

    request_id: UUID
    question: str
    memory_messages: tuple[ChatMessage, ...] = ()
    summary: str | None = None

    def __post_init__(self) -> None:
        normalized_question = self.question.strip()
        normalized_summary = self.summary.strip() if self.summary is not None else None
        if not normalized_question or len(normalized_question) > 10_000:
            raise ValueError("QueryRewriteRequest.question 长度无效")
        if normalized_summary == "":
            raise ValueError("QueryRewriteRequest.summary 不能为空")
        object.__setattr__(self, "question", normalized_question)
        object.__setattr__(self, "summary", normalized_summary)


@dataclass(frozen=True, slots=True)
class QueryRewriteResult:
    """Validated standalone question and the queries that must be retrieved."""

    rewritten_question: str
    sub_questions: tuple[str, ...]
    model_id: str | None = None
    degradation_reason: QueryRewriteDegradationReason | None = None

    def __post_init__(self) -> None:
        rewritten_question = self.rewritten_question.strip()
        sub_questions = tuple(question.strip() for question in self.sub_questions)
        model_id = self.model_id.strip() if self.model_id is not None else None
        if not rewritten_question or len(rewritten_question) > 10_000:
            raise ValueError("QueryRewriteResult.rewritten_question 长度无效")
        if not 1 <= len(sub_questions) <= 3:
            raise ValueError("QueryRewriteResult.sub_questions 必须包含 1 到 3 个问题")
        if any(not question or len(question) > 10_000 for question in sub_questions):
            raise ValueError("QueryRewriteResult.sub_questions 包含空值或超长值")
        if len(set(sub_questions)) != len(sub_questions):
            raise ValueError("QueryRewriteResult.sub_questions 不能重复")
        if (model_id is None) == (self.degradation_reason is None):
            raise ValueError("QueryRewriteResult 必须明确模型成功或降级结果")
        object.__setattr__(self, "rewritten_question", rewritten_question)
        object.__setattr__(self, "sub_questions", sub_questions)
        object.__setattr__(self, "model_id", model_id)


class QueryRewriter(Protocol):
    """Application port for contextual question rewrite and decomposition."""

    async def rewrite(self, request: QueryRewriteRequest) -> QueryRewriteResult: ...
