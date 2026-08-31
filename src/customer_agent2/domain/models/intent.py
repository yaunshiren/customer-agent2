"""Framework-independent intent classification and routing contracts."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from customer_agent2.domain.models.chat import TokenUsage
from customer_agent2.domain.models.errors import ModelErrorCode


class IntentRoute(StrEnum):
    """The three accepted M4-C terminal routes."""

    SYSTEM_DIRECT = "system_direct"
    KNOWLEDGE_BASE = "knowledge_base"
    CLARIFICATION = "clarification"


class IntentDecisionReason(StrEnum):
    """Stable reasons for the final route selected from classifier scores."""

    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"
    EXPLICIT_CLARIFICATION = "explicit_clarification"
    CLASSIFIER_FALLBACK = "classifier_fallback"


class GuidanceReason(StrEnum):
    """Public reasons that require another user message."""

    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"
    EXPLICIT_CLARIFICATION = "explicit_clarification"


class IntentDegradationReason(StrEnum):
    """Content-free classifier failures that trigger authorized-scope retrieval."""

    MODEL_FAILURE = "intent_classifier_model_failure"
    PROTOCOL = "intent_classifier_protocol"
    TIMEOUT = "intent_classifier_timeout"


@dataclass(frozen=True, slots=True)
class IntentDefinition:
    """One versioned route description loaded from the intent tree."""

    route: IntentRoute
    description: str

    def __post_init__(self) -> None:
        description = self.description.strip()
        if not description or len(description) > 1000:
            raise ValueError("IntentDefinition.description 长度无效")
        object.__setattr__(self, "description", description)


@dataclass(frozen=True, slots=True)
class IntentTree:
    """Strict initial route tree used to construct the classifier prompt."""

    version: str
    definitions: tuple[IntentDefinition, ...]

    def __post_init__(self) -> None:
        version = self.version.strip()
        routes = tuple(definition.route for definition in self.definitions)
        if not version or len(version) > 100:
            raise ValueError("IntentTree.version 长度无效")
        if routes != tuple(IntentRoute):
            raise ValueError("IntentTree 必须按固定顺序包含三个 M4-C 路由")
        object.__setattr__(self, "version", version)


@dataclass(frozen=True, slots=True)
class IntentClassificationRequest:
    """One rewritten standalone question ready for intent classification."""

    request_id: UUID
    question: str

    def __post_init__(self) -> None:
        question = self.question.strip()
        if not question or len(question) > 10_000:
            raise ValueError("IntentClassificationRequest.question 长度无效")
        object.__setattr__(self, "question", question)


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    """One normalized classifier score."""

    route: IntentRoute
    score: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("IntentCandidate.score 必须在 0 到 1 之间")


def select_intent_route(
    candidates: tuple[IntentCandidate, ...],
    *,
    high_confidence_threshold: float,
    ambiguity_margin: float,
) -> tuple[IntentRoute, IntentDecisionReason]:
    """Apply the shared deterministic policy to one complete score snapshot."""
    if not 0 <= high_confidence_threshold <= 1:
        raise ValueError("high_confidence_threshold 必须在 0 到 1 之间")
    if not 0 <= ambiguity_margin <= 1:
        raise ValueError("ambiguity_margin 必须在 0 到 1 之间")
    if len(candidates) != len(IntentRoute) or {candidate.route for candidate in candidates} != set(
        IntentRoute
    ):
        raise ValueError("Intent 候选必须恰好包含三个唯一路由")

    order = {route: index for index, route in enumerate(IntentRoute)}
    ranked = tuple(
        sorted(candidates, key=lambda candidate: (-candidate.score, order[candidate.route]))
    )
    top, second = ranked[:2]
    if top.score < high_confidence_threshold:
        return IntentRoute.CLARIFICATION, IntentDecisionReason.LOW_CONFIDENCE
    difference = top.score - second.score
    if difference < ambiguity_margin and not math.isclose(
        difference,
        ambiguity_margin,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        return IntentRoute.CLARIFICATION, IntentDecisionReason.AMBIGUOUS
    if top.route is IntentRoute.CLARIFICATION:
        return IntentRoute.CLARIFICATION, IntentDecisionReason.EXPLICIT_CLARIFICATION
    return top.route, IntentDecisionReason.HIGH_CONFIDENCE


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """Validated route decision, including observable classifier fallback."""

    route: IntentRoute
    reason: IntentDecisionReason
    candidates: tuple[IntentCandidate, ...]
    guidance_message: str | None = None
    classifier_model_id: str | None = None
    classifier_finish_reason: str | None = None
    usage: TokenUsage | None = None
    degradation_reason: IntentDegradationReason | None = None
    model_error_code: ModelErrorCode | None = None

    def __post_init__(self) -> None:
        message = self.guidance_message.strip() if self.guidance_message is not None else None
        model_id = (
            self.classifier_model_id.strip() if self.classifier_model_id is not None else None
        )
        finish_reason = (
            self.classifier_finish_reason.strip()
            if self.classifier_finish_reason is not None
            else None
        )
        if self.degradation_reason is None:
            if len(self.candidates) != len(IntentRoute):
                raise ValueError("正常 IntentDecision 必须包含三个候选")
            if {candidate.route for candidate in self.candidates} != set(IntentRoute):
                raise ValueError("IntentDecision.candidates 路由集合无效")
            if model_id is None or not model_id or finish_reason is None or not finish_reason:
                raise ValueError("正常 IntentDecision 必须包含分类模型结果")
            if self.reason is IntentDecisionReason.CLASSIFIER_FALLBACK:
                raise ValueError("正常 IntentDecision 不能使用降级原因")
            if self.model_error_code is not None:
                raise ValueError("正常 IntentDecision 不能包含模型错误代码")
        else:
            if (
                self.route is not IntentRoute.KNOWLEDGE_BASE
                or self.reason is not IntentDecisionReason.CLASSIFIER_FALLBACK
                or self.candidates
                or model_id is not None
                or finish_reason is not None
                or self.usage is not None
            ):
                raise ValueError("降级 IntentDecision 必须进入知识库兜底")
        if self.route is IntentRoute.CLARIFICATION:
            if message is None or not message or len(message) > 1000:
                raise ValueError("澄清路由必须包含不超过 1000 字符的问题")
        elif message is not None:
            raise ValueError("非澄清路由不能包含 guidance_message")
        object.__setattr__(self, "guidance_message", message)
        object.__setattr__(self, "classifier_model_id", model_id)
        object.__setattr__(self, "classifier_finish_reason", finish_reason)


class IntentClassifier(Protocol):
    """Application port for one bounded route decision."""

    async def classify(self, request: IntentClassificationRequest) -> IntentDecision: ...
