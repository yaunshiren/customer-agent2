"""Content-safe 150-case intent-route evaluation."""

import math
from collections import Counter
from collections.abc import Callable
from time import perf_counter
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.domain.models import (
    IntentCandidate,
    IntentClassificationRequest,
    IntentClassifier,
    IntentDegradationReason,
    IntentRoute,
    ModelErrorCode,
    select_intent_route,
)
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_FULL_CASES,
    EXPECTED_NO_RAG_CASES,
    EXPECTED_RAG_CASES,
    FullEvaluationDataset,
    FullEvaluationSample,
)


class FullIntentRunError(RuntimeError):
    """Sanitized fatal error that stops the live run at first degradation."""

    def __init__(self, query_id: str, error_code: str) -> None:
        super().__init__(f"完整 Intent 评测在 {query_id} 安全终止: {error_code}")
        self.query_id = query_id
        self.error_code = error_code


class FullIntentFailedAttempt(BaseModel):
    """One content-free failed live attempt retained for resume auditing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=100)
    error_code: str = Field(min_length=1, max_length=100)


class IntentCandidateScores(BaseModel):
    """Content-free route scores retained for deterministic threshold replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_direct: float = Field(ge=0, le=1)
    knowledge_base: float = Field(ge=0, le=1)
    clarification: float = Field(ge=0, le=1)

    def as_candidates(self) -> tuple[IntentCandidate, ...]:
        """Return scores in the domain route order used by online selection."""
        return tuple(IntentCandidate(route, getattr(self, route.value)) for route in IntentRoute)


class FullIntentCheckpoint(BaseModel):
    """Sanitized successful prefix and failed attempts for exact paid-call resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    model_id: str = Field(min_length=1, max_length=200)
    intent_tree_version: str = Field(min_length=1, max_length=100)
    intent_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    high_confidence_threshold: float = Field(ge=0, le=1)
    ambiguity_margin: float = Field(ge=0, le=1)
    timeout_seconds: float = Field(gt=0)
    max_output_tokens: int = Field(ge=1)
    reasoning_enabled: bool
    completed_cases: tuple["FullIntentCaseResult", ...] = ()
    failed_attempts: tuple[FullIntentFailedAttempt, ...] = ()

    @model_validator(mode="after")
    def validate_checkpoint(self) -> Self:
        if len(self.completed_cases) > EXPECTED_FULL_CASES:
            raise ValueError("Intent checkpoint 成功样本数不能超过完整数据集")
        if any(case.degradation_reason is not None for case in self.completed_cases):
            raise ValueError("Intent checkpoint 只能把成功结果标记为已完成")
        return self


class IntentSliceMetrics(BaseModel):
    """Accuracy counts for one documented evaluation slice."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sample_count: int = Field(ge=1)
    correct_count: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.correct_count > self.sample_count:
            raise ValueError("Intent 正确数不能超过样本数")
        if not math.isclose(
            self.accuracy,
            self.correct_count / self.sample_count,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Intent accuracy 与计数不一致")
        return self


class FullIntentEvaluationConfiguration(BaseModel):
    """Exact model and classifier controls attached to a live report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, max_length=200)
    intent_tree_version: str = Field(default="m4-c-v1", min_length=1, max_length=100)
    intent_tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    high_confidence_threshold: float = Field(ge=0, le=1)
    ambiguity_margin: float = Field(ge=0, le=1)
    timeout_seconds: float = Field(gt=0)
    max_output_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0, le=2)
    reasoning_enabled: bool


class FullIntentCaseResult(BaseModel):
    """Sanitized per-case route decision without the original question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=100)
    intent_l1: str = Field(min_length=1, max_length=100)
    expected_route: IntentRoute
    actual_route: IntentRoute
    correct: bool
    decision_reason: str = Field(min_length=1, max_length=100)
    degradation_reason: IntentDegradationReason | None = None
    model_error_code: ModelErrorCode | None = None
    candidate_scores: IntentCandidateScores | None = None
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        expected_correct = (
            self.degradation_reason is None and self.actual_route is self.expected_route
        )
        if self.correct != expected_correct:
            raise ValueError("Intent correct 必须排除降级并匹配期望路由")
        if (self.input_tokens is None) != (self.output_tokens is None):
            raise ValueError("Intent 输入和输出 Token 必须同时存在或缺失")
        if self.degradation_reason is not None and self.input_tokens is not None:
            raise ValueError("Intent 降级结果不能声明成功 Token 用量")
        if self.degradation_reason is None and self.model_error_code is not None:
            raise ValueError("正常 Intent 结果不能声明模型错误代码")
        if self.degradation_reason is not None and self.candidate_scores is not None:
            raise ValueError("Intent 降级结果不能声明候选分数")
        return self


class FullIntentReport(BaseModel):
    """Aggregate and per-case intent evaluation over all 150 samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    configuration: FullIntentEvaluationConfiguration | None = None
    sample_count: int = Field(ge=1)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    overall: IntentSliceMetrics
    rag: IntentSliceMetrics
    no_rag: IntentSliceMetrics
    by_intent_l1: dict[str, IntentSliceMetrics]
    confusion: dict[IntentRoute, dict[IntentRoute, int]]
    latency_ms_p50: float | None = Field(default=None, ge=0)
    latency_ms_p95: float | None = Field(default=None, ge=0)
    cases: tuple[FullIntentCaseResult, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.sample_count != EXPECTED_FULL_CASES or len(self.cases) != self.sample_count:
            raise ValueError("完整 Intent 报告必须覆盖固定 150 条样本")
        if self.successful_calls + self.failed_calls != self.sample_count:
            raise ValueError("Intent 成功和失败数必须覆盖全部样本")
        if self.rag.sample_count != EXPECTED_RAG_CASES:
            raise ValueError("Intent RAG 切片必须包含 132 条")
        if self.no_rag.sample_count != EXPECTED_NO_RAG_CASES:
            raise ValueError("Intent no-rag 切片必须包含 18 条")
        if sum(item.sample_count for item in self.by_intent_l1.values()) != self.sample_count:
            raise ValueError("Intent L1 切片必须覆盖全部样本")
        if sum(sum(row.values()) for row in self.confusion.values()) != self.sample_count:
            raise ValueError("Intent 混淆矩阵必须覆盖全部样本")
        return self


async def run_full_intent_evaluation(
    dataset: FullEvaluationDataset,
    classifier: IntentClassifier,
    *,
    abort_on_degradation: bool = True,
    configuration: FullIntentEvaluationConfiguration | None = None,
    initial_cases: tuple[FullIntentCaseResult, ...] = (),
    on_case: Callable[[FullIntentCaseResult], None] | None = None,
) -> FullIntentReport:
    """Classify the remaining full-set queries and checkpoint each sanitized result."""
    _validate_initial_case_prefix(dataset, initial_cases)
    new_cases = await run_intent_case_evaluation(
        dataset.samples[len(initial_cases) :],
        classifier,
        abort_on_degradation=abort_on_degradation,
        on_case=on_case,
    )
    return build_full_intent_report(
        dataset,
        (*initial_cases, *new_cases),
        configuration=configuration,
    )


async def run_intent_case_evaluation(
    samples: tuple[FullEvaluationSample, ...],
    classifier: IntentClassifier,
    *,
    abort_on_degradation: bool = True,
    on_case: Callable[[FullIntentCaseResult], None] | None = None,
) -> tuple[FullIntentCaseResult, ...]:
    """Classify an explicit sample subset while preserving full-run safety semantics."""
    cases: list[FullIntentCaseResult] = []
    for sample in samples:
        expected_route = (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        started = perf_counter()
        decision = await classifier.classify(IntentClassificationRequest(uuid4(), sample.query))
        latency_ms = (perf_counter() - started) * 1000
        degradation = decision.degradation_reason
        usage = decision.usage
        case_result = FullIntentCaseResult(
            query_id=sample.query_id,
            intent_l1=sample.intent_l1,
            expected_route=expected_route,
            actual_route=decision.route,
            correct=degradation is None and decision.route is expected_route,
            decision_reason=decision.reason.value,
            degradation_reason=degradation,
            model_error_code=decision.model_error_code,
            candidate_scores=(
                IntentCandidateScores(
                    **{candidate.route.value: candidate.score for candidate in decision.candidates}
                )
                if degradation is None
                else None
            ),
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
        )
        if on_case is not None:
            on_case(case_result)
        if degradation is not None and abort_on_degradation:
            error_code = decision.model_error_code
            raise FullIntentRunError(
                sample.query_id,
                error_code.value if error_code is not None else degradation.value,
            )
        cases.append(case_result)

    return tuple(cases)


def build_full_intent_report(
    dataset: FullEvaluationDataset,
    cases: tuple[FullIntentCaseResult, ...],
    *,
    configuration: FullIntentEvaluationConfiguration | None = None,
) -> FullIntentReport:
    """Build the strict 150-case report from cached cases in dataset order."""
    _validate_complete_case_sequence(dataset, cases)

    return _build_report(
        dataset.dataset_id,
        configuration,
        cases,
        limitations=(
            "全量集没有 clarification 路由真值, 不报告 clarification 准确率。",
            "requires_rag=true 映射 knowledge_base, no-rag 映射 system_direct。",
            "Intent 只评估路由决策, 不评估最终答案质量或工具执行。",
        ),
    )


def replay_full_intent_thresholds(
    report: FullIntentReport,
    *,
    high_confidence_threshold: float,
    ambiguity_margin: float,
) -> FullIntentReport:
    """Recompute routes from one paid run without making additional model calls."""
    if report.configuration is None:
        raise ValueError("Intent 阈值重放需要完整评测配置")
    replayed: list[FullIntentCaseResult] = []
    for case in report.cases:
        if case.degradation_reason is not None:
            replayed.append(case)
            continue
        if case.candidate_scores is None:
            raise ValueError("Intent 阈值重放需要每条成功样本的候选分数")
        route, reason = select_intent_route(
            case.candidate_scores.as_candidates(),
            high_confidence_threshold=high_confidence_threshold,
            ambiguity_margin=ambiguity_margin,
        )
        replayed.append(
            FullIntentCaseResult(
                **case.model_dump(exclude={"actual_route", "correct", "decision_reason"}),
                actual_route=route,
                correct=route is case.expected_route,
                decision_reason=reason.value,
            )
        )

    configuration = FullIntentEvaluationConfiguration(
        **report.configuration.model_dump(
            exclude={"high_confidence_threshold", "ambiguity_margin"}
        ),
        high_confidence_threshold=high_confidence_threshold,
        ambiguity_margin=ambiguity_margin,
    )
    return _build_report(
        report.dataset_id,
        configuration,
        tuple(replayed),
        limitations=(
            *report.limitations,
            "阈值重放复用同一轮模型分数, 不产生额外模型调用或新的独立样本。",
        ),
    )


def _build_report(
    dataset_id: str,
    configuration: FullIntentEvaluationConfiguration | None,
    case_results: tuple[FullIntentCaseResult, ...],
    *,
    limitations: tuple[str, ...],
) -> FullIntentReport:
    rag_cases = tuple(
        case for case in case_results if case.expected_route is IntentRoute.KNOWLEDGE_BASE
    )
    no_rag_cases = tuple(
        case for case in case_results if case.expected_route is IntentRoute.SYSTEM_DIRECT
    )
    intent_l1_values = tuple(dict.fromkeys(case.intent_l1 for case in case_results))
    latencies = tuple(case.latency_ms for case in case_results)
    failures = sum(case.degradation_reason is not None for case in case_results)
    return FullIntentReport(
        dataset_id=dataset_id,
        configuration=configuration,
        sample_count=len(case_results),
        successful_calls=len(case_results) - failures,
        failed_calls=failures,
        input_tokens=sum(case.input_tokens or 0 for case in case_results),
        output_tokens=sum(case.output_tokens or 0 for case in case_results),
        overall=_slice_metrics(case_results),
        rag=_slice_metrics(rag_cases),
        no_rag=_slice_metrics(no_rag_cases),
        by_intent_l1={
            intent_l1: _slice_metrics(
                tuple(case for case in case_results if case.intent_l1 == intent_l1)
            )
            for intent_l1 in intent_l1_values
        },
        confusion=_confusion(case_results),
        latency_ms_p50=_percentile(latencies, 0.50),
        latency_ms_p95=_percentile(latencies, 0.95),
        cases=case_results,
        limitations=limitations,
    )


def _validate_initial_case_prefix(
    dataset: FullEvaluationDataset,
    cases: tuple[FullIntentCaseResult, ...],
) -> None:
    if len(cases) > len(dataset.samples):
        raise ValueError("Intent checkpoint 超过数据集长度")
    for sample, case in zip(dataset.samples, cases, strict=False):
        expected_route = (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        if (
            case.query_id != sample.query_id
            or case.intent_l1 != sample.intent_l1
            or case.expected_route is not expected_route
            or case.degradation_reason is not None
        ):
            raise ValueError("Intent checkpoint 必须是当前数据集的连续成功前缀")


def _validate_complete_case_sequence(
    dataset: FullEvaluationDataset,
    cases: tuple[FullIntentCaseResult, ...],
) -> None:
    if len(cases) != len(dataset.samples):
        raise ValueError("完整 Intent 报告必须包含数据集全部样本")
    for sample, case in zip(dataset.samples, cases, strict=True):
        expected_route = (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        if (
            case.query_id != sample.query_id
            or case.intent_l1 != sample.intent_l1
            or case.expected_route is not expected_route
        ):
            raise ValueError("完整 Intent 缓存必须按固定数据集顺序匹配标签")


def _slice_metrics(cases: tuple[FullIntentCaseResult, ...]) -> IntentSliceMetrics:
    correct = sum(case.correct for case in cases)
    return IntentSliceMetrics(
        sample_count=len(cases),
        correct_count=correct,
        accuracy=correct / len(cases),
    )


def _confusion(
    cases: tuple[FullIntentCaseResult, ...],
) -> dict[IntentRoute, dict[IntentRoute, int]]:
    counts = Counter((case.expected_route, case.actual_route) for case in cases)
    return {
        expected: {actual: counts[(expected, actual)] for actual in IntentRoute}
        for expected in IntentRoute
    }


def _percentile(values: tuple[float, ...], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
