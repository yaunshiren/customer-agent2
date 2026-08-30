"""Content-safe 150-case intent-route evaluation."""

import math
from collections import Counter
from time import perf_counter
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.domain.models import (
    IntentClassificationRequest,
    IntentClassifier,
    IntentDegradationReason,
    IntentRoute,
)
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_FULL_CASES,
    EXPECTED_NO_RAG_CASES,
    EXPECTED_RAG_CASES,
    FullEvaluationDataset,
)


class FullIntentRunError(RuntimeError):
    """Sanitized fatal error that stops the live run at first degradation."""

    def __init__(self, query_id: str, error_code: str) -> None:
        super().__init__(f"完整 Intent 评测在 {query_id} 安全终止: {error_code}")
        self.query_id = query_id
        self.error_code = error_code


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
        return self


class FullIntentReport(BaseModel):
    """Aggregate and per-case intent evaluation over all 150 samples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
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
) -> FullIntentReport:
    """Classify every full-set query once and count degradation as incorrect."""
    cases: list[FullIntentCaseResult] = []
    for sample in dataset.samples:
        expected_route = (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        started = perf_counter()
        decision = await classifier.classify(IntentClassificationRequest(uuid4(), sample.query))
        latency_ms = (perf_counter() - started) * 1000
        degradation = decision.degradation_reason
        if degradation is not None and abort_on_degradation:
            raise FullIntentRunError(sample.query_id, degradation.value)
        usage = decision.usage
        cases.append(
            FullIntentCaseResult(
                query_id=sample.query_id,
                intent_l1=sample.intent_l1,
                expected_route=expected_route,
                actual_route=decision.route,
                correct=degradation is None and decision.route is expected_route,
                decision_reason=decision.reason.value,
                degradation_reason=degradation,
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens if usage is not None else None,
                output_tokens=usage.output_tokens if usage is not None else None,
            )
        )

    case_results = tuple(cases)
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
        dataset_id=dataset.dataset_id,
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
        limitations=(
            "全量集没有 clarification 路由真值, 不报告 clarification 准确率。",
            "requires_rag=true 映射 knowledge_base, no-rag 映射 system_direct。",
            "Intent 只评估路由决策, 不评估最终答案质量或工具执行。",
        ),
    )


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
