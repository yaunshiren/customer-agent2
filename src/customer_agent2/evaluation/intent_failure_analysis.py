"""Content-free failure slicing for one complete Intent report."""

from collections import Counter, defaultdict
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.domain.models import IntentRoute
from customer_agent2.evaluation.full_dataset import FullEvaluationDataset
from customer_agent2.evaluation.full_intent import FullIntentCaseResult, FullIntentReport


class IntentFailureBucket(BaseModel):
    """One stable failure slice identified without retaining question text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_l1: str = Field(min_length=1, max_length=100)
    intent_l2: str = Field(min_length=1, max_length=100)
    expected_route: IntentRoute
    actual_route: IntentRoute
    decision_reason: str = Field(min_length=1, max_length=100)
    sample_count: int = Field(ge=1)
    query_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_query_ids(self) -> Self:
        if self.sample_count != len(self.query_ids) or len(set(self.query_ids)) != len(
            self.query_ids
        ):
            raise ValueError("Intent 失败切片数量必须与唯一 Query ID 一致")
        return self


class IntentFailureAnalysis(BaseModel):
    """Aggregate failure taxonomy used to pre-register the next experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=200)
    source_model_id: str | None = Field(default=None, max_length=200)
    source_intent_tree_version: str | None = Field(default=None, max_length=100)
    sample_count: int = Field(ge=1)
    correct_count: int = Field(ge=0)
    incorrect_count: int = Field(ge=0)
    over_retrieval_count: int = Field(ge=0)
    under_retrieval_count: int = Field(ge=0)
    incorrect_clarification_count: int = Field(ge=0)
    incorrect_by_intent_l1: dict[str, int]
    incorrect_by_decision_reason: dict[str, int]
    buckets: tuple[IntentFailureBucket, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.correct_count + self.incorrect_count != self.sample_count:
            raise ValueError("Intent 成功和失败切片必须覆盖完整报告")
        if sum(self.incorrect_by_intent_l1.values()) != self.incorrect_count:
            raise ValueError("Intent L1 失败切片必须覆盖全部错误")
        if sum(self.incorrect_by_decision_reason.values()) != self.incorrect_count:
            raise ValueError("Intent 决策原因切片必须覆盖全部错误")
        if sum(bucket.sample_count for bucket in self.buckets) != self.incorrect_count:
            raise ValueError("Intent 失败 Bucket 必须覆盖全部错误")
        query_ids = tuple(query_id for bucket in self.buckets for query_id in bucket.query_ids)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("Intent 失败 Bucket 不能重复 Query ID")
        return self


def analyze_full_intent_failures(
    dataset: FullEvaluationDataset,
    report: FullIntentReport,
) -> IntentFailureAnalysis:
    """Join immutable labels to sanitized cases and emit no question or answer text."""
    if report.dataset_id != dataset.dataset_id:
        raise ValueError("Intent 报告与评测数据集 ID 不一致")
    if len(report.cases) != len(dataset.samples):
        raise ValueError("Intent 报告与评测数据集样本数不一致")

    failed_cases: list[FullIntentCaseResult] = []
    groups: defaultdict[tuple[str, str, IntentRoute, IntentRoute, str], list[str]] = defaultdict(
        list
    )
    for sample, case in zip(dataset.samples, report.cases, strict=True):
        expected_route = (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        if (
            case.query_id != sample.query_id
            or case.intent_l1 != sample.intent_l1
            or case.expected_route is not expected_route
        ):
            raise ValueError("Intent 报告样本顺序或标签与固定数据集不一致")
        if case.correct:
            continue
        failed_cases.append(case)
        groups[
            (
                sample.intent_l1,
                sample.intent_l2,
                case.expected_route,
                case.actual_route,
                case.decision_reason,
            )
        ].append(case.query_id)

    buckets = tuple(
        IntentFailureBucket(
            intent_l1=key[0],
            intent_l2=key[1],
            expected_route=key[2],
            actual_route=key[3],
            decision_reason=key[4],
            sample_count=len(query_ids),
            query_ids=tuple(query_ids),
        )
        for key, query_ids in sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))
    )
    failure_results = tuple(failed_cases)
    configuration = report.configuration
    return IntentFailureAnalysis(
        dataset_id=dataset.dataset_id,
        source_model_id=configuration.model_id if configuration is not None else None,
        source_intent_tree_version=(
            configuration.intent_tree_version if configuration is not None else None
        ),
        sample_count=report.sample_count,
        correct_count=report.overall.correct_count,
        incorrect_count=len(failure_results),
        over_retrieval_count=sum(
            case.expected_route is IntentRoute.SYSTEM_DIRECT
            and case.actual_route is IntentRoute.KNOWLEDGE_BASE
            for case in failure_results
        ),
        under_retrieval_count=sum(
            case.expected_route is IntentRoute.KNOWLEDGE_BASE
            and case.actual_route is IntentRoute.SYSTEM_DIRECT
            for case in failure_results
        ),
        incorrect_clarification_count=sum(
            case.actual_route is IntentRoute.CLARIFICATION for case in failure_results
        ),
        incorrect_by_intent_l1=dict(Counter(case.intent_l1 for case in failure_results)),
        incorrect_by_decision_reason=dict(
            Counter(case.decision_reason for case in failure_results)
        ),
        buckets=buckets,
        limitations=(
            "报告只保存聚合计数和 Query ID, 不保存 Query、Ground Truth 或模型原始响应。",
            "M5-C 没有保存候选分数, 不能从本报告可靠重放其他阈值。",
            "requires_rag 仅作为当前二分类路由真值, 不等同于最终答案质量。",
        ),
    )
