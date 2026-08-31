"""Budget-aware staged Intent evaluation with exact content-free cache reuse."""

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.domain.models import IntentRoute
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_FULL_CASES,
    FullEvaluationDataset,
    FullEvaluationSample,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    FullIntentReport,
    IntentSliceMetrics,
)

EXPECTED_FAILURE_CASES = 22
EXPECTED_GUARD_CASES = 18
EXPECTED_SCREEN_CASES = EXPECTED_FAILURE_CASES + EXPECTED_GUARD_CASES


class IntentEvaluationStage(StrEnum):
    """The three cumulative paid-call gates for one candidate tree."""

    FAILURES = "failures"
    GUARD = "guard"
    FULL = "full"


class M5DIntentStageManifest(BaseModel):
    """Versioned Query-ID-only stage membership."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=100)
    failure_query_ids: tuple[str, ...]
    guard_query_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        if len(self.failure_query_ids) != EXPECTED_FAILURE_CASES:
            raise ValueError("M5-D 第一阶段必须恰好包含 22 条历史错误")
        if len(self.guard_query_ids) != EXPECTED_GUARD_CASES:
            raise ValueError("M5-D 第二阶段必须恰好增加 18 条回归保护")
        combined = (*self.failure_query_ids, *self.guard_query_ids)
        if len(set(combined)) != len(combined):
            raise ValueError("M5-D 阶段 Query ID 不能重复")
        return self


class StagedIntentCache(BaseModel):
    """Candidate-specific successful cases reusable across cumulative stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=200)
    manifest_version: str = Field(min_length=1, max_length=100)
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: FullIntentEvaluationConfiguration
    completed_cases: tuple[FullIntentCaseResult, ...] = ()
    failed_attempts: tuple[FullIntentFailedAttempt, ...] = ()

    @model_validator(mode="after")
    def validate_cases(self) -> Self:
        if len(self.completed_cases) > EXPECTED_FULL_CASES:
            raise ValueError("Intent 阶段缓存不能超过完整数据集")
        query_ids = tuple(case.query_id for case in self.completed_cases)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Intent 阶段缓存不能重复 Query ID")
        if any(
            case.degradation_reason is not None or case.candidate_scores is None
            for case in self.completed_cases
        ):
            raise ValueError("Intent 阶段缓存只保存带候选分数的成功调用")
        return self


class IntentStageGateCheck(BaseModel):
    """One pre-registered integer gate with its observed value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=100)
    comparison: Literal["eq", "gte", "gt"]
    actual: int = Field(ge=0)
    required: int = Field(ge=0)
    passed: bool

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        expected = {
            "eq": self.actual == self.required,
            "gte": self.actual >= self.required,
            "gt": self.actual > self.required,
        }[self.comparison]
        if self.passed != expected:
            raise ValueError("Intent 阶段门禁结果与比较规则不一致")
        return self


class IntentStageGate(BaseModel):
    """All checks that must pass before the next paid stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    checks: tuple[IntentStageGateCheck, ...]

    @model_validator(mode="after")
    def validate_passed(self) -> Self:
        if not self.checks or self.passed != all(check.passed for check in self.checks):
            raise ValueError("Intent 阶段总门禁必须由全部子检查决定")
        return self


class StagedIntentReport(BaseModel):
    """Sanitized report for one cumulative stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=200)
    manifest_version: str = Field(min_length=1, max_length=100)
    stage: IntentEvaluationStage
    configuration: FullIntentEvaluationConfiguration
    target_sample_count: int = Field(ge=1, le=EXPECTED_FULL_CASES)
    cached_target_before_count: int = Field(ge=0, le=EXPECTED_FULL_CASES)
    new_calls: int = Field(ge=0, le=EXPECTED_FULL_CASES)
    total_cached_count: int = Field(ge=0, le=EXPECTED_FULL_CASES)
    failed_attempt_count: int = Field(ge=0)
    total_attempt_count: int = Field(ge=0)
    remaining_full_calls: int = Field(ge=0, le=EXPECTED_FULL_CASES)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    overall: IntentSliceMetrics
    rag: IntentSliceMetrics
    no_rag: IntentSliceMetrics
    by_intent_l1: dict[str, IntentSliceMetrics]
    under_retrieval_count: int = Field(ge=0)
    over_retrieval_count: int = Field(ge=0)
    clarification_count: int = Field(ge=0)
    gate: IntentStageGate
    cases: tuple[FullIntentCaseResult, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.target_sample_count != len(self.cases):
            raise ValueError("Intent 阶段报告必须覆盖当前阶段全部样本")
        if self.cached_target_before_count + self.new_calls != self.target_sample_count:
            raise ValueError("Intent 阶段缓存复用数与新增调用数必须覆盖目标")
        if self.total_cached_count < self.target_sample_count:
            raise ValueError("Intent 总缓存数不能小于当前阶段目标")
        if self.total_attempt_count != self.total_cached_count + self.failed_attempt_count:
            raise ValueError("Intent 总尝试数必须包含成功缓存和失败尝试")
        if self.remaining_full_calls != EXPECTED_FULL_CASES - self.total_cached_count:
            raise ValueError("Intent 剩余完整调用数与总缓存数不一致")
        if sum(item.sample_count for item in self.by_intent_l1.values()) != len(self.cases):
            raise ValueError("Intent 阶段 L1 切片必须覆盖全部目标")
        return self


def load_m5d_stage_manifest(content: str) -> M5DIntentStageManifest:
    """Parse a strict stage manifest without accepting extra fields."""
    return M5DIntentStageManifest.model_validate_json(content)


def staged_manifest_fingerprint(manifest: M5DIntentStageManifest) -> str:
    """Hash normalized stage membership for exact cache compatibility."""
    return _model_fingerprint(manifest)


def baseline_report_fingerprint(report: FullIntentReport) -> str:
    """Hash the parsed baseline so stage membership cannot drift silently."""
    return _model_fingerprint(report)


def validate_m5d_stage_manifest(
    manifest: M5DIntentStageManifest,
    dataset: FullEvaluationDataset,
    baseline: FullIntentReport,
) -> None:
    """Prove that failures and guards still match the committed M5-C baseline."""
    if baseline.dataset_id != dataset.dataset_id:
        raise ValueError("M5-D Baseline 与固定数据集 ID 不一致")
    baseline_by_id = {case.query_id: case for case in baseline.cases}
    samples_by_id = {sample.query_id: sample for sample in dataset.samples}
    if set(baseline_by_id) != set(samples_by_id):
        raise ValueError("M5-D Baseline 与固定数据集 Query ID 不一致")
    expected_failures = {case.query_id for case in baseline.cases if not case.correct}
    if set(manifest.failure_query_ids) != expected_failures:
        raise ValueError("M5-D 第一阶段必须与 M5-C 的 22 条错误完全一致")
    if any(not baseline_by_id[query_id].correct for query_id in manifest.guard_query_ids):
        raise ValueError("M5-D 回归保护只能选择 M5-C 原本正确的样本")

    failure_samples = tuple(samples_by_id[query_id] for query_id in manifest.failure_query_ids)
    guard_samples = tuple(samples_by_id[query_id] for query_id in manifest.guard_query_ids)
    if Counter(sample.intent_l1 for sample in failure_samples) != {
        "SUPPORT": 9,
        "FEEDBACK": 10,
        "CHAT": 3,
    }:
        raise ValueError("M5-D 失败集 L1 分层发生漂移")
    if _route_counts(failure_samples) != {
        IntentRoute.KNOWLEDGE_BASE: 11,
        IntentRoute.SYSTEM_DIRECT: 11,
    }:
        raise ValueError("M5-D 失败集路由分层发生漂移")
    if Counter(sample.intent_l1 for sample in guard_samples) != {
        "SUPPORT": 6,
        "FEEDBACK": 5,
        "CHAT": 7,
    }:
        raise ValueError("M5-D 回归集 L1 分层发生漂移")
    if _route_counts(guard_samples) != {
        IntentRoute.KNOWLEDGE_BASE: 11,
        IntentRoute.SYSTEM_DIRECT: 7,
    }:
        raise ValueError("M5-D 回归集路由分层发生漂移")


def validate_staged_intent_cache(
    cache: StagedIntentCache,
    dataset: FullEvaluationDataset,
) -> None:
    """Reject cached cases whose labels no longer match the immutable dataset."""
    if cache.dataset_id != dataset.dataset_id:
        raise ValueError("Intent 阶段缓存与数据集 ID 不一致")
    samples_by_id = {sample.query_id: sample for sample in dataset.samples}
    if any(attempt.query_id not in samples_by_id for attempt in cache.failed_attempts):
        raise ValueError("Intent 阶段缓存的失败记录包含未知 Query ID")
    for case in cache.completed_cases:
        sample = samples_by_id.get(case.query_id)
        if sample is None:
            raise ValueError("Intent 阶段缓存包含未知 Query ID")
        expected_route = _expected_route(sample)
        if case.intent_l1 != sample.intent_l1 or case.expected_route is not expected_route:
            raise ValueError("Intent 阶段缓存标签与固定数据集不一致")


def stage_query_ids(
    stage: IntentEvaluationStage,
    manifest: M5DIntentStageManifest,
    dataset: FullEvaluationDataset,
) -> tuple[str, ...]:
    """Return the cumulative target membership for one stage."""
    if stage is IntentEvaluationStage.FAILURES:
        return manifest.failure_query_ids
    if stage is IntentEvaluationStage.GUARD:
        return (*manifest.failure_query_ids, *manifest.guard_query_ids)
    return tuple(sample.query_id for sample in dataset.samples)


def pending_stage_samples(
    stage: IntentEvaluationStage,
    manifest: M5DIntentStageManifest,
    dataset: FullEvaluationDataset,
    cache: StagedIntentCache,
) -> tuple[FullEvaluationSample, ...]:
    """Return only target cases not already paid for under the exact cache fingerprint."""
    validate_staged_intent_cache(cache, dataset)
    completed = {case.query_id for case in cache.completed_cases}
    samples_by_id = {sample.query_id: sample for sample in dataset.samples}
    return tuple(
        samples_by_id[query_id]
        for query_id in stage_query_ids(stage, manifest, dataset)
        if query_id not in completed
    )


def ensure_stage_unlocked(
    stage: IntentEvaluationStage,
    manifest: M5DIntentStageManifest,
    dataset: FullEvaluationDataset,
    cache: StagedIntentCache,
) -> None:
    """Prevent paying for a later stage until all earlier gates pass."""
    if stage is IntentEvaluationStage.FAILURES:
        return
    case_by_id = {case.query_id: case for case in cache.completed_cases}
    failure_cases = _require_cached_cases(manifest.failure_query_ids, case_by_id)
    failure_gate = evaluate_stage_gate(
        IntentEvaluationStage.FAILURES,
        failure_cases,
        manifest,
    )
    if not failure_gate.passed:
        raise ValueError("M5-D 失败集门禁未通过, 不允许产生回归集费用")
    if stage is IntentEvaluationStage.GUARD:
        return
    screen_ids = (*manifest.failure_query_ids, *manifest.guard_query_ids)
    screen_cases = _require_cached_cases(screen_ids, case_by_id)
    screen_gate = evaluate_stage_gate(IntentEvaluationStage.GUARD, screen_cases, manifest)
    if not screen_gate.passed:
        raise ValueError("M5-D 回归门禁未通过, 不允许产生完整集费用")


def build_staged_intent_report(
    stage: IntentEvaluationStage,
    manifest: M5DIntentStageManifest,
    dataset: FullEvaluationDataset,
    cache: StagedIntentCache,
    *,
    cached_target_before_count: int,
    new_calls: int,
) -> StagedIntentReport:
    """Build one cumulative sanitized stage report from reusable cached results."""
    validate_staged_intent_cache(cache, dataset)
    query_ids = stage_query_ids(stage, manifest, dataset)
    case_by_id = {case.query_id: case for case in cache.completed_cases}
    cases = _require_cached_cases(query_ids, case_by_id)
    rag_cases = tuple(case for case in cases if case.expected_route is IntentRoute.KNOWLEDGE_BASE)
    no_rag_cases = tuple(case for case in cases if case.expected_route is IntentRoute.SYSTEM_DIRECT)
    l1_values = tuple(dict.fromkeys(case.intent_l1 for case in cases))
    return StagedIntentReport(
        dataset_id=dataset.dataset_id,
        manifest_version=manifest.version,
        stage=stage,
        configuration=cache.configuration,
        target_sample_count=len(cases),
        cached_target_before_count=cached_target_before_count,
        new_calls=new_calls,
        total_cached_count=len(cache.completed_cases),
        failed_attempt_count=len(cache.failed_attempts),
        total_attempt_count=len(cache.completed_cases) + len(cache.failed_attempts),
        remaining_full_calls=EXPECTED_FULL_CASES - len(cache.completed_cases),
        input_tokens=sum(case.input_tokens or 0 for case in cases),
        output_tokens=sum(case.output_tokens or 0 for case in cases),
        overall=_slice_metrics(cases),
        rag=_slice_metrics(rag_cases),
        no_rag=_slice_metrics(no_rag_cases),
        by_intent_l1={
            intent_l1: _slice_metrics(tuple(case for case in cases if case.intent_l1 == intent_l1))
            for intent_l1 in l1_values
        },
        under_retrieval_count=_under_retrieval_count(cases),
        over_retrieval_count=sum(
            case.expected_route is IntentRoute.SYSTEM_DIRECT
            and case.actual_route is IntentRoute.KNOWLEDGE_BASE
            for case in cases
        ),
        clarification_count=sum(case.actual_route is IntentRoute.CLARIFICATION for case in cases),
        gate=evaluate_stage_gate(stage, cases, manifest),
        cases=cases,
        limitations=(
            "阶段报告只保存 Query ID、路由、分数、延迟和 Token, 不保存问题或模型原文。",
            "失败集和回归集用于候选筛选, 只有完整 150 条可用于最终晋级结论。",
            "同一候选的完整配置指纹不变时复用缓存, 不重复产生调用费用。",
            "尝试次数来自本地逐条回调, 用于审计但不能替代供应商账单。",
        ),
    )


def evaluate_stage_gate(
    stage: IntentEvaluationStage,
    cases: tuple[FullIntentCaseResult, ...],
    manifest: M5DIntentStageManifest,
) -> IntentStageGate:
    """Evaluate the pre-registered stop/go rules for one completed stage."""
    case_by_id = {case.query_id: case for case in cases}
    failure_cases = _require_cached_cases(manifest.failure_query_ids, case_by_id)
    failure_rag = tuple(
        case for case in failure_cases if case.expected_route is IntentRoute.KNOWLEDGE_BASE
    )
    failure_direct = tuple(
        case for case in failure_cases if case.expected_route is IntentRoute.SYSTEM_DIRECT
    )
    checks = [
        _gate_check("failure_rag_correct", "gte", sum(case.correct for case in failure_rag), 8),
        _gate_check(
            "failure_no_rag_correct",
            "gte",
            sum(case.correct for case in failure_direct),
            8,
        ),
        _gate_check(
            "failure_under_retrieval",
            "eq",
            _under_retrieval_count(failure_cases),
            0,
        ),
    ]
    if stage in {IntentEvaluationStage.GUARD, IntentEvaluationStage.FULL}:
        guard_cases = _require_cached_cases(manifest.guard_query_ids, case_by_id)
        checks.extend(
            (
                _gate_check(
                    "guard_correct",
                    "gte",
                    sum(case.correct for case in guard_cases),
                    17,
                ),
                _gate_check(
                    "guard_under_retrieval",
                    "eq",
                    _under_retrieval_count(guard_cases),
                    0,
                ),
            )
        )
    if stage is IntentEvaluationStage.FULL:
        rag_cases = tuple(
            case for case in cases if case.expected_route is IntentRoute.KNOWLEDGE_BASE
        )
        no_rag_cases = tuple(
            case for case in cases if case.expected_route is IntentRoute.SYSTEM_DIRECT
        )
        checks.extend(
            (
                _gate_check("full_overall_correct", "gt", sum(case.correct for case in cases), 128),
                _gate_check(
                    "full_rag_correct", "gte", sum(case.correct for case in rag_cases), 121
                ),
                _gate_check(
                    "full_no_rag_correct",
                    "gt",
                    sum(case.correct for case in no_rag_cases),
                    7,
                ),
                _gate_check(
                    "full_under_retrieval",
                    "eq",
                    _under_retrieval_count(cases),
                    0,
                ),
            )
        )
    gate_checks = tuple(checks)
    return IntentStageGate(
        passed=all(check.passed for check in gate_checks),
        checks=gate_checks,
    )


def ordered_full_cache_cases(
    dataset: FullEvaluationDataset,
    cache: StagedIntentCache,
) -> tuple[FullIntentCaseResult, ...]:
    """Return all cached cases in immutable dataset order for the final report."""
    validate_staged_intent_cache(cache, dataset)
    case_by_id = {case.query_id: case for case in cache.completed_cases}
    return _require_cached_cases(
        tuple(sample.query_id for sample in dataset.samples),
        case_by_id,
    )


def _model_fingerprint(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_route(sample: FullEvaluationSample) -> IntentRoute:
    return IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT


def _route_counts(samples: tuple[FullEvaluationSample, ...]) -> dict[IntentRoute, int]:
    return dict(Counter(_expected_route(sample) for sample in samples))


def _require_cached_cases(
    query_ids: tuple[str, ...],
    case_by_id: dict[str, FullIntentCaseResult],
) -> tuple[FullIntentCaseResult, ...]:
    missing = tuple(query_id for query_id in query_ids if query_id not in case_by_id)
    if missing:
        raise ValueError(f"Intent 阶段缓存尚缺 {len(missing)} 条目标样本")
    return tuple(case_by_id[query_id] for query_id in query_ids)


def _slice_metrics(cases: tuple[FullIntentCaseResult, ...]) -> IntentSliceMetrics:
    if not cases:
        raise ValueError("Intent 阶段切片不能为空")
    correct = sum(case.correct for case in cases)
    return IntentSliceMetrics(
        sample_count=len(cases),
        correct_count=correct,
        accuracy=correct / len(cases),
    )


def _under_retrieval_count(cases: tuple[FullIntentCaseResult, ...]) -> int:
    return sum(
        case.expected_route is IntentRoute.KNOWLEDGE_BASE
        and case.actual_route is IntentRoute.SYSTEM_DIRECT
        for case in cases
    )


def _gate_check(
    name: str,
    comparison: Literal["eq", "gte", "gt"],
    actual: int,
    required: int,
) -> IntentStageGateCheck:
    passed = {
        "eq": actual == required,
        "gte": actual >= required,
        "gt": actual > required,
    }[comparison]
    return IntentStageGateCheck(
        name=name,
        comparison=comparison,
        actual=actual,
        required=required,
        passed=passed,
    )
