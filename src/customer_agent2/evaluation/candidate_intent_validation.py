"""Frozen, budget-aware validation for the M5-D v4 Intent boundary candidate."""

import hashlib
import json
from collections import Counter
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.domain.models import IntentRoute
from customer_agent2.evaluation.full_dataset import FullEvaluationDataset
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    IntentEvaluationSample,
    IntentSliceMetrics,
)
from customer_agent2.evaluation.staged_intent import IntentStageGate, IntentStageGateCheck

EXPECTED_DEVELOPMENT_CASES = 4
EXPECTED_CHALLENGE_CASES = 20
EXPECTED_TOTAL_CASES = EXPECTED_DEVELOPMENT_CASES + EXPECTED_CHALLENGE_CASES
V4_DEVELOPMENT_QUERY_IDS = ("S3-08", "S12-06", "C2-02", "C2-04")


class CandidateIntentValidationStage(StrEnum):
    """Paid stages for one focused candidate validation."""

    DEVELOPMENT = "development"
    CHALLENGE = "challenge"


class CandidateIntentChallengeSample(BaseModel):
    """One newly authored route-only challenge without answer gold text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(pattern=r"^V4-(KB|DIRECT)-\d{2}$")
    query: str = Field(min_length=2, max_length=1_000)
    intent_l1: Literal["SUPPORT", "CHAT"]
    expected_route: Literal[IntentRoute.KNOWLEDGE_BASE, IntentRoute.SYSTEM_DIRECT]
    slice: str = Field(min_length=1, max_length=100)

    @property
    def requires_rag(self) -> bool:
        """Expose the shared evaluation sample contract."""
        return self.expected_route is IntentRoute.KNOWLEDGE_BASE


class CandidateIntentValidationManifest(BaseModel):
    """Frozen development IDs and new boundary challenge cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1, max_length=100)
    development_query_ids: tuple[str, ...]
    challenge_samples: tuple[CandidateIntentChallengeSample, ...]

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> Self:
        if self.development_query_ids != V4_DEVELOPMENT_QUERY_IDS:
            raise ValueError("v4 开发探针必须保持预登记的 4 条边界样本及顺序")
        if len(self.challenge_samples) != EXPECTED_CHALLENGE_CASES:
            raise ValueError("v4 冻结挑战集必须恰好包含 20 条")
        query_ids = tuple(sample.query_id for sample in self.challenge_samples)
        normalized_queries = tuple(
            _normalize_query(sample.query) for sample in self.challenge_samples
        )
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("v4 挑战集 Query ID 不能重复")
        if len(set(normalized_queries)) != len(normalized_queries):
            raise ValueError("v4 挑战集问题不能重复")
        route_counts = Counter(sample.expected_route for sample in self.challenge_samples)
        if route_counts != {
            IntentRoute.KNOWLEDGE_BASE: 10,
            IntentRoute.SYSTEM_DIRECT: 10,
        }:
            raise ValueError("v4 挑战集必须保持 10 条检索与 10 条直接回答")
        if any(
            (sample.expected_route is IntentRoute.KNOWLEDGE_BASE) != (sample.intent_l1 == "SUPPORT")
            for sample in self.challenge_samples
        ):
            raise ValueError("v4 挑战集 SUPPORT/CHAT 必须与检索/直接回答边界一致")
        return self


class CandidateIntentValidationCache(BaseModel):
    """Successful paid results bound to the exact v4 validation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=200)
    manifest_version: str = Field(min_length=1, max_length=100)
    validation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: FullIntentEvaluationConfiguration
    completed_cases: tuple[FullIntentCaseResult, ...] = ()
    failed_attempts: tuple[FullIntentFailedAttempt, ...] = ()

    @model_validator(mode="after")
    def validate_cache_shape(self) -> Self:
        if len(self.completed_cases) > EXPECTED_TOTAL_CASES:
            raise ValueError("v4 验证缓存不能超过 24 条成功样本")
        query_ids = tuple(case.query_id for case in self.completed_cases)
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("v4 验证缓存不能重复 Query ID")
        if any(
            case.degradation_reason is not None or case.candidate_scores is None
            for case in self.completed_cases
        ):
            raise ValueError("v4 验证缓存只保存带候选分数的成功调用")
        return self


class CandidateIntentValidationReport(BaseModel):
    """Content-free result for one focused v4 paid stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=200)
    manifest_version: str = Field(min_length=1, max_length=100)
    stage: CandidateIntentValidationStage
    configuration: FullIntentEvaluationConfiguration
    target_sample_count: int = Field(ge=1, le=EXPECTED_CHALLENGE_CASES)
    cached_target_before_count: int = Field(ge=0, le=EXPECTED_CHALLENGE_CASES)
    new_calls: int = Field(ge=0, le=EXPECTED_CHALLENGE_CASES)
    total_cached_count: int = Field(ge=0, le=EXPECTED_TOTAL_CASES)
    failed_attempt_count: int = Field(ge=0)
    total_attempt_count: int = Field(ge=0)
    remaining_candidate_calls: int = Field(ge=0, le=EXPECTED_TOTAL_CASES)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    overall: IntentSliceMetrics
    rag: IntentSliceMetrics
    no_rag: IntentSliceMetrics
    by_slice: dict[str, IntentSliceMetrics]
    under_retrieval_count: int = Field(ge=0)
    over_retrieval_count: int = Field(ge=0)
    clarification_count: int = Field(ge=0)
    gate: IntentStageGate
    cases: tuple[FullIntentCaseResult, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report_counts(self) -> Self:
        if self.target_sample_count != len(self.cases):
            raise ValueError("v4 阶段报告必须覆盖当前阶段全部样本")
        if self.cached_target_before_count + self.new_calls != self.target_sample_count:
            raise ValueError("v4 阶段缓存复用数与新增调用数必须覆盖当前阶段")
        if self.total_attempt_count != self.total_cached_count + self.failed_attempt_count:
            raise ValueError("v4 总尝试数必须包含成功缓存和失败尝试")
        if self.remaining_candidate_calls != EXPECTED_TOTAL_CASES - self.total_cached_count:
            raise ValueError("v4 剩余调用数与总缓存数不一致")
        if sum(metric.sample_count for metric in self.by_slice.values()) != len(self.cases):
            raise ValueError("v4 报告切片必须覆盖当前阶段全部样本")
        return self


def load_candidate_intent_validation_manifest(
    content: str,
) -> CandidateIntentValidationManifest:
    """Load the strict v4 validation manifest."""
    return CandidateIntentValidationManifest.model_validate_json(content)


def validate_candidate_intent_validation_manifest(
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
) -> None:
    """Prove development labels and challenge independence from the 150-case snapshot."""
    sample_by_id = {sample.query_id: sample for sample in dataset.samples}
    if any(query_id not in sample_by_id for query_id in manifest.development_query_ids):
        raise ValueError("v4 开发探针包含固定数据集之外的 Query ID")
    expected_routes = (
        IntentRoute.KNOWLEDGE_BASE,
        IntentRoute.KNOWLEDGE_BASE,
        IntentRoute.SYSTEM_DIRECT,
        IntentRoute.SYSTEM_DIRECT,
    )
    actual_routes = tuple(
        _route_for_requires_rag(sample_by_id[query_id].requires_rag)
        for query_id in manifest.development_query_ids
    )
    if actual_routes != expected_routes:
        raise ValueError("v4 开发探针标签发生漂移")
    original_ids = set(sample_by_id)
    original_queries = {_normalize_query(sample.query) for sample in dataset.samples}
    if any(sample.query_id in original_ids for sample in manifest.challenge_samples):
        raise ValueError("v4 挑战集 Query ID 不能复用原 150 条")
    if any(
        _normalize_query(sample.query) in original_queries for sample in manifest.challenge_samples
    ):
        raise ValueError("v4 挑战集不能原样复用原 150 条问题")


def candidate_intent_validation_fingerprint(
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
) -> str:
    """Bind the cache to the manifest and exact source development cases."""
    sample_by_id = {sample.query_id: sample for sample in dataset.samples}
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "development_samples": [
            sample_by_id[query_id].model_dump(mode="json")
            for query_id in manifest.development_query_ids
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def candidate_stage_samples(
    stage: CandidateIntentValidationStage,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
) -> tuple[IntentEvaluationSample, ...]:
    """Resolve one stage without mixing development and challenge denominators."""
    if stage is CandidateIntentValidationStage.CHALLENGE:
        return tuple(manifest.challenge_samples)
    sample_by_id = {sample.query_id: sample for sample in dataset.samples}
    return tuple(sample_by_id[query_id] for query_id in manifest.development_query_ids)


def validate_candidate_intent_validation_cache(
    cache: CandidateIntentValidationCache,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
) -> None:
    """Reject cached routes whose frozen labels no longer match."""
    expected = {
        sample.query_id: (
            sample.intent_l1,
            _route_for_requires_rag(sample.requires_rag),
        )
        for stage in CandidateIntentValidationStage
        for sample in candidate_stage_samples(stage, manifest, dataset)
    }
    if cache.dataset_id != manifest.version:
        raise ValueError("v4 验证缓存与清单版本不一致")
    if any(attempt.query_id not in expected for attempt in cache.failed_attempts):
        raise ValueError("v4 验证缓存的失败记录包含未知 Query ID")
    for case in cache.completed_cases:
        labels = expected.get(case.query_id)
        if labels is None or labels != (case.intent_l1, case.expected_route):
            raise ValueError("v4 验证缓存标签与冻结清单不一致")


def pending_candidate_stage_samples(
    stage: CandidateIntentValidationStage,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
    cache: CandidateIntentValidationCache,
) -> tuple[IntentEvaluationSample, ...]:
    """Return only unpaid cases for the selected focused stage."""
    validate_candidate_intent_validation_cache(cache, manifest, dataset)
    completed = {case.query_id for case in cache.completed_cases}
    return tuple(
        sample
        for sample in candidate_stage_samples(stage, manifest, dataset)
        if sample.query_id not in completed
    )


def ensure_candidate_validation_stage_unlocked(
    stage: CandidateIntentValidationStage,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
    cache: CandidateIntentValidationCache,
) -> None:
    """Block challenge spend until all four development boundaries pass."""
    if stage is CandidateIntentValidationStage.DEVELOPMENT:
        return
    development_cases = _require_cases(
        tuple(
            sample.query_id
            for sample in candidate_stage_samples(
                CandidateIntentValidationStage.DEVELOPMENT,
                manifest,
                dataset,
            )
        ),
        cache,
    )
    if not _evaluate_gate(development_cases).passed:
        raise ValueError("v4 开发探针未全部通过, 不允许产生挑战集费用")


def build_candidate_intent_validation_report(
    stage: CandidateIntentValidationStage,
    manifest: CandidateIntentValidationManifest,
    dataset: FullEvaluationDataset,
    cache: CandidateIntentValidationCache,
    *,
    cached_target_before_count: int,
    new_calls: int,
) -> CandidateIntentValidationReport:
    """Build a strict sanitized report for one v4 stage."""
    validate_candidate_intent_validation_cache(cache, manifest, dataset)
    samples = candidate_stage_samples(stage, manifest, dataset)
    cases = _require_cases(tuple(sample.query_id for sample in samples), cache)
    case_by_id = {case.query_id: case for case in cases}
    slices: dict[str, list[FullIntentCaseResult]] = {}
    for sample in samples:
        slice_name = _sample_slice(sample)
        slices.setdefault(slice_name, []).append(case_by_id[sample.query_id])
    rag_cases = tuple(case for case in cases if case.expected_route is IntentRoute.KNOWLEDGE_BASE)
    direct_cases = tuple(case for case in cases if case.expected_route is IntentRoute.SYSTEM_DIRECT)
    return CandidateIntentValidationReport(
        dataset_id=manifest.version,
        manifest_version=manifest.version,
        stage=stage,
        configuration=cache.configuration,
        target_sample_count=len(cases),
        cached_target_before_count=cached_target_before_count,
        new_calls=new_calls,
        total_cached_count=len(cache.completed_cases),
        failed_attempt_count=len(cache.failed_attempts),
        total_attempt_count=len(cache.completed_cases) + len(cache.failed_attempts),
        remaining_candidate_calls=EXPECTED_TOTAL_CASES - len(cache.completed_cases),
        input_tokens=sum(case.input_tokens or 0 for case in cases),
        output_tokens=sum(case.output_tokens or 0 for case in cases),
        overall=_slice_metrics(cases),
        rag=_slice_metrics(rag_cases),
        no_rag=_slice_metrics(direct_cases),
        by_slice={name: _slice_metrics(tuple(items)) for name, items in slices.items()},
        under_retrieval_count=_under_retrieval_count(cases),
        over_retrieval_count=_over_retrieval_count(cases),
        clarification_count=sum(case.actual_route is IntentRoute.CLARIFICATION for case in cases),
        gate=_evaluate_gate(cases),
        cases=cases,
        limitations=(
            "报告不保存问题、模型原文、API Key 或 Workspace ID。",
            "4 条开发探针来自已查看的 150 条, 不是独立验证。",
            "20 条挑战样本是冻结的新边界用例, 不是生产流量的无偏随机样本。",
            "通过两个阶段只支持进入影子或小流量验证, 不自动替换线上默认树。",
        ),
    )


def _evaluate_gate(cases: tuple[FullIntentCaseResult, ...]) -> IntentStageGate:
    checks = (
        _gate_check("all_routes_correct", "eq", sum(case.correct for case in cases), len(cases)),
        _gate_check("under_retrieval", "eq", _under_retrieval_count(cases), 0),
        _gate_check("over_retrieval", "eq", _over_retrieval_count(cases), 0),
        _gate_check(
            "clarification",
            "eq",
            sum(case.actual_route is IntentRoute.CLARIFICATION for case in cases),
            0,
        ),
    )
    return IntentStageGate(passed=all(check.passed for check in checks), checks=checks)


def _gate_check(
    name: str,
    comparison: Literal["eq", "gte", "gt"],
    actual: int,
    required: int,
) -> IntentStageGateCheck:
    passed = {"eq": actual == required, "gte": actual >= required, "gt": actual > required}[
        comparison
    ]
    return IntentStageGateCheck(
        name=name,
        comparison=comparison,
        actual=actual,
        required=required,
        passed=passed,
    )


def _require_cases(
    query_ids: tuple[str, ...],
    cache: CandidateIntentValidationCache,
) -> tuple[FullIntentCaseResult, ...]:
    case_by_id = {case.query_id: case for case in cache.completed_cases}
    missing = tuple(query_id for query_id in query_ids if query_id not in case_by_id)
    if missing:
        raise ValueError(f"v4 验证缓存尚缺 {len(missing)} 条当前阶段样本")
    return tuple(case_by_id[query_id] for query_id in query_ids)


def _sample_slice(sample: IntentEvaluationSample) -> str:
    if isinstance(sample, CandidateIntentChallengeSample):
        return sample.slice
    return {
        "S3-08": "known_mixed_product_comparison",
        "S12-06": "known_cross_ecosystem",
        "C2-02": "known_third_party_only",
        "C2-04": "known_generic_brand_opinion",
    }[sample.query_id]


def _slice_metrics(cases: tuple[FullIntentCaseResult, ...]) -> IntentSliceMetrics:
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


def _over_retrieval_count(cases: tuple[FullIntentCaseResult, ...]) -> int:
    return sum(
        case.expected_route is IntentRoute.SYSTEM_DIRECT
        and case.actual_route is IntentRoute.KNOWLEDGE_BASE
        for case in cases
    )


def _route_for_requires_rag(requires_rag: bool) -> IntentRoute:
    return IntentRoute.KNOWLEDGE_BASE if requires_rag else IntentRoute.SYSTEM_DIRECT


def _normalize_query(query: str) -> str:
    return "".join(query.split()).casefold()
