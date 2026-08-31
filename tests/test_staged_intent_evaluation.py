"""Deterministic tests for budget-aware staged Intent evaluation."""

from pathlib import Path

import pytest

from customer_agent2.domain.models import IntentRoute
from customer_agent2.evaluation.full_dataset import (
    FullEvaluationDataset,
    FullEvaluationSample,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentFailedAttempt,
    FullIntentReport,
    IntentCandidateScores,
    build_full_intent_report,
)
from customer_agent2.evaluation.staged_intent import (
    IntentEvaluationStage,
    M5DIntentStageManifest,
    StagedIntentCache,
    build_staged_intent_report,
    ensure_stage_unlocked,
    load_m5d_stage_manifest,
    ordered_full_cache_cases,
    pending_stage_samples,
    stage_query_ids,
    validate_m5d_stage_manifest,
)

PROJECT_ROOT = Path(__file__).parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "evaluation" / "datasets" / "ragenteval-v1"
BASELINE_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "m5c-full-intent.json"
STAGE_MANIFEST = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-stages.json"


def _dataset() -> FullEvaluationDataset:
    return load_full_evaluation_assets(SNAPSHOT_ROOT).dataset


def _baseline() -> FullIntentReport:
    return FullIntentReport.model_validate_json(BASELINE_REPORT.read_text(encoding="utf-8"))


def _manifest() -> M5DIntentStageManifest:
    return load_m5d_stage_manifest(STAGE_MANIFEST.read_text(encoding="utf-8"))


def _configuration() -> FullIntentEvaluationConfiguration:
    return FullIntentEvaluationConfiguration(
        model_id="fake-fast",
        intent_tree_version="m5-d-v2-candidate",
        intent_tree_sha256="1" * 64,
        high_confidence_threshold=0.75,
        ambiguity_margin=0.10,
        timeout_seconds=60,
        max_output_tokens=256,
        temperature=0,
        reasoning_enabled=False,
    )


def _cache(cases: tuple[FullIntentCaseResult, ...] = ()) -> StagedIntentCache:
    return StagedIntentCache(
        dataset_id="ragenteval-v1-all",
        manifest_version="m5-d-stages-v1",
        manifest_sha256="2" * 64,
        baseline_report_sha256="3" * 64,
        configuration=_configuration(),
        completed_cases=cases,
    )


def _case(
    sample: FullEvaluationSample,
    actual_route: IntentRoute | None = None,
) -> FullIntentCaseResult:
    expected_route = (
        IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
    )
    actual = actual_route or expected_route
    scores = {route.value: 0.9 if route is actual else 0.1 for route in IntentRoute}
    return FullIntentCaseResult(
        query_id=sample.query_id,
        intent_l1=sample.intent_l1,
        expected_route=expected_route,
        actual_route=actual,
        correct=actual is expected_route,
        decision_reason=(
            "explicit_clarification" if actual is IntentRoute.CLARIFICATION else "high_confidence"
        ),
        candidate_scores=IntentCandidateScores(**scores),
        latency_ms=1,
        input_tokens=10,
        output_tokens=2,
    )


def _cases_for_ids(
    query_ids: tuple[str, ...],
    dataset: FullEvaluationDataset,
) -> tuple[FullIntentCaseResult, ...]:
    samples = {sample.query_id: sample for sample in dataset.samples}
    return tuple(_case(samples[query_id]) for query_id in query_ids)


def test_stage_manifest_matches_baseline_and_fixed_strata() -> None:
    dataset = _dataset()
    manifest = _manifest()

    validate_m5d_stage_manifest(manifest, dataset, _baseline())

    assert len(manifest.failure_query_ids) == 22
    assert len(manifest.guard_query_ids) == 18
    assert len(stage_query_ids(IntentEvaluationStage.GUARD, manifest, dataset)) == 40
    assert len(stage_query_ids(IntentEvaluationStage.FULL, manifest, dataset)) == 150


def test_stage_cache_reuses_query_ids_and_only_adds_missing_calls() -> None:
    dataset = _dataset()
    manifest = _manifest()
    empty = _cache()

    first_pending = pending_stage_samples(
        IntentEvaluationStage.FAILURES,
        manifest,
        dataset,
        empty,
    )
    partial = _cache(_cases_for_ids(manifest.failure_query_ids[:5], dataset))
    resumed_pending = pending_stage_samples(
        IntentEvaluationStage.FAILURES,
        manifest,
        dataset,
        partial,
    )

    assert len(first_pending) == 22
    assert tuple(sample.query_id for sample in resumed_pending) == manifest.failure_query_ids[5:]

    failure_cache = _cache(_cases_for_ids(manifest.failure_query_ids, dataset))
    ensure_stage_unlocked(IntentEvaluationStage.GUARD, manifest, dataset, failure_cache)
    assert (
        len(pending_stage_samples(IntentEvaluationStage.GUARD, manifest, dataset, failure_cache))
        == 18
    )

    screen_ids = (*manifest.failure_query_ids, *manifest.guard_query_ids)
    screen_cache = _cache(_cases_for_ids(screen_ids, dataset))
    ensure_stage_unlocked(IntentEvaluationStage.FULL, manifest, dataset, screen_cache)
    assert (
        len(pending_stage_samples(IntentEvaluationStage.FULL, manifest, dataset, screen_cache))
        == 110
    )


def test_failed_first_gate_blocks_guard_calls() -> None:
    dataset = _dataset()
    manifest = _manifest()
    samples = {sample.query_id: sample for sample in dataset.samples}
    cases = list(_cases_for_ids(manifest.failure_query_ids, dataset))
    first_rag_index = next(
        index
        for index, case in enumerate(cases)
        if case.expected_route is IntentRoute.KNOWLEDGE_BASE
    )
    cases[first_rag_index] = _case(
        samples[cases[first_rag_index].query_id],
        IntentRoute.SYSTEM_DIRECT,
    )
    cache = _cache(tuple(cases))

    report = build_staged_intent_report(
        IntentEvaluationStage.FAILURES,
        manifest,
        dataset,
        cache,
        cached_target_before_count=0,
        new_calls=22,
    )

    assert report.gate.passed is False
    assert report.under_retrieval_count == 1
    with pytest.raises(ValueError, match="门禁未通过"):
        ensure_stage_unlocked(IntentEvaluationStage.GUARD, manifest, dataset, cache)


def test_guard_report_passes_with_18_reused_regression_cases() -> None:
    dataset = _dataset()
    manifest = _manifest()
    screen_ids = (*manifest.failure_query_ids, *manifest.guard_query_ids)
    cache = _cache(_cases_for_ids(screen_ids, dataset))

    report = build_staged_intent_report(
        IntentEvaluationStage.GUARD,
        manifest,
        dataset,
        cache,
        cached_target_before_count=22,
        new_calls=18,
    )

    assert report.target_sample_count == 40
    assert report.overall.correct_count == 40
    assert report.gate.passed is True
    assert report.failed_attempt_count == 0
    assert report.total_attempt_count == 40
    assert report.remaining_full_calls == 110


def test_stage_report_discloses_historical_failed_attempts() -> None:
    dataset = _dataset()
    manifest = _manifest()
    cases = _cases_for_ids(manifest.failure_query_ids, dataset)
    cache = StagedIntentCache(
        **_cache(cases).model_dump(exclude={"failed_attempts"}),
        failed_attempts=(
            FullIntentFailedAttempt(
                query_id=manifest.failure_query_ids[0],
                error_code="timeout",
            ),
        ),
    )

    report = build_staged_intent_report(
        IntentEvaluationStage.FAILURES,
        manifest,
        dataset,
        cache,
        cached_target_before_count=22,
        new_calls=0,
    )

    assert report.failed_attempt_count == 1
    assert report.total_attempt_count == 23


def test_full_stage_orders_cache_and_builds_strict_report_without_recalling() -> None:
    dataset = _dataset()
    manifest = _manifest()
    reversed_cases = tuple(_case(sample) for sample in reversed(dataset.samples))
    cache = _cache(reversed_cases)

    ordered = ordered_full_cache_cases(dataset, cache)
    stage_report = build_staged_intent_report(
        IntentEvaluationStage.FULL,
        manifest,
        dataset,
        cache,
        cached_target_before_count=40,
        new_calls=110,
    )
    full_report = build_full_intent_report(
        dataset,
        ordered,
        configuration=cache.configuration,
    )

    assert tuple(case.query_id for case in ordered) == tuple(
        sample.query_id for sample in dataset.samples
    )
    assert stage_report.remaining_full_calls == 0
    assert stage_report.gate.passed is True
    assert full_report.overall.correct_count == 150
    serialized = stage_report.model_dump_json()
    assert dataset.samples[0].query not in serialized
    assert dataset.samples[0].ground_truth not in serialized
