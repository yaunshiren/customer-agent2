"""Deterministic tests for the focused M5-D v4 validation protocol."""

from pathlib import Path

import pytest

from customer_agent2.domain.models import (
    IntentCandidate,
    IntentClassificationRequest,
    IntentDecision,
    IntentDecisionReason,
    IntentRoute,
    ModelError,
    TokenUsage,
)
from customer_agent2.evaluation.candidate_intent_validation import (
    EXPECTED_CHALLENGE_CASES,
    EXPECTED_DEVELOPMENT_CASES,
    EXPECTED_TOTAL_CASES,
    CandidateIntentValidationCache,
    CandidateIntentValidationManifest,
    CandidateIntentValidationStage,
    build_candidate_intent_validation_report,
    candidate_intent_validation_fingerprint,
    candidate_stage_samples,
    ensure_candidate_validation_stage_unlocked,
    load_candidate_intent_validation_manifest,
    pending_candidate_stage_samples,
    validate_candidate_intent_validation_manifest,
)
from customer_agent2.evaluation.candidate_intent_validation_cli import (
    load_candidate_intent_validation_cache,
    new_candidate_intent_validation_cache,
    validate_candidate_intent_validation_paths,
    write_candidate_intent_validation_cache,
)
from customer_agent2.evaluation.full_dataset import (
    FullEvaluationDataset,
    load_full_evaluation_assets,
)
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    IntentCandidateScores,
    IntentEvaluationSample,
    run_intent_case_evaluation,
)
from customer_agent2.infrastructure.intents import (
    intent_tree_fingerprint,
    load_intent_tree_json,
)
from tests.settings import IsolatedSettings

PROJECT_ROOT = Path(__file__).parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "evaluation" / "datasets" / "ragenteval-v1"
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-v4-validation.json"
V3_TREE_PATH = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-tree-v3.json"
V4_TREE_PATH = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-tree-v4.json"


def _dataset() -> FullEvaluationDataset:
    return load_full_evaluation_assets(SNAPSHOT_ROOT).dataset


def _manifest() -> CandidateIntentValidationManifest:
    return load_candidate_intent_validation_manifest(MANIFEST_PATH.read_text(encoding="utf-8"))


def _configuration() -> FullIntentEvaluationConfiguration:
    return FullIntentEvaluationConfiguration(
        model_id="fake-fast",
        intent_tree_version="m5-d-v4-candidate",
        intent_tree_sha256="4" * 64,
        high_confidence_threshold=0.75,
        ambiguity_margin=0.10,
        timeout_seconds=60,
        max_output_tokens=256,
        temperature=0,
        reasoning_enabled=False,
    )


def _cache(
    cases: tuple[FullIntentCaseResult, ...] = (),
) -> CandidateIntentValidationCache:
    manifest = _manifest()
    dataset = _dataset()
    return CandidateIntentValidationCache(
        dataset_id=manifest.version,
        manifest_version=manifest.version,
        validation_sha256=candidate_intent_validation_fingerprint(manifest, dataset),
        configuration=_configuration(),
        completed_cases=cases,
    )


def _case(
    sample: IntentEvaluationSample,
    actual_route: IntentRoute | None = None,
) -> FullIntentCaseResult:
    expected_route = (
        IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
    )
    actual = actual_route or expected_route
    return FullIntentCaseResult(
        query_id=sample.query_id,
        intent_l1=sample.intent_l1,
        expected_route=expected_route,
        actual_route=actual,
        correct=actual is expected_route,
        decision_reason="high_confidence",
        candidate_scores=IntentCandidateScores(
            **{route.value: 0.9 if route is actual else 0.05 for route in IntentRoute}
        ),
        latency_ms=1,
        input_tokens=10,
        output_tokens=2,
    )


class FixedCandidateClassifier:
    """Return the expected route for one focused challenge sample."""

    def __init__(self, route: IntentRoute) -> None:
        self._route = route

    async def classify(self, request: IntentClassificationRequest) -> IntentDecision:
        assert request.question
        return IntentDecision(
            route=self._route,
            reason=IntentDecisionReason.HIGH_CONFIDENCE,
            candidates=tuple(
                IntentCandidate(route, 0.9 if route is self._route else 0.05)
                for route in IntentRoute
            ),
            classifier_model_id="fake-fast",
            classifier_finish_reason="stop",
            usage=TokenUsage(10, 2),
        )


def test_v4_manifest_is_new_balanced_and_frozen() -> None:
    manifest = _manifest()
    dataset = _dataset()

    validate_candidate_intent_validation_manifest(manifest, dataset)

    assert len(manifest.development_query_ids) == EXPECTED_DEVELOPMENT_CASES
    assert len(manifest.challenge_samples) == EXPECTED_CHALLENGE_CASES
    assert sum(sample.requires_rag for sample in manifest.challenge_samples) == 10
    assert all(
        sample.query not in {original.query for original in dataset.samples}
        for sample in manifest.challenge_samples
    )


def test_v4_candidate_changes_only_versioned_route_descriptions() -> None:
    v3 = load_intent_tree_json(V3_TREE_PATH.read_text(encoding="utf-8"))
    v4 = load_intent_tree_json(V4_TREE_PATH.read_text(encoding="utf-8"))

    assert v4.version == "m5-d-v4-candidate"
    assert tuple(item.route for item in v4.definitions) == tuple(
        item.route for item in v3.definitions
    )
    assert intent_tree_fingerprint(v4) != intent_tree_fingerprint(v3)
    descriptions = "".join(item.description for item in v4.definitions)
    assert "至少包含一个可能由授权知识库覆盖的具体商品或型号" in descriptions
    assert "泛化的品牌主观评价" in descriptions


@pytest.mark.asyncio
async def test_v4_challenge_sample_uses_shared_live_evaluation_contract() -> None:
    sample = _manifest().challenge_samples[0]

    cases = await run_intent_case_evaluation(
        (sample,),
        FixedCandidateClassifier(sample.expected_route),
    )

    assert len(cases) == 1
    assert cases[0].query_id == sample.query_id
    assert cases[0].correct is True
    assert cases[0].candidate_scores is not None


def test_v4_stages_require_only_four_then_twenty_calls() -> None:
    manifest = _manifest()
    dataset = _dataset()
    development = candidate_stage_samples(
        CandidateIntentValidationStage.DEVELOPMENT,
        manifest,
        dataset,
    )
    challenge = candidate_stage_samples(
        CandidateIntentValidationStage.CHALLENGE,
        manifest,
        dataset,
    )

    assert (
        len(
            pending_candidate_stage_samples(
                CandidateIntentValidationStage.DEVELOPMENT,
                manifest,
                dataset,
                _cache(),
            )
        )
        == EXPECTED_DEVELOPMENT_CASES
    )
    development_cache = _cache(tuple(_case(sample) for sample in development))
    ensure_candidate_validation_stage_unlocked(
        CandidateIntentValidationStage.CHALLENGE,
        manifest,
        dataset,
        development_cache,
    )
    assert (
        len(
            pending_candidate_stage_samples(
                CandidateIntentValidationStage.CHALLENGE,
                manifest,
                dataset,
                development_cache,
            )
        )
        == EXPECTED_CHALLENGE_CASES
    )
    assert len((*development, *challenge)) == EXPECTED_TOTAL_CASES


def test_v4_failed_development_boundary_blocks_challenge_spend() -> None:
    manifest = _manifest()
    dataset = _dataset()
    development = candidate_stage_samples(
        CandidateIntentValidationStage.DEVELOPMENT,
        manifest,
        dataset,
    )
    cases = [_case(sample) for sample in development]
    cases[0] = _case(development[0], IntentRoute.SYSTEM_DIRECT)
    cache = _cache(tuple(cases))

    report = build_candidate_intent_validation_report(
        CandidateIntentValidationStage.DEVELOPMENT,
        manifest,
        dataset,
        cache,
        cached_target_before_count=0,
        new_calls=EXPECTED_DEVELOPMENT_CASES,
    )

    assert report.gate.passed is False
    assert report.under_retrieval_count == 1
    with pytest.raises(ValueError, match="不允许产生挑战集费用"):
        ensure_candidate_validation_stage_unlocked(
            CandidateIntentValidationStage.CHALLENGE,
            manifest,
            dataset,
            cache,
        )


def test_v4_challenge_report_is_strict_and_content_free() -> None:
    manifest = _manifest()
    dataset = _dataset()
    development = candidate_stage_samples(
        CandidateIntentValidationStage.DEVELOPMENT,
        manifest,
        dataset,
    )
    challenge = candidate_stage_samples(
        CandidateIntentValidationStage.CHALLENGE,
        manifest,
        dataset,
    )
    cache = _cache(tuple(_case(sample) for sample in (*development, *challenge)))

    report = build_candidate_intent_validation_report(
        CandidateIntentValidationStage.CHALLENGE,
        manifest,
        dataset,
        cache,
        cached_target_before_count=0,
        new_calls=EXPECTED_CHALLENGE_CASES,
    )
    serialized = report.model_dump_json()

    assert report.target_sample_count == EXPECTED_CHALLENGE_CASES
    assert report.total_cached_count == EXPECTED_TOTAL_CASES
    assert report.remaining_candidate_calls == 0
    assert report.rag.correct_count == 10
    assert report.no_rag.correct_count == 10
    assert report.gate.passed is True
    assert all(sample.query not in serialized for sample in manifest.challenge_samples)


def test_v4_cache_identity_rejects_manifest_drift(tmp_path: Path) -> None:
    manifest = _manifest()
    dataset = _dataset()
    tree = load_intent_tree_json(V4_TREE_PATH.read_text(encoding="utf-8"))
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    expected = new_candidate_intent_validation_cache(
        settings,
        manifest,
        dataset,
        intent_tree_version=tree.version,
        intent_tree_sha256=intent_tree_fingerprint(tree),
        timeout_seconds=60,
    )
    cache_path = tmp_path / "v4.cache.json"
    write_candidate_intent_validation_cache(cache_path, expected)

    assert load_candidate_intent_validation_cache(cache_path, expected) == expected

    changed_manifest = manifest.model_copy(update={"version": "m5-d-v4-validation-drift"})
    changed = new_candidate_intent_validation_cache(
        settings,
        changed_manifest,
        dataset,
        intent_tree_version=tree.version,
        intent_tree_sha256=intent_tree_fingerprint(tree),
        timeout_seconds=60,
    )
    with pytest.raises(ModelError, match="清单、模型、候选树或参数"):
        load_candidate_intent_validation_cache(cache_path, changed)


def test_v4_outputs_cannot_overwrite_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    tree = tmp_path / "tree.json"
    cache = tmp_path / "cache.json"
    output = tmp_path / "report.json"

    validate_candidate_intent_validation_paths(
        snapshot=tmp_path / "snapshot",
        manifest=manifest,
        intent_tree=tree,
        cache=cache,
        output=output,
    )

    with pytest.raises(ModelError):
        validate_candidate_intent_validation_paths(
            snapshot=tmp_path / "snapshot",
            manifest=manifest,
            intent_tree=tree,
            cache=cache,
            output=cache,
        )
