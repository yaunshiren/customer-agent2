"""Deterministic tests for the protected 150-case Intent evaluator."""

import math
from pathlib import Path

import pytest

from customer_agent2.domain.models import (
    IntentCandidate,
    IntentClassificationRequest,
    IntentDecision,
    IntentDecisionReason,
    IntentDegradationReason,
    IntentRoute,
    ModelErrorCode,
    TokenUsage,
)
from customer_agent2.evaluation.full_dataset import load_full_evaluation_assets
from customer_agent2.evaluation.full_intent import (
    FullIntentCaseResult,
    FullIntentEvaluationConfiguration,
    FullIntentReport,
    FullIntentRunError,
    replay_full_intent_thresholds,
    run_full_intent_evaluation,
)

SNAPSHOT_ROOT = Path(__file__).parents[1] / "evaluation" / "datasets" / "ragenteval-v1"


class FixedIntentClassifier:
    def __init__(
        self,
        routes: dict[str, IntentRoute],
        *,
        degrade_question: str | None = None,
    ) -> None:
        self._routes = routes
        self._degrade_question = degrade_question
        self.calls = 0

    async def classify(self, request: IntentClassificationRequest) -> IntentDecision:
        self.calls += 1
        if request.question == self._degrade_question:
            return IntentDecision(
                route=IntentRoute.KNOWLEDGE_BASE,
                reason=IntentDecisionReason.CLASSIFIER_FALLBACK,
                candidates=(),
                degradation_reason=IntentDegradationReason.PROTOCOL,
                model_error_code=ModelErrorCode.PROTOCOL,
            )
        route = self._routes[request.question]
        return IntentDecision(
            route=route,
            reason=IntentDecisionReason.HIGH_CONFIDENCE,
            candidates=tuple(
                IntentCandidate(candidate, 0.9 if candidate is route else 0.1)
                for candidate in IntentRoute
            ),
            classifier_model_id="fake-fast",
            classifier_finish_reason="stop",
            usage=TokenUsage(10, 2),
        )


@pytest.mark.asyncio
async def test_full_intent_run_covers_all_slices_and_sanitizes_report() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    classifier = FixedIntentClassifier(routes)

    report = await run_full_intent_evaluation(assets.dataset, classifier)

    assert classifier.calls == 150
    assert report.successful_calls == 150
    assert report.failed_calls == 0
    assert report.overall.accuracy == 1
    assert report.rag.sample_count == 132
    assert report.no_rag.sample_count == 18
    assert report.input_tokens == 1500
    assert report.output_tokens == 300
    assert report.cases[0].candidate_scores is not None
    assert report.cases[0].candidate_scores.knowledge_base in {0.1, 0.9}
    serialized = report.model_dump_json()
    assert assets.dataset.samples[0].query not in serialized
    assert assets.dataset.samples[0].ground_truth not in serialized


@pytest.mark.asyncio
async def test_full_intent_run_counts_wrong_route_in_fixed_denominator() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    wrong_sample = next(sample for sample in assets.dataset.samples if not sample.requires_rag)
    routes[wrong_sample.query] = IntentRoute.KNOWLEDGE_BASE

    report = await run_full_intent_evaluation(
        assets.dataset,
        FixedIntentClassifier(routes),
    )

    assert report.overall.correct_count == 149
    assert math.isclose(report.overall.accuracy, 149 / 150)
    assert report.no_rag.correct_count == 17
    assert report.confusion[IntentRoute.SYSTEM_DIRECT][IntentRoute.KNOWLEDGE_BASE] == 1


@pytest.mark.asyncio
async def test_full_intent_threshold_replay_reuses_scores_without_calls() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    source = await run_full_intent_evaluation(
        assets.dataset,
        FixedIntentClassifier(routes),
        configuration=FullIntentEvaluationConfiguration(
            model_id="fake-fast",
            high_confidence_threshold=0.75,
            ambiguity_margin=0.10,
            timeout_seconds=60,
            max_output_tokens=256,
            temperature=0,
            reasoning_enabled=False,
        ),
    )

    replayed = replay_full_intent_thresholds(
        source,
        high_confidence_threshold=0.95,
        ambiguity_margin=0.10,
    )

    assert replayed.overall.correct_count == 0
    assert all(case.actual_route is IntentRoute.CLARIFICATION for case in replayed.cases)
    assert replayed.input_tokens == source.input_tokens
    assert replayed.output_tokens == source.output_tokens
    assert replayed.configuration is not None
    assert replayed.configuration.high_confidence_threshold == 0.95


def test_m5c_report_without_scores_rejects_threshold_replay() -> None:
    report_path = Path(__file__).parents[1] / "evaluation" / "reports" / "m5c-full-intent.json"
    report = FullIntentReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="候选分数"):
        replay_full_intent_thresholds(
            report,
            high_confidence_threshold=0.70,
            ambiguity_margin=0.05,
        )


@pytest.mark.asyncio
async def test_full_intent_run_stops_at_first_degradation() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    failed_sample = assets.dataset.samples[2]
    classifier = FixedIntentClassifier(routes, degrade_question=failed_sample.query)
    recorded: list[FullIntentCaseResult] = []

    with pytest.raises(FullIntentRunError, match=failed_sample.query_id) as captured:
        await run_full_intent_evaluation(
            assets.dataset,
            classifier,
            on_case=recorded.append,
        )

    assert classifier.calls == 3
    assert len(recorded) == 3
    assert recorded[-1].degradation_reason is IntentDegradationReason.PROTOCOL
    assert captured.value.error_code == ModelErrorCode.PROTOCOL.value


@pytest.mark.asyncio
async def test_full_intent_run_resumes_successful_prefix_without_recalling_it() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    first_run = await run_full_intent_evaluation(
        assets.dataset,
        FixedIntentClassifier(routes),
    )
    resumed_classifier = FixedIntentClassifier(routes)

    resumed = await run_full_intent_evaluation(
        assets.dataset,
        resumed_classifier,
        initial_cases=first_run.cases[:144],
    )

    assert resumed_classifier.calls == 6
    assert resumed.sample_count == 150
    assert resumed.overall.accuracy == 1


@pytest.mark.asyncio
async def test_full_intent_run_rejects_non_prefix_checkpoint() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    routes = {
        sample.query: (
            IntentRoute.KNOWLEDGE_BASE if sample.requires_rag else IntentRoute.SYSTEM_DIRECT
        )
        for sample in assets.dataset.samples
    }
    report = await run_full_intent_evaluation(assets.dataset, FixedIntentClassifier(routes))

    with pytest.raises(ValueError, match="连续成功前缀"):
        await run_full_intent_evaluation(
            assets.dataset,
            FixedIntentClassifier(routes),
            initial_cases=report.cases[1:2],
        )
