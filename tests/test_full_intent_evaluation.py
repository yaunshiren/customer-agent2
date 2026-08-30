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
    TokenUsage,
)
from customer_agent2.evaluation.full_dataset import load_full_evaluation_assets
from customer_agent2.evaluation.full_intent import (
    FullIntentRunError,
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

    with pytest.raises(FullIntentRunError, match=failed_sample.query_id):
        await run_full_intent_evaluation(assets.dataset, classifier)

    assert classifier.calls == 3
