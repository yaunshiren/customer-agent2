"""Deterministic tests for the bounded M5-B Rerank OFF/ON smoke runner."""

import json
from math import isclose
from pathlib import Path
from typing import cast

import pytest

from customer_agent2.domain.models import (
    ModelError,
    ModelErrorCode,
    RerankItem,
    RerankRequest,
    RerankResult,
)
from customer_agent2.evaluation import (
    RerankSmokeDataset,
    load_rerank_smoke_dataset,
    run_rerank_smoke,
)

FIXTURE_PATH = Path(__file__).parents[1] / "evaluation" / "fixtures" / "rerank_smoke_cases.json"


class PerfectRerankModel:
    def __init__(self, dataset: RerankSmokeDataset, *, failing_query: str | None = None) -> None:
        self._relevant_by_query = {case.query: set(case.relevant_ids) for case in dataset.cases}
        self._failing_query = failing_query
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "perfect-test-rerank"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.calls += 1
        if request.query == self._failing_query:
            raise ModelError(
                ModelErrorCode.UNAVAILABLE,
                "合成失败",
                retryable=True,
            )
        relevant = self._relevant_by_query[request.query]
        ordered = sorted(
            enumerate(request.documents),
            key=lambda item: (item[1].document_id not in relevant, item[0]),
        )
        return RerankResult(
            model_id=self.model_id,
            items=tuple(
                RerankItem(
                    original_index=index,
                    document_id=document.document_id,
                    score=1 - position / 20,
                )
                for position, (index, document) in enumerate(ordered[: request.result_limit])
            ),
            total_tokens=10,
        )


class AuthenticationFailureModel:
    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_id(self) -> str:
        return "authentication-failure-test"

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.calls += 1
        raise ModelError(
            ModelErrorCode.AUTHENTICATION,
            "合成认证失败",
            retryable=False,
        )


def test_fixed_dataset_has_twenty_safe_stratified_cases() -> None:
    dataset = load_rerank_smoke_dataset(FIXTURE_PATH)

    assert dataset.dataset_id == "m5b-rerank-smoke-v1"
    assert len(dataset.cases) == 20
    assert all(len(case.candidate_ids) == 10 for case in dataset.cases)
    assert sorted(
        case.candidate_ids.index(case.relevant_ids[0]) + 1 for case in dataset.cases
    ) == sorted(list(range(1, 11)) * 2)


def test_dataset_rejects_a_non_twenty_case_variant() -> None:
    payload = cast(
        dict[str, object],
        json.loads(FIXTURE_PATH.read_text(encoding="utf-8")),
    )
    cases = cast(list[object], payload["cases"])
    cases.pop()

    with pytest.raises(ValueError, match="20"):
        RerankSmokeDataset.model_validate(payload)


@pytest.mark.asyncio
async def test_smoke_reports_full_denominator_metrics_and_no_content() -> None:
    dataset = load_rerank_smoke_dataset(FIXTURE_PATH)
    model = PerfectRerankModel(dataset)

    report = await run_rerank_smoke(dataset, model)

    assert model.calls == 20
    assert report.live_calls == 20
    assert report.successful_calls == 20
    assert report.failed_calls == 0
    assert report.total_tokens == 200
    assert isclose(report.off_metrics.hit_at_1, 0.1)
    assert isclose(report.off_metrics.hit_at_3, 0.3)
    assert isclose(report.off_metrics.mrr_at_10, 0.2928968254)
    assert isclose(report.on_metrics.hit_at_1, 1.0)
    assert isclose(report.on_metrics.hit_at_3, 1.0)
    assert isclose(report.on_metrics.mrr_at_10, 1.0)
    assert (report.wins, report.ties, report.losses) == (18, 2, 0)
    assert report.latency_ms_p50 is not None
    assert report.latency_ms_p95 is not None
    serialized = report.model_dump_json()
    assert dataset.cases[0].query not in serialized
    assert dataset.documents[0].text not in serialized


@pytest.mark.asyncio
async def test_known_model_failure_counts_as_miss_without_retry() -> None:
    dataset = load_rerank_smoke_dataset(FIXTURE_PATH)
    model = PerfectRerankModel(dataset, failing_query=dataset.cases[1].query)

    report = await run_rerank_smoke(dataset, model)

    assert model.calls == 20
    assert report.live_calls == 20
    assert report.successful_calls == 19
    assert report.failed_calls == 1
    assert report.total_tokens == 190
    assert isclose(report.on_metrics.hit_at_1, 0.95)
    assert isclose(report.on_metrics.hit_at_3, 0.95)
    assert isclose(report.on_metrics.mrr_at_10, 0.95)
    assert (report.wins, report.ties, report.losses) == (17, 2, 1)
    assert report.cases[1].error_code is ModelErrorCode.UNAVAILABLE
    assert report.cases[1].on_first_relevant_rank is None


@pytest.mark.asyncio
async def test_non_retryable_run_error_stops_after_first_call() -> None:
    dataset = load_rerank_smoke_dataset(FIXTURE_PATH)
    model = AuthenticationFailureModel()

    with pytest.raises(ModelError, match="合成认证失败"):
        await run_rerank_smoke(dataset, model)

    assert model.calls == 1
