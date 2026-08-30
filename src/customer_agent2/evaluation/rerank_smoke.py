"""Fixed 20-case Rerank OFF/ON smoke runner with bounded live calls."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customer_agent2.config import Settings
from customer_agent2.domain.models import (
    ModelError,
    ModelErrorCode,
    RerankDocument,
    RerankModel,
    RerankRequest,
)
from customer_agent2.infrastructure.models import DashScopeRerankModel

_EXPECTED_CASES = 20
_EXPECTED_CANDIDATES = 10
_FATAL_RUN_ERROR_CODES = frozenset(
    {
        ModelErrorCode.AUTHENTICATION,
        ModelErrorCode.CONFIGURATION,
        ModelErrorCode.PROTOCOL,
        ModelErrorCode.QUOTA_EXHAUSTED,
    }
)


class SmokeDocument(BaseModel):
    """One synthetic, non-sensitive candidate shared by smoke cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=2000)


class SmokeCase(BaseModel):
    """One query, fixed candidate order, and manual relevance labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    candidate_ids: tuple[str, ...]
    relevant_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if len(self.candidate_ids) != _EXPECTED_CANDIDATES:
            raise ValueError(f"每条 Smoke 必须恰好包含 {_EXPECTED_CANDIDATES} 个候选")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("Smoke candidate_ids 不能重复")
        if not self.relevant_ids or len(set(self.relevant_ids)) != len(self.relevant_ids):
            raise ValueError("Smoke relevant_ids 必须非空且不能重复")
        if not set(self.relevant_ids).issubset(self.candidate_ids):
            raise ValueError("Smoke relevant_ids 必须属于 candidate_ids")
        return self


class RerankSmokeDataset(BaseModel):
    """Versioned fixed dataset accepted by the M5-B experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=100)
    documents: tuple[SmokeDocument, ...]
    cases: tuple[SmokeCase, ...]

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if len(self.cases) != _EXPECTED_CASES:
            raise ValueError(f"Rerank Smoke 必须恰好包含 {_EXPECTED_CASES} 条样本")
        document_ids = tuple(document.document_id for document in self.documents)
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("Smoke documents ID 不能重复")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("Smoke case_id 不能重复")
        available = set(document_ids)
        if any(not set(case.candidate_ids).issubset(available) for case in self.cases):
            raise ValueError("Smoke 候选引用了不存在的文档")
        return self


class RankingMetrics(BaseModel):
    """Aggregate metrics over the complete 20-case denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_at_1: float = Field(ge=0, le=1)
    hit_at_3: float = Field(ge=0, le=1)
    mrr_at_10: float = Field(ge=0, le=1)


class RerankCaseComparison(BaseModel):
    """Content-free per-case result safe to commit in a report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    off_first_relevant_rank: int = Field(ge=1, le=_EXPECTED_CANDIDATES)
    on_first_relevant_rank: int | None = Field(default=None, ge=1, le=_EXPECTED_CANDIDATES)
    error_code: ModelErrorCode | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.on_first_relevant_rank is None) == (self.error_code is None):
            raise ValueError("ON 名次和错误码必须且只能存在一个")
        return self


class RerankSmokeReport(BaseModel):
    """Sanitized aggregate and per-case output from one bounded run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    model_id: str
    sample_count: int = Field(ge=1)
    candidates_per_case: int = Field(ge=1)
    top_n: int = Field(ge=1)
    live_calls: int = Field(ge=0)
    successful_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    off_metrics: RankingMetrics
    on_metrics: RankingMetrics
    wins: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses: int = Field(ge=0)
    latency_ms_p50: float | None = Field(default=None, ge=0)
    latency_ms_p95: float | None = Field(default=None, ge=0)
    cases: tuple[RerankCaseComparison, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.sample_count != _EXPECTED_CASES or len(self.cases) != self.sample_count:
            raise ValueError("报告样本数必须与固定数据集一致")
        if self.live_calls != self.successful_calls + self.failed_calls:
            raise ValueError("真实调用成功和失败数关系无效")
        if self.wins + self.ties + self.losses != self.sample_count:
            raise ValueError("胜平负数量必须覆盖全部样本")
        return self


def load_rerank_smoke_dataset(path: Path) -> RerankSmokeDataset:
    """Load and strictly validate one versioned synthetic dataset."""
    return RerankSmokeDataset.model_validate_json(path.read_text(encoding="utf-8"))


async def run_rerank_smoke(
    dataset: RerankSmokeDataset,
    model: RerankModel,
) -> RerankSmokeReport:
    """Run at most one ON request per case and compare it with the fixed OFF order."""
    documents = {document.document_id: document.text for document in dataset.documents}
    comparisons: list[RerankCaseComparison] = []
    live_calls = 0
    total_tokens = 0
    latencies: list[float] = []

    for case in dataset.cases:
        off_rank = _first_relevant_rank(case.candidate_ids, set(case.relevant_ids))
        request = RerankRequest(
            query=case.query,
            documents=tuple(
                RerankDocument(document_id, documents[document_id])
                for document_id in case.candidate_ids
            ),
            top_n=_EXPECTED_CANDIDATES,
        )
        started = perf_counter()
        live_calls += 1
        try:
            result = await model.rerank(request)
        except ModelError as error:
            if error.code in _FATAL_RUN_ERROR_CODES:
                raise
            comparisons.append(
                RerankCaseComparison(
                    case_id=case.case_id,
                    off_first_relevant_rank=off_rank,
                    error_code=error.code,
                )
            )
            continue
        latency_ms = (perf_counter() - started) * 1000
        latencies.append(latency_ms)
        case_tokens = result.total_tokens or 0
        total_tokens += case_tokens
        on_ids = tuple(item.document_id for item in result.items)
        comparisons.append(
            RerankCaseComparison(
                case_id=case.case_id,
                off_first_relevant_rank=off_rank,
                on_first_relevant_rank=_first_relevant_rank(on_ids, set(case.relevant_ids)),
                latency_ms=latency_ms,
                total_tokens=result.total_tokens,
            )
        )

    comparison_tuple = tuple(comparisons)
    successful_calls = sum(item.error_code is None for item in comparison_tuple)
    off_ranks = tuple(item.off_first_relevant_rank for item in comparison_tuple)
    on_ranks = tuple(item.on_first_relevant_rank for item in comparison_tuple)
    wins, ties, losses = _wins_ties_losses(comparison_tuple)
    return RerankSmokeReport(
        dataset_id=dataset.dataset_id,
        model_id=model.model_id,
        sample_count=len(dataset.cases),
        candidates_per_case=_EXPECTED_CANDIDATES,
        top_n=_EXPECTED_CANDIDATES,
        live_calls=live_calls,
        successful_calls=successful_calls,
        failed_calls=live_calls - successful_calls,
        total_tokens=total_tokens,
        off_metrics=_metrics(off_ranks),
        on_metrics=_metrics(on_ranks),
        wins=wins,
        ties=ties,
        losses=losses,
        latency_ms_p50=_percentile(latencies, 0.50),
        latency_ms_p95=_percentile(latencies, 0.95),
        cases=comparison_tuple,
        limitations=(
            "20 条合成客服样本只用于工程 Smoke, 不代表生产流量。",
            "候选顺序经过分层安排, 不代表真实 pgvector 排名分布。",
            "M5-C 仍需使用 150 条固定检索集完成正式结论。",
        ),
    )


def _first_relevant_rank(ordered_ids: tuple[str, ...], relevant_ids: set[str]) -> int:
    for rank, document_id in enumerate(ordered_ids, start=1):
        if document_id in relevant_ids:
            return rank
    raise ValueError("排序结果没有包含人工标注的相关候选")


def _metrics(ranks: tuple[int | None, ...]) -> RankingMetrics:
    denominator = len(ranks)
    return RankingMetrics(
        hit_at_1=sum(rank == 1 for rank in ranks) / denominator,
        hit_at_3=sum(rank is not None and rank <= 3 for rank in ranks) / denominator,
        mrr_at_10=sum(0 if rank is None else 1 / rank for rank in ranks) / denominator,
    )


def _wins_ties_losses(
    comparisons: tuple[RerankCaseComparison, ...],
) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for comparison in comparisons:
        on_rank = comparison.on_first_relevant_rank
        if on_rank is None:
            losses += 1
        elif on_rank < comparison.off_first_relevant_rank:
            wins += 1
        elif on_rank == comparison.off_first_relevant_rank:
            ties += 1
        else:
            losses += 1
    return wins, ties, losses


def _percentile(values: list[float], fraction: float) -> float | None:
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


def _live_model(settings: Settings) -> DashScopeRerankModel:
    api_key = settings.dashscope_api_key
    base_url = settings.dashscope_rerank_api_base_url
    if not api_key.get_secret_value().strip() or base_url is None:
        raise ModelError(
            ModelErrorCode.CONFIGURATION,
            "真实 Rerank Smoke 需要本地 API Key 和 Workspace ID",
            retryable=False,
        )
    return DashScopeRerankModel(
        api_key=api_key,
        base_url=base_url,
        model_id=settings.rerank_model,
        timeout_seconds=settings.rerank_timeout_seconds,
    )


async def _run_cli(cases_path: Path, output_path: Path) -> RerankSmokeReport:
    dataset = load_rerank_smoke_dataset(cases_path)
    model = _live_model(Settings())
    try:
        report = await run_rerank_smoke(dataset, model)
    finally:
        await model.aclose()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run the explicitly authorized live Smoke and write a sanitized JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="允许最多 20 次真实计费调用")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/fixtures/rerank_smoke_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/reports/m5b-rerank-smoke.json"),
    )
    arguments = parser.parse_args()
    if not arguments.live:
        parser.error("必须显式传入 --live 才允许真实 Rerank 调用")
    try:
        report = asyncio.run(_run_cli(arguments.cases, arguments.output))
    except ModelError as error:
        parser.exit(
            status=2,
            message=f"Rerank Smoke 已安全终止: {error.code.value}\n",
        )
    print(
        json.dumps(
            {
                "dataset_id": report.dataset_id,
                "model_id": report.model_id,
                "live_calls": report.live_calls,
                "successful_calls": report.successful_calls,
                "failed_calls": report.failed_calls,
                "total_tokens": report.total_tokens,
                "off_metrics": report.off_metrics.model_dump(),
                "on_metrics": report.on_metrics.model_dump(),
                "output": str(arguments.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
