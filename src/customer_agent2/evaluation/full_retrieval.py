"""Content-safe 132-case retrieval and optional Rerank evaluation."""

import math
from collections.abc import Iterable
from time import perf_counter
from typing import Literal, Self, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from customer_agent2.application import RetrievalPostProcessor
from customer_agent2.application.services import VectorRetrievalUseCase
from customer_agent2.domain.models import (
    ModelError,
    RetrievalError,
    VectorSearchCandidate,
    VectorSearchRequest,
    VectorSearchScope,
)
from customer_agent2.evaluation.full_dataset import (
    EXPECTED_RAG_CASES,
    FullEvaluationDataset,
)

_TOP_K = 10


class FullRetrievalRunError(RuntimeError):
    """Sanitized fatal error that stops a paid ON run at the first degradation."""

    def __init__(self, query_id: str, error_code: str) -> None:
        super().__init__(f"完整评测在 {query_id} 安全终止: {error_code}")
        self.query_id = query_id
        self.error_code = error_code


class RetrievalEvaluationMetrics(BaseModel):
    """Document-level aggregate metrics over the fixed RAG denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hit_at_1: float = Field(ge=0, le=1)
    hit_at_3: float = Field(ge=0, le=1)
    hit_at_5: float = Field(ge=0, le=1)
    hit_at_10: float = Field(ge=0, le=1)
    recall_at_3: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    mrr_at_10: float = Field(ge=0, le=1)


class FullRetrievalCaseResult(BaseModel):
    """Sanitized per-query result without query, answer, or chunk content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(min_length=1, max_length=100)
    required_document_ids: tuple[str, ...]
    off_document_ids: tuple[str, ...] = ()
    on_document_ids: tuple[str, ...] | None = None
    retrieval_error_code: str | None = Field(default=None, min_length=1, max_length=100)
    rerank_attempted: bool = False
    rerank_error_code: str | None = Field(default=None, min_length=1, max_length=100)
    retrieval_latency_ms: float | None = Field(default=None, ge=0)
    rerank_latency_ms: float | None = Field(default=None, ge=0)
    rerank_total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def discard_serialized_computed_fields(cls, value: object) -> object:
        """Allow strict reports to round-trip their four deterministic fields."""
        if not isinstance(value, dict):
            return value
        sanitized = cast(dict[str, object], value).copy()
        for field_name in (
            "off_first_relevant_rank",
            "on_first_relevant_rank",
            "off_missing_required_document_ids",
            "on_missing_required_document_ids",
        ):
            sanitized.pop(field_name, None)
        return sanitized

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if not self.required_document_ids or len(set(self.required_document_ids)) != len(
            self.required_document_ids
        ):
            raise ValueError("required_document_ids 必须非空且不能重复")
        for document_ids in (self.off_document_ids, self.on_document_ids):
            if document_ids is not None and (
                len(document_ids) > _TOP_K or len(set(document_ids)) != len(document_ids)
            ):
                raise ValueError("评测文档排名最多 10 个且不能重复")
        if self.retrieval_error_code is not None:
            if self.off_document_ids or self.retrieval_latency_ms is not None:
                raise ValueError("检索失败不能同时保存候选或成功延迟")
            if self.rerank_attempted:
                raise ValueError("检索失败后不能调用 Rerank")
        elif self.retrieval_latency_ms is None:
            raise ValueError("检索成功必须记录延迟")
        if self.rerank_attempted:
            if (self.on_document_ids is None) == (self.rerank_error_code is None):
                raise ValueError("Rerank 成功结果和错误码必须且只能存在一个")
            if self.rerank_error_code is None and self.rerank_latency_ms is None:
                raise ValueError("Rerank 成功必须记录延迟")
        elif any(
            value is not None
            for value in (
                self.rerank_error_code,
                self.rerank_latency_ms,
                self.rerank_total_tokens,
            )
        ):
            raise ValueError("未调用 Rerank 时不能记录 Rerank 结果")
        return self

    @computed_field
    @property
    def off_first_relevant_rank(self) -> int | None:
        """Return the first required document rank in the OFF output."""
        return _first_relevant_rank(self.off_document_ids, set(self.required_document_ids))

    @computed_field
    @property
    def on_first_relevant_rank(self) -> int | None:
        """Return the first required document rank in the ON output."""
        return _first_relevant_rank(self.on_document_ids or (), set(self.required_document_ids))

    @computed_field
    @property
    def off_missing_required_document_ids(self) -> tuple[str, ...]:
        """Return required documents absent from the OFF TopK."""
        return _missing(self.required_document_ids, self.off_document_ids)

    @computed_field
    @property
    def on_missing_required_document_ids(self) -> tuple[str, ...] | None:
        """Return required documents absent from ON, or None when ON was not enabled."""
        if self.on_document_ids is None:
            return None
        return _missing(self.required_document_ids, self.on_document_ids)


class FullRetrievalReport(BaseModel):
    """Aggregate and per-case output for one fixed full-dataset run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    rerank_enabled: bool
    sample_count: int = Field(ge=1)
    retrieval_successes: int = Field(ge=0)
    retrieval_failures: int = Field(ge=0)
    rerank_live_calls: int = Field(ge=0)
    rerank_successes: int = Field(ge=0)
    rerank_failures: int = Field(ge=0)
    rerank_total_tokens: int = Field(ge=0)
    off_metrics: RetrievalEvaluationMetrics
    on_metrics: RetrievalEvaluationMetrics | None = None
    wins: int | None = Field(default=None, ge=0)
    ties: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)
    retrieval_latency_ms_p50: float | None = Field(default=None, ge=0)
    retrieval_latency_ms_p95: float | None = Field(default=None, ge=0)
    rerank_latency_ms_p50: float | None = Field(default=None, ge=0)
    rerank_latency_ms_p95: float | None = Field(default=None, ge=0)
    cases: tuple[FullRetrievalCaseResult, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.sample_count != EXPECTED_RAG_CASES or len(self.cases) != self.sample_count:
            raise ValueError("完整检索报告必须覆盖固定 132 条 RAG 样本")
        if self.retrieval_successes + self.retrieval_failures != self.sample_count:
            raise ValueError("检索成功和失败数必须覆盖全部样本")
        if self.rerank_live_calls != self.rerank_successes + self.rerank_failures:
            raise ValueError("Rerank 成功和失败数关系无效")
        comparisons = (self.wins, self.ties, self.losses)
        if self.rerank_enabled:
            if self.on_metrics is None or any(value is None for value in comparisons):
                raise ValueError("Rerank ON 报告必须包含 ON 指标和胜平负")
            if sum(value for value in comparisons if value is not None) != self.sample_count:
                raise ValueError("胜平负必须覆盖全部样本")
        elif self.on_metrics is not None or any(value is not None for value in comparisons):
            raise ValueError("Rerank OFF 报告不能包含 ON 指标")
        return self


async def run_full_retrieval_evaluation(
    dataset: FullEvaluationDataset,
    retrieval: VectorRetrievalUseCase,
    postprocessor: RetrievalPostProcessor,
    scope: VectorSearchScope,
    *,
    enable_rerank: bool,
    abort_on_rerank_degradation: bool = True,
) -> FullRetrievalReport:
    """Run each RAG query once and optionally Rerank its exact fused candidates."""
    cases: list[FullRetrievalCaseResult] = []
    for sample in dataset.rag_samples:
        started = perf_counter()
        try:
            search_result = await retrieval.search(VectorSearchRequest(sample.query, scope))
            fusion = postprocessor.fuse((search_result,))
        except RetrievalError as error:
            cases.append(
                FullRetrievalCaseResult(
                    query_id=sample.query_id,
                    required_document_ids=sample.expected_doc_ids,
                    retrieval_error_code=error.code.value,
                )
            )
            continue
        except ModelError as error:
            cases.append(
                FullRetrievalCaseResult(
                    query_id=sample.query_id,
                    required_document_ids=sample.expected_doc_ids,
                    retrieval_error_code=f"model_{error.code.value}",
                )
            )
            continue

        retrieval_latency_ms = (perf_counter() - started) * 1000
        off_ids = rank_documents_from_top_chunks(fusion.candidates)
        if not enable_rerank or not fusion.candidates:
            cases.append(
                FullRetrievalCaseResult(
                    query_id=sample.query_id,
                    required_document_ids=sample.expected_doc_ids,
                    off_document_ids=off_ids,
                    on_document_ids=() if enable_rerank else None,
                    retrieval_latency_ms=retrieval_latency_ms,
                )
            )
            continue

        rerank_started = perf_counter()
        reranked = await postprocessor.rerank(uuid4(), sample.query, fusion.candidates)
        rerank_latency_ms = (perf_counter() - rerank_started) * 1000
        if reranked.degradation_reason is not None:
            if abort_on_rerank_degradation:
                raise FullRetrievalRunError(
                    sample.query_id,
                    reranked.degradation_reason.value,
                )
            cases.append(
                FullRetrievalCaseResult(
                    query_id=sample.query_id,
                    required_document_ids=sample.expected_doc_ids,
                    off_document_ids=off_ids,
                    rerank_attempted=True,
                    rerank_error_code=reranked.degradation_reason.value,
                    retrieval_latency_ms=retrieval_latency_ms,
                )
            )
            continue
        cases.append(
            FullRetrievalCaseResult(
                query_id=sample.query_id,
                required_document_ids=sample.expected_doc_ids,
                off_document_ids=off_ids,
                on_document_ids=rank_documents_from_top_chunks(reranked.candidates),
                rerank_attempted=True,
                retrieval_latency_ms=retrieval_latency_ms,
                rerank_latency_ms=rerank_latency_ms,
                rerank_total_tokens=reranked.total_tokens,
            )
        )

    return _build_report(dataset.dataset_id, tuple(cases), enable_rerank)


def merge_full_retrieval_reports(
    off_report: FullRetrievalReport,
    live_report: FullRetrievalReport,
) -> FullRetrievalReport:
    """Attach corrected deterministic OFF rankings to an existing paid ON run."""
    if off_report.rerank_enabled or not live_report.rerank_enabled:
        raise ValueError("合并报告必须按 OFF、ON 顺序提供")
    if off_report.dataset_id != live_report.dataset_id:
        raise ValueError("OFF 和 ON 报告的数据集不一致")
    if off_report.retrieval_failures or live_report.retrieval_failures:
        raise ValueError("存在检索失败时不能合并 OFF/ON 报告")
    off_cases = {case.query_id: case for case in off_report.cases}
    if set(off_cases) != {case.query_id for case in live_report.cases}:
        raise ValueError("OFF 和 ON 报告的 Query ID 不一致")

    merged_cases: list[FullRetrievalCaseResult] = []
    for live_case in live_report.cases:
        off_case = off_cases[live_case.query_id]
        if off_case.required_document_ids != live_case.required_document_ids:
            raise ValueError("OFF 和 ON 报告的 required 文档不一致")
        merged_cases.append(
            FullRetrievalCaseResult(
                query_id=live_case.query_id,
                required_document_ids=live_case.required_document_ids,
                off_document_ids=off_case.off_document_ids,
                on_document_ids=live_case.on_document_ids,
                rerank_attempted=live_case.rerank_attempted,
                rerank_error_code=live_case.rerank_error_code,
                retrieval_latency_ms=off_case.retrieval_latency_ms,
                rerank_latency_ms=live_case.rerank_latency_ms,
                rerank_total_tokens=live_case.rerank_total_tokens,
            )
        )
    return _build_report(
        live_report.dataset_id,
        tuple(merged_cases),
        True,
        extra_limitations=(
            "付费 ON 完成后修正 OFF 的 Chunk TopK 统计, OFF 使用同配置确定性本地复算。",
        ),
    )


def _build_report(
    dataset_id: str,
    case_results: tuple[FullRetrievalCaseResult, ...],
    enable_rerank: bool,
    *,
    extra_limitations: tuple[str, ...] = (),
) -> FullRetrievalReport:
    retrieval_latencies = tuple(
        case.retrieval_latency_ms for case in case_results if case.retrieval_latency_ms is not None
    )
    rerank_latencies = tuple(
        case.rerank_latency_ms for case in case_results if case.rerank_latency_ms is not None
    )
    retrieval_failures = sum(case.retrieval_error_code is not None for case in case_results)
    rerank_live_calls = sum(case.rerank_attempted for case in case_results)
    rerank_failures = sum(case.rerank_error_code is not None for case in case_results)
    wins, ties, losses = _wins_ties_losses(case_results) if enable_rerank else (None, None, None)
    return FullRetrievalReport(
        dataset_id=dataset_id,
        rerank_enabled=enable_rerank,
        sample_count=len(case_results),
        retrieval_successes=len(case_results) - retrieval_failures,
        retrieval_failures=retrieval_failures,
        rerank_live_calls=rerank_live_calls,
        rerank_successes=rerank_live_calls - rerank_failures,
        rerank_failures=rerank_failures,
        rerank_total_tokens=sum(case.rerank_total_tokens or 0 for case in case_results),
        off_metrics=calculate_retrieval_metrics(case_results, ranking="off"),
        on_metrics=(
            calculate_retrieval_metrics(case_results, ranking="on") if enable_rerank else None
        ),
        wins=wins,
        ties=ties,
        losses=losses,
        retrieval_latency_ms_p50=_percentile(retrieval_latencies, 0.50),
        retrieval_latency_ms_p95=_percentile(retrieval_latencies, 0.95),
        rerank_latency_ms_p50=_percentile(rerank_latencies, 0.50),
        rerank_latency_ms_p95=_percentile(rerank_latencies, 0.95),
        cases=case_results,
        limitations=(
            "检索指标只覆盖 requires_rag=true 的 132 条样本。",
            "主指标只使用 required 文档标签, nice 标签仅供后续诊断。",
            "文档级排名取该文档首个 Chunk 在 TopK 中的位置。",
            *extra_limitations,
        ),
    )


def calculate_retrieval_metrics(
    cases: Iterable[FullRetrievalCaseResult],
    *,
    ranking: Literal["off", "on"],
) -> RetrievalEvaluationMetrics:
    """Calculate fixed document-level metrics, counting failures as misses."""
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("检索指标分母不能为空")
    rankings = tuple(
        case.off_document_ids if ranking == "off" else (case.on_document_ids or ())
        for case in materialized
    )
    required_sets = tuple(set(case.required_document_ids) for case in materialized)
    ranks = tuple(
        _first_relevant_rank(document_ids, required)
        for document_ids, required in zip(rankings, required_sets, strict=True)
    )
    denominator = len(materialized)
    return RetrievalEvaluationMetrics(
        hit_at_1=_hit_at(ranks, 1) / denominator,
        hit_at_3=_hit_at(ranks, 3) / denominator,
        hit_at_5=_hit_at(ranks, 5) / denominator,
        hit_at_10=_hit_at(ranks, 10) / denominator,
        recall_at_3=_mean_recall(rankings, required_sets, 3),
        recall_at_5=_mean_recall(rankings, required_sets, 5),
        recall_at_10=_mean_recall(rankings, required_sets, 10),
        mrr_at_10=sum(0.0 if rank is None or rank > 10 else 1 / rank for rank in ranks)
        / denominator,
    )


def rank_documents_from_top_chunks(
    candidates: Iterable[VectorSearchCandidate],
    *,
    top_k: int = _TOP_K,
) -> tuple[str, ...]:
    """Deduplicate document IDs only after enforcing the Chunk TopK cutoff."""
    if top_k < 1:
        raise ValueError("Chunk TopK 必须大于 0")
    document_ids: list[str] = []
    seen: set[str] = set()
    for chunk_rank, candidate in enumerate(candidates, start=1):
        if chunk_rank > top_k:
            break
        source_key = candidate.source_key
        if source_key not in seen:
            seen.add(source_key)
            document_ids.append(source_key)
    return tuple(document_ids)


def _first_relevant_rank(ordered_ids: tuple[str, ...], relevant_ids: set[str]) -> int | None:
    return next(
        (
            rank
            for rank, document_id in enumerate(ordered_ids, start=1)
            if document_id in relevant_ids
        ),
        None,
    )


def _missing(required_ids: tuple[str, ...], ranked_ids: tuple[str, ...]) -> tuple[str, ...]:
    present = set(ranked_ids)
    return tuple(document_id for document_id in required_ids if document_id not in present)


def _hit_at(ranks: tuple[int | None, ...], cutoff: int) -> int:
    return sum(rank is not None and rank <= cutoff for rank in ranks)


def _mean_recall(
    rankings: tuple[tuple[str, ...], ...],
    required_sets: tuple[set[str], ...],
    cutoff: int,
) -> float:
    return sum(
        len(set(document_ids[:cutoff]).intersection(required)) / len(required)
        for document_ids, required in zip(rankings, required_sets, strict=True)
    ) / len(rankings)


def _wins_ties_losses(
    cases: tuple[FullRetrievalCaseResult, ...],
) -> tuple[int, int, int]:
    wins = ties = losses = 0
    for case in cases:
        off_rank = case.off_first_relevant_rank
        on_rank = case.on_first_relevant_rank
        if on_rank == off_rank:
            ties += 1
        elif on_rank is not None and (off_rank is None or on_rank < off_rank):
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


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
