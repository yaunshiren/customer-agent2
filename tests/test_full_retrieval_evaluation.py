"""Deterministic tests for the 132-case document-level evaluation runner."""

import hashlib
import math
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from customer_agent2.application import RetrievalPostProcessor
from customer_agent2.domain.models import (
    DocumentFormat,
    EmbeddingIndexConfiguration,
    RerankItem,
    RerankRequest,
    RerankResult,
    RetrievalError,
    RetrievalErrorCode,
    RetrievedChunkSource,
    VectorSearchCandidate,
    VectorSearchRequest,
    VectorSearchResult,
    VectorSearchScope,
)
from customer_agent2.evaluation.full_dataset import load_full_evaluation_assets
from customer_agent2.evaluation.full_retrieval import (
    FullRetrievalCaseResult,
    FullRetrievalReport,
    calculate_retrieval_metrics,
    merge_full_retrieval_reports,
    rank_documents_from_top_chunks,
    run_full_retrieval_evaluation,
)

SNAPSHOT_ROOT = Path(__file__).parents[1] / "evaluation" / "datasets" / "ragenteval-v1"
INDEX = EmbeddingIndexConfiguration("embedding", "revision", 8, True)
KNOWLEDGE_BASE_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
SCOPE = VectorSearchScope((KNOWLEDGE_BASE_ID,))


class GoldRetrieval:
    """Return each query's required document IDs without reading chunk content."""

    def __init__(
        self,
        query_documents: dict[str, tuple[str, ...]],
        *,
        fail_query: str | None = None,
    ) -> None:
        self._query_documents = query_documents
        self._fail_query = fail_query
        self.calls = 0

    async def search(self, request: VectorSearchRequest) -> VectorSearchResult:
        self.calls += 1
        if request.query == self._fail_query:
            raise RetrievalError(
                RetrievalErrorCode.PERSISTENCE_FAILURE,
                "测试检索失败",
                retryable=True,
            )
        candidates = tuple(
            _candidate(request.query, document_id, rank)
            for rank, document_id in enumerate(
                self._query_documents[request.query],
                start=1,
            )
        )
        return VectorSearchResult(INDEX, candidates)


class IdentityRerank:
    """Return the exact candidate order and one synthetic token per call."""

    model_id = "fake-rerank"

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, request: RerankRequest) -> RerankResult:
        self.calls += 1
        return RerankResult(
            model_id=self.model_id,
            items=tuple(
                RerankItem(index, document.document_id, 1.0 - index / 100)
                for index, document in enumerate(request.documents[: request.result_limit])
            ),
            total_tokens=1,
        )


def _candidate(query: str, document_id: str, rank: int) -> VectorSearchCandidate:
    content = f"{query}-{document_id}"
    chunk_id = uuid5(NAMESPACE_URL, f"{query}:{document_id}")
    document_uuid = uuid5(NAMESPACE_URL, document_id)
    distance = rank / 100
    return VectorSearchCandidate(
        rank=rank,
        chunk_id=chunk_id,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=document_uuid,
        document_version_id=uuid5(NAMESPACE_URL, f"version:{document_id}"),
        source_key=document_id,
        display_name=f"{document_id}.md",
        document_format=DocumentFormat.MARKDOWN,
        media_type="text/markdown",
        parser_name="test",
        parser_version="1",
        chunk_index=0,
        content=content,
        token_count=1,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        section=None,
        page_number=None,
        source=RetrievedChunkSource(0, 0, 1, 1, (), 0),
        cosine_distance=distance,
        similarity=1 - distance,
    )


def _postprocessor(model: IdentityRerank) -> RetrievalPostProcessor:
    return RetrievalPostProcessor(
        model,
        rrf_k=60,
        rerank_candidate_limit=40,
        context_top_k=10,
        max_chunks_per_document=2,
        rerank_timeout_seconds=10,
    )


def test_document_metrics_count_missing_required_documents() -> None:
    cases = (
        FullRetrievalCaseResult(
            query_id="A",
            required_document_ids=("D1", "D2"),
            off_document_ids=("D1", "OTHER"),
            retrieval_latency_ms=1,
        ),
        FullRetrievalCaseResult(
            query_id="B",
            required_document_ids=("D3",),
            off_document_ids=(),
            retrieval_latency_ms=2,
        ),
    )

    metrics = calculate_retrieval_metrics(cases, ranking="off")

    assert metrics.hit_at_1 == 0.5
    assert metrics.hit_at_10 == 0.5
    assert metrics.recall_at_10 == 0.25
    assert metrics.mrr_at_10 == 0.5
    assert cases[0].off_missing_required_document_ids == ("D2",)


def test_document_ranking_applies_chunk_top_k_before_document_deduplication() -> None:
    candidates = tuple(_candidate("query", f"DOC-{(rank - 1) // 2}", rank) for rank in range(1, 12))

    ranking = rank_documents_from_top_chunks(candidates, top_k=10)

    assert ranking == ("DOC-0", "DOC-1", "DOC-2", "DOC-3", "DOC-4")


@pytest.mark.asyncio
async def test_full_off_run_uses_only_132_rag_samples_and_sanitizes_report() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    query_documents = {
        sample.query: sample.expected_doc_ids for sample in assets.dataset.rag_samples
    }
    retrieval = GoldRetrieval(query_documents)
    rerank = IdentityRerank()

    report = await run_full_retrieval_evaluation(
        assets.dataset,
        retrieval,
        _postprocessor(rerank),
        SCOPE,
        enable_rerank=False,
    )

    assert retrieval.calls == 132
    assert rerank.calls == 0
    assert report.sample_count == 132
    assert report.retrieval_failures == 0
    assert report.off_metrics.hit_at_1 == 1
    assert report.off_metrics.recall_at_10 == 1
    serialized = report.model_dump_json()
    assert assets.dataset.rag_samples[0].query not in serialized
    assert assets.dataset.rag_samples[0].ground_truth not in serialized
    assert '"off_first_relevant_rank":1' in serialized
    assert '"off_missing_required_document_ids":[]' in serialized
    assert FullRetrievalReport.model_validate_json(serialized) == report


@pytest.mark.asyncio
async def test_full_on_run_reuses_one_retrieval_and_tracks_tokens() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    query_documents = {
        sample.query: sample.expected_doc_ids for sample in assets.dataset.rag_samples
    }
    retrieval = GoldRetrieval(query_documents)
    rerank = IdentityRerank()

    report = await run_full_retrieval_evaluation(
        assets.dataset,
        retrieval,
        _postprocessor(rerank),
        SCOPE,
        enable_rerank=True,
    )

    assert retrieval.calls == 132
    assert rerank.calls == 132
    assert report.rerank_live_calls == 132
    assert report.rerank_total_tokens == 132
    assert report.on_metrics == report.off_metrics
    assert (report.wins, report.ties, report.losses) == (0, 132, 0)

    corrected_off = await run_full_retrieval_evaluation(
        assets.dataset,
        retrieval,
        _postprocessor(rerank),
        SCOPE,
        enable_rerank=False,
    )
    merged = merge_full_retrieval_reports(corrected_off, report)
    assert merged.rerank_live_calls == 132
    assert merged.rerank_total_tokens == 132
    assert merged.off_metrics == corrected_off.off_metrics
    assert merged.on_metrics == report.on_metrics


@pytest.mark.asyncio
async def test_full_run_counts_retrieval_failure_as_a_miss() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    query_documents = {
        sample.query: sample.expected_doc_ids for sample in assets.dataset.rag_samples
    }
    failed_sample = assets.dataset.rag_samples[0]
    retrieval = GoldRetrieval(query_documents, fail_query=failed_sample.query)
    rerank = IdentityRerank()

    report = await run_full_retrieval_evaluation(
        assets.dataset,
        retrieval,
        _postprocessor(rerank),
        SCOPE,
        enable_rerank=False,
    )

    assert report.retrieval_successes == 131
    assert report.retrieval_failures == 1
    assert math.isclose(report.off_metrics.hit_at_10, 131 / 132)
    assert report.cases[0].retrieval_error_code == "persistence_failure"
