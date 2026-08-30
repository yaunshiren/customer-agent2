"""Tests for idempotent M5-C corpus import orchestration."""

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from customer_agent2.domain.models import (
    DocumentIngestionRequest,
    EmbeddingIndexConfiguration,
    IngestionResult,
)
from customer_agent2.evaluation.full_corpus import import_full_evaluation_corpus
from customer_agent2.evaluation.full_dataset import (
    EvaluationCategory,
    load_full_evaluation_assets,
)

SNAPSHOT_ROOT = Path(__file__).parents[1] / "evaluation" / "datasets" / "ragenteval-v1"
INDEX = EmbeddingIndexConfiguration("embedding", "revision", 768, True)


class FakeCorpusState:
    def __init__(self, active_hashes: dict[tuple[UUID, str], str]) -> None:
        self.active_hashes = active_hashes
        self.knowledge_base_ids: dict[EvaluationCategory, UUID] = {
            category: uuid5(NAMESPACE_URL, category)
            for category in (
                "01_product",
                "02_manual",
                "03_policy",
                "04_faq",
            )
        }

    async def ensure_knowledge_bases(
        self,
        index_configuration: EmbeddingIndexConfiguration,
    ) -> dict[EvaluationCategory, UUID]:
        assert index_configuration == INDEX
        return self.knowledge_base_ids

    async def active_content_sha256(
        self,
        knowledge_base_id: UUID,
        source_key: str,
    ) -> str | None:
        return self.active_hashes.get((knowledge_base_id, source_key))


class RecordingIngestion:
    def __init__(self) -> None:
        self.requests: list[DocumentIngestionRequest] = []

    async def ingest(self, request: DocumentIngestionRequest) -> IngestionResult:
        self.requests.append(request)
        identity = f"{request.knowledge_base_id}:{request.source_key}"
        return IngestionResult(
            knowledge_base_id=request.knowledge_base_id,
            document_id=uuid5(NAMESPACE_URL, identity),
            version_id=uuid5(NAMESPACE_URL, f"version:{identity}"),
            version_number=1,
            chunk_count=2,
            content_sha256=hashlib.sha256(request.source.content).hexdigest(),
        )


@pytest.mark.asyncio
async def test_import_skips_identical_active_version_and_uses_business_doc_ids() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    skipped = assets.documents[0]
    state = FakeCorpusState({})
    state.active_hashes[(state.knowledge_base_ids[skipped.category], skipped.document_id)] = (
        skipped.content_sha256
    )
    ingestion = RecordingIngestion()

    report = await import_full_evaluation_corpus(assets, ingestion, state, INDEX)

    assert report.document_count == 116
    assert report.skipped_documents == 1
    assert report.imported_documents == 115
    assert report.imported_chunks == 230
    assert len(ingestion.requests) == 115
    assert skipped.document_id not in {request.source_key for request in ingestion.requests}
    assert all(
        request.source.declared_media_type == "text/markdown" for request in ingestion.requests
    )
    assert {request.source_key for request in ingestion.requests} == {
        document.document_id for document in assets.documents[1:]
    }


@pytest.mark.asyncio
async def test_second_identical_import_is_a_complete_noop() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    state = FakeCorpusState({})
    state.active_hashes.update(
        {
            (state.knowledge_base_ids[document.category], document.document_id): (
                document.content_sha256
            )
            for document in assets.documents
        }
    )
    ingestion = RecordingIngestion()

    report = await import_full_evaluation_corpus(assets, ingestion, state, INDEX)

    assert report.imported_documents == 0
    assert report.skipped_documents == 116
    assert report.imported_chunks == 0
    assert ingestion.requests == []
