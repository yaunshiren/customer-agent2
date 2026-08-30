"""Validation tests for the versioned 150-case M5-C snapshot."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from customer_agent2.evaluation import (
    EXPECTED_DOCUMENTS,
    EXPECTED_FULL_CASES,
    EXPECTED_NO_RAG_CASES,
    EXPECTED_RAG_CASES,
    EXPECTED_SMOKE_CASES,
    FullEvaluationSample,
    load_full_evaluation_assets,
)

SNAPSHOT_ROOT = Path(__file__).parents[1] / "evaluation" / "datasets" / "ragenteval-v1"


def test_versioned_full_assets_are_complete_and_cross_referenced() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)

    assert len(assets.dataset.samples) == EXPECTED_FULL_CASES
    assert len(assets.dataset.smoke_query_ids) == EXPECTED_SMOKE_CASES
    assert len(assets.dataset.rag_samples) == EXPECTED_RAG_CASES
    assert len(assets.dataset.samples) - len(assets.dataset.rag_samples) == EXPECTED_NO_RAG_CASES
    assert len(assets.documents) == EXPECTED_DOCUMENTS
    assert {document.category for document in assets.documents} == {
        "01_product",
        "02_manual",
        "03_policy",
        "04_faq",
    }
    assert any(document.document_id == "PRODUCT_MAPPING" for document in assets.documents)


def test_full_sample_rejects_inconsistent_rag_and_document_labels() -> None:
    with pytest.raises(ValidationError, match="requires_rag"):
        FullEvaluationSample.model_validate(
            {
                "query_id": "TEST-1",
                "query": "测试问题",
                "intent_l1": "SUPPORT",
                "intent_l2": "S1_测试",
                "difficulty": "easy",
                "requires_rag": False,
                "expected_answer_type": "factual",
                "expected_doc_ids": ["DOC_1"],
                "trap_type": "test",
                "ground_truth": "测试答案",
                "eval_metrics": ["recall@5"],
            }
        )


def test_full_assets_reject_missing_dataset_files(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    with pytest.raises(ValueError, match=r"eval_set_v1_all\.jsonl"):
        load_full_evaluation_assets(snapshot)
