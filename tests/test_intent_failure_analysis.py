"""Offline failure slicing and M5-D candidate-tree tests."""

from pathlib import Path

from customer_agent2.evaluation.full_dataset import load_full_evaluation_assets
from customer_agent2.evaluation.full_intent import FullIntentReport
from customer_agent2.evaluation.intent_failure_analysis import analyze_full_intent_failures
from customer_agent2.infrastructure.intents import load_intent_tree_json

PROJECT_ROOT = Path(__file__).parents[1]
SNAPSHOT_ROOT = PROJECT_ROOT / "evaluation" / "datasets" / "ragenteval-v1"
BASELINE_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "m5c-full-intent.json"
CANDIDATE_TREE = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-tree-v2.json"


def test_m5c_failure_analysis_matches_fixed_report_and_excludes_content() -> None:
    assets = load_full_evaluation_assets(SNAPSHOT_ROOT)
    report = FullIntentReport.model_validate_json(BASELINE_REPORT.read_text(encoding="utf-8"))

    analysis = analyze_full_intent_failures(assets.dataset, report)

    assert analysis.correct_count == 128
    assert analysis.incorrect_count == 22
    assert analysis.over_retrieval_count == 5
    assert analysis.under_retrieval_count == 0
    assert analysis.incorrect_clarification_count == 17
    assert analysis.incorrect_by_intent_l1 == {"SUPPORT": 9, "FEEDBACK": 10, "CHAT": 3}
    assert analysis.incorrect_by_decision_reason == {
        "explicit_clarification": 13,
        "high_confidence": 5,
        "low_confidence": 4,
    }
    serialized = analysis.model_dump_json()
    for sample in assets.dataset.samples:
        assert sample.query not in serialized
        assert sample.ground_truth not in serialized


def test_m5d_candidate_tree_is_separate_and_strictly_valid() -> None:
    tree = load_intent_tree_json(CANDIDATE_TREE.read_text(encoding="utf-8"))

    assert tree.version == "m5-d-v2-candidate"
    assert "功能建议" in tree.definitions[0].description
    assert "通用规则" in tree.definitions[1].description
    assert "仅当" in tree.definitions[2].description
