"""Exact cache identity, path, and paid-call tests for the staged Intent CLI."""

from pathlib import Path

import pytest

from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.full_intent import FullIntentReport
from customer_agent2.evaluation.staged_intent import StagedIntentCache, load_m5d_stage_manifest
from customer_agent2.evaluation.staged_intent_cli import (
    load_staged_intent_cache,
    new_staged_intent_cache,
    staged_intent_cache_lock,
    validate_paid_call_acknowledgement,
    validate_staged_intent_paths,
    write_staged_intent_cache,
)
from customer_agent2.infrastructure.intents import (
    intent_tree_fingerprint,
    load_intent_tree_json,
)
from tests.settings import IsolatedSettings

PROJECT_ROOT = Path(__file__).parents[1]
BASELINE_REPORT = PROJECT_ROOT / "evaluation" / "reports" / "m5c-full-intent.json"
STAGE_MANIFEST = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-stages.json"
CANDIDATE_TREE = PROJECT_ROOT / "evaluation" / "config" / "m5d-intent-tree-v2.json"


def _new_cache(tree_content: str) -> StagedIntentCache:
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    baseline = FullIntentReport.model_validate_json(BASELINE_REPORT.read_text(encoding="utf-8"))
    manifest = load_m5d_stage_manifest(STAGE_MANIFEST.read_text(encoding="utf-8"))
    tree = load_intent_tree_json(tree_content)
    return new_staged_intent_cache(
        settings,
        "ragenteval-v1-all",
        manifest,
        baseline,
        intent_tree_version=tree.version,
        intent_tree_sha256=intent_tree_fingerprint(tree),
        timeout_seconds=60,
    )


def test_staged_cache_round_trips_and_rejects_tree_drift(tmp_path: Path) -> None:
    tree_content = CANDIDATE_TREE.read_text(encoding="utf-8")
    expected = _new_cache(tree_content)
    path = tmp_path / "staged.cache.json"
    write_staged_intent_cache(path, expected)

    assert load_staged_intent_cache(path, expected) == expected

    changed = tree_content.replace("通用建议", "一般建议")
    with pytest.raises(ModelError) as captured:
        load_staged_intent_cache(path, _new_cache(changed))

    assert captured.value.code is ModelErrorCode.CONFIGURATION


def test_paid_call_acknowledgement_must_equal_uncached_count() -> None:
    validate_paid_call_acknowledgement(22, 22)

    with pytest.raises(ModelError, match="精确确认 22") as captured:
        validate_paid_call_acknowledgement(150, 22)

    assert captured.value.code is ModelErrorCode.CONFIGURATION


def test_staged_cache_lock_rejects_concurrent_process_and_releases(tmp_path: Path) -> None:
    cache = tmp_path / "candidate.cache.json"
    lock = tmp_path / "candidate.cache.json.lock"

    with staged_intent_cache_lock(cache):
        assert lock.exists()
        with pytest.raises(ModelError, match="另一个进程"), staged_intent_cache_lock(cache):
            pytest.fail("重复阶段进程不应进入锁内")

    assert not lock.exists()

    with pytest.raises(RuntimeError, match="synthetic"), staged_intent_cache_lock(cache):
        raise RuntimeError("synthetic")
    assert not lock.exists()


def test_stage_outputs_cannot_overwrite_inputs_or_each_other(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    manifest = tmp_path / "manifest.json"
    tree = tmp_path / "tree.json"
    cache = tmp_path / "cache.json"
    output = tmp_path / "stage.json"
    full = tmp_path / "full.json"

    validate_staged_intent_paths(
        snapshot=tmp_path / "snapshot",
        baseline_report=baseline,
        stage_manifest=manifest,
        intent_tree=tree,
        cache=cache,
        output=output,
        full_output=full,
    )

    with pytest.raises(ModelError):
        validate_staged_intent_paths(
            snapshot=tmp_path / "snapshot",
            baseline_report=baseline,
            stage_manifest=manifest,
            intent_tree=tree,
            cache=cache,
            output=cache,
            full_output=full,
        )

    snapshot = tmp_path / "snapshot"
    with pytest.raises(ModelError):
        validate_staged_intent_paths(
            snapshot=snapshot,
            baseline_report=baseline,
            stage_manifest=manifest,
            intent_tree=tree,
            cache=snapshot / "cache.json",
            output=output,
            full_output=full,
        )
