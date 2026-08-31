"""Checkpoint and exact-configuration tests for the paid Intent CLI."""

from pathlib import Path

import pytest

from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.full_intent_cli import (
    load_intent_checkpoint,
    new_intent_checkpoint,
    validate_intent_experiment_paths,
    write_intent_checkpoint,
)
from customer_agent2.infrastructure.intents import (
    load_default_intent_tree,
    load_intent_tree_json,
)
from tests.settings import IsolatedSettings


def test_intent_checkpoint_round_trips_without_credentials(tmp_path: Path) -> None:
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    path = tmp_path / "intent.checkpoint.json"
    checkpoint = new_intent_checkpoint(
        settings,
        "dataset-v1",
        load_default_intent_tree(),
        timeout_seconds=60,
    )

    write_intent_checkpoint(path, checkpoint)
    loaded = load_intent_checkpoint(
        path,
        settings,
        "dataset-v1",
        load_default_intent_tree(),
        timeout_seconds=60,
    )

    assert loaded == checkpoint
    assert loaded.completed_cases == ()
    assert loaded.failed_attempts == ()
    assert loaded.reasoning_enabled is False


def test_intent_checkpoint_rejects_changed_experiment_configuration(tmp_path: Path) -> None:
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    path = tmp_path / "intent.checkpoint.json"
    write_intent_checkpoint(
        path,
        new_intent_checkpoint(
            settings,
            "dataset-v1",
            load_default_intent_tree(),
            timeout_seconds=20,
        ),
    )

    with pytest.raises(ModelError) as captured:
        load_intent_checkpoint(
            path,
            settings,
            "dataset-v1",
            load_default_intent_tree(),
            timeout_seconds=60,
        )

    assert captured.value.code is ModelErrorCode.CONFIGURATION


def test_intent_checkpoint_rejects_changed_tree_semantics(tmp_path: Path) -> None:
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    path = tmp_path / "intent.checkpoint.json"
    baseline = load_default_intent_tree()
    write_intent_checkpoint(
        path,
        new_intent_checkpoint(settings, "dataset-v1", baseline, timeout_seconds=60),
    )
    changed = load_intent_tree_json(
        '{"version":"m4-c-v1","routes":['
        '{"name":"system_direct","description":"changed"},'
        '{"name":"knowledge_base","description":"knowledge"},'
        '{"name":"clarification","description":"clarify"}]}'
    )

    with pytest.raises(ModelError) as captured:
        load_intent_checkpoint(
            path,
            settings,
            "dataset-v1",
            changed,
            timeout_seconds=60,
        )

    assert captured.value.code is ModelErrorCode.CONFIGURATION


def test_candidate_tree_cannot_overwrite_m5c_baseline_paths() -> None:
    with pytest.raises(ModelError) as captured:
        validate_intent_experiment_paths(
            Path("evaluation/config/m5d-intent-tree-v2.json"),
            Path("evaluation/reports/m5c-full-intent.json"),
            Path("evaluation/reports/m5d-full-intent-v2.checkpoint.json"),
        )

    assert captured.value.code is ModelErrorCode.CONFIGURATION

    validate_intent_experiment_paths(
        Path("evaluation/config/m5d-intent-tree-v2.json"),
        Path("evaluation/reports/m5d-full-intent-v2.json"),
        Path("evaluation/reports/m5d-full-intent-v2.checkpoint.json"),
    )
