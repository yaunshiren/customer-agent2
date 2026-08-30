"""Checkpoint and exact-configuration tests for the paid Intent CLI."""

from pathlib import Path

import pytest

from customer_agent2.domain.models import ModelError, ModelErrorCode
from customer_agent2.evaluation.full_intent_cli import (
    load_intent_checkpoint,
    new_intent_checkpoint,
    write_intent_checkpoint,
)
from tests.settings import IsolatedSettings


def test_intent_checkpoint_round_trips_without_credentials(tmp_path: Path) -> None:
    settings = IsolatedSettings(chat_model_fast="qwen3.8-flash")
    path = tmp_path / "intent.checkpoint.json"
    checkpoint = new_intent_checkpoint(
        settings,
        "dataset-v1",
        timeout_seconds=60,
    )

    write_intent_checkpoint(path, checkpoint)
    loaded = load_intent_checkpoint(
        path,
        settings,
        "dataset-v1",
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
        new_intent_checkpoint(settings, "dataset-v1", timeout_seconds=20),
    )

    with pytest.raises(ModelError) as captured:
        load_intent_checkpoint(
            path,
            settings,
            "dataset-v1",
            timeout_seconds=60,
        )

    assert captured.value.code is ModelErrorCode.CONFIGURATION
