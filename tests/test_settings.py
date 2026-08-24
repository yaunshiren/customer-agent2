"""Configuration validation tests."""

import pytest
from pydantic import ValidationError

from customer_agent2.config import Settings


def test_settings_normalize_api_prefix() -> None:
    settings = Settings(api_prefix=" /api/v1/ ")

    assert settings.api_prefix == "/api/v1"


def test_settings_reject_root_api_prefix() -> None:
    with pytest.raises(ValidationError, match="API_PREFIX 不能只包含"):
        Settings(api_prefix="/")


def test_settings_reject_recall_budget_below_top_k() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_RECALL_BUDGET"):
        Settings(retrieval_recall_budget=4, retrieval_context_top_k=5)


def test_settings_reject_rerank_limit_below_top_k() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_CANDIDATE_LIMIT"):
        Settings(retrieval_rerank_candidate_limit=4, retrieval_context_top_k=5)
