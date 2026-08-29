"""Configuration validation tests."""

import pytest
from pydantic import ValidationError

from tests.settings import IsolatedSettings as Settings


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
        Settings(
            retrieval_rerank_candidate_limit=4,
            retrieval_context_top_k=5,
        )


def test_settings_bound_hnsw_search_effort() -> None:
    assert Settings().retrieval_hnsw_ef_search == 100

    with pytest.raises(ValidationError, match="retrieval_hnsw_ef_search"):
        Settings(retrieval_hnsw_ef_search=1001)


def test_settings_parse_typed_infrastructure_urls() -> None:
    settings = Settings.model_validate(
        {
            "database_url": "postgresql+asyncpg://app:password@db:5432/app",
            "redis_url": "redis://cache:6379/2",
        }
    )

    assert settings.database_url.scheme == "postgresql+asyncpg"
    assert settings.redis_url.scheme == "redis"


def test_settings_reject_non_async_database_driver() -> None:
    with pytest.raises(ValidationError, match=r"postgresql\+asyncpg"):
        Settings.model_validate({"database_url": "postgresql://app:password@db:5432/app"})


def test_settings_reject_empty_redis_prefix() -> None:
    with pytest.raises(ValidationError, match="REDIS_KEY_PREFIX"):
        Settings(redis_key_prefix="  ")


def test_settings_keep_api_key_secret_and_separate_chat_profiles() -> None:
    settings = Settings.model_validate(
        {
            "dashscope_api_key": "test-secret-key",
            "chat_model_final": "final-model",
            "chat_model_fast": "fast-model",
        }
    )

    assert str(settings.dashscope_api_key) == "**********"
    assert "test-secret-key" not in repr(settings)
    assert settings.chat_model_final == "final-model"
    assert settings.chat_model_fast == "fast-model"


def test_settings_reject_first_packet_timeout_above_total_timeout() -> None:
    with pytest.raises(ValidationError, match="FIRST_PACKET"):
        Settings(
            llm_timeout_seconds=10,
            llm_first_packet_timeout_seconds=11,
        )


def test_settings_keep_rag_deadline_at_least_as_large_as_model_timeout() -> None:
    assert Settings().rag_global_timeout_seconds == 120

    with pytest.raises(ValidationError, match="RAG_GLOBAL_TIMEOUT_SECONDS"):
        Settings(llm_timeout_seconds=100, rag_global_timeout_seconds=99)


def test_settings_expose_confirmed_memory_baseline() -> None:
    settings = Settings()

    assert settings.memory_recent_turns == 6
    assert settings.memory_summary_trigger_turns == 12
    assert settings.memory_summary_timeout_seconds == 30
    assert settings.memory_summary_max_output_tokens == 512

    with pytest.raises(ValidationError, match="MEMORY_SUMMARY_TRIGGER_TURNS"):
        Settings(memory_recent_turns=6, memory_summary_trigger_turns=6)


def test_settings_expose_bounded_query_rewrite_baseline() -> None:
    settings = Settings()

    assert settings.query_rewrite_timeout_seconds == 20
    assert settings.query_rewrite_max_output_tokens == 512
    assert settings.query_rewrite_max_sub_questions == 3

    with pytest.raises(ValidationError, match="QUERY_REWRITE_TIMEOUT_SECONDS"):
        Settings(query_rewrite_timeout_seconds=120)
    with pytest.raises(ValidationError, match="query_rewrite_max_sub_questions"):
        Settings(query_rewrite_max_sub_questions=4)


def test_settings_expose_confirmed_intent_baseline() -> None:
    settings = Settings()

    assert settings.intent_high_confidence_threshold == 0.75
    assert settings.intent_ambiguity_margin == 0.10
    assert settings.intent_timeout_seconds == 20
    assert settings.intent_max_output_tokens == 256

    with pytest.raises(ValidationError, match="INTENT_TIMEOUT_SECONDS"):
        Settings(intent_timeout_seconds=120)
    with pytest.raises(ValidationError, match="INTENT_AMBIGUITY_MARGIN"):
        Settings(intent_high_confidence_threshold=0.2, intent_ambiguity_margin=0.3)


def test_settings_expose_m5a_postprocessing_baseline() -> None:
    settings = Settings()

    assert settings.retrieval_rrf_k == 60
    assert settings.retrieval_rerank_candidate_limit == 40
    assert settings.retrieval_context_top_k == 10
    assert settings.retrieval_max_chunks_per_document == 2
    assert settings.rerank_timeout_seconds == 10

    with pytest.raises(ValidationError, match="RERANK_TIMEOUT_SECONDS"):
        Settings(rerank_timeout_seconds=120)


def test_settings_protect_fixed_embedding_baseline_capabilities() -> None:
    with pytest.raises(ValidationError, match="DIMENSION"):
        Settings(local_embedding_dimension=1024)

    with pytest.raises(ValidationError, match="MAX_TOKENS"):
        Settings(local_embedding_max_tokens=1024)


def test_settings_reject_empty_embedding_revision() -> None:
    with pytest.raises(ValidationError, match="不能为空"):
        Settings(local_embedding_revision=" ")


def test_settings_type_and_bound_upload_file_size() -> None:
    assert Settings().upload_max_file_mb == 50
    assert Settings.model_validate({"upload_max_file_mb": "25"}).upload_max_file_mb == 25

    with pytest.raises(ValidationError, match="upload_max_file_mb"):
        Settings(upload_max_file_mb=0)


def test_settings_expose_positive_multiformat_parser_limits() -> None:
    settings = Settings()

    assert settings.document_max_extracted_chars == 5_000_000
    assert settings.document_max_pdf_pages == 1000
    assert settings.document_max_docx_entries == 2000
    assert settings.document_max_docx_uncompressed_mb == 200
    assert settings.document_max_docx_expansion_ratio == 100
    assert settings.document_max_csv_rows == 10_000
    assert settings.document_max_csv_columns == 200

    with pytest.raises(ValidationError, match="document_max_pdf_pages"):
        Settings(document_max_pdf_pages=0)


def test_settings_validate_confirmed_chunk_budget() -> None:
    settings = Settings()
    assert settings.chunk_target_tokens == 400
    assert settings.chunk_overlap_tokens == 64

    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(chunk_target_tokens=64, chunk_overlap_tokens=64)

    with pytest.raises(ValidationError, match="CHUNK_TARGET_TOKENS"):
        Settings(chunk_target_tokens=513)


def test_settings_require_credentials_when_rerank_is_enabled() -> None:
    with pytest.raises(ValidationError, match="DASHSCOPE_API_KEY"):
        Settings.model_validate(
            {
                "rerank_enabled": True,
                "dashscope_api_key": "",
                "dashscope_workspace_id": "workspace",
            }
        )

    with pytest.raises(ValidationError, match="M5-A"):
        Settings.model_validate(
            {
                "rerank_enabled": True,
                "dashscope_api_key": "test-key",
                "dashscope_workspace_id": "workspace",
            }
        )

    with pytest.raises(ValidationError, match="DASHSCOPE_WORKSPACE_ID"):
        Settings.model_validate(
            {
                "rerank_enabled": True,
                "dashscope_api_key": "test-key",
                "dashscope_workspace_id": "",
            }
        )
