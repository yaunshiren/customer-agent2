"""Static contract tests for the accepted persistence schema."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKeyConstraint, Index, String
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from customer_agent2.infrastructure.persistence import EMBEDDING_DIMENSION, Base


def find_index(table_name: str, index_name: str) -> Index:
    """Return a named metadata index or fail with a useful assertion."""
    table = Base.metadata.tables[table_name]
    index = next((candidate for candidate in table.indexes if candidate.name == index_name), None)
    assert index is not None
    return index


def test_metadata_contains_all_accepted_business_tables() -> None:
    assert set(Base.metadata.tables) == {
        "chunks",
        "conversations",
        "conversation_summaries",
        "document_versions",
        "documents",
        "knowledge_bases",
        "messages",
        "rag_runs",
    }


def test_embedding_column_and_hnsw_index_match_the_accepted_baseline() -> None:
    embedding_type = Base.metadata.tables["chunks"].c.embedding.type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == EMBEDDING_DIMENSION == 768

    index = find_index("chunks", "ix_chunks_embedding_hnsw_cosine")
    sql = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert "USING hnsw" in sql
    assert "vector_cosine_ops" in sql
    assert "ef_construction = 64" in sql
    assert "m = 16" in sql


def test_schema_prevents_cross_knowledge_base_document_links() -> None:
    version_table = Base.metadata.tables["document_versions"]
    chunk_table = Base.metadata.tables["chunks"]

    version_fk = next(
        constraint
        for constraint in version_table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    )
    chunk_fk = next(
        constraint
        for constraint in chunk_table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    )

    assert tuple(version_fk.column_keys) == ("document_id", "knowledge_base_id")
    assert tuple(chunk_fk.column_keys) == ("document_version_id", "knowledge_base_id")
    assert version_fk.ondelete == "CASCADE"
    assert chunk_fk.ondelete == "CASCADE"


def test_only_one_active_version_is_allowed_per_document() -> None:
    index = find_index("document_versions", "ux_document_versions_one_active")
    sql = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert index.unique is True
    assert "WHERE status = 'active'" in sql


def test_only_one_running_rag_run_is_allowed_per_conversation() -> None:
    index = find_index("rag_runs", "ux_rag_runs_one_running_per_conversation")
    sql = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert index.unique is True
    assert "WHERE status = 'running'" in sql


def test_rag_run_schema_accepts_only_consistent_intent_terminal_routes() -> None:
    table = Base.metadata.tables["rag_runs"]
    constraints = {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }

    intent_route_type = table.c.intent_route.type
    assert isinstance(intent_route_type, String)
    assert intent_route_type.length == 32
    assert "clarification" in constraints["ck_rag_runs_status_allowed"]
    route_constraint = constraints["ck_rag_runs_intent_route_matches_status"]
    assert "system_direct" in route_constraint
    assert "knowledge_base" in route_constraint
    assert "clarification" in route_constraint


def test_conversation_deletion_cascades_runs_and_messages() -> None:
    run_table = Base.metadata.tables["rag_runs"]
    message_table = Base.metadata.tables["messages"]
    run_fk = next(
        constraint
        for constraint in run_table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    )
    message_fks = {
        constraint.referred_table.name: constraint
        for constraint in message_table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    }

    assert run_fk.referred_table.name == "conversations"
    assert run_fk.ondelete == "CASCADE"
    assert message_fks["conversations"].ondelete == "CASCADE"
    assert message_fks["rag_runs"].ondelete == "SET NULL"


def test_conversation_deletion_cascades_its_summary() -> None:
    summary_table = Base.metadata.tables["conversation_summaries"]
    summary_fk = next(
        constraint
        for constraint in summary_table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    )

    assert summary_fk.referred_table.name == "conversations"
    assert summary_fk.ondelete == "CASCADE"
