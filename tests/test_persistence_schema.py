"""Static contract tests for the accepted M2-A persistence schema."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKeyConstraint, Index
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from customer_agent2.infrastructure.persistence import EMBEDDING_DIMENSION, Base


def find_index(table_name: str, index_name: str) -> Index:
    """Return a named metadata index or fail with a useful assertion."""
    table = Base.metadata.tables[table_name]
    index = next((candidate for candidate in table.indexes if candidate.name == index_name), None)
    assert index is not None
    return index


def test_metadata_contains_only_the_m2a_business_tables() -> None:
    assert set(Base.metadata.tables) == {
        "chunks",
        "document_versions",
        "documents",
        "knowledge_bases",
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
