"""Create the versioned document and vector index schema.

Revision ID: 0002_document_index_schema
Revises: 0001_infrastructure_baseline
Create Date: 2026-08-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002_document_index_schema"
down_revision: str | Sequence[str] | None = "0001_infrastructure_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create knowledge-base, document-version, chunk, and vector-index storage."""
    op.create_table(
        "knowledge_bases",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("embedding_model_id", sa.String(length=255), nullable=False),
        sa.Column("embedding_model_revision", sa.String(length=255), nullable=False),
        sa.Column(
            "embedding_dimension",
            sa.Integer(),
            server_default=sa.text("768"),
            nullable=False,
        ),
        sa.Column(
            "embedding_normalized",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "embedding_dimension = 768",
            name=op.f("ck_knowledge_bases_embedding_dimension_fixed"),
        ),
        sa.CheckConstraint(
            "embedding_normalized",
            name=op.f("ck_knowledge_bases_embedding_must_be_normalized"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(embedding_model_id)) > 0",
            name=op.f("ck_knowledge_bases_embedding_model_id_not_blank"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(embedding_model_revision)) > 0",
            name=op.f("ck_knowledge_bases_embedding_model_revision_not_blank"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(name)) > 0",
            name=op.f("ck_knowledge_bases_name_not_blank"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(slug)) > 0",
            name=op.f("ck_knowledge_bases_slug_not_blank"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_bases")),
        sa.UniqueConstraint("slug", name=op.f("uq_knowledge_bases_slug")),
    )

    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("knowledge_base_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_key", sa.String(length=1024), nullable=False),
        sa.Column("display_name", sa.String(length=500), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(btrim(display_name)) > 0",
            name=op.f("ck_documents_display_name_not_blank"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(source_key)) > 0",
            name=op.f("ck_documents_source_key_not_blank"),
        ),
        sa.ForeignKeyConstraint(
            ["knowledge_base_id"],
            ["knowledge_bases.id"],
            name=op.f("fk_documents_knowledge_base_id_knowledge_bases"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_documents_id_knowledge_base_id",
        ),
        sa.UniqueConstraint(
            "knowledge_base_id",
            "source_key",
            name="uq_documents_knowledge_base_source_key",
        ),
    )
    op.create_index("ix_documents_knowledge_base_id", "documents", ["knowledge_base_id"])

    op.create_table(
        "document_versions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("knowledge_base_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default=sa.text("'building'"),
            nullable=False,
        ),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=255), nullable=True),
        sa.Column("object_uri", sa.Text(), nullable=True),
        sa.Column("parser_name", sa.String(length=100), nullable=True),
        sa.Column("parser_version", sa.String(length=100), nullable=True),
        sa.Column(
            "source_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_document_versions_content_sha256_format"),
        ),
        sa.CheckConstraint(
            "status IN ('building', 'active', 'failed', 'superseded')",
            name=op.f("ck_document_versions_status_allowed"),
        ),
        sa.CheckConstraint(
            "version_number > 0",
            name=op.f("ck_document_versions_version_number_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "knowledge_base_id"],
            ["documents.id", "documents.knowledge_base_id"],
            name="fk_document_versions_document_knowledge_base",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_versions")),
        sa.UniqueConstraint(
            "document_id",
            "version_number",
            name="uq_document_versions_document_version_number",
        ),
        sa.UniqueConstraint(
            "id",
            "knowledge_base_id",
            name="uq_document_versions_id_knowledge_base_id",
        ),
    )
    op.create_index(
        "ix_document_versions_knowledge_base_id",
        "document_versions",
        ["knowledge_base_id"],
    )
    op.create_index(
        "ux_document_versions_one_active",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )

    op.create_table(
        "chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("knowledge_base_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("section", sa.String(length=500), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column(
            "source_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(dim=768), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name=op.f("ck_chunks_chunk_index_nonnegative"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(content)) > 0",
            name=op.f("ck_chunks_content_not_blank"),
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name=op.f("ck_chunks_content_sha256_format"),
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number > 0",
            name=op.f("ck_chunks_page_number_positive"),
        ),
        sa.CheckConstraint(
            "token_count > 0",
            name=op.f("ck_chunks_token_count_positive"),
        ),
        sa.ForeignKeyConstraint(
            ["document_version_id", "knowledge_base_id"],
            ["document_versions.id", "document_versions.knowledge_base_id"],
            name="fk_chunks_document_version_knowledge_base",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chunks")),
        sa.UniqueConstraint(
            "document_version_id",
            "chunk_index",
            name="uq_chunks_document_version_chunk_index",
        ),
    )
    op.create_index(
        "ix_chunks_knowledge_base_document_version",
        "chunks",
        ["knowledge_base_id", "document_version_id"],
    )
    op.create_index(
        "ix_chunks_embedding_hnsw_cosine",
        "chunks",
        ["embedding"],
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_using="hnsw",
        postgresql_with={"ef_construction": 64, "m": 16},
    )


def downgrade() -> None:
    """Remove M2-A business tables while retaining the pgvector extension."""
    op.drop_index("ix_chunks_embedding_hnsw_cosine", table_name="chunks")
    op.drop_index("ix_chunks_knowledge_base_document_version", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(
        "ux_document_versions_one_active",
        table_name="document_versions",
        postgresql_where=sa.text("status = 'active'"),
    )
    op.drop_index(
        "ix_document_versions_knowledge_base_id",
        table_name="document_versions",
    )
    op.drop_table("document_versions")
    op.drop_index("ix_documents_knowledge_base_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("knowledge_bases")
