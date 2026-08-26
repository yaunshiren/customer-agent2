"""Create minimal conversation, message, and RAG Run storage.

Revision ID: 0003_conversation_rag_run
Revises: 0002_document_index_schema
Create Date: 2026-08-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_conversation_rag_run"
down_revision: str | Sequence[str] | None = "0002_document_index_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the M3-C conversation and minimal execution trace tables."""
    op.create_table(
        "conversations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conversations")),
    )

    op.create_table(
        "rag_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("request_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "knowledge_base_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'running'"),
            nullable=False,
        ),
        sa.Column("model_id", sa.String(length=255), nullable=True),
        sa.Column("finish_reason", sa.String(length=100), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "trace",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "source_chunk_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            server_default=sa.text("'{}'::uuid[]"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=100), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'no_context', 'failed', 'cancelled')",
            name=op.f("ck_rag_runs_status_allowed"),
        ),
        sa.CheckConstraint(
            "cardinality(knowledge_base_ids) > 0",
            name=op.f("ck_rag_runs_knowledge_base_ids_not_empty"),
        ),
        sa.CheckConstraint(
            "(status = 'running' AND finished_at IS NULL) "
            "OR (status <> 'running' AND finished_at IS NOT NULL)",
            name=op.f("ck_rag_runs_finished_at_matches_status"),
        ),
        sa.CheckConstraint(
            "status <> 'completed' OR "
            "(model_id IS NOT NULL AND finish_reason IS NOT NULL "
            "AND char_length(btrim(model_id)) > 0 "
            "AND char_length(btrim(finish_reason)) > 0)",
            name=op.f("ck_rag_runs_completed_model_result_present"),
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name=op.f("ck_rag_runs_input_tokens_nonnegative"),
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name=op.f("ck_rag_runs_output_tokens_nonnegative"),
        ),
        sa.CheckConstraint(
            "(status IN ('failed', 'cancelled') AND error_code IS NOT NULL "
            "AND char_length(btrim(error_code)) > 0) "
            "OR (status NOT IN ('failed', 'cancelled') AND error_code IS NULL)",
            name=op.f("ck_rag_runs_error_code_matches_status"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_rag_runs_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rag_runs")),
        sa.UniqueConstraint("request_id", name=op.f("uq_rag_runs_request_id")),
    )
    op.create_index(
        "ix_rag_runs_conversation_started_at",
        "rag_runs",
        ["conversation_id", "started_at"],
    )
    op.create_index(
        "ux_rag_runs_one_running_per_conversation",
        "rag_runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status = 'running'"),
    )

    op.create_table(
        "messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("rag_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "char_length(btrim(content)) > 0",
            name=op.f("ck_messages_content_not_blank"),
        ),
        sa.CheckConstraint("ordinal > 0", name=op.f("ck_messages_ordinal_positive")),
        sa.CheckConstraint(
            "role IN ('user', 'assistant')",
            name=op.f("ck_messages_role_allowed"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_messages_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["rag_run_id"],
            ["rag_runs.id"],
            name=op.f("fk_messages_rag_run_id_rag_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_messages")),
        sa.UniqueConstraint(
            "conversation_id",
            "ordinal",
            name="uq_messages_conversation_ordinal",
        ),
        sa.UniqueConstraint("rag_run_id", "role", name="uq_messages_rag_run_role"),
    )
    op.create_index(
        "ix_messages_conversation_ordinal",
        "messages",
        ["conversation_id", "ordinal"],
    )


def downgrade() -> None:
    """Remove M3-C state while retaining document and vector storage."""
    op.drop_index("ix_messages_conversation_ordinal", table_name="messages")
    op.drop_table("messages")
    op.drop_index(
        "ux_rag_runs_one_running_per_conversation",
        table_name="rag_runs",
        postgresql_where=sa.text("status = 'running'"),
    )
    op.drop_index("ix_rag_runs_conversation_started_at", table_name="rag_runs")
    op.drop_table("rag_runs")
    op.drop_table("conversations")
