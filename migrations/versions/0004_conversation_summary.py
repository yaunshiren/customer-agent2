"""Create durable conversation summaries.

Revision ID: 0004_conversation_summary
Revises: 0003_conversation_rag_run
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_conversation_summary"
down_revision: str | Sequence[str] | None = "0003_conversation_rag_run"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the one-row-per-conversation M4-A summary table."""
    op.create_table(
        "conversation_summaries",
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("summarized_through_ordinal", sa.Integer(), nullable=False),
        sa.Column("source_message_count", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model_id", sa.String(length=255), nullable=False),
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
            "summarized_through_ordinal > 0",
            name=op.f("ck_conversation_summaries_summarized_through_ordinal_positive"),
        ),
        sa.CheckConstraint(
            "source_message_count > 0 AND source_message_count % 2 = 0",
            name=op.f("ck_conversation_summaries_source_message_count_complete_turns"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(content)) > 0",
            name=op.f("ck_conversation_summaries_content_not_blank"),
        ),
        sa.CheckConstraint(
            "char_length(btrim(model_id)) > 0",
            name=op.f("ck_conversation_summaries_model_id_not_blank"),
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_conversation_summaries_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "conversation_id",
            name=op.f("pk_conversation_summaries"),
        ),
    )


def downgrade() -> None:
    """Remove summaries while retaining conversations, messages, and RAG Runs."""
    op.drop_table("conversation_summaries")
