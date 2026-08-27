"""Add intent route and clarification RAG Run terminal state.

Revision ID: 0005_intent_routing
Revises: 0004_conversation_summary
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_intent_routing"
down_revision: str | Sequence[str] | None = "0004_conversation_summary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Persist M4-C route decisions and a distinct clarification terminal state."""
    op.add_column("rag_runs", sa.Column("intent_route", sa.String(length=32), nullable=True))
    op.execute(
        "UPDATE rag_runs SET intent_route = 'knowledge_base' "
        "WHERE status IN ('completed', 'no_context')"
    )
    op.drop_constraint(op.f("ck_rag_runs_status_allowed"), "rag_runs", type_="check")
    op.drop_constraint(
        op.f("ck_rag_runs_completed_model_result_present"),
        "rag_runs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_rag_runs_status_allowed"),
        "rag_runs",
        "status IN ('running', 'completed', 'no_context', 'clarification', 'failed', 'cancelled')",
    )
    op.create_check_constraint(
        op.f("ck_rag_runs_completed_model_result_present"),
        "rag_runs",
        "status NOT IN ('completed', 'clarification') OR "
        "(model_id IS NOT NULL AND finish_reason IS NOT NULL "
        "AND char_length(btrim(model_id)) > 0 "
        "AND char_length(btrim(finish_reason)) > 0)",
    )
    op.create_check_constraint(
        op.f("ck_rag_runs_intent_route_matches_status"),
        "rag_runs",
        "(status = 'completed' AND intent_route IN ('system_direct', 'knowledge_base')) "
        "OR (status = 'no_context' AND intent_route = 'knowledge_base') "
        "OR (status = 'clarification' AND intent_route = 'clarification') "
        "OR (status IN ('running', 'failed', 'cancelled') AND intent_route IS NULL)",
    )


def downgrade() -> None:
    """Map clarification to completed before restoring the pre-M4-C constraints."""
    op.drop_constraint(
        op.f("ck_rag_runs_intent_route_matches_status"),
        "rag_runs",
        type_="check",
    )
    op.execute("UPDATE rag_runs SET status = 'completed' WHERE status = 'clarification'")
    op.drop_constraint(op.f("ck_rag_runs_status_allowed"), "rag_runs", type_="check")
    op.drop_constraint(
        op.f("ck_rag_runs_completed_model_result_present"),
        "rag_runs",
        type_="check",
    )
    op.create_check_constraint(
        op.f("ck_rag_runs_status_allowed"),
        "rag_runs",
        "status IN ('running', 'completed', 'no_context', 'failed', 'cancelled')",
    )
    op.create_check_constraint(
        op.f("ck_rag_runs_completed_model_result_present"),
        "rag_runs",
        "status <> 'completed' OR "
        "(model_id IS NOT NULL AND finish_reason IS NOT NULL "
        "AND char_length(btrim(model_id)) > 0 "
        "AND char_length(btrim(finish_reason)) > 0)",
    )
    op.drop_column("rag_runs", "intent_route")
