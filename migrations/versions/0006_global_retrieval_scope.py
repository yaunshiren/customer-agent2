"""Allow Ragent-style global retrieval runs.

Revision ID: 0006_global_retrieval_scope
Revises: 0005_intent_routing
Create Date: 2026-09-01
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0006_global_retrieval_scope"
down_revision: str | None = "0005_intent_routing"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Use an empty UUID array to record a global retrieval request."""
    op.drop_constraint(
        op.f("ck_rag_runs_knowledge_base_ids_not_empty"),
        "rag_runs",
        type_="check",
    )


def downgrade() -> None:
    """Restore the former caller-scoped non-empty requirement."""
    op.create_check_constraint(
        op.f("ck_rag_runs_knowledge_base_ids_not_empty"),
        "rag_runs",
        "cardinality(knowledge_base_ids) > 0",
    )
