"""Create the pgvector extension infrastructure baseline.

Revision ID: 0001_infrastructure_baseline
Revises:
Create Date: 2026-08-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001_infrastructure_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Ensure pgvector is installed before later vector tables are introduced."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Keep the shared extension installed when reverting this empty baseline."""
