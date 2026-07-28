"""carries: the volume stat running backs are actually judged on

Revision ID: 0004
Revises: 0003

`carries` was in every weekly parquet we already download and was being dropped. For a
running back it is the whole story - target share is close to irrelevant, and the model
was ranking RBs off receiving volume plus rushing *yards*, with no usage signal at all.
Measured on 2021-2025, prior-season volume correlates ~0.83 with the next season, so
omitting it for an entire position was the largest gap in the feature set.
"""

import sqlalchemy as sa

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "player_stats",
        sa.Column("carries", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "player_context",
        sa.Column("prior_carries_per_game", sa.Float(), nullable=False, server_default="0"),
    )
    # Share of the team's carries: separates a genuine lead back from someone on a team
    # that simply runs a lot, the same way target share does for receivers.
    op.add_column(
        "player_context",
        sa.Column("prior_carry_share", sa.Float(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("player_context", "prior_carry_share")
    op.drop_column("player_context", "prior_carries_per_game")
    op.drop_column("player_stats", "carries")
