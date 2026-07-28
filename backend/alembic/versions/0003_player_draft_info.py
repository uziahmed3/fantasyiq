"""players: draft capital and rookie season

Revision ID: 0003
Revises: 0002

Draft round/pick and rookie season belong on the player, not only in an external feed:
they never change, they are the single most important input for projecting a rookie, and
keeping them in the database means the preseason model works without a live dependency
on the bios download.
"""

import sqlalchemy as sa

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("players", sa.Column("draft_round", sa.Integer(), nullable=True))
    op.add_column("players", sa.Column("draft_pick", sa.Integer(), nullable=True))
    op.add_column("players", sa.Column("rookie_season", sa.Integer(), nullable=True))
    # "every rookie entering season X" is a query the preseason pipeline runs directly.
    op.create_index("ix_players_rookie_season", "players", ["rookie_season"])


def downgrade() -> None:
    op.drop_index("ix_players_rookie_season", table_name="players")
    op.drop_column("players", "rookie_season")
    op.drop_column("players", "draft_pick")
    op.drop_column("players", "draft_round")
