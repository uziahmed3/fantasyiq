"""player_context: everything needed to project a player with no games this season

Revision ID: 0002
Revises: 0001

One row per (player, season) describing what we knew *before* that season started:
prior-season production, role (depth chart, snap share, target share), draft capital,
and the team's passing situation. This is what makes a week-1 projection possible at
all, and the only thing a rookie has.
"""

import sqlalchemy as sa

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "player_context",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "player_id",
            sa.Integer(),
            sa.ForeignKey("players.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # The season this context is used to project (features come from season - 1).
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("team", sa.String(8), nullable=True),
        # ---- carryover production from the prior season ----
        sa.Column("prior_games", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("prior_points_per_game", sa.Float(), nullable=False, server_default="0"),
        sa.Column("prior_targets_per_game", sa.Float(), nullable=False, server_default="0"),
        sa.Column("prior_yards_per_game", sa.Float(), nullable=False, server_default="0"),
        # Share of the team's targets - separates "good player" from "player on a team
        # that throws a lot", which raw target counts conflate.
        sa.Column("prior_target_share", sa.Float(), nullable=False, server_default="0"),
        # Late-season form: the last 4 games carry more signal than a full-season average
        # for a player whose role changed mid-year.
        sa.Column("prior_last4_points_per_game", sa.Float(), nullable=False, server_default="0"),
        # ---- playing time ----
        sa.Column("prior_snap_share", sa.Float(), nullable=True),
        # ---- role going into the season ----
        sa.Column("depth_chart_rank", sa.Integer(), nullable=True),
        # ---- draft capital / experience (the only signal a rookie has) ----
        sa.Column("draft_round", sa.Integer(), nullable=True),
        sa.Column("draft_pick", sa.Integer(), nullable=True),
        sa.Column("rookie_season", sa.Integer(), nullable=True),
        sa.Column("years_experience", sa.Integer(), nullable=True),
        sa.Column("is_rookie", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("age", sa.Float(), nullable=True),
        # ---- team / QB situation ----
        sa.Column("team_pass_attempts_prior", sa.Float(), nullable=False, server_default="0"),
        sa.Column("team_points_prior", sa.Float(), nullable=False, server_default="0"),
        # A new starting QB is one of the biggest swing factors for a receiver, and it is
        # knowable before week 1.
        sa.Column("qb_changed", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        # Natural key, so the pipeline can upsert idempotently like every other table.
        sa.UniqueConstraint("player_id", "season", name="uq_player_context_player_season"),
    )
    op.create_index("ix_player_context_season", "player_context", ["season"])
    op.create_index("ix_player_context_player_season", "player_context", ["player_id", "season"])
    # Preseason training scans "every rookie in season X" and "every player on team Y".
    op.create_index("ix_player_context_season_rookie", "player_context", ["season", "is_rookie"])


def downgrade() -> None:
    op.drop_table("player_context")
