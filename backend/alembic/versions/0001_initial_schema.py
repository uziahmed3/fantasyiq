"""initial schema: players, player_stats, predictions, users

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("external_id", sa.String(32), nullable=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("team", sa.String(8), nullable=True),
        sa.Column("position", sa.String(4), nullable=False),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("height_inches", sa.Integer(), nullable=True),
        sa.Column("weight_lbs", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_players_external_id", "players", ["external_id"], unique=True)
    op.create_index("ix_players_position_team", "players", ["position", "team"])
    op.create_index("ix_players_name", "players", ["name"])

    op.create_table(
        "player_stats",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "player_id",
            sa.Integer(),
            sa.ForeignKey("players.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("opponent", sa.String(8), nullable=True),
        sa.Column("is_home", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("targets", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("receptions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("yards", sa.Float(), nullable=False, server_default="0"),
        sa.Column("touchdowns", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("snap_share", sa.Float(), nullable=True),
        sa.Column("fantasy_points", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint(
            "player_id", "season", "week", name="uq_player_stats_player_season_week"
        ),
    )
    op.create_index(
        "ix_player_stats_player_season_week", "player_stats", ["player_id", "season", "week"]
    )
    op.create_index("ix_player_stats_season_week", "player_stats", ["season", "week"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "player_id",
            sa.Integer(),
            sa.ForeignKey("players.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("season", sa.Integer(), nullable=False),
        sa.Column("week", sa.Integer(), nullable=False),
        sa.Column("opponent", sa.String(8), nullable=True),
        sa.Column("prediction", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(48), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_predictions_lookup", "predictions", ["player_id", "season", "week", "model_version"]
    )
    op.create_index("ix_predictions_week_model", "predictions", ["season", "week", "model_version"])

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("predictions")
    op.drop_table("player_stats")
    op.drop_table("players")
