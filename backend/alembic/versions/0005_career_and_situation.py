"""career history, volume/efficiency split, and situation features

Revision ID: 0005
Revises: 0004

The model saw exactly one prior season, so one down year erased a career. Justin
Jefferson averaged 19.5, 21.5, 20.4, 18.6 and then 11.9 - and projected 11.91, copying
the outlier verbatim. His target share was still 0.315, the highest on his team, on 141
targets: the role never collapsed, only the efficiency did.

That diagnosis drives the whole design here:

  * career_* columns weight every past season by recency (heavy on the last three),
    by games played (a 10-game season is a noisier estimate of a rate), and translate
    each one to next season's age via the age curve.
  * volume is taken from last season, which is genuinely stable year to year
    (correlation ~0.83 on 2021-2025), while efficiency - points per target, the noisiest
    thing in football - is taken from the career weighted mean instead.
  * situation columns describe what changed around the player: how good the quarterback
    is (not merely whether he changed), how much target share left the roster, and who
    is still there competing for it.
"""

import sqlalchemy as sa

from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

# (column, type, server_default)
NEW_COLUMNS = [
    # ---- career history, age-adjusted and injury-weighted ----
    ("career_weighted_ppg", sa.Float, "0"),
    ("career_weighted_targets_per_game", sa.Float, "0"),
    ("career_weighted_carries_per_game", sa.Float, "0"),
    ("career_weighted_target_share", sa.Float, "0"),
    ("career_best_ppg", sa.Float, "0"),
    ("career_seasons", sa.Integer, "0"),
    ("career_games", sa.Integer, "0"),
    # ---- volume vs efficiency ----
    # Points per target last season, and across the career. The gap between them is the
    # bounce-back signal: a big negative gap means the efficiency, not the role, broke.
    ("prior_points_per_target", sa.Float, "0"),
    ("career_points_per_target", sa.Float, "0"),
    ("efficiency_delta", sa.Float, "0"),
    # ---- situation ----
    # The quarterback's quality, not just whether he changed.
    ("qb_quality", sa.Float, "0"),
    # Target/carry share that belonged to players who are no longer on the roster - the
    # "the WR1 left, someone inherits those looks" signal.
    ("team_departed_target_share", sa.Float, "0"),
    ("team_departed_carry_share", sa.Float, "0"),
    # The strongest remaining competitor at the same position on the same team.
    ("teammate_top_target_share", sa.Float, "0"),
    ("teammate_top_carry_share", sa.Float, "0"),
]


def upgrade() -> None:
    for name, coltype, default in NEW_COLUMNS:
        op.add_column(
            "player_context",
            sa.Column(name, coltype(), nullable=False, server_default=default),
        )


def downgrade() -> None:
    for name, _coltype, _default in reversed(NEW_COLUMNS):
        op.drop_column("player_context", name)
