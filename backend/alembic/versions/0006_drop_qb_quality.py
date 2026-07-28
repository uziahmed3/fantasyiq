"""Drop qb_quality.

The feature rated each team's quarterback by his own fantasy points. That only works if
quarterback fantasy points mean something, and in this project they do not: ingestion
covers receiving and rushing, so a quarterback's "points" were essentially his scrambles.
Minnesota scored 2.2 and Cincinnati 0.9 - the opposite of any reasonable reading, and a
feature the model was free to split on.

Rating quarterbacks honestly needs passing yards and touchdowns ingested. Until that
exists, team_pass_attempts_prior carries the team-volume signal and qb_changed flags the
risk of a new starter, both of which are computed from data actually present. A missing
feature is better than a confidently wrong one.

Reversible: the downgrade restores the column with its old default. It will be full of
zeros, because the data it was derived from was never meaningful - so the pipeline would
have to be rerun to repopulate it, and there is no reason to.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

TABLE = "player_context"
COLUMN = "qb_quality"


def _has_column(table: str, column: str) -> bool:
    """SQLite and Postgres both reach this migration; neither supports IF EXISTS here.

    The local no-Docker path runs on a database that may have been created before 0005
    ever added the column, so dropping it unconditionally would fail on a fresh install.
    """
    inspector = sa.inspect(op.get_bind())
    return column in {c["name"] for c in inspector.get_columns(table)}


def upgrade() -> None:
    if not _has_column(TABLE, COLUMN):
        return
    # batch_alter_table because SQLite cannot DROP COLUMN before 3.35 and Alembic's batch
    # mode handles the table-rebuild dance; on Postgres it compiles to a plain ALTER.
    with op.batch_alter_table(TABLE) as batch:
        batch.drop_column(COLUMN)


def downgrade() -> None:
    if _has_column(TABLE, COLUMN):
        return
    with op.batch_alter_table(TABLE) as batch:
        batch.add_column(
            sa.Column(COLUMN, sa.Float(), nullable=False, server_default="0"),
        )
