from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class PlayerContext(Base, TimestampMixin):
    """What was knowable about a player *before* a season started.

    The in-season model works off rolling 3-game form, which does not exist in week 1 and
    never exists for a rookie. This table is the other half of the problem: prior-season
    production, role, draft capital and team situation - all of it available before a
    single game is played.
    """

    __tablename__ = "player_context"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )
    # The season being projected; features are derived from season - 1.
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    team: Mapped[str | None] = mapped_column(String(8))

    # ---- carryover production ----
    prior_games: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    prior_points_per_game: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_targets_per_game: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_yards_per_game: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_target_share: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_carries_per_game: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_carry_share: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    prior_last4_points_per_game: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # ---- playing time ----
    prior_snap_share: Mapped[float | None] = mapped_column(Float)

    # ---- role ----
    depth_chart_rank: Mapped[int | None] = mapped_column(Integer)

    # ---- draft capital / experience ----
    draft_round: Mapped[int | None] = mapped_column(Integer)
    draft_pick: Mapped[int | None] = mapped_column(Integer)
    rookie_season: Mapped[int | None] = mapped_column(Integer)
    years_experience: Mapped[int | None] = mapped_column(Integer)
    is_rookie: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    age: Mapped[float | None] = mapped_column(Float)

    # ---- team situation ----
    team_pass_attempts_prior: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    team_points_prior: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    qb_changed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    player: Mapped["Player"] = relationship(back_populates="contexts")  # noqa: F821

    __table_args__ = (
        UniqueConstraint("player_id", "season", name="uq_player_context_player_season"),
        Index("ix_player_context_season", "season"),
        Index("ix_player_context_player_season", "player_id", "season"),
        Index("ix_player_context_season_rookie", "season", "is_rookie"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<PlayerContext player={self.player_id} season={self.season}>"
