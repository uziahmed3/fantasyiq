from sqlalchemy import Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class PlayerStats(Base, TimestampMixin):
    """One row per player per game week. The fact table."""

    __tablename__ = "player_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    opponent: Mapped[str | None] = mapped_column(String(8))
    is_home: Mapped[bool] = mapped_column(default=True, nullable=False)

    targets: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    receptions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    yards: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    touchdowns: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    snap_share: Mapped[float | None] = mapped_column(Float)
    fantasy_points: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    player: Mapped["Player"] = relationship(back_populates="stats")  # noqa: F821

    __table_args__ = (
        # Idempotent upserts from the pipeline + prevents duplicate ingests.
        UniqueConstraint("player_id", "season", "week", name="uq_player_stats_player_season_week"),
        # Hot path: "last N weeks for player X" -> index order matches the ORDER BY.
        Index("ix_player_stats_player_season_week", "player_id", "season", "week"),
        # Rankings/leaderboards scan a single week across all players.
        Index("ix_player_stats_season_week", "season", "week"),
    )
