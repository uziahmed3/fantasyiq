from sqlalchemy import Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Player(Base, TimestampMixin):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Stable external id from the NFL data source - lets the pipeline upsert idempotently.
    external_id: Mapped[str | None] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    team: Mapped[str | None] = mapped_column(String(8))
    position: Mapped[str] = mapped_column(String(4), nullable=False)
    age: Mapped[int | None] = mapped_column(Integer)
    height_inches: Mapped[int | None] = mapped_column(Integer)
    weight_lbs: Mapped[int | None] = mapped_column(Integer)

    stats: Mapped[list["PlayerStats"]] = relationship(  # noqa: F821
        back_populates="player", cascade="all, delete-orphan", lazy="selectin"
    )
    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        back_populates="player", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Rankings filter by position and sort/paginate - composite index covers it.
        Index("ix_players_position_team", "position", "team"),
        Index("ix_players_name", "name"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Player {self.id} {self.name} {self.position}>"
