from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Prediction(Base):
    """Append-only prediction log. Never updated - a new model version writes a new row,
    which is what makes model performance auditable after the fact."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int] = mapped_column(
        ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[int] = mapped_column(Integer, nullable=False)
    week: Mapped[int] = mapped_column(Integer, nullable=False)
    opponent: Mapped[str | None] = mapped_column(String(8))
    prediction: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(48), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    player: Mapped["Player"] = relationship(back_populates="predictions")  # noqa: F821

    __table_args__ = (
        Index(
            "ix_predictions_lookup",
            "player_id",
            "season",
            "week",
            "model_version",
        ),
        Index("ix_predictions_week_model", "season", "week", "model_version"),
    )
