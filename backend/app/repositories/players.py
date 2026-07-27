"""Data access layer.

Routers never write SQL and never touch the ORM directly; they call repositories.
That boundary is what makes the API testable without a database and lets query
tuning happen in one place.
"""

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models import Player, PlayerStats, Prediction


class PlayerRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, player_id: int) -> Player | None:
        return self.db.get(Player, player_id)

    def get_by_external_id(self, external_id: str) -> Player | None:
        return self.db.scalar(select(Player).where(Player.external_id == external_id))

    def search(
        self,
        position: str | None = None,
        team: str | None = None,
        name: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Player], int]:
        stmt: Select = select(Player)
        if position:
            stmt = stmt.where(Player.position == position.upper())
        if team:
            stmt = stmt.where(Player.team == team.upper())
        if name:
            stmt = stmt.where(Player.name.ilike(f"%{name}%"))

        total = self.db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = self.db.scalars(stmt.order_by(Player.name).limit(limit).offset(offset)).all()
        return list(rows), total


class StatsRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def for_player(
        self, player_id: int, season: int | None = None, limit: int = 20
    ) -> list[PlayerStats]:
        stmt = select(PlayerStats).where(PlayerStats.player_id == player_id)
        if season is not None:
            stmt = stmt.where(PlayerStats.season == season)
        # Index ix_player_stats_player_season_week serves this ordering directly.
        stmt = stmt.order_by(PlayerStats.season.desc(), PlayerStats.week.desc()).limit(limit)
        return list(self.db.scalars(stmt).all())

    def last_n_before_week(
        self, player_id: int, season: int, week: int, n: int = 3
    ) -> list[PlayerStats]:
        """Strictly prior weeks only - leaking the target week into features is the
        single most common way an ML project reports a fake RMSE."""
        stmt = (
            select(PlayerStats)
            .where(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < week,
            )
            .order_by(PlayerStats.week.desc())
            .limit(n)
        )
        return list(self.db.scalars(stmt).all())

    def season_aggregate(self, player_id: int, season: int, before_week: int) -> tuple[float, int]:
        row = self.db.execute(
            select(func.avg(PlayerStats.fantasy_points), func.count(PlayerStats.id)).where(
                PlayerStats.player_id == player_id,
                PlayerStats.season == season,
                PlayerStats.week < before_week,
            )
        ).one()
        return float(row[0] or 0.0), int(row[1] or 0)

    def opponent_defense_rank(self, season: int, opponent: str, before_week: int) -> int:
        """Cheap proxy for strength of schedule: rank of fantasy points allowed.

        Real version would precompute this in the feature pipeline; doing it here keeps
        the prediction path self-contained and it is fast enough behind the cache.
        """
        if not opponent or opponent.upper() == "UNK":
            return 16
        subq = (
            select(
                PlayerStats.opponent.label("opp"),
                func.avg(PlayerStats.fantasy_points).label("pts_allowed"),
            )
            .where(PlayerStats.season == season, PlayerStats.week < before_week)
            .group_by(PlayerStats.opponent)
            .subquery()
        )
        rows = (
            self.db.execute(select(subq.c.opp).order_by(subq.c.pts_allowed.desc())).scalars().all()
        )
        opp = opponent.upper()
        return rows.index(opp) + 1 if opp in rows else 16


class PredictionRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def record(self, **kwargs) -> Prediction:
        pred = Prediction(**kwargs)
        self.db.add(pred)
        self.db.commit()
        self.db.refresh(pred)
        return pred

    def latest_for_week(
        self,
        season: int,
        week: int,
        model_version: str,
        position: str | None = None,
        limit: int = 50,
    ) -> list[tuple[Prediction, Player]]:
        """Latest prediction per player for a week (dedupes repeated runs of the same model)."""
        newest = (
            select(
                Prediction.player_id,
                func.max(Prediction.created_at).label("newest"),
            )
            .where(
                Prediction.season == season,
                Prediction.week == week,
                Prediction.model_version == model_version,
            )
            .group_by(Prediction.player_id)
            .subquery()
        )
        stmt = (
            select(Prediction, Player)
            .join(Player, Player.id == Prediction.player_id)
            .join(
                newest,
                (newest.c.player_id == Prediction.player_id)
                & (newest.c.newest == Prediction.created_at),
            )
            .where(
                Prediction.season == season,
                Prediction.week == week,
                Prediction.model_version == model_version,
            )
        )
        if position:
            stmt = stmt.where(Player.position == position.upper())
        stmt = stmt.order_by(Prediction.prediction.desc()).limit(limit)
        return list(self.db.execute(stmt).all())

    def history(self, player_id: int, limit: int = 20) -> list[Prediction]:
        return list(
            self.db.scalars(
                select(Prediction)
                .where(Prediction.player_id == player_id)
                .order_by(Prediction.created_at.desc())
                .limit(limit)
            ).all()
        )
