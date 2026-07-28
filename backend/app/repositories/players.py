"""Data access layer.

Routers never write SQL and never touch the ORM directly; they call repositories.
That boundary is what makes the API testable without a database and lets query
tuning happen in one place.
"""

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models import Player, PlayerContext, PlayerStats, Prediction

# Week 0 means "the season as a whole" - see pipeline/preseason.py.
SEASON_PROJECTION_WEEK = 0

# The regular season. Playoff weeks exist in the data but are not what fantasy is scored
# on, and counting them produced impossible totals like "19 games last season".
REGULAR_SEASON_WEEKS = 18

# FLEX is any of these. Quarterbacks are deliberately absent: points here are computed
# from receiving and rushing only, so a QB total would omit passing and mislead.
FLEX_POSITIONS = ("WR", "RB", "TE")


def position_filter(position: str) -> tuple[str, ...]:
    """Resolve a position selector to the set of positions it covers."""
    upper = position.upper()
    return FLEX_POSITIONS if upper == "FLEX" else (upper,)


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

    def season_has_games(self, season: int) -> bool:
        """Has this season actually started? Drives the UI's weekly/season-long switch."""
        return bool(
            self.db.scalar(select(func.count(PlayerStats.id)).where(PlayerStats.season == season))
        )

    def games_before_week(self, player_id: int, season: int, week: int) -> int:
        """How many games this player has already played this season.

        This is the routing signal: zero means the in-season features would all be zero,
        so the preseason model should answer instead.
        """
        return int(
            self.db.scalar(
                select(func.count(PlayerStats.id)).where(
                    PlayerStats.player_id == player_id,
                    PlayerStats.season == season,
                    PlayerStats.week < week,
                )
            )
            or 0
        )

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


class LeaderRepository:
    """Completed-season actuals. No model, no projection - just what happened.

    Regular season only, and aggregated in SQL so a full-league leaderboard is one query
    rather than pulling every game row into Python.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def _base(self, season: int, position: str):
        return (
            select(
                Player.id.label("player_id"),
                Player.name,
                Player.team,
                Player.position,
                func.count(PlayerStats.id).label("games"),
                func.sum(PlayerStats.fantasy_points).label("total_points"),
                func.avg(PlayerStats.fantasy_points).label("points_per_game"),
                func.sum(PlayerStats.targets).label("targets"),
                func.sum(PlayerStats.receptions).label("receptions"),
                func.sum(PlayerStats.yards).label("yards"),
                func.sum(PlayerStats.touchdowns).label("touchdowns"),
            )
            .join(PlayerStats, PlayerStats.player_id == Player.id)
            .where(
                PlayerStats.season == season,
                PlayerStats.week <= REGULAR_SEASON_WEEKS,
                Player.position.in_(position_filter(position)),
            )
            .group_by(Player.id, Player.name, Player.team, Player.position)
        )

    def count(self, season: int, position: str) -> int:
        subquery = self._base(season, position).subquery()
        return int(self.db.scalar(select(func.count()).select_from(subquery)) or 0)

    def leaders(self, season: int, position: str, limit: int = 20, offset: int = 0) -> list:
        stmt = (
            self._base(season, position)
            .order_by(func.sum(PlayerStats.fantasy_points).desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.execute(stmt).all())

    def latest_completed_season(self) -> int | None:
        return self.db.scalar(select(func.max(PlayerStats.season)))


class ContextRepository:
    """Preseason context: what was knowable before the season started."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get(self, player_id: int, season: int) -> PlayerContext | None:
        return self.db.scalar(
            select(PlayerContext).where(
                PlayerContext.player_id == player_id, PlayerContext.season == season
            )
        )

    def latest_at_or_before(self, player_id: int, season: int) -> PlayerContext | None:
        """Fall back to the most recent earlier context if this season has none built yet.

        Stale context still beats no projection at all - and the response reports which
        season the context came from, so a caller can see it is not current.
        """
        return self.db.scalar(
            select(PlayerContext)
            .where(PlayerContext.player_id == player_id, PlayerContext.season <= season)
            .order_by(PlayerContext.season.desc())
            .limit(1)
        )


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

    def next_unplayed_season(self) -> int:
        """The season the draft board is for: the one after the latest with games."""
        latest = self.db.scalar(select(func.max(PlayerStats.season)))
        return int(latest) + 1 if latest is not None else 0

    def season_board_count(self, season: int, model_version: str, position: str) -> int:
        return int(
            self.db.scalar(
                select(func.count(Prediction.id))
                .join(Player, Player.id == Prediction.player_id)
                .where(
                    Prediction.season == season,
                    Prediction.week == SEASON_PROJECTION_WEEK,
                    Prediction.model_version == model_version,
                    Player.position.in_(position_filter(position)),
                )
            )
            or 0
        )

    def season_board(
        self,
        season: int,
        model_version: str,
        position: str = "FLEX",
        limit: int = 20,
        offset: int = 0,
    ) -> list[tuple[Prediction, Player, PlayerContext | None]]:
        """The draft board: season-long projections, stored at the week=0 sentinel.

        Joined to player_context so the board can show whether a number rests on a real
        prior season or only on where the player was drafted.
        """
        stmt = (
            select(Prediction, Player, PlayerContext)
            .join(Player, Player.id == Prediction.player_id)
            .outerjoin(
                PlayerContext,
                (PlayerContext.player_id == Prediction.player_id)
                & (PlayerContext.season == Prediction.season),
            )
            .where(
                Prediction.season == season,
                Prediction.week == SEASON_PROJECTION_WEEK,
                Prediction.model_version == model_version,
                Player.position.in_(position_filter(position)),
            )
            .order_by(Prediction.prediction.desc())
            .limit(limit)
            .offset(offset)
        )
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
