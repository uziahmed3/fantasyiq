"""Seed the database with synthetic seasons. No network required.

Why this exists: the real ingest depends on reaching nflverse, which some networks block
outright. Rather than leaving someone with an empty dashboard and no way to see the
system work, this generates plausible seasons and pushes them through the *real* load.py
upsert path - same SQL, same idempotency, same schema. Only the origin of the numbers
differs, and every surface labels it clearly.

## The generative model, and why it is built this way

Players persist across seasons with a career arc, because the preseason model trains on
(season S-1 -> season S) pairs. An earlier version of this file regenerated every season
from the same seed, which made a player's talent *identical* year to year. The preseason
model then reported an RMSE of 0.66 and a 57% lift over carry-forward - which was not
skill, it was the model rediscovering a constant. Anyone who probed that number would
have found nothing behind it.

So talent now moves the way it actually does:

  * a persistent per-player base talent
  * an age curve - rising to a peak around 26, declining after
  * genuine year-to-year noise (injury, scheme, quarterback, luck)
  * roughly 12% of each position group turns over per season: rookies in, veterans out
  * depth-chart rank follows talent rank within team and position, so role is earned
    rather than assigned

The result is a dataset where last season predicts this season only *partially* - which
is the real problem, and gives an honest read on whether a model beats carry-forward.

    python -m seed_demo                          # 2022, 2023, 2024
    python -m seed_demo --seasons 2021,2022,2023,2024
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import load
from logging_setup import log

TEAMS = [
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LAC",
    "LAR",
    "LV",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SEA",
    "SF",
    "TB",
    "TEN",
    "WAS",
]

# (players per team, mean targets/game for a median starter at that position)
POSITION_PLAN = {"WR": (4, 7.5), "RB": (3, 4.0), "TE": (2, 4.5)}

FIRST = [
    "Marcus",
    "Jalen",
    "Tyler",
    "Devin",
    "Cameron",
    "Isaiah",
    "Brandon",
    "Xavier",
    "Trey",
    "Amari",
    "Deion",
    "Kyren",
    "Rashid",
    "Elijah",
    "Terrion",
    "Josiah",
    "Malik",
    "Dontae",
    "Kaden",
    "Jaylen",
    "Corey",
    "Zion",
    "Nico",
    "Braylon",
    "Emeka",
    "Rome",
    "Ladd",
    "Jaxon",
    "Keon",
    "Xavien",
    "Tank",
    "Quentin",
]
LAST = [
    "Whitfield",
    "Okafor",
    "Camarillo",
    "Boateng",
    "Lindqvist",
    "Vasquez",
    "Ferreira",
    "Nakamura",
    "Delacroix",
    "Adeyemi",
    "Kowalski",
    "Petrov",
    "Silva",
    "Aguilar",
    "Bennings",
    "Tavares",
    "Osei",
    "Rahimi",
    "Castellano",
    "Mbeki",
    "Sorensen",
    "Ibarra",
    "Duplessis",
    "Fontaine",
    "Achebe",
    "Bergstrom",
    "Calderon",
    "Dimitrov",
]

# Every synthetic player carries this prefix, which is what makes them separable from
# real ones after the fact - see --purge and the mixing guard in main().
DEMO_ID_PREFIX = "DEMO-"

TURNOVER_RATE = 0.12  # fraction of each position group replaced by rookies per season
PEAK_AGE = 26.0


@dataclass
class Career:
    """One player, persisting across seasons."""

    external_id: str
    name: str
    position: str
    team: str
    base_talent: float
    entry_season: int
    entry_age: int
    draft_round: int | None
    draft_pick: int | None
    exit_season: int | None = None
    _noise: dict[int, float] = field(default_factory=dict)

    def age_in(self, season: int) -> int:
        return self.entry_age + (season - self.entry_season)

    def active_in(self, season: int) -> bool:
        if season < self.entry_season:
            return False
        return self.exit_season is None or season <= self.exit_season

    def talent_in(self, season: int, rng: np.random.Generator) -> float:
        """Base talent bent by an age curve and a year-specific shock.

        The shock is cached per season so repeated calls agree, and is precisely why last
        season only partially predicts this one.
        """
        if season not in self._noise:
            self._noise[season] = float(rng.normal(0.0, 0.26))
        age = self.age_in(season)
        curve = max(0.35, 1.0 - 0.011 * (age - PEAK_AGE) ** 2 / 2.0)
        rookie_penalty = 0.78 if season == self.entry_season else 1.0
        return max(
            0.3, self.base_talent * curve * rookie_penalty * float(np.exp(self._noise[season]))
        )


def _draft_slot(talent: float, base_volume: float, rng: np.random.Generator):
    """Better players tend to go earlier - with heavy noise, because draft position is a
    genuinely noisy predictor and the model should have to cope with that."""
    percentile = float(np.clip(talent / (base_volume * 1.8), 0.0, 1.0))
    noisy = float(np.clip(percentile + rng.normal(0, 0.22), 0.0, 1.0))
    if noisy > 0.93:
        return 1, int(rng.integers(1, 33))
    if noisy > 0.80:
        return 2, int(rng.integers(33, 65))
    if noisy > 0.62:
        return 3, int(rng.integers(65, 105))
    if noisy > 0.42:
        return int(rng.integers(4, 6)), int(rng.integers(105, 180))
    if noisy > 0.20:
        return int(rng.integers(6, 8)), int(rng.integers(180, 260))
    return None, None  # undrafted


def build_universe(seasons: list[int], rng: np.random.Generator) -> list[Career]:
    """Create the league for the first season, then age it forward with turnover."""
    careers: list[Career] = []
    used_names: set[str] = set()
    counter = 0

    def new_career(
        position: str, team: str, base_volume: float, season: int, rookie: bool
    ) -> Career:
        nonlocal counter
        counter += 1
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in used_names:
                used_names.add(name)
                break
        talent = max(0.4, float(rng.normal(base_volume, base_volume * 0.42)))
        rnd, pick = _draft_slot(talent, base_volume, rng)
        return Career(
            external_id=f"{DEMO_ID_PREFIX}{counter:05d}",
            name=name,
            position=position,
            team=team,
            base_talent=talent,
            entry_season=season,
            entry_age=int(rng.integers(21, 23)) if rookie else int(rng.integers(22, 31)),
            draft_round=rnd,
            draft_pick=pick,
        )

    first = min(seasons)
    for team in TEAMS:
        for position, (count, base_volume) in POSITION_PLAN.items():
            for depth in range(count):
                # Depth 0 is the best at the position; talent scales down the chart.
                careers.append(
                    new_career(position, team, base_volume / (1.0 + 0.5 * depth), first, False)
                )

    for season in sorted(seasons)[1:]:
        for team in TEAMS:
            for position, (_count, base_volume) in POSITION_PLAN.items():
                group = [
                    c
                    for c in careers
                    if c.team == team and c.position == position and c.active_in(season - 1)
                ]
                if not group:
                    continue
                n_out = max(1, int(round(len(group) * TURNOVER_RATE)))
                if rng.random() < 0.65:  # not every group turns over every year
                    for leaver in sorted(group, key=lambda c: -c.age_in(season))[:n_out]:
                        leaver.exit_season = season - 1
                        careers.append(new_career(position, team, base_volume * 0.85, season, True))
    return careers


def generate(
    seasons: list[int], weeks: int = 17, seed: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly stats, roster rows) across every requested season."""
    rng = np.random.default_rng(seed)
    careers = build_universe(seasons, rng)
    defence = {t: float(rng.normal(1.0, 0.09)) for t in TEAMS}

    rows, roster_rows = [], []
    for season in sorted(seasons):
        active = [c for c in careers if c.active_in(season)]
        talents = {c.external_id: c.talent_in(season, rng) for c in active}

        # Depth chart is earned: rank by this season's talent within team and position.
        ranks: dict[str, int] = {}
        for team in TEAMS:
            for position in POSITION_PLAN:
                group = [c for c in active if c.team == team and c.position == position]
                for i, c in enumerate(sorted(group, key=lambda c: -talents[c.external_id])):
                    ranks[c.external_id] = i + 1

        for c in active:
            talent = talents[c.external_id]
            roster_rows.append(
                {
                    "player_id": c.external_id,
                    "name": c.name,
                    "position": c.position,
                    "team": c.team,
                    "season": season,
                    "age": c.age_in(season),
                    "height_inches": 68 + (abs(hash(c.external_id)) % 11),
                    "weight_lbs": 180 + (abs(hash(c.external_id)) % 70),
                    "depth_chart_rank": ranks[c.external_id],
                    "draft_round": c.draft_round,
                    "draft_pick": c.draft_pick,
                    "rookie_season": c.entry_season,
                }
            )

            depth_factor = 1.0 / (1.0 + 0.45 * (ranks[c.external_id] - 1))
            for week in range(1, weeks + 1):
                opponent = TEAMS[(TEAMS.index(c.team) + week * 3) % len(TEAMS)]
                if opponent == c.team:
                    opponent = TEAMS[(TEAMS.index(c.team) + 1) % len(TEAMS)]

                miss_rate = 0.05 + (0.03 if c.age_in(season) > 30 else 0.0)
                if rng.random() < miss_rate:
                    targets = receptions = touchdowns = 0
                    yards = 0.0
                else:
                    volume = talent * depth_factor * defence[opponent]
                    targets = max(0, int(rng.normal(volume, max(0.8, volume * 0.4))))
                    catch_rate = float(np.clip(rng.normal(0.64, 0.10), 0.15, 0.95))
                    receptions = int(round(targets * catch_rate))
                    mean_ypr = 11.5 if c.position != "RB" else 7.5
                    ypr = float(np.clip(rng.normal(mean_ypr, 3.0), 1.0, 30.0))
                    rush = max(0.0, rng.normal(30, 20)) if c.position == "RB" else 0.0
                    yards = max(0.0, round(receptions * ypr + rush, 1))
                    touchdowns = int(rng.poisson(float(np.clip(volume / 22.0, 0.01, 1.1))))

                # PPR, matching pipeline/config.py SCORING.
                points = round(receptions * 1.0 + yards * 0.1 + touchdowns * 6.0, 2)
                rows.append(
                    {
                        "player_id": c.external_id,
                        "name": c.name,
                        "position": c.position,
                        "team": c.team,
                        "season": season,
                        "week": week,
                        "opponent": opponent,
                        "is_home": week % 2 == 0,
                        "targets": targets,
                        "receptions": receptions,
                        "yards": yards,
                        "touchdowns": touchdowns,
                        "fantasy_points": points,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(roster_rows)


def _census(engine) -> tuple[int, int]:
    """(real players, synthetic players) currently in the database."""
    from sqlalchemy import text

    with engine.connect() as conn:
        try:
            real = conn.execute(
                text("SELECT COUNT(*) FROM players WHERE external_id NOT LIKE :p"),
                {"p": f"{DEMO_ID_PREFIX}%"},
            ).scalar_one()
            fake = conn.execute(
                text("SELECT COUNT(*) FROM players WHERE external_id LIKE :p"),
                {"p": f"{DEMO_ID_PREFIX}%"},
            ).scalar_one()
        except Exception:  # noqa: BLE001 - table may not exist yet
            return 0, 0
    return int(real), int(fake)


def purge(engine) -> int:
    """Remove every synthetic player and everything hanging off them.

    Exists because synthetic and real data got mixed once: a demo seed ran against a
    database that already held real seasons, and the leaderboard then showed invented
    names next to real ones. Cascades handle stats, context and predictions.
    """
    from sqlalchemy import text

    with engine.begin() as conn:
        ids = [
            r[0]
            for r in conn.execute(
                text("SELECT id FROM players WHERE external_id LIKE :p"),
                {"p": f"{DEMO_ID_PREFIX}%"},
            )
        ]
        if not ids:
            return 0
        # Delete children explicitly: SQLite does not enforce ON DELETE CASCADE unless
        # foreign_keys pragma is on, and relying on that here would be fragile.
        for table in ("player_stats", "player_context", "predictions"):
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                conn.execute(
                    text(
                        f"DELETE FROM {table} WHERE player_id IN "  # noqa: S608 - ints only
                        "(" + ",".join(str(int(x)) for x in chunk) + ")"
                    )
                )
        conn.execute(
            text("DELETE FROM players WHERE external_id LIKE :p"),
            {"p": f"{DEMO_ID_PREFIX}%"},
        )
    log.info("demo_purged", players=len(ids))
    return len(ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed synthetic seasons (no network)")
    parser.add_argument(
        "--seasons",
        default="2022,2023,2024",
        help="Comma-separated. The preseason model trains on (prior -> next) pairs, so "
        "three seasons gives two pairs: one to train on, one to hold out.",
    )
    parser.add_argument("--weeks", type=int, default=17)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Delete all synthetic players and exit - use this to clean real data that "
        "got mixed with demo data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Seed even though the database already contains real players",
    )
    args = parser.parse_args(argv)

    engine_for_checks = load.get_engine()

    if args.purge:
        removed = purge(engine_for_checks)
        print(
            f"\nRemoved {removed} synthetic players and their stats, context and " f"predictions.\n"
        )
        return 0

    real, fake = _census(engine_for_checks)
    if real and not args.force:
        print(
            f"\nRefusing to seed: this database already holds {real} real players.\n\n"
            "Mixing synthetic and real data puts invented names on your leaderboard next\n"
            "to real ones, and quietly corrupts model training. Pick one:\n\n"
            "  python local.py --reset            start clean, then seed demo data\n"
            "  python -m seed_demo --purge        remove synthetic rows, keep real data\n"
            "  python -m seed_demo --force        seed anyway (you know what you are doing)\n"
        )
        return 1
    if fake:
        log.info("reseeding_over_existing_demo_data", synthetic_players=fake)

    seasons = sorted(int(s) for s in args.seasons.split(",") if s.strip())
    weekly, rosters = generate(seasons, args.weeks, args.seed)

    engine = load.get_engine()
    # Latest roster row per player wins, matching how the real loader treats identity.
    latest = rosters.sort_values("season").drop_duplicates("player_id", keep="last")
    id_map = load.upsert_players(engine, weekly, latest)
    stat_rows = load.upsert_stats(engine, weekly, id_map)

    rookies = (
        rosters[rosters["rookie_season"] == rosters["season"]].groupby("season").size().to_dict()
    )
    log.info(
        "demo_seed_complete",
        seasons=seasons,
        players=len(id_map),
        stat_rows=stat_rows,
        rookies_per_season=rookies,
        note="SYNTHETIC DATA - not real NFL results",
    )
    print(
        f"\nSeeded {len(id_map)} players and {stat_rows} weekly stat lines "
        f"across synthetic seasons {seasons}."
    )
    print(f"Rookies per season: {rookies}")
    print(
        "These are NOT real NFL numbers. For real data, from the repository root:\n"
        "  python local.py --seasons 2021,2022,2023,2024,2025\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
