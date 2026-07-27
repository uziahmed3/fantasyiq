"""Seed the database with a synthetic season. No network required.

Why this exists: the real ingest depends on reaching nflverse, which some networks
block outright. Rather than leaving someone with an empty dashboard and no way to see
the system work, this generates a plausible season and pushes it through the *real*
`load.py` upsert path — same SQL, same idempotency, same schema. Only the origin of
the numbers differs, and every surface labels it clearly.

The generative model: each player has a latent talent level; weekly target volume is
talent plus noise; catch rate and yards-per-reception are drawn per game; points follow
from the same PPR rules the real cleaner applies, scaled by matchup. Enough signal that
a model beats the naive baseline, enough noise that it cannot beat it by much — which
is also true of real fantasy football.

    python -m seed_demo               # 2024, 16 teams x 9 skill players, 17 weeks
    python -m seed_demo --season 2025 --weeks 12
"""

from __future__ import annotations

import argparse
import sys

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

# Roughly realistic squad shape and per-position usage.
POSITION_PLAN = {
    "WR": (4, 7.5),  # (players per team, mean targets/game for a median player)
    "RB": (3, 4.0),
    "TE": (2, 4.5),
}

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
]


def generate(
    season: int = 2024, weeks: int = 17, seed: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # Each defence has a multiplier: some are genuinely easier to score on.
    defence_strength = {t: float(rng.normal(1.0, 0.09)) for t in TEAMS}

    players, rows = [], []
    used_names: set[str] = set()
    pid = 0

    for team in TEAMS:
        for position, (count, base_volume) in POSITION_PLAN.items():
            # Depth chart: WR1 sees far more volume than WR4.
            for depth in range(count):
                pid += 1
                while True:
                    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                    if name not in used_names:
                        used_names.add(name)
                        break
                external_id = f"DEMO-{pid:05d}"
                depth_factor = 1.0 / (1.0 + 0.55 * depth)
                talent = max(0.4, float(rng.normal(base_volume * depth_factor, base_volume * 0.18)))

                players.append(
                    {
                        "player_id": external_id,
                        "name": name,
                        "position": position,
                        "team": team,
                        "age": int(rng.integers(22, 34)),
                        "height_inches": int(rng.integers(68, 79)),
                        "weight_lbs": int(rng.integers(180, 250)),
                    }
                )

                for week in range(1, weeks + 1):
                    opponent = TEAMS[(TEAMS.index(team) + week * 3) % len(TEAMS)]
                    if opponent == team:
                        opponent = TEAMS[(TEAMS.index(team) + 1) % len(TEAMS)]

                    # ~6% chance of missing a game; a zero line is realistic and
                    # exercises the cold-start branch of the feature builder.
                    if rng.random() < 0.06:
                        targets = receptions = touchdowns = 0
                        yards = 0.0
                    else:
                        matchup = defence_strength[opponent]
                        targets = max(0, int(rng.normal(talent * matchup, talent * 0.35)))
                        catch_rate = float(np.clip(rng.normal(0.64, 0.10), 0.15, 0.95))
                        receptions = int(round(targets * catch_rate))
                        mean_ypr = 11.5 if position != "RB" else 7.5
                        ypr = float(np.clip(rng.normal(mean_ypr, 3.0), 1.0, 30.0))
                        # RBs get rushing yardage on top of receiving.
                        rush = rng.normal(28, 18) if position == "RB" else 0.0
                        yards = round(receptions * ypr + rush, 1)
                        yards = max(0.0, yards)
                        touchdowns = int(rng.poisson(np.clip(talent / 22.0, 0.01, 1.1)))

                    # PPR, matching pipeline/config.py SCORING.
                    points = round(receptions * 1.0 + yards * 0.1 + touchdowns * 6.0, 2)

                    rows.append(
                        {
                            "player_id": external_id,
                            "name": name,
                            "position": position,
                            "team": team,
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

    return pd.DataFrame(rows), pd.DataFrame(players)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a synthetic season (no network)")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--weeks", type=int, default=17)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args(argv)

    weekly, rosters = generate(args.season, args.weeks, args.seed)
    engine = load.get_engine()

    id_map = load.upsert_players(engine, weekly, rosters)
    stat_rows = load.upsert_stats(engine, weekly, id_map)

    log.info(
        "demo_seed_complete",
        season=args.season,
        players=len(id_map),
        stat_rows=stat_rows,
        note="SYNTHETIC DATA - not real NFL results",
    )
    print(
        f"\nSeeded {len(id_map)} players and {stat_rows} weekly stat lines "
        f"for a synthetic {args.season} season."
    )
    print("These are NOT real NFL numbers. For real data:  python -m run_weekly\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
