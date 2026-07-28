import pandas as pd
import pytest

from clean import clean_rosters, clean_weekly, compute_fantasy_points
from features import FEATURE_ORDER
from scheduler import next_run

RAW = pd.DataFrame(
    [
        # duplicate player/season/week - the fuller row must win
        {
            "player_id": "00-1",
            "player_display_name": "A Receiver",
            "position": "WR",
            "recent_team": "MIN",
            "season": 2023,
            "week": 1,
            "opponent_team": "tb",
            "targets": 9,
            "receptions": 7,
            "receiving_yards": 84.0,
            "receiving_tds": 1,
            "rushing_yards": 0.0,
            "rushing_tds": 0,
        },
        {
            "player_id": "00-1",
            "player_display_name": "A Receiver",
            "position": "WR",
            "recent_team": "MIN",
            "season": 2023,
            "week": 1,
            "opponent_team": "TB",
            "targets": 9,
            "receptions": 7,
            "receiving_yards": 12.0,
            "receiving_tds": 0,
            "rushing_yards": 0.0,
            "rushing_tds": 0,
        },
        # nulls and a negative that must be clamped
        {
            "player_id": "00-2",
            "player_display_name": "B Back",
            "position": "RB",
            "recent_team": "CIN",
            "season": 2023,
            "week": 1,
            "opponent_team": None,
            "targets": None,
            "receptions": -1,
            "receiving_yards": None,
            "receiving_tds": None,
            "rushing_yards": 55.0,
            "rushing_tds": 1,
        },
        # unjoinable: no team
        {
            "player_id": "00-3",
            "player_display_name": "C Ghost",
            "position": "TE",
            "recent_team": None,
            "season": 2023,
            "week": 1,
            "opponent_team": "GB",
            "targets": 3,
            "receptions": 2,
            "receiving_yards": 20.0,
            "receiving_tds": 0,
            "rushing_yards": 0.0,
            "rushing_tds": 0,
        },
    ]
)


def test_dedupe_keeps_fuller_row():
    out = clean_weekly(RAW)
    row = out[out["player_id"] == "00-1"]
    assert len(row) == 1
    assert row.iloc[0]["yards"] == 84.0


def test_drops_rows_without_team():
    assert "00-3" not in set(clean_weekly(RAW)["player_id"])


def test_negatives_clamped_and_nulls_zeroed():
    out = clean_weekly(RAW)
    b = out[out["player_id"] == "00-2"].iloc[0]
    assert b["receptions"] == 0
    assert b["targets"] == 0
    assert b["opponent"] == "UNK"


def test_ppr_scoring_is_recomputed():
    # 7 rec + 84 yds + 1 TD = 7 + 8.4 + 6 = 21.4
    out = clean_weekly(RAW)
    assert out[out["player_id"] == "00-1"].iloc[0]["fantasy_points"] == pytest.approx(21.4)


def test_scoring_handles_missing_columns():
    assert compute_fantasy_points(pd.DataFrame({"receptions": [3]})).iloc[0] == 3.0


def test_output_schema_is_stable():
    expected = {
        "player_id",
        "name",
        "position",
        "team",
        "season",
        "week",
        "opponent",
        "is_home",
        "targets",
        "receptions",
        "yards",
        "touchdowns",
        "fantasy_points",
    }
    assert set(clean_weekly(RAW).columns) == expected


def test_clean_rosters_dedupes_and_renames():
    rosters = pd.DataFrame(
        [
            {
                "player_id": "00-1",
                "player_name": "A Receiver",
                "position": "WR",
                "team": "MIN",
                "age": 25.0,
                "height": 73,
                "weight": 195,
            },
            {
                "player_id": "00-1",
                "player_name": "A Receiver",
                "position": "WR",
                "team": "MIN",
                "age": 26.0,
                "height": 73,
                "weight": 198,
            },
        ]
    )
    out = clean_rosters(rosters)
    assert len(out) == 1
    assert out.iloc[0]["age"] == 26.0
    assert {"name", "height_inches", "weight_lbs"} <= set(out.columns)


def test_feature_order_matches_ml_contract():
    """The pipeline and the ML service must agree on feature order, or every batch
    projection is silently wrong."""
    ml_contract = (
        "targets_last_3",
        "receptions_last_3",
        "yards_last_3",
        "touchdowns_last_3",
        "fantasy_points_last_3",
        "fantasy_points_last_1",
        "season_avg_points",
        "games_played",
        "opponent_rank",
        "is_home",
    )
    assert ml_contract == FEATURE_ORDER


@pytest.mark.parametrize(
    "now,expected_weekday",
    [("2024-09-04T12:00:00Z", 1), ("2024-09-03T08:00:00Z", 1), ("2024-09-03T10:00:00Z", 1)],
)
def test_scheduler_always_targets_tuesday(now, expected_weekday):
    from datetime import datetime

    dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    assert next_run(dt).weekday() == expected_weekday
    assert next_run(dt) > dt


# ---------------------------------------------------------------- week clamping
def test_next_week_clamps_to_the_regular_season(tmp_path):
    """Real 2025 data runs to week 22 (playoffs), and the old code projected week 23 -
    a week that does not exist, so 530 predictions landed somewhere nothing reads."""
    from sqlalchemy import create_engine, text

    import run_weekly

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/w.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE player_stats (player_id INT, season INT, week INT)"))
        # A completed season including playoff weeks.
        for wk in range(1, 23):
            conn.execute(text("INSERT INTO player_stats VALUES (1, 2025, :w)"), {"w": wk})
    assert run_weekly._next_week(engine, 2025) == run_weekly.REGULAR_SEASON_WEEKS


def test_next_week_is_the_following_week_midseason(tmp_path):
    from sqlalchemy import create_engine, text

    import run_weekly

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/w2.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE player_stats (player_id INT, season INT, week INT)"))
        for wk in range(1, 8):
            conn.execute(text("INSERT INTO player_stats VALUES (1, 2025, :w)"), {"w": wk})
    assert run_weekly._next_week(engine, 2025) == 8


def test_quarterbacks_are_ingested_by_default():
    """qb_changed needs QB rows to detect a change of starter. Without them the real run
    logged no_qb_data on every season and the feature was silently dead."""
    import importlib
    import os

    os.environ.pop("INGEST_POSITIONS", None)
    import config

    importlib.reload(config)
    assert "QB" in config.POSITIONS


# ---------------------------------------------------------------- demo/real isolation
def test_synthetic_players_all_carry_the_demo_prefix():
    """The prefix is the only thing that makes synthetic rows separable from real ones
    after they are in the database, so every generated player must have it."""
    import seed_demo

    weekly, rosters = seed_demo.generate([2023], weeks=2, seed=1)
    assert weekly["player_id"].str.startswith(seed_demo.DEMO_ID_PREFIX).all()
    assert rosters["player_id"].str.startswith(seed_demo.DEMO_ID_PREFIX).all()


def test_purge_removes_synthetic_rows_and_keeps_real_ones(tmp_path):
    """Real 2021-2025 data got mixed with a demo seed once; purge is the recovery path."""
    from sqlalchemy import create_engine, text

    import seed_demo

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/mixed.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE players (id INTEGER PRIMARY KEY, external_id TEXT)"))
        conn.execute(text("CREATE TABLE player_stats (player_id INT, season INT)"))
        conn.execute(text("CREATE TABLE player_context (player_id INT, season INT)"))
        conn.execute(text("CREATE TABLE predictions (player_id INT, season INT)"))
        conn.execute(text("INSERT INTO players VALUES (1, '00-0036322')"))  # real
        conn.execute(text("INSERT INTO players VALUES (2, 'DEMO-00001')"))  # synthetic
        for table in ("player_stats", "player_context", "predictions"):
            conn.execute(text(f"INSERT INTO {table} VALUES (1, 2024)"))
            conn.execute(text(f"INSERT INTO {table} VALUES (2, 2024)"))

    removed = seed_demo.purge(engine)
    assert removed == 1
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM players")).scalar_one() == 1
        assert conn.execute(text("SELECT external_id FROM players")).scalar_one() == "00-0036322"
        # Children of the synthetic player must go too, and the real one's must survive.
        for table in ("player_stats", "player_context", "predictions"):
            rows = conn.execute(text(f"SELECT player_id FROM {table}")).scalars().all()
            assert rows == [1], f"{table} not cleaned correctly: {rows}"


def test_census_separates_real_from_synthetic(tmp_path):
    from sqlalchemy import create_engine, text

    import seed_demo

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/census.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE players (id INTEGER PRIMARY KEY, external_id TEXT)"))
        conn.execute(text("INSERT INTO players VALUES (1, '00-1'), (2, 'DEMO-1'), (3, 'DEMO-2')"))
    assert seed_demo._census(engine) == (1, 2)


def test_build_all_includes_the_season_being_projected(tmp_path, monkeypatch):
    """The draft-board season has no games, so it never appears in seasons_with_data.
    Leaving it out meant `context --all` silently skipped the one season that matters -
    and stale rows survived a regular-season fix, showing "21 games last season"."""
    from sqlalchemy import create_engine, text

    import context

    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/ba.db")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE player_stats (player_id INT, season INT, week INT)"))
        for season in (2023, 2024, 2025):
            conn.execute(text("INSERT INTO player_stats VALUES (1, :s, 1)"), {"s": season})

    built = []
    monkeypatch.setattr(context, "build", lambda _e, season, **kw: built.append(season) or 1)
    context.build_all(engine)
    # 2023 is the earliest (nothing before it); 2026 is the projection season.
    assert built == [2024, 2025, 2026]
