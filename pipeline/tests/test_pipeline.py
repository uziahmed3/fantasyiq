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
