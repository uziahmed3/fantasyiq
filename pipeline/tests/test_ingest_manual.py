"""The offline ingest path.

Exists because the network path cannot be tested in CI without hitting nflverse, and
because the manual path is the one people fall back to when a corporate proxy blocks
the container - which is exactly when you least want it to be broken.
"""

import pandas as pd
import pytest

import ingest


@pytest.fixture
def manual_dir(tmp_path, monkeypatch):
    d = tmp_path / "manual"
    d.mkdir()
    monkeypatch.setattr(ingest, "MANUAL_DIR", d)
    return d


def _weekly_frame(season: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "00-0036322",
                "player_display_name": "Justin Jefferson",
                "position": "WR",
                "recent_team": "MIN",
                "season": season,
                "week": wk,
                "opponent_team": "GB",
                "targets": 10 + wk,
                "receptions": 7,
                "receiving_yards": 90.0 + wk,
                "receiving_tds": 1,
                "rushing_yards": 0.0,
                "rushing_tds": 0,
                "fantasy_points_ppr": 20.0,
            }
            for wk in (1, 2, 3)
        ]
    )


def _roster_frame(season: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "00-0036322",
                "player_name": "Justin Jefferson",
                "position": "WR",
                "team": "MIN",
                "season": season,
                "age": 25.0,
                "height": 73,
                "weight": 195,
            }
        ]
    )


def _seed(manual_dir, seasons):
    for s in seasons:
        _weekly_frame(s).to_parquet(manual_dir / f"player_stats_{s}.parquet", index=False)
        _roster_frame(s).to_parquet(manual_dir / f"roster_{s}.parquet", index=False)


def test_urls_are_the_real_nflverse_paths():
    pairs = dict(ingest.manual_files([2024]))
    assert pairs["player_stats_2024.parquet"].endswith(
        "/releases/download/player_stats/player_stats_2024.parquet"
    )
    assert pairs["roster_2024.parquet"].endswith("/releases/download/rosters/roster_2024.parquet")


def test_status_reports_missing_files(manual_dir):
    present, missing = ingest.manual_status([2024])
    assert present == []
    assert sorted(missing) == ["player_stats_2024.parquet", "roster_2024.parquet"]


def test_status_reports_present_files(manual_dir):
    _seed(manual_dir, [2024])
    present, missing = ingest.manual_status([2024])
    assert missing == []
    assert len(present) == 2


def test_manual_source_raises_with_actionable_message(manual_dir):
    with pytest.raises(ingest.MissingManualData) as exc:
        ingest.load_raw([2024], source="manual")
    assert "--urls" in str(exc.value)


def test_manual_load_concatenates_seasons(manual_dir):
    _seed(manual_dir, [2023, 2024])
    weekly, rosters, source = ingest.load_raw([2023, 2024], source="manual")
    assert source == "manual"
    assert len(weekly) == 6  # 3 weeks x 2 seasons
    assert set(weekly["season"]) == {2023, 2024}
    assert len(rosters) == 2


def test_auto_prefers_manual_and_never_touches_network(manual_dir, monkeypatch):
    _seed(manual_dir, [2024])

    def explode(*_a, **_k):
        raise AssertionError("auto must not hit the network when manual files are complete")

    monkeypatch.setattr(ingest, "_fetch_weekly", explode)
    monkeypatch.setattr(ingest, "_fetch_rosters", explode)

    _weekly, _rosters, source = ingest.load_raw([2024], source="auto")
    assert source == "manual"


def test_auto_falls_back_to_network_when_files_absent(manual_dir, monkeypatch):
    monkeypatch.setattr(ingest, "_fetch_weekly", lambda s: _weekly_frame(s[0]))
    monkeypatch.setattr(ingest, "_fetch_rosters", lambda s: _roster_frame(s[0]))
    _weekly, _rosters, source = ingest.load_raw([2024], source="auto")
    assert source == "network"


def test_run_end_to_end_from_manual_files(manual_dir, tmp_path, monkeypatch):
    _seed(manual_dir, [2024])
    raw = tmp_path / "raw"
    monkeypatch.setattr(ingest, "RAW_DIR", raw)
    monkeypatch.setattr(ingest, "_fetch_weekly", lambda *_: pytest.fail("should not fetch"))

    result = ingest.run([2024], source="manual")
    assert result["weekly_rows"] == 3
    assert (raw / "weekly.parquet").exists()
    assert (raw / "rosters.parquet").exists()

    written = pd.read_parquet(raw / "weekly.parquet")
    assert set(written.columns) <= set(ingest.WEEKLY_COLUMNS)
    assert written["position"].unique().tolist() == ["WR"]


def test_print_urls_lists_every_file(manual_dir, capsys):
    _seed(manual_dir, [2024])
    ingest.print_urls([2024, 2025])
    out = capsys.readouterr().out
    assert "player_stats_2024.parquet" in out
    assert "roster_2025.parquet" in out
    assert "[have]" in out and "[NEED]" in out
