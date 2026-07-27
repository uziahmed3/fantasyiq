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


def test_derive_age_from_birth_date():
    """The roster release ships birth_date, not age - nfl_data_py used to compute this
    for us. Regression guard for the reimplementation."""
    rosters = pd.DataFrame(
        [
            {"player_id": "1", "season": 2024, "birth_date": "1999-06-16"},
            {"player_id": "2", "season": 2024, "birth_date": "1995-01-01"},
            {"player_id": "3", "season": 2024, "birth_date": None},
        ]
    )
    out = ingest._derive_age(rosters, [2024])
    # Measured at 1 September of the season.
    assert out.loc[0, "age"] == pytest.approx(25.2, abs=0.2)
    assert out.loc[1, "age"] == pytest.approx(29.7, abs=0.2)
    assert pd.isna(out.loc[2, "age"])


def test_derive_age_is_a_noop_when_age_present():
    rosters = pd.DataFrame(
        [{"player_id": "1", "season": 2024, "age": 27.0, "birth_date": "1997-01-01"}]
    )
    assert ingest._derive_age(rosters, [2024]).loc[0, "age"] == 27.0


def test_derive_age_is_a_noop_without_birth_date():
    rosters = pd.DataFrame([{"player_id": "1", "season": 2024}])
    assert "age" not in ingest._derive_age(rosters, [2024]).columns


def test_download_is_cached(tmp_path, monkeypatch):
    """A re-run after a later stage fails must not re-download."""
    dest = tmp_path / "player_stats_2024.parquet"
    dest.write_bytes(b"already here")
    calls = []
    monkeypatch.setattr(ingest.urllib.request, "urlopen", lambda *a, **k: calls.append(1))
    assert ingest._download("https://example.invalid/x.parquet", dest) == dest
    assert calls == [], "cached file must not trigger a download"
