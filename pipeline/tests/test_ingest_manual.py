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
    # Canonical (current) naming is what the manual listing shows.
    assert pairs["player_stats_2024.parquet"].endswith(
        "/releases/download/stats_player/stats_player_week_2024.parquet"
    )
    assert pairs["roster_2024.parquet"].endswith("/releases/download/rosters/roster_2024.parquet")


def test_weekly_url_keeps_a_fallback_candidate():
    """nflverse renamed this release once already; the old path stays as a fallback so a
    rename does not look like 'no data for this season'."""
    assert len(ingest.WEEKLY_URL_CANDIDATES) >= 2
    assert any("stats_player_week_" in u for u in ingest.WEEKLY_URL_CANDIDATES)
    assert any("player_stats/player_stats_" in u for u in ingest.WEEKLY_URL_CANDIDATES)


def test_download_any_falls_through_to_the_next_candidate(tmp_path, monkeypatch):
    dest = tmp_path / "x.parquet"
    tried = []

    def fake_download(url, d):
        tried.append(url)
        if "stats_player_week" in url:
            raise OSError("404")
        d.write_bytes(b"ok")
        return d

    monkeypatch.setattr(ingest, "_download", fake_download)
    out = ingest._download_any(ingest.WEEKLY_URL_CANDIDATES, 2025, dest)
    assert out == dest
    assert len(tried) == 2, "must try the fallback after the first 404"


def test_download_any_raises_when_every_candidate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "_download", lambda *a, **k: (_ for _ in ()).throw(OSError("404")))
    with pytest.raises(ingest.AllCandidatesFailed) as exc:
        ingest._download_any(ingest.WEEKLY_URL_CANDIDATES, 2025, tmp_path / "y.parquet")
    # The message must name what was tried, or debugging this is guesswork.
    assert "stats_player_week_2025" in str(exc.value)
    assert "player_stats_2025" in str(exc.value)


def test_normalise_weekly_maps_new_schema_names():
    """The new release renamed several columns; downstream must see one stable shape."""
    new_schema = pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_name": "A",
                "team": "MIN",
                "opponent": "GB",
                "season": 2025,
                "week": 1,
                "targets": 8,
                "rec": 6,
                "rec_yds": 70.0,
                "rec_td": 1,
                "rush_yds": 0.0,
                "rush_td": 0,
                "position": "WR",
            }
        ]
    )
    out = ingest._normalise_weekly(new_schema)
    for expected in [
        "player_display_name",
        "recent_team",
        "opponent_team",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ]:
        assert expected in out.columns, f"{expected} missing after normalisation"


def test_normalise_weekly_leaves_canonical_schema_alone():
    old = pd.DataFrame(
        [
            {
                "player_id": "1",
                "player_display_name": "A",
                "position": "WR",
                "season": 2024,
                "week": 1,
                "recent_team": "MIN",
                "opponent_team": "GB",
                "receptions": 6,
                "receiving_yards": 70.0,
                "receiving_tds": 1,
            }
        ]
    )
    out = ingest._normalise_weekly(old)
    assert out.equals(old)


def test_normalise_rosters_maps_gsis_id():
    """The roster release keys on gsis_id while the weekly stats call it player_id.
    Getting this wrong crashed the real ingest with KeyError: ['player_id']."""
    real_shape = pd.DataFrame(
        [
            {
                "season": 2024,
                "team": "MIN",
                "position": "WR",
                "full_name": "Justin Jefferson",
                "gsis_id": "00-0036322",
                "height": 73,
                "weight": 195,
                "birth_date": "1999-06-16",
                "years_exp": 4,
                "draft_number": 22,
                "rookie_year": 2020,
            }
        ]
    )
    out = ingest._normalise_rosters(real_shape)
    assert out.loc[0, "player_id"] == "00-0036322"
    assert out.loc[0, "player_name"] == "Justin Jefferson"
    assert out.loc[0, "draft_pick"] == 22
    assert out.loc[0, "rookie_season"] == 2020
    assert out.loc[0, "years_of_experience"] == 4


def test_normalise_rosters_raises_a_schema_error_not_a_keyerror():
    """A download that succeeds but returns unexpected columns must say so, rather than
    being reported as a network failure."""
    with pytest.raises(ingest.SchemaMismatch) as exc:
        ingest._normalise_rosters(pd.DataFrame([{"some_other_id": "x", "team": "MIN"}]))
    assert "not a network problem" in str(exc.value)
    assert "some_other_id" in str(exc.value)


def test_normalise_weekly_requires_its_key_columns():
    with pytest.raises(ingest.SchemaMismatch):
        ingest._normalise_weekly(pd.DataFrame([{"targets": 5}]))


def test_weekly_accepts_the_new_release_schema():
    new = pd.DataFrame(
        [
            {
                "gsis_id": "00-1",
                "player_name": "A",
                "position": "WR",
                "season": 2025,
                "week": 1,
                "team": "MIN",
                "opponent": "GB",
                "rec": 6,
                "rec_yds": 70.0,
                "rec_td": 1,
                "rush_yds": 0.0,
                "rush_td": 0,
                "targets": 8,
            }
        ]
    )
    out = ingest._normalise_weekly(new)
    for expected in (
        "player_id",
        "player_display_name",
        "recent_team",
        "opponent_team",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ):
        assert expected in out.columns, f"{expected} missing"


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
    assert "stats_player_week_2024.parquet" in out
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
