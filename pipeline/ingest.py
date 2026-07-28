"""Extract: pull weekly NFL stats and roster data, land them raw as parquet.

Raw data is written to disk before any transformation. That separation means a bad
cleaning rule is re-runnable without re-downloading, and the raw files are the evidence
when a number in the dashboard looks wrong.

Two sources, in priority order:

  1. MANUAL_DIR (default /data/manual) - if the expected parquet files are already
     sitting there, they are used and nothing touches the network.
  2. The same files downloaded from the nflverse GitHub release over HTTPS.

Both paths read byte-identical parquet files, so behaviour cannot diverge between them.

We fetch the release assets directly rather than depending on nfl_data_py: that package
hard-pins pandas<2.0, which conflicts with every other component here, and we use
exactly two of its endpoints. A stdlib download plus pd.read_parquet is fewer moving
parts than a dependency we would have to hold pandas back for.

The manual path exists because corporate networks routinely allow a browser download
while blocking the same host from inside a container - the proxy is configured in the
browser and nowhere else. Rather than fight that, `python -m ingest --urls` prints the
exact files to download by hand, and dropping them in one folder makes the pipeline
work offline. Same code path afterwards.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from config import DATA_DIR, MANUAL_DIR, POSITIONS, RAW_DIR, SEASONS
from logging_setup import log

WEEKLY_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "recent_team",
    "season",
    "week",
    "opponent_team",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rushing_yards",
    "rushing_tds",
    "fantasy_points_ppr",
]

ROSTER_COLUMNS = ["player_id", "player_name", "position", "team", "age", "height", "weight"]

# Cache for automatically downloaded release assets (distinct from MANUAL_DIR, which
# holds files the user placed there by hand).
DOWNLOAD_DIR = DATA_DIR / "downloads"

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"

# nflverse renamed this release: weekly player stats moved from `player_stats` to
# `stats_player`, and the old files stop being updated around 2024. Rather than pin one
# name and break the next time upstream reorganises, try candidates in order and use the
# first that downloads. The failure mode we are avoiding is the bad one: a 404 for a
# recent season that looks like "no data for 2025" instead of "wrong URL".
WEEKLY_URL_CANDIDATES = (
    f"{NFLVERSE}/stats_player/stats_player_week_{{season}}.parquet",
    f"{NFLVERSE}/player_stats/player_stats_{{season}}.parquet",
)
ROSTER_URL_CANDIDATES = (f"{NFLVERSE}/rosters/roster_{{season}}.parquet",)

# Context data for preseason projections. players.parquet and draft_picks.parquet are
# single combined files; the rest are per-season.
PLAYERS_URL = f"{NFLVERSE}/players/players.parquet"
DRAFT_URL = f"{NFLVERSE}/draft_picks/draft_picks.parquet"
SNAPS_URL_CANDIDATES = (f"{NFLVERSE}/snap_counts/snap_counts_{{season}}.parquet",)
DEPTH_URL_CANDIDATES = (f"{NFLVERSE}/depth_charts/depth_charts_{{season}}.parquet",)

# Kept for the manual-download listing and for tests that assert the canonical names.
WEEKLY_URL = WEEKLY_URL_CANDIDATES[0]
ROSTER_URL = ROSTER_URL_CANDIDATES[0]


class MissingManualData(FileNotFoundError):
    pass


def manual_files(seasons: list[int]) -> list[tuple[str, str]]:
    """(filename, download URL) pairs the manual path expects, in MANUAL_DIR."""
    pairs: list[tuple[str, str]] = []
    for season in seasons:
        pairs.append((f"player_stats_{season}.parquet", WEEKLY_URL.format(season=season)))
        pairs.append((f"roster_{season}.parquet", ROSTER_URL.format(season=season)))
    return pairs


def manual_status(seasons: list[int]) -> tuple[list[str], list[str]]:
    """Returns (present, missing) filenames for the manual path."""
    present, missing = [], []
    for filename, _url in manual_files(seasons):
        (present if (MANUAL_DIR / filename).exists() else missing).append(filename)
    return present, missing


def print_urls(seasons: list[int]) -> None:
    present, missing = manual_status(seasons)
    print(f"\nManual download folder: {MANUAL_DIR}")
    print("(on the host that is  ./data/manual/  next to docker-compose.yml)\n")
    print("Download these in your browser and drop them in that folder:\n")
    for filename, url in manual_files(seasons):
        mark = "have" if filename in present else "NEED"
        print(f"  [{mark}] {url}")
    print()
    if missing:
        print(f"{len(missing)} file(s) still missing: {', '.join(missing)}")
    else:
        print("All files present - `python -m run_weekly` will run fully offline.")
    print()


def _load_manual(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    present, missing = manual_status(seasons)
    if missing:
        raise MissingManualData(
            f"{len(missing)} file(s) missing from {MANUAL_DIR}: {', '.join(missing)}. "
            "Run `python -m ingest --urls` for the download links."
        )
    log.info("using_manual_data", directory=str(MANUAL_DIR), files=len(present))
    weekly = pd.concat(
        [pd.read_parquet(MANUAL_DIR / f"player_stats_{s}.parquet") for s in seasons],
        ignore_index=True,
    )
    rosters = pd.concat(
        [pd.read_parquet(MANUAL_DIR / f"roster_{s}.parquet") for s in seasons],
        ignore_index=True,
    )
    return weekly, rosters


class AllCandidatesFailed(RuntimeError):
    pass


def _download_any(urls: tuple[str, ...], season: int | None, dest: Path) -> Path:
    """Try each candidate URL in order; return the first that downloads.

    A 404 means "this release no longer carries that file", which is a naming problem,
    not a missing-data problem - so we keep going rather than reporting no data.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    errors: list[str] = []
    for template in urls:
        url = template.format(season=season) if season is not None else template
        try:
            return _download(url, dest)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
            log.warning("download_candidate_failed", url=url, error=str(exc))
    raise AllCandidatesFailed("none of the candidate URLs worked:\n  " + "\n  ".join(errors))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _download(url: str, dest: Path) -> Path:
    """Fetch one nflverse parquet release asset to disk.

    Downloads are cached, so a re-run after a failed later stage does not re-fetch.
    """
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("download_cached", file=dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("download_start", url=url)
    request = urllib.request.Request(url, headers={"User-Agent": "fantasyiq/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)  # atomic: a killed download never leaves a truncated parquet
    log.info("download_done", file=dest.name, mb=round(dest.stat().st_size / 1e6, 1))
    return dest


def _fetch_weekly(seasons: list[int]) -> pd.DataFrame:
    log.info("fetch_weekly_start", seasons=seasons)
    frames = [
        pd.read_parquet(
            _download_any(WEEKLY_URL_CANDIDATES, s, DOWNLOAD_DIR / f"player_stats_{s}.parquet")
        )
        for s in seasons
    ]
    return _normalise_weekly(pd.concat(frames, ignore_index=True))


def _normalise_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile column names across the old `player_stats` and new `stats_player` schemas.

    Upstream renamed several columns in the move. Mapping them here means the cleaner and
    everything downstream see one stable shape regardless of which release answered.
    """
    aliases = {
        "player_name": "player_display_name",
        "player_display_name": "player_display_name",
        "team": "recent_team",
        "recent_team": "recent_team",
        "opponent": "opponent_team",
        "opponent_team": "opponent_team",
        "receiving_tds": "receiving_tds",
        "rec_td": "receiving_tds",
        "receiving_yards": "receiving_yards",
        "rec_yds": "receiving_yards",
        "receptions": "receptions",
        "rec": "receptions",
        "targets": "targets",
        "rushing_yards": "rushing_yards",
        "rush_yds": "rushing_yards",
        "rushing_tds": "rushing_tds",
        "rush_td": "rushing_tds",
        "fantasy_points_ppr": "fantasy_points_ppr",
    }
    renames = {
        src: dst
        for src, dst in aliases.items()
        if src in df.columns and src != dst and dst not in df.columns
    }
    if renames:
        log.info("weekly_columns_renamed", mapping=renames)
        df = df.rename(columns=renames)
    return df


def _fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            _download_any(ROSTER_URL_CANDIDATES, s, DOWNLOAD_DIR / f"roster_{s}.parquet")
        )
        for s in seasons
    ]
    return pd.concat(frames, ignore_index=True)


def _derive_age(rosters: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """The roster release ships birth_date, not age. nfl_data_py used to compute this;
    now we do, against 1 September of the season (roughly week 1)."""
    if "age" in rosters.columns or "birth_date" not in rosters.columns:
        return rosters
    out = rosters.copy()
    season = out["season"] if "season" in out.columns else max(seasons)
    born = pd.to_datetime(out["birth_date"], errors="coerce")
    season_str = pd.Series(season, index=out.index).astype("Int64").astype(str)
    start = pd.to_datetime(season_str + "-09-01", errors="coerce")
    out["age"] = ((start - born).dt.days / 365.25).round(1)
    return out


def _load_network(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        return _fetch_weekly(seasons), _fetch_rosters(seasons)
    except Exception as exc:  # noqa: BLE001 - any transport failure gets the same advice
        log.error("network_ingest_failed", error=str(exc), error_type=type(exc).__name__)
        print(
            "\n"
            "Could not download NFL data. On a corporate network this is usually the\n"
            "proxy blocking the container (your browser may well reach the same URL).\n"
            "\n"
            "Offline route - download these by hand, then re-run:\n",
            file=sys.stderr,
        )
        print_urls(seasons)
        raise


def load_raw(seasons: list[int], source: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Returns (weekly, rosters, source_used)."""
    if source == "manual":
        weekly, rosters = _load_manual(seasons)
        return weekly, rosters, "manual"
    if source == "network":
        weekly, rosters = _load_network(seasons)
        return weekly, rosters, "network"

    # auto: prefer local files if they are all there, otherwise go to the network.
    _present, missing = manual_status(seasons)
    if not missing:
        weekly, rosters = _load_manual(seasons)
        return weekly, rosters, "manual"
    weekly, rosters = _load_network(seasons)
    return weekly, rosters, "network"


def run(seasons: list[int] | None = None, source: str = "auto") -> dict[str, int]:
    seasons = seasons or SEASONS
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    weekly, rosters, source_used = load_raw(seasons, source)

    keep = [c for c in WEEKLY_COLUMNS if c in weekly.columns]
    missing_cols = [c for c in WEEKLY_COLUMNS if c not in weekly.columns]
    if missing_cols:
        # Loud, not silent: a renamed upstream column would otherwise become a
        # column of zeros and quietly degrade every projection.
        log.warning("weekly_columns_missing", columns=missing_cols)
    weekly = weekly[keep]
    weekly = weekly[weekly["position"].isin(POSITIONS)]
    weekly_path = RAW_DIR / "weekly.parquet"
    weekly.to_parquet(weekly_path, index=False)

    rosters = _derive_age(rosters, seasons)
    roster_cols = [c for c in ROSTER_COLUMNS if c in rosters.columns]
    rosters = rosters[roster_cols]
    roster_path = RAW_DIR / "rosters.parquet"
    rosters.to_parquet(roster_path, index=False)

    log.info(
        "ingest_done",
        source=source_used,
        seasons=seasons,
        weekly_rows=len(weekly),
        roster_rows=len(rosters),
        weekly_path=str(weekly_path),
    )
    return {"weekly_rows": len(weekly), "roster_rows": len(rosters)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest NFL weekly stats and rosters")
    parser.add_argument("seasons", nargs="*", type=int, help="Seasons (default: INGEST_SEASONS)")
    parser.add_argument(
        "--source",
        choices=["auto", "manual", "network"],
        default="auto",
        help="auto = use ./data/manual if complete, else download",
    )
    parser.add_argument(
        "--urls",
        action="store_true",
        help="Print the files to download by hand and exit (no network access)",
    )
    args = parser.parse_args(argv)
    seasons = args.seasons or SEASONS

    if args.urls:
        print_urls(seasons)
        return 0

    run(seasons, source=args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
