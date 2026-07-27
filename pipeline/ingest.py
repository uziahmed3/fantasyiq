"""Extract: pull weekly NFL stats and roster data, land them raw as parquet.

Raw data is written to disk before any transformation. That separation means a bad
cleaning rule is re-runnable without re-downloading, and the raw files are the evidence
when a number in the dashboard looks wrong.

Two sources, in priority order:

  1. MANUAL_DIR (default /data/manual) - if the expected parquet files are already
     sitting there, they are used and nothing touches the network.
  2. nflverse over HTTPS, via nfl_data_py.

The manual path exists because corporate networks routinely allow a browser download
while blocking the same host from inside a container - the proxy is configured in the
browser and nowhere else. Rather than fight that, `python -m ingest --urls` prints the
exact files to download by hand, and dropping them in one folder makes the pipeline
work offline. Same code path afterwards.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from config import MANUAL_DIR, POSITIONS, RAW_DIR, SEASONS
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

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
WEEKLY_URL = f"{NFLVERSE}/player_stats/player_stats_{{season}}.parquet"
ROSTER_URL = f"{NFLVERSE}/rosters/roster_{{season}}.parquet"


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_weekly(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    log.info("fetch_weekly_start", seasons=seasons)
    return nfl.import_weekly_data(seasons, downcast=True)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _fetch_rosters(seasons: list[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    return nfl.import_seasonal_rosters(seasons)


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
