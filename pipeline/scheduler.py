"""Minimal in-container scheduler for local development.

Deliberately dumb: sleep until the next target time, run the job, log the outcome. In AWS
this process is replaced by an EventBridge rule firing an ECS scheduled task (see
infra/terraform/pipeline.tf) so there is no long-lived container to babysit; the Airflow
DAG in dags/ is the alternative if the pipeline grows more than a handful of stages.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone

import run_weekly
from logging_setup import log

# Tuesday 09:00 UTC - after Monday Night Football has settled.
RUN_WEEKDAY = int(os.getenv("PIPELINE_WEEKDAY", "1"))  # 0=Monday
RUN_HOUR = int(os.getenv("PIPELINE_HOUR", "9"))
RUN_ON_BOOT = os.getenv("PIPELINE_RUN_ON_BOOT", "false").lower() == "true"


def next_run(now: datetime) -> datetime:
    target = now.replace(hour=RUN_HOUR, minute=0, second=0, microsecond=0)
    days_ahead = (RUN_WEEKDAY - now.weekday()) % 7
    target += timedelta(days=days_ahead)
    if target <= now:
        target += timedelta(days=7)
    return target


def main() -> None:
    log.info("scheduler_started", weekday=RUN_WEEKDAY, hour=RUN_HOUR, run_on_boot=RUN_ON_BOOT)
    if RUN_ON_BOOT:
        run_weekly.main([])
    while True:
        now = datetime.now(timezone.utc)
        upcoming = next_run(now)
        sleep_for = (upcoming - now).total_seconds()
        log.info(
            "scheduler_sleeping", next_run=upcoming.isoformat(), hours=round(sleep_for / 3600, 2)
        )
        time.sleep(min(sleep_for, 3600))  # wake hourly so restarts do not lose the schedule
        if datetime.now(timezone.utc) >= upcoming:
            code = run_weekly.main([])
            log.info("scheduled_run_finished", exit_code=code)


if __name__ == "__main__":
    main()
