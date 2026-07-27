# Manual data drop

Only needed if the pipeline container cannot reach the internet — common on corporate
networks, where the proxy is configured in your browser but not inside Docker.

Get the exact list of files and URLs for your configured seasons:

```bash
docker compose run --rm pipeline python -m ingest --urls
```

Download each one in your browser and save it into **this folder**, keeping the
filename exactly as-is (e.g. `player_stats_2024.parquet`, `roster_2024.parquet`).

Then run the pipeline normally — it detects the files and skips the network entirely:

```bash
docker compose run --rm pipeline python -m run_weekly
```

Files here are gitignored; only this README is tracked.
