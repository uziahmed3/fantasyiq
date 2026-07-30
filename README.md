# FantasyIQ — NFL Fantasy Projection Platform

An analytics platform that projects NFL fantasy points — season-long rankings for draft
decisions, and weekly projections for start/sit decisions once the season is under way.
The model is one component of it, not the whole thing: there is a REST API, a normalised
Postgres schema, an idempotent ETL pipeline, a separately deployed inference service, a
cache-aside layer, containerised local dev, CI, and monitoring with alerts.

```
                                   Users
                                     │
                        ┌────────────┴────────────┐
                        │  backend  (FastAPI)     │  REST API, /api/v1
                        └───┬─────────────────┬───┘
                            │                 │
          ┌─────────────────┴──────┐   ┌──────┴───────────────────┐
          │ Redis                  │   │ ml-service               │
          │ cache-aside, TTL 1h    │   │ XGBoost / PyTorch / Ridge│
          │ (in-process fallback)  │   │ versioned artifacts      │
          └────────────────────────┘   └──────┬───────────────────┘
                            │                 │
          ┌─────────────────┴──────┐   ┌──────┴───────────────────┐
          │ Postgres 16            │   │ model volume             │
          │ players / stats /      │   │ versioned .joblib / .pt  │
          │ context / predictions  │   └──────────────────────────┘
          └─────────────────┬──────┘
                            │
          ┌─────────────────┴────────────────────────────────┐
          │ pipeline  (weekly, scheduled)                    │
          │ ingest → clean → load → context → batch score    │
          │ source: nflverse                                 │
          └──────────────────────────────────────────────────┘
```

## Run it

Two ways. Pick based on whether you can run Docker.

### No Docker — Python only

Needs nothing but Python 3.9+. No Postgres, no Redis, no Node, no admin rights.

```bash
python local.py --demo     # synthetic data, zero network, fastest way to see it work
python local.py            # real NFL data
python local.py --help     # everything else
```

(`run-local.ps1` and `run-local.sh` are one-line wrappers around the same script.)

Dashboard at **http://localhost:8000/app/**, API docs at `/docs`.

Same application code as the Docker stack, with two substitutions selected by
configuration rather than a code fork:

| | Docker | Local |
| --- | --- | --- |
| Database | Postgres 16 | SQLite file (`fantasyiq.db`) |
| Cache | Redis | in-process TTL cache |
| Front end | React + Vite (nginx) | zero-build HTML served at `/app` |

`GET /info` reports which combination is live. The pipeline SQL is written to standard
SQL — `CURRENT_TIMESTAMP` not `NOW()`, no `LEAST`/`GREATEST`, no `string_to_array` — so
the identical upsert and window-function queries run on both engines.

```bash
python local.py --reset        # wipe the database and models
python local.py --serve-only   # skip setup, just start the servers
python local.py --data-urls    # list files to download by hand
python local.py --offline      # use files already in data/manual/
```

The runner is Python rather than a shell script on purpose: it is the entry point
everything else depends on, and Python is the one language guaranteed to be present
and testable on every platform. A PowerShell script cannot be exercised from a Linux
CI runner; this can.

### With Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) running.

**Windows (PowerShell)** — one command does everything: build, boot, pull real NFL data,
train all three models, print the accuracy comparison, generate projections.

```powershell
.\run.ps1
```

```powershell
.\run.ps1 -Seasons 2025    # one season only (faster first run)
.\run.ps1 -Stop            # shut down
.\run.ps1 -Logs            # tail API + ML logs
.\run.ps1 -Nuke            # shut down and wipe all data
```

**macOS / Linux**

```bash
cp .env.example .env
make up        # postgres, redis, ml-service, backend, frontend, prometheus, grafana
make ingest    # pull real NFL data, load Postgres, generate weekly projections
make train     # ridge / XGBoost / PyTorch bake-off, writes versioned artifacts
make test      # backend + ml-service + pipeline test suites
```

First run pulls ~2GB of base images and a few hundred MB of NFL data — budget 10-15
minutes. After that, `docker compose up -d` is seconds.

### If your network blocks the download

Corporate proxies commonly allow a browser download while blocking the same host from
inside a container — the proxy is configured in the browser and nowhere else. The
pipeline has an offline path for exactly this:

```powershell
.\run.ps1 -DataUrls    # prints the exact nflverse files to download
# save each into .\data\manual\ using your browser
.\run.ps1 -Offline     # runs with zero network access
```

```bash
docker compose run --rm pipeline python -m ingest --urls   # same list
docker compose run --rm pipeline python -m run_weekly --source manual
```

`ingest.py` in `auto` mode (the default) uses `data/manual/` whenever the files are all
present and only falls back to the network otherwise, so once the parquet files are on
disk everything downstream is identical.

| Surface | URL |
| --- | --- |
| API docs (OpenAPI) | http://localhost:8000/docs |
| ML service docs | http://localhost:9000/docs |
| Dashboard | http://localhost:3000 |
| Grafana | http://localhost:3001 (`admin` / `admin`) |
| Prometheus | http://localhost:9090 |

The stack works before `make train`: with no artifact on disk the ML service serves a
documented heuristic baseline (`heuristic_fallback_v0`) rather than failing, so
`docker compose up` always produces a working system.

## API

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/api/v1/players` | filter by position / team / name, paginated |
| `GET` | `/api/v1/players/{id}` | |
| `GET` | `/api/v1/players/{id}/stats` | game log, newest first |
| `GET` | `/api/v1/players/{id}/predictions` | prediction audit history |
| `POST` | `/api/v1/predict` | `?refresh=true` bypasses the cache |
| `GET` | `/api/v1/players/{id}/prediction` | GET wrapper, CDN/browser-cacheable |
| `POST` | `/api/v1/compare` | 2–8 players, returned sorted |
| `GET` | `/api/v1/rankings` | weekly leaderboard — pure indexed read |
| `POST` | `/api/v1/auth/register` · `/auth/token` · `GET /auth/me` | JWT |
| `GET` | `/health` · `/ready` · `/metrics` | liveness · dependency check · Prometheus |

```bash
curl -X POST localhost:8000/api/v1/predict -H 'Content-Type: application/json' \
  -d '{"player_id":15,"week":6,"season":2023,"opponent":"GB"}'
```

```json
{
  "player_id": 15, "player": "Justin Jefferson", "season": 2023, "week": 6,
  "opponent": "GB", "prediction": 18.7, "confidence": 0.84,
  "model_version": "xgboost_v1", "source": "model"
}
```

Re-run the same request and `source` becomes `"cache"`.

## Design decisions worth asking about

**The ML service is a separate deployable.** It scales on CPU, the API scales on IO, and
a bad model rollout is reverted with an env var instead of an API redeploy. The cost is a
network hop, so `MLClient` owns the timeout, the retry policy, and the failure mapping —
an unreachable model surfaces as `503`, never `500`, because the API is healthy and a
dependency is not.

**Features are built from strictly prior weeks.** `WHERE week < :week`, and in SQL
`ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING`. Leaking the target week into the features is
the most common way an ML project reports a fake RMSE, so the window frame is written to
be reviewable. Evaluation splits chronologically, never randomly.

**One feature contract, enforced at the boundary.** `FEATURE_ORDER` in
`ml-service/app/features.py` is the single source of truth; training builds its design
matrix from it and serving builds its input row from it. A renamed or reordered feature
raises `FeatureContractError` at the service boundary rather than silently producing a
wrong number. Artifacts record the contract version they were trained against and refuse
to load under a mismatched one. `pipeline/tests/test_pipeline.py` asserts the pipeline's
copy of the order matches.

**The cache is cache-aside with TTLs, and every failure path degrades to a miss.** Redis
being down slows the API; it never breaks it. Keys are scoped by model version so
promoting a model cannot serve the previous model's values. Invalidation is
`SCAN`-based (never `KEYS`) and lives with the writer — the weekly pipeline busts the
prefix after it writes, because the reader has no way to know a batch job ran.

**`/rankings` reads the predictions table instead of invoking the model.** The weekly
pipeline batch-scores every eligible player in one vectorised call, so the dashboard's
front page costs zero inferences no matter how much traffic it gets.

**Migrations are versioned and drift is a CI failure.** CI runs
`upgrade head → downgrade base → upgrade head`, then autogenerates a migration and fails
the build if the diff is non-empty — proving the ORM and the schema actually agree.

**Health and readiness are different endpoints and used differently.** The ALB health
check hits `/health` (is this task alive), not `/ready` (are dependencies reachable). If
Postgres is down, `/ready` fails on every task, and draining them all would turn a
degraded API into a completely unavailable one.

**Autoscaling signals match the bottleneck.** The API scales on
`ALBRequestCountPerTarget`, because it is IO-bound waiting on Postgres and the ML
service — CPU stays flat while latency climbs. The ML service scales on CPU, because
inference genuinely is CPU-bound.

**Secrets stay out of the repo and out of images.** `.env` is gitignored, the JWT signing
key is generated at runtime if unset, and nothing secret is baked into a Dockerfile.

## Database

```
players ──┬── player_stats   UNIQUE (player_id, season, week)
          └── predictions    append-only
users
```

Indexes exist for named access patterns, not by reflex:

| Index | Serves |
| --- | --- |
| `ix_player_stats_player_season_week` | "last N weeks for player X" — column order matches the `ORDER BY`, so no sort step |
| `ix_player_stats_season_week` | single-week leaderboard scans across all players |
| `ix_predictions_lookup` | latest prediction for a player/week/model |
| `ix_players_position_team` | `/rankings` and `/players` filters |
| `uq_player_stats_player_season_week` | makes the ETL's `ON CONFLICT DO UPDATE` upsert possible |

`predictions` is append-only: a new model version writes a new row rather than updating
one, which is what makes "how did last week's projections actually do" answerable later.

Postgres can be set to log any statement slower than 500ms (`log_min_duration_statement`)
— that is how the next missing index gets found rather than guessed at.

## Pipeline

```
nfl_data_py → ingest.py → raw parquet → clean.py → clean parquet
                                                        │
                                          load.py (ON CONFLICT DO UPDATE)
                                                        │
                                    features.py (SQL windows) → /predict/batch
                                                        │
                                          predictions table → cache bust
```

Raw data lands on disk before any transformation, so a bad cleaning rule is re-runnable
without re-downloading. Every write is an upsert against a natural key, so the job is
safe to re-run — which it will be, on retries and backfills. Stages are independently
importable, so a failure at `load` resumes from `load`.

Fantasy points are **recomputed** from components under `config.SCORING` rather than
trusted from the feed, so the number is reproducible and switching to standard or
half-PPR scoring is a one-line diff plus a backfill.

`scheduler.py` runs the job locally and `pipeline/dags/fantasyiq_weekly.py` is the Airflow
version for when per-stage retries and a backfill UI become worth it. Both call the same
functions — no duplicated pipeline logic.

## Models

Three models, one bake-off, judged on a chronological holdout:

```
make train
model            framework        RMSE      MAE   vs naive
baseline_v1      sklearn        5.1454   3.9009     +14.9%
xgboost_v1       xgboost        5.3412   4.0840     +11.7%
naive_last3_avg  none           6.0497   4.6056          -
```

Those numbers are from the synthetic generator (no database required), where the data is
close to linear and ridge therefore wins — which is the point of keeping a baseline. On
real ingested data the ordering changes; `make train` reprints the table and names the
winner, and promoting it is an `ACTIVE_MODEL_VERSION` change, not a code change.

`confidence` is a documented heuristic combining the model's validation residual spread
with how much history the player has — a week-1 projection off zero games reports low
confidence. It is deliberately **not** described as a calibrated probability, because it
is not one. Conformal prediction intervals would be the honest upgrade.

## Testing

```
backend/tests      30 tests  — API contract, feature windows, cache behaviour, auth
ml-service/tests   12 tests  — feature contract, batch/single parity, graceful degradation
pipeline/tests     11 tests  — dedupe, null/negative handling, PPR scoring, contract match
```

Unit tests run against SQLite + fakeredis + a stubbed ML client, so the whole API surface
is testable with zero infrastructure. CI additionally spins up real Postgres and Redis
containers for migrations and drift checks, builds all four images, and boots the compose
stack to assert the API responds correctly end to end.

Some tests are there specifically to pin down behaviour that is easy to regress:

- a cache hit must not invoke the model (`test_second_call_hits_cache_and_skips_model`)
- a new model version must not read the old version's cached values
- Redis raising must degrade to a miss, not an exception
- features for week 3 must not see weeks 3 or 4
- batch and single-item inference must agree
- an unknown model version must degrade to the fallback, not 500
- bad password and unknown user must return identical responses (no account enumeration)

Load profile in `loadtest/locustfile.py` weights hot players against a long tail, so the
measured cache hit rate means something:

```bash
locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 1000 -r 50 -t 3m
```

## Monitoring

Prometheus scrapes both services. Custom metrics: cache hit/miss/error, predictions by
`(model_version, source)`, prediction failures by reason, downstream ML latency histogram,
ETL rows written. The Grafana dashboard is provisioned from
`monitoring/grafana/dashboards/fantasyiq.json` — no click-ops.

Alerts are symptom-based (5xx rate, p95 latency, prediction failures, cache hit-rate
collapse) rather than cause-based — including the failure mode that otherwise goes
unnoticed for a full week: the weekly ETL not running.

## Containers

```bash
docker compose up --build
```

Brings up Postgres, Redis, the ML service, the API, the pipeline and the frontend, plus
Prometheus and Grafana. Model artifacts live on a named volume rather than inside the
image, which is what makes rolling a model back a config change (`ACTIVE_MODEL_VERSION`)
instead of a rebuild.

Images are built and smoke-tested in CI on every push — see `.github/workflows/ci.yml`.

## Known limitations

Stated because they are the honest answers to the obvious follow-up questions:

- **`is_home` is hardcoded `true`.** the nflverse weekly release does not expose it; a
  schedule join would fill it properly. Defaulted rather than fabricated.
- **`opponent_rank` in the training set is computed over the full season**, which leaks a
  little end-of-season information. The fix is an expanding season-to-date rank, at the
  cost of very noisy early-season values. The serving path already computes it from prior
  weeks only.
- **`confidence` is not calibrated** — see above.
- **Only receiving/rushing scoring.** Passing stats are ingested but not scored, so QB
  projections are not meaningful yet.
- **One NAT gateway, not one per AZ.** Saves ~$32/month; makes outbound traffic
  single-AZ-dependent. Deliberate for this workload.
- **No end-to-end backtest harness yet.** The next thing worth building: replay a full
  season week by week and score the projections against what actually happened.

## Layout

```
backend/       FastAPI · SQLAlchemy · Alembic · repositories · services · 30 tests
ml-service/    inference API · model registry · 3 trainers · bake-off · 12 tests
pipeline/      ingest / clean / load / features · scheduler · Airflow DAG · 11 tests
frontend/      React + Vite dashboard, nginx-served
monitoring/    Prometheus config + alert rules, provisioned Grafana dashboard
loadtest/      Locust profile
.github/workflows/  ci.yml — lint · 3 test suites · integration · image build
```

Projections are model output, not advice.
