# Interview prep — what you will actually be asked

For each component: why it exists, and the questions that follow. Answers are pointers to
real code in this repo, because "let me show you the file" beats a rehearsed paragraph.

---

## Backend API

**"Walk me through what happens on `POST /api/v1/predict`."**
Router validates the body with Pydantic → `PredictionService` looks the player up (404 if
absent) → builds a feature vector from Postgres using only weeks strictly before the
target week → checks Redis → on a miss, calls the ML service over HTTP → writes a row to
the append-only `predictions` table → writes the cache with a TTL → returns. On a hit it
short-circuits after the Redis read. `app/services/predictions.py`.

**"Why repositories instead of querying in the route handler?"**
Routers never touch the ORM, so query tuning happens in one place and the API is testable
without a database. It also keeps `EXPLAIN`-able SQL together instead of scattered across
handlers.

**"Why is a failing ML service a 503 and not a 500?"**
500 means "this service is broken"; 503 means "this service is fine, a dependency is
not." That distinction drives whether the on-call person looks at the API or at the model
service, and it stops the ALB from cycling healthy tasks.

**"How do you version the API?"**
Path-prefixed `/api/v1`, set from config. Breaking changes go to `/v2` and run alongside;
additive changes ship in place.

**Follow-ups you should be ready for:** why sync SQLAlchemy rather than async (the DB
driver is not the bottleneck here, and sync is simpler to reason about — but async would
matter under much higher concurrency); why `POST /predict` and `GET .../prediction` both
exist (the GET is CDN/browser cacheable, the POST is the one clients use for compare).

---

## Database design

**"Why is `predictions` append-only?"**
So "how did last week's projections actually do" is answerable. Updating in place destroys
exactly the record you need to evaluate a model after the fact. `model_version` on the row
is what lets you compare two models on identical weeks.

**"Justify each index."**
`ix_player_stats_player_season_week` — column order matches the `ORDER BY` on the hot
"last N weeks for player X" query, so Postgres can walk the index instead of sorting.
`ix_player_stats_season_week` — the leaderboard scans one week across all players.
`uq_player_stats_player_season_week` — not for reads at all; it is what makes the ETL's
`ON CONFLICT DO UPDATE` upsert possible.

**"How would you find a missing index in production?"**
`log_min_duration_statement = 500` is set in the RDS parameter group (`data.tf`), so slow
statements land in CloudWatch Logs. Then `EXPLAIN (ANALYZE, BUFFERS)` on the offender and
look for a Seq Scan or a Sort node that an index could remove. `pg_stat_statements` for
aggregate offenders.

**"This table gets to 50M rows. What breaks?"**
Nothing in the point lookups — they are index-bound. The leaderboard scan is the first
thing to hurt; partition `player_stats` by season, since every query filters on it.

**"Why not just store CSVs?"**
Concurrent writers, constraint enforcement, indexed lookups, transactions, and the
ability to compute rolling features in SQL next to the data instead of shipping every row
to Python.

---

## Data pipeline

**"What happens if the job runs twice?"**
Nothing bad — that is the design requirement, not a nice-to-have. Every write is
`INSERT ... ON CONFLICT DO UPDATE` against a natural key (`load.py`). Retries and
backfills are safe.

**"It failed halfway. What now?"**
Raw data is already on disk, so re-run with `--skip-ingest`. Stages are independently
importable functions, so `load` resumes at `load` without re-downloading.

**"Why recompute fantasy points instead of using the feed's number?"**
Reproducibility and league-format independence. `config.SCORING` is the whole scoring
rulebook; switching to half-PPR is a one-line diff plus a backfill. Trusting the feed
means you cannot explain your own numbers.

**"Cron or Airflow?"**
Both are here. EventBridge → Fargate task in AWS, because the job runs for minutes once a
week and an idle service the other 167 hours is waste. The Airflow DAG becomes worth its
operational cost once you want per-stage retries, a backfill UI, and dependency
visualisation. Both call the same functions — no duplicated logic.

**"How do you know the data is good?"**
`clean.py` enforces explicit rules and the tests in `pipeline/tests/` pin each one:
dedupe keeps the fuller row, negatives clamp, nulls zero, unjoinable rows drop. The
honest gap: there is no volume-anomaly check yet ("this week has 40% fewer rows than
usual" should fail the run).

---

## ML service

**"Why a separate service instead of loading the model in the API?"**
Different scaling signals (CPU vs IO), independent deploys, and model rollback without an
API deploy. The cost is a network hop and a new failure mode, which is why `MLClient`
owns timeouts and maps failures to 503.

**"How do you prevent training/serving skew?"**
One `FEATURE_ORDER` tuple is the single source of truth. Training builds its matrix from
it; serving builds its row from it. A missing, extra, or reordered feature raises at the
boundary. Artifacts record the contract version they were trained under and refuse to
load under a mismatch. A test in the *pipeline* asserts its copy matches too.

**"How do you roll back a bad model?"**
Change `ACTIVE_MODEL_VERSION` and restart. Artifacts live on EFS, not in the image, so
there is no rebuild and no retrain. Old versions stay on disk precisely for this.

**"Why not random train/test split?"**
Because the task is forecasting. A random split lets week 12 leak into training and
inflates the score. `time_split()` holds out the last four weeks.

**"Is `confidence` a probability?"**
No, and it does not claim to be. It combines validation residual spread with how much
history the player has, so a week-1 projection off zero games reads as low confidence.
Conformal prediction intervals would be the real fix.

**"XGBoost lost to ridge in your table. Why ship XGBoost?"**
On the synthetic generator the data is near-linear, so ridge should win — that is what the
baseline is for. On real ingested data the ordering changes; `make train` reprints the
table and names the winner. The point is that the comparison exists and promotion is a
config change.

---

## Caching

**"Why cache-aside rather than write-through?"**
Predictions are derived data with a natural TTL. Cache-aside keeps the write path simple
and means a cold cache is a latency event, never a correctness one.

**"What happens when Redis is down?"**
Every read and write path catches `RedisError` and degrades to a miss. Redis being down
makes the API slower, never broken. There is a test for exactly this.

**"How do you invalidate?"**
TTL for normal ageing, plus explicit `SCAN`-based prefix deletion from the pipeline after
it writes new data. Invalidation belongs to the writer — the reader has no way to know a
batch job ran. Never `KEYS`; it blocks the Redis event loop.

**"What if you promote a new model?"**
Cache keys include `model_version`, so the new model cannot serve the old model's values.
There is a test asserting the keys differ.

**"What is your hit rate and how do you know?"**
`fantasyiq_cache_events_total{result="hit"|"miss"|"error"}` in Prometheus, on the Grafana
dashboard, with an alert if it collapses below 50% (usually means eviction pressure or a
key-shape change).

---

## Docker & AWS

**"Why is the ML image built with a CPU-only torch wheel?"**
~200MB instead of ~2.5GB of CUDA the service will never use on Fargate. Image size is
deploy time and cold-start time.

**"Why do containers run as non-root?"**
A container escape starting from uid 10001 is strictly less useful to an attacker than one
starting from root. Cheap to do, so there is no reason not to.

**"Explain your subnet tiers."**
Public: ALB and NAT only. Private: ECS tasks, no inbound internet route. Isolated: RDS and
ElastiCache, no NAT at all. Security groups reference each other by ID, never by CIDR, so
"Postgres accepts traffic from the ECS security group" stays true when subnets change.

**"Where do secrets live?"**
Generated by Terraform into Secrets Manager, injected by ECS at task start via `secrets`
(not `environment`), so they never appear in the task definition JSON or `docker inspect`.
CI authenticates to AWS with OIDC — no long-lived keys in GitHub.

**"Why request-count autoscaling on the API but CPU on the ML service?"**
The API is IO-bound waiting on Postgres and the model service; its CPU stays flat while
latency climbs, so CPU is a useless trigger. Inference genuinely is CPU-bound. Match the
signal to the bottleneck.

**"What is the single point of failure?"**
The one NAT gateway (deliberate, ~$32/month saved) and single-AZ RDS in non-prod
(`db_multi_az` flips it). Both are stated in the README rather than hidden.

---

## CI/CD

**"What runs on a pull request?"**
Ruff lint + format check, three unit test suites, an integration job against real Postgres
and Redis that applies migrations forward/backward/forward and fails on schema drift, four
Docker builds, Terraform fmt + validate, and a compose smoke test that boots the stack and
exercises the API.

**"How do you catch schema drift?"**
CI autogenerates a migration against the migrated database and fails the build if the diff
is non-empty. An empty diff proves the ORM and the migrations agree — otherwise they
silently diverge until someone deploys.

**"Migration ordering — how do you avoid downtime?"**
Migrations run as a one-off task *before* the new app code rolls out, and they are
additive-only, so old and new code can both run against the new schema during the
rollout. Destructive changes are a separate, later migration.

**"How do you roll back?"**
ECS deployment circuit breaker handles tasks that never go healthy. For tasks that are
healthy but wrong, the deploy job smoke-tests `/health` and `/ready` and reverts to the
previous task definition revision on failure. Images are SHA-tagged and ECR tags are
immutable, so a revision always means the same bits.

---

## Monitoring

**"What do you alert on?"**
Symptoms, not causes: 5xx rate, p95 latency, prediction failure rate, cache hit-rate
collapse. High CPU is not an alert — it is a dashboard panel. Nobody should be woken up
for a number that users cannot feel.

**"How would you debug a latency spike?"**
Grafana: is it all endpoints or one? Then split cache hit rate (a drop means every request
is paying for inference) against ML service p95 (a rise means the model service is the
bottleneck) against RDS CPU and connections. The custom `fantasyiq_ml_service_latency`
histogram exists specifically to separate "our code is slow" from "our dependency is
slow."

**"What are you not monitoring that you should be?"**
Prediction accuracy drift. Every projection is stored with a model version and the actual
result lands the following week, so the join is trivial — a "rolling 4-week RMSE by model
version" panel is the obvious next build, and it is the metric that tells you the model
has gone stale rather than the service.

---

## Questions to ask them

- How do you decide a model is good enough to ship, and who signs off?
- What does the path from merge to production look like, and how long does it take?
- Where does the on-call boundary sit between the platform team and service owners?
