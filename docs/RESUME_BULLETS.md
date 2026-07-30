# Resume bullets — FantasyIQ

Every number below was measured on this repo. Sources are listed at the bottom so you can
reproduce any of them before an interview. **Do not use a number you haven't re-run.**

---

## Recommended version (4 bullets)

Use this if the project gets a normal amount of space. It covers backend, data, ML and
infra in that order — which is the order a backend SWE screener cares about.

> **FantasyIQ — NFL Fantasy Football Analytics Platform**
> *Python, FastAPI, PostgreSQL, Redis, XGBoost, Docker, GitHub Actions, Prometheus*
>
> - Built a 3-service platform (REST API, ML inference service, ETL pipeline) exposing 14
>   versioned endpoints that serve season-long draft rankings and weekly start/sit
>   projections, backed by PostgreSQL with 30K+ player-week records across 5 NFL seasons.
> - Designed a two-model architecture routing on data availability rather than calendar
>   date, so rookies and week-1 players are served by a preseason model instead of failing
>   for lack of recent form; enforced a shared feature contract across training and serving
>   to prevent skew, with artifacts rejected at load time on version mismatch.
> - Implemented cache-aside caching with a pluggable Redis/in-memory backend, cutting
>   repeat prediction latency 92% (101ms → 8ms) by eliminating an inter-service HTTP call
>   and model inference per request.
> - Containerized the stack with Docker Compose (API, ML service, Postgres, Redis,
>   Prometheus/Grafana) and set up 7-job GitHub Actions CI running 152 tests, linting,
>   image builds, and an integration job that applies migrations forward and backward
>   against real Postgres to catch schema drift.

## Compact version (3 bullets)

When space is tight. Merges infra into the API bullet and keeps the two strongest claims.

> - Built a 3-service NFL analytics platform (FastAPI, PostgreSQL, Redis, Docker) serving
>   14 REST endpoints over 30K+ player-week records, with 7-job CI running 152 tests on
>   every push.
> - Architected dual prediction models routing on data availability, with a shared feature
>   contract preventing training/serving skew; improved holdout accuracy 30% over baseline
>   (RMSE 4.04 → 2.83) on a season fully withheld from training.
> - Reduced repeat prediction latency 92% (101ms → 8ms) via cache-aside caching, and
>   diagnosed a systematic model bias using per-player SHAP attribution that global feature
>   importance could not surface.

## If the role leans ML / data

Swap the third bullet for these two:

> - Engineered 31 preseason features including career production weighted by recency,
>   games played and an age curve, improving accuracy 29.9% over a carry-forward baseline
>   (RMSE 4.04 → 2.83, R² 0.67) using season-level holdout validation to prevent leakage.
> - Identified and corrected a systematic ranking bias via per-player SHAP attribution —
>   the model penalized established players for draft position years after production made
>   it irrelevant — using monotonic constraints and evidence-weighted feature decay to fix
>   it while improving accuracy.

## If the role leans infra / DevOps

Swap the fourth bullet for:

> - Containerized a 6-service stack with Docker Compose, instrumented it with Prometheus
>   metrics and provisioned Grafana dashboards, and built CI that boots the entire stack
>   and smoke-tests the API on every push; designed model artifacts to live on a mounted
>   volume so a bad model rolls back via environment variable without a rebuild.

---

## Why these are written the way they are

**They lead with the system, not the model.** Nearly every new-grad resume says "trained a
machine learning model to predict X." That reads as a course project. "Built a 3-service
platform serving 14 versioned endpoints" reads as engineering, and for a backend role
that's the difference.

**Every bullet names a decision, not just an activity.** "Used Redis for caching" is a
tool list. "Cache-aside with a pluggable backend, cutting repeat latency 92% by
eliminating an inter-service call" says you understood *why* and measured the result.

**The ATS keywords are real.** Python, FastAPI, PostgreSQL, Redis, Docker, CI/CD, REST,
XGBoost, Prometheus — all present, all things you actually did and can discuss.
Keyword-stuffing tech you can't talk about is how people get destroyed in phone screens,
which is exactly why Terraform and AWS were cut from this project rather than left in as
resume decoration.

**Each bullet is a question you want to be asked:**

| Bullet | The question it invites | Your answer lives in |
|---|---|---|
| 3-service platform | "Why not a monolith?" | Guide, Part 5 |
| Two-model routing | "Why two models?" | Guide, Part 2 — this is your best material |
| Cache-aside 92% | "Why cache-aside, not write-through?" | `INTERVIEW.md`, Caching |
| Docker / CI | "How would you deploy this?" | Treat it as a design question — see `INTERVIEW.md`, Docker |

---

## What NOT to put on there

These are the claims that would fall apart under one follow-up question:

- ❌ **"Deployed to production"**, **"serving live traffic"**, or any cloud provider name —
  none of it is deployed anywhere. "Containerized with Docker Compose" is true and enough.
  If a job description demands cloud experience, say what you'd do rather than what you did.
- ❌ **Any user count, DAU, or adoption metric** — there are no users.
- ❌ **"99.9% uptime"** — nothing has been running long enough to have uptime.
- ❌ **"Improved model accuracy by 57%"** — that was the synthetic-data number that turned
  out to be a bug in my generator. The real number is 29.9%.
- ❌ **In-season model accuracy figures** — that artifact isn't currently trained in
  `models/`. Retrain first if you want to cite it.
- ❌ **"Scaled to X requests/sec"** — you have a Locust file but no load-test results
  worth quoting. Run it if you want the number.

---

## Sources for every number

Re-run these before an interview so the figures are ones you've personally seen.

| Claim | How to verify |
|---|---|
| 30,710 player-week records, 5 seasons, 1,627 players | `SELECT COUNT(*) FROM player_stats;` — seasons 2021–2025 |
| 14 REST endpoints | `GET /docs` on the running API, or count routes under `/api/v1` |
| 152 tests, 7 CI jobs | `pytest` in `backend/`, `ml-service/`, `pipeline/` (61 + 45 + 46); jobs in `.github/workflows/ci.yml` |
| RMSE 4.04 → 2.83, +29.9%, R² 0.67 | `models/preseason_v1.json`, or rerun `python -m train.train_preseason` |
| 101ms → 8ms, 92% | Repeat `POST /api/v1/predict` with an identical body; the response's `source` field flips to `cache`. Measured locally on SQLite with the in-memory backend — say "local benchmark" if pressed |
| 31 / 10 features | `len(PRESEASON_FEATURE_ORDER)`, `len(FEATURE_ORDER)` in `ml-service/app/features.py` |

**On the latency number specifically:** the honest framing is that the cache-miss path
makes an HTTP call to the ML service, runs inference, and writes a row; the hit path skips
all three. That structural difference is what produces the gap, and it would hold on real
infrastructure — but the absolute figures come from a local single-client benchmark, so
present them as such rather than as production numbers.
