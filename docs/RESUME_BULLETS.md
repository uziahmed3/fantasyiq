# Resume bullets — FantasyIQ (software engineering roles)

Written for general SWE and backend roles. The machine learning is present because it's
differentiating, but it's framed as engineering — interfaces, dispatch, correctness,
performance — because that's what a SWE screener is reading for.

Every number was measured on this repo. Sources are at the bottom. **Do not use a number
you haven't re-run.**

---

## Recommended — 5 bullets

A flagship project earns five lines. Bullet 2 is the one that says *how* the prediction
works — without it a reader cannot tell a trained model from a pile of if-statements.

> **FantasyIQ — Full-Stack NFL Analytics Platform** · [github.com/uziahmed3/fantasyiq](https://github.com/uziahmed3/fantasyiq)
> *Python, JavaScript, FastAPI, PostgreSQL, Redis, XGBoost, React, Docker, pytest, GitHub Actions*
>
> - Built a 3-service application (REST API, ML inference service, ETL pipeline) with a
>   React dashboard, exposing 14 versioned endpoints over a normalized PostgreSQL schema of
>   30K+ records; designed idempotent upserts so re-runs are safe and an append-only
>   prediction log so past outputs stay auditable.
> - Trained XGBoost gradient-boosted regression models to project player scoring, with 31
>   engineered features — career production weighted by recency, games played and an age
>   curve; usage share; depth-chart role — improving accuracy **29.9% over a carry-forward
>   baseline (RMSE 4.04 → 2.83, R² 0.67)**, validated on a season withheld from training
>   entirely to prevent leakage.
> - Designed a dispatch layer selecting between two models at request time based on data
>   availability, behind a single endpoint, plus a versioned interface contract validated at
>   the service boundary — incompatible artifacts fail fast on load instead of silently
>   returning wrong results.
> - Cut repeat-request latency 92% (101ms → 8ms) with a cache-aside layer behind a
>   pluggable Redis/in-memory interface, eliminating a cross-service HTTP call and model
>   inference per request.
> - Wrote 152 tests and a 7-job CI pipeline (lint, 3 test suites, integration, image build,
>   full-stack smoke test), including a migration forward/backward check that fails the
>   build when the ORM and schema drift apart.

## Tight — 4 bullets

Merges the dispatch design into the modelling bullet. Use when space is normal.

> - Built a 3-service application (REST API, ML inference service, ETL pipeline) with a
>   React dashboard, exposing 14 versioned endpoints over a normalized PostgreSQL schema of
>   30K+ records, with idempotent upserts and an append-only prediction log.
> - Trained XGBoost regression models on 31 engineered features (career production weighted
>   by recency and injury, usage share, depth-chart role), improving accuracy 29.9% over a
>   carry-forward baseline (RMSE 4.04 → 2.83) on a fully held-out season; a dispatch layer
>   selects between two models at request time by data availability, behind a versioned
>   contract that fails fast on artifact mismatch.
> - Cut repeat-request latency 92% (101ms → 8ms) with a cache-aside layer behind a
>   pluggable Redis/in-memory interface, eliminating a cross-service call and inference per
>   request.
> - Wrote 152 tests and a 7-job CI pipeline including a migration forward/backward check
>   that fails the build when the ORM and schema drift apart.

## Optional extra — the debugging one

The most distinctive line available to you, because almost no new-grad resume describes
finding a bug that never threw an error.

> - Diagnosed a silent correctness defect in model output — no exception, just wrong
>   rankings — by building a per-prediction SHAP attribution endpoint that decomposed each
>   result into the feature contributions that produced it; corrected the bias with
>   monotonic constraints and evidence-weighted feature decay.

## Compact — 3 bullets

> - Built a 3-service full-stack application (FastAPI, PostgreSQL, Redis, React, Docker)
>   serving 14 versioned REST endpoints over a normalized 30K-record schema, with 152 tests
>   and 7-job CI on every push.
> - Trained XGBoost models on 31 engineered features to project player scoring, improving
>   accuracy 29.9% over baseline (RMSE 4.04 → 2.83) on a held-out season, served through a
>   request-time dispatch layer with a versioned contract validated at the service boundary.
> - Cut repeat-request latency 92% (101ms → 8ms) via a pluggable cache-aside layer, and
>   diagnosed a silent correctness bug by building per-prediction SHAP attribution tooling.

## Ultra-compact — 2 bullets

When the project is one of five and space is tight.

> - Built a 3-service full-stack NFL analytics platform (FastAPI, PostgreSQL, Redis, React,
>   Docker) — 14 REST endpoints, 30K-record normalized schema, 152 tests, 7-job CI.
> - Trained XGBoost models on 31 engineered features, improving prediction accuracy 29.9%
>   over baseline (RMSE 4.04 → 2.83) on a held-out season, and cut repeat-request latency
>   92% (101ms → 8ms) with a pluggable cache-aside layer.

---

## Why they're written this way

**The method is named, then framed as engineering.** Both halves matter. "Trained XGBoost
regression models on 31 engineered features" tells a reader what you actually did — without
it, "two prediction strategies" could mean if-statements. But the *dispatch* between them,
and the contract that validates artifacts at the boundary, are described as software design
because that's what a SWE interviewer is evaluating. Naming the technique earns credibility;
framing it as system design earns the interview.

**Specifically say "XGBoost", not "machine learning".** Vague ML claims invite the question
"what model did you use?" and a weak answer sinks the bullet. Naming the algorithm, the
feature count, the validation scheme and the baseline comparison signals you know what you
trained and how you knew it worked.

**The bullets name decisions, not tools.** "Used Redis for caching" is a tool list.
"Cache-aside behind a pluggable interface, cutting latency 92% by eliminating a
cross-service call" says you knew why, chose a pattern, and measured the result. That's the
difference between a bullet that survives follow-up and one that doesn't.

**Correctness gets equal billing with features.** Idempotent upserts, an append-only audit
log, fail-fast contract validation, and a migration drift check are all about *not being
silently wrong*. Most new-grad resumes are entirely about things built and never about
things kept correct — and senior engineers reading resumes notice that.

**The keywords are all defensible.** Python, JavaScript, FastAPI, PostgreSQL, Redis,
XGBoost, React, Docker, pytest, CI/CD, REST — every one is something you did and can
discuss. That's the same reason Terraform and AWS were removed from the project entirely
rather than kept as resume decoration.

**Each bullet invites a question you can answer:**

| Bullet | The question it invites | Where your answer lives |
|---|---|---|
| 3-service platform | "Why not one service?" | Guide, Part 5 |
| XGBoost + 31 features | "Why XGBoost, and how did you validate?" | Guide, Part 5 and Part 6 |
| Dispatch + contract | "Why two models?" | Guide, Part 2 — your best material |
| Cache-aside 92% | "Why cache-aside, not write-through?" | `INTERVIEW.md`, Caching |
| Tests + CI | "What does your CI actually catch?" | `INTERVIEW.md`, CI/CD |
| Attribution bug | "Walk me through it." | Guide, Part 4, story 1 |

---

## If a job description leans ML

The bullets above already name the method, which is enough for most SWE postings. If a role
is explicitly ML-focused, expand bullet 2 into two and lead with them:

> - Engineered 31 features for a season-projection model — career production weighted by
>   recency, games played and an age curve; volume/efficiency split; usage share and
>   depth-chart role — improving accuracy 29.9% over a carry-forward baseline (RMSE 4.04 →
>   2.83, R² 0.67) with season-level holdout validation to prevent leakage.
> - Corrected a systematic ranking bias found via per-prediction SHAP attribution, applying
>   XGBoost monotonic constraints so production features cannot lower a projection and
>   decaying draft-capital features as real game evidence accumulates.

---

## What NOT to put on there

Each of these collapses under one follow-up question:

- ❌ **"Deployed to production"**, **"serving live traffic"**, or any cloud provider name —
  it isn't deployed anywhere. "Containerized with Docker Compose" is true and enough.
- ❌ **Any user count, DAU, or adoption metric** — there are no users.
- ❌ **"99.9% uptime"** — nothing has run long enough to have uptime.
- ❌ **"Improved accuracy 57%"** — that was the synthetic-data figure that turned out to be
  a bug in the generator. The real number is 29.9%.
- ❌ **In-season model accuracy** — that artifact isn't currently trained in `models/`.
  Retrain before citing it.
- ❌ **"Handles X requests/sec"** — there's a Locust file but no load-test results worth
  quoting. Run it if you want the number.
- ❌ **"Agile", "collaborated with stakeholders", "cross-functional"** on a solo project.
  Interviewers know what a personal project is; padding it reads as insecurity.

---

## Sources for every number

Re-run these before an interview so every figure is one you've personally seen.

| Claim | How to verify |
|---|---|
| 30,710 records, 5 seasons, 1,627 players | `SELECT COUNT(*) FROM player_stats;` — seasons 2021–2025 |
| 14 REST endpoints | `GET /docs` on the running API, or count routes under `/api/v1` |
| 152 tests, 7 CI jobs | `pytest` in `backend/`, `ml-service/`, `pipeline/` (61 + 45 + 46); jobs in `.github/workflows/ci.yml` |
| RMSE 4.04 → 2.83, +29.9%, R² 0.67 | `models/preseason_v1.json`, or rerun `python -m train.train_preseason` |
| 101ms → 8ms, 92% | Repeat `POST /api/v1/predict` with an identical body; the response's `source` field flips to `cache`. Local single-client benchmark — say so if pressed |
| 31 features / 2 models | `len(PRESEASON_FEATURE_ORDER)`, `len(FEATURE_ORDER)` in `ml-service/app/features.py` |
| XGBoost, trained 2022–24, holdout 2025 | `models/preseason_v1.json` — `framework`, `train_seasons`, `holdout_season` |
| React dashboard | `frontend/src/` — Vite app, 4 components; plus a zero-build HTML dashboard served at `/app` |

**On the latency number:** the cache-miss path makes an HTTP call to the ML service, runs
inference, and writes a row; the hit path skips all three. That structural difference is
what produces the gap and would hold anywhere — but the absolute figures come from a local
single-client benchmark, so present them as such.
