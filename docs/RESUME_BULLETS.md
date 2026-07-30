# Resume bullets — FantasyIQ (software engineering roles)

Written for general SWE and backend roles. The machine learning is present because it's
differentiating, but it's framed as engineering — interfaces, dispatch, correctness,
performance — because that's what a SWE screener is reading for.

Every number was measured on this repo. Sources are at the bottom. **Do not use a number
you haven't re-run.**

---

## Recommended — 4 bullets

> **FantasyIQ — Full-Stack NFL Analytics Platform** · [github.com/uziahmed3/fantasyiq](https://github.com/uziahmed3/fantasyiq)
> *Python, JavaScript, FastAPI, PostgreSQL, Redis, React, Docker, pytest, GitHub Actions*
>
> - Built a 3-service application (REST API, inference service, ETL pipeline) with a React
>   dashboard, exposing 14 versioned endpoints over a normalized PostgreSQL schema of 30K+
>   records; designed idempotent upserts so re-runs are safe and an append-only prediction
>   log so past outputs stay auditable.
> - Designed a dispatch layer that selects between two prediction strategies at request
>   time based on data availability, behind a single endpoint, plus a versioned interface
>   contract validated at the service boundary — incompatible artifacts fail fast on load
>   instead of silently returning wrong results.
> - Cut repeat-request latency 92% (101ms → 8ms) with a cache-aside layer behind a
>   pluggable Redis/in-memory interface, eliminating a cross-service HTTP call and model
>   inference per request; the same abstraction lets the stack run with or without Redis.
> - Wrote 152 tests and a 7-job CI pipeline (lint, 3 test suites, integration, image build,
>   full-stack smoke test), including a migration forward/backward check that fails the
>   build when the ORM and schema drift apart.

## Optional 5th bullet — the debugging one

Add this if the project gets 5 lines. It's the most distinctive thing on the page, because
almost no new-grad resume describes finding a bug that didn't throw an error.

> - Diagnosed a silent correctness defect in production rankings — no exception, just wrong
>   output — by building a per-feature attribution endpoint that decomposed each result into
>   the inputs that produced it; the fix improved prediction accuracy 29.9% over baseline
>   (RMSE 4.04 → 2.83).

## Compact — 3 bullets

> - Built a 3-service full-stack application (FastAPI, PostgreSQL, Redis, React, Docker)
>   serving 14 versioned REST endpoints over a normalized 30K-record schema, with 152 tests
>   and 7-job CI on every push.
> - Designed request-time dispatch between two prediction strategies behind one endpoint,
>   with a versioned interface contract validated at the service boundary so incompatible
>   artifacts fail fast rather than silently returning wrong results.
> - Cut repeat-request latency 92% (101ms → 8ms) via a cache-aside layer behind a pluggable
>   backend interface, and diagnosed a silent correctness bug by building per-result
>   attribution tooling.

## Ultra-compact — 2 bullets

When the project is one of five and space is tight.

> - Built a 3-service full-stack NFL analytics platform (FastAPI, PostgreSQL, Redis, React,
>   Docker) — 14 REST endpoints, 30K-record normalized schema, 152 tests, 7-job CI.
> - Cut repeat-request latency 92% (101ms → 8ms) with a pluggable cache-aside layer, and
>   improved prediction accuracy 29.9% over baseline after diagnosing a silent correctness
>   bug with custom attribution tooling.

---

## Why they're written this way

**The ML is framed as engineering.** "Two-model architecture" sounds like a data science
project. "Request-time dispatch between two strategies behind one endpoint" is the same
fact described as software design, and it's what a SWE interviewer is actually evaluating.
Same for the feature contract: it's an interface validated at a boundary with fail-fast on
mismatch, which is a correctness argument, not an ML one.

**The bullets name decisions, not tools.** "Used Redis for caching" is a tool list.
"Cache-aside behind a pluggable interface, cutting latency 92% by eliminating a
cross-service call" says you knew why, chose a pattern, and measured the result. That's the
difference between a bullet that survives follow-up and one that doesn't.

**Correctness gets equal billing with features.** Idempotent upserts, an append-only audit
log, fail-fast contract validation, and a migration drift check are all about *not being
silently wrong*. Most new-grad resumes are entirely about things built and never about
things kept correct — and senior engineers reading resumes notice that.

**The keywords are all defensible.** Python, JavaScript, FastAPI, PostgreSQL, Redis, React,
Docker, pytest, CI/CD, REST — every one is something you did and can discuss. That's the
same reason Terraform and AWS were removed from the project entirely rather than kept as
resume decoration.

**Each bullet invites a question you can answer:**

| Bullet | The question it invites | Where your answer lives |
|---|---|---|
| 3-service platform | "Why not one service?" | Guide, Part 5 |
| Dispatch + contract | "Why two models?" | Guide, Part 2 — your best material |
| Cache-aside 92% | "Why cache-aside, not write-through?" | `INTERVIEW.md`, Caching |
| Tests + CI | "What does your CI actually catch?" | `INTERVIEW.md`, CI/CD |
| Attribution bug | "Walk me through it." | Guide, Part 4, story 1 |

---

## If a job description mentions ML

Most SWE postings don't, and leading with ML for a SWE role works against you. But if the
posting explicitly asks for it, swap bullet 2 for:

> - Engineered 31 features including career production weighted by recency, games played
>   and an age curve, improving accuracy 29.9% over a carry-forward baseline (RMSE 4.04 →
>   2.83, R² 0.67) with season-level holdout validation to prevent leakage; applied
>   monotonic constraints to eliminate a systematic ranking bias.

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
| 31 features / 2 strategies | `len(PRESEASON_FEATURE_ORDER)`, `len(FEATURE_ORDER)` in `ml-service/app/features.py` |
| React dashboard | `frontend/src/` — Vite app, 4 components; plus a zero-build HTML dashboard served at `/app` |

**On the latency number:** the cache-miss path makes an HTTP call to the ML service, runs
inference, and writes a row; the hit path skips all three. That structural difference is
what produces the gap and would hold anywhere — but the absolute figures come from a local
single-client benchmark, so present them as such.
