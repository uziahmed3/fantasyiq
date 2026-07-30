# FantasyIQ — how to talk about this project

`docs/INTERVIEW.md` is the component-by-component drill: "justify each index", "why 503
not 500". This document is the layer above it — how to open, the stories worth telling,
and how to handle the hard questions honestly.

Every number here was measured, not estimated. If you cannot reproduce a number live, do
not say it.

---

## The 30-second version

> It's a fantasy football projection platform. A pipeline ingests five seasons of NFL
> play data into Postgres, a FastAPI service exposes projections over a REST API, and a
> separate ML service serves the models. The interesting part isn't the model — it's that
> the whole thing is a system: versioned model artifacts you can roll back with an
> environment variable, a feature contract shared between training and serving so they
> can't drift, and per-player explanations so the projections aren't a black box.

Stop there. Let them pick what to dig into. The most common follow-up is "why two
services", and you have a real answer (below).

## The 2-minute version

Add the shape of the problem, because it's what makes the design non-obvious:

> There are actually two different prediction problems. During the season you have
> rolling form — what did he do the last three games. But in August that data doesn't
> exist, and for a rookie it never exists. Those need different features and a different
> target, so they're two models behind one endpoint. The router picks based on whether
> the player has games yet, not on what week it is — because week 1 for a veteran and
> week 6 for someone returning from injury are the same problem.

That paragraph does a lot of work. It shows you found a real modeling distinction rather
than fitting one model to everything, and "route on data availability, not the calendar"
is a design instinct that generalizes.

---

## What actually exists

Measured on the repo as of the last commit:

| | |
|---|---|
| Python | 86 files, ~11,000 lines |
| Tests | 152 passing, ~2,400 lines across 15 files |
| API | 14 REST endpoints, versioned under `/api/v1` |
| Data | 30,710 weekly stat rows, 1,627 players, 2021–2025 |
| Migrations | 6 Alembic revisions |
| Infra | 79 Terraform resources across 12 `.tf` files |
| CI | 8 jobs — lint, 3 test suites, integration, terraform validate, docker build, compose smoke |
| Services | Postgres, Redis, ML service, API, pipeline, frontend, Prometheus, Grafana |

**Preseason model** (`preseason_v1`, XGBoost, 31 features): trained on 2022–2024, held out
2025 entirely. RMSE 2.83, MAE 2.17, R² 0.67 — **29.9% better than carry-forward**, which
is the honest baseline (just use last season's points per game, what a human does for
free).

**In-season model** (10 features): implemented, tested, chronologically split. Be careful
here — the artifact currently in `models/` is the preseason one. Retrain and read the real
numbers before you quote any. Saying "the in-season path is built and tested but I've only
validated it on backfilled data, never on a live week" is a *good* answer. Quoting a
number you didn't measure is the one thing that ends an interview badly.

### Request flow, the thing you'll be asked to draw

```
Browser ──> FastAPI (api/v1)
              │
              ├─ Repository layer ──> Postgres
              │                        players, player_stats,
              │                        player_context, predictions
              ├─ Cache (Redis, or in-memory) ── cache-aside, TTL
              │
              └─ MLClient ──HTTP──> ML service
                                      registry: versioned artifacts on disk
                                      feature contract validated on load
```

The pipeline writes to Postgres out of band — ingest → clean → context → batch predict.
The API never trains and never calls the model on the hot leaderboard path; the
leaderboard is a pure indexed read of pre-computed rows. Say that out loud, because
"where does the model actually run" is a question people ask to see if you understand
your own latency budget.

---

## The five stories

Interviews are won on specifics. Each of these is a real thing that happened, with a
number attached. Pick the one that fits the question rather than reciting all five.

### 1. The projection that was backwards — *use this for "tell me about a hard bug"*

**Situation.** The draft board had Puka Nacua fifth among receivers. He had the highest
career scoring rate in that group — 24.2 points per game — and the second-highest number
from the previous season. He was ranked below players he beat on every production stat.

**What I did.** Rather than guess, I dumped his full feature row next to the player ranked
above him. The production features all favored Nacua. So I pulled exact per-feature
contributions out of the model — tree SHAP values, which for a single player sum to that
player's prediction. The gap was almost entirely draft position: pick 177 versus pick 20,
worth 2.4 points. More than their entire difference in actual football.

**The root cause.** Draft position is a proxy for talent nobody has measured yet. For a
rookie it's nearly the only signal. For a player with 44 games it's been superseded — you
don't need to guess anymore, you can look. But across the training set high picks
outproduce low picks on average, so the model learned to reward draft position for
*everyone*. With only ~1,300 training rows it was never going to discover the interaction
"use this only when production is missing" by itself.

**The fix.** I made the shrink explicit instead of hoping the model would learn it: draft
capital blends toward league-average in proportion to games played, and is gone by game
32. One function, called from both the training script and the serving path, so they
can't diverge.

**The result.** Nacua moved from WR5 to a four-way tie at WR1–4. Holdout RMSE improved
from 2.886 to 2.835, and lift over the baseline went from 28.6% to 29.9%. Rookie accuracy
was unchanged — they keep their full draft signal, which is correct.

**Why this story lands.** It's not "I read a stack trace." Nothing crashed. The system was
confidently, silently wrong, and finding it required building the tool that could see
inside the model. If they ask one follow-up it'll be *"how did you know it was wrong and
not just surprising?"* — and the answer is that a player can legitimately be ranked below
someone with better stats, but not below someone he beats on *every* input. That's not a
judgment call, it's an ordering violation.

### 2. The model couldn't count above 20 — *use this for "what do you know about ML that isn't the tutorial"*

Same investigation, second finding. Gradient-boosted trees split on thresholds, so they
cannot extrapolate past the top of their training range. Above about 20 points per game
the training rows thin out, the production splits stop separating anyone, and whatever
secondary features still vary — target share, snap share — end up deciding the order.
Exactly the tier a drafter cares most about.

The fix was monotonic constraints: XGBoost's `monotone_constraints` enforces at split time
that increasing a feature can never decrease the output. Production features are
constrained upward, cost and competition features downward. Age is deliberately left
unconstrained, because it genuinely peaks and declines — forcing a direction would encode
a belief I know to be false.

It cost nothing measurable in accuracy and removed a whole class of nonsense. There's a
test that sweeps career scoring rate across its full range against a real trained artifact
and asserts the curve never falls.

### 3. The feature that was quietly lying — *use this for "tell me about a tradeoff"*

I built a `qb_quality` feature: rate each team's quarterback by his fantasy points, since
a receiver's ceiling is capped by his quarterback. It read 2.2 for Minnesota and 0.9 for
Cincinnati — roughly the opposite of reality.

The cause: the pipeline only ingests receiving and rushing statistics. So a quarterback's
"fantasy points" were essentially his scrambles. The feature was structurally meaningless
and the model was free to split on it.

Two options: ingest passing data, or delete the feature. I deleted it — migration 0006
drops the column. A feature that *looks* like signal but isn't is worse than a missing
one, because nothing errors and it silently corrupts every ranking.

**The generalizable point:** I'd rather ship 31 trustworthy inputs than 32 where one is
lying. Interviewers like this because most candidates only talk about adding things.

### 4. Every prediction 500'd in production — *use this for "debugging under pressure"*

The API called the ML service over HTTP and every request failed. Both services were
healthy in isolation.

Cause: `httpx` respects `HTTP_PROXY` / `ALL_PROXY` from the environment by default. On a
corporate network those are set, so internal service-to-service calls — to a host that
resolves fine locally — were being routed out through the proxy and dropped.

Fix: `trust_env=False` on the internal client. But the more useful part of the fix was
what it exposed about error handling. A client-construction failure was surfacing as a
500, and `/ready` could raise. So: dependency failures became 503 (this service is fine,
its dependency isn't — which tells on-call *where to look*), and the readiness probe was
made incapable of throwing, because a health check that 500s is worse than useless.

### 5. I fooled myself with my own test data — *use this for "tell me about a mistake"*

Before real data, I generated synthetic seasons. The model scored RMSE 0.66 and +57%
over baseline. I was pleased, then suspicious — those numbers were too good.

The generator reused one random seed per season, so a player's underlying talent was
*identical* year to year. The model wasn't learning football, it was learning that column
A equals column B. I rewrote it with career arcs and year-to-year shocks. Honest result:
+22%.

Then, once on real data, I caught a second version of the same self-deception: my
evaluation script was comparing the in-season and preseason models in one table and
printing "best: preseason_v1, set ACTIVE_MODEL_VERSION=preseason_v1". Following my own
tooling's advice would have broken the weekly endpoint — the two models have different
feature contracts and different targets and aren't comparable at all. Split into two
tables with separate promotion hints.

**Say the lesson plainly:** a result that looks too good is a bug report about your
evaluation, not a win. I now distrust my own metrics first.

---

## Design decisions you must be able to defend

Do not memorize these. Understand the tradeoff, because they'll push on it.

**Two services instead of one.** Different scaling signals — the model is CPU-bound, the
API is IO-bound — plus independent deploys and model rollback without an API deploy. The
cost is a network hop and a new failure mode, which is exactly why `MLClient` owns
timeouts and maps failures to 503. *If they push:* for this traffic level a monolith would
be fine; I'd justify the split on rollback and deploy independence, not performance.

**Feature contract as a shared tuple.** `FEATURE_ORDER` is the single source of truth.
Training builds its matrix from it, serving builds its row from it, and artifacts record
which contract version they were trained under and refuse to load on mismatch. This is the
standard way training/serving skew kills a model in production — a reordered column
produces no error, just quietly wrong numbers forever.

**Append-only predictions table.** So "how did last week's projections actually do" stays
answerable. Updating in place destroys the record you need to evaluate the model later.
`model_version` on each row is what lets you compare two models across identical weeks.

**Chronological splits, never random.** It's forecasting. A random split lets week 12 leak
into training and inflates the score. The preseason model holds out the most recent season
*entirely* — predicting a season you trained on tells you nothing about next August.

**Cache-aside, with the mode in the key.** Two models serve the same endpoint, so the
cache key includes which one produced the value; otherwise a preseason number could be
served as an in-season one. Same interface backs Redis and an in-memory dict, which is
what lets the whole stack run without Docker.

**Value over replacement on the board.** Raw projected points aren't comparable across
positions — a 15-point tight end is worth more than a 15-point running back because the
replacement-level tight end is so much worse. VOR is what makes "who do I take next"
answerable. This one reads as domain understanding, which is rarer than framework
knowledge.

**Explanations as a separate endpoint.** Attribution walks every tree for every feature.
Folding it into the leaderboard would make every page load pay for explanations of twenty
players when the user wants one. It's a drill-down, cached on (player, season, model
version) because those inputs are fixed.

---

## The limitations — say these before they find them

Volunteering a weakness reads as confidence. Being caught not knowing one reads as the
opposite. Any of these is a good answer to "what would you do next".

- **Age is overweighted.** It contributes about +0.9 for a 23-year-old versus a
  24-year-old, which is too much that far from the peak. It's intentionally unconstrained
  because the age curve is genuinely non-monotonic, so it's free to overfit. I'd bucket it
  or fit an explicit curve.
- **No projected usage.** Every input is last season's actuals plus the current depth
  chart. No ADP, no beat-reported role changes. This matters most in August — exactly when
  a draft board is used — and stops mattering once real games accumulate.
- **`confidence` is not a probability** and doesn't claim to be. It's a heuristic over
  residual spread and how much history a player has. Conformal prediction intervals are
  the real fix.
- **The in-season path has never run on a live week.** Validated on backfilled data only.
- **Single-region, no blue/green.** ECS rolling deploys with health checks; a bad model is
  rolled back by env var, but a bad *image* is a redeploy.
- **AWS infra is written and validated, not continuously running** — say this plainly if
  asked whether it's deployed. `terraform validate` runs in CI. Claiming production uptime
  you don't have is the fastest way to lose the room.

---

## Questions you should expect

**"Why is the model any good? Anyone can predict last season's average."**
That's precisely the baseline I measure against, and it's a hard one — carry-forward gets
RMSE 4.04. The model gets 2.83, about 30% better, on a season it never saw. The gain comes
from weighting a player's whole career by recency, games played and an age curve, rather
than trusting one season. Justin Jefferson had one down year after four strong ones;
last-season-only ranked him 40th, the career features put him back in the top ten.

**"How do you know it's not overfitting?"**
The holdout is a whole season the model never saw, not a random slice. And it's compared
against a baseline on the same holdout, so improvement is relative to something real
rather than an absolute number I could tune.

**"What would you change if this had 1,000× the traffic?"**
The leaderboard is already a pre-computed indexed read, so it scales with read replicas.
The per-player explain endpoint is the expensive path — I'd precompute the top drivers
during the batch job and store them, turning it into a read too. Beyond that, partition
`player_stats` by season, since every query filters on it.

**"Did you build this yourself, or did AI write it?"**
Answer honestly — everyone is using these tools and pretending otherwise is a bad look.
What matters is whether you can defend the decisions, and you can: why draft capital
decays over 32 games, why the split is chronological, why `qb_quality` was deleted rather
than patched. A good framing: *"I used AI heavily as a pair programmer. The architecture
decisions and the debugging are mine — the Nacua bug is a good example, because no tool
tells you the ranking is wrong. You have to notice it and go find out why."* Then offer to
walk through the code. Nobody who wrote nothing can do that.

---

## Before the interview

1. **Run it.** `python local.py`, open the dashboard, click a player, see the breakdown.
   Being able to demo beats any description.
2. **Retrain and read the real numbers** so every figure you quote is one you've seen
   today, especially the in-season model.
3. **Be able to open three files from memory:** `ml-service/app/features.py` (the
   contract), `backend/app/services/predictions.py` (the request path),
   `pipeline/career.py` (the career weighting). "Let me show you" beats a rehearsed
   paragraph.
4. **Practice the Nacua story out loud until it's 90 seconds.** It's your best material
   and it's currently too long.
5. **Read `docs/INTERVIEW.md`** for the component-level drill — indexes, HTTP status
   codes, cache strategy, CI.

---

## Resume bullets, with the numbers backed

> Built a fantasy football analytics platform (FastAPI, PostgreSQL, Redis, Docker, AWS)
> serving REST projections from versioned XGBoost models, with a two-model architecture
> routing on data availability so rookies and week-1 players are handled by a separate
> preseason model.

> Engineered career-history features weighted by recency, games played and an age curve,
> improving holdout accuracy 29.9% over a carry-forward baseline (RMSE 4.04 → 2.83) on a
> season fully held out from training.

> Diagnosed a systematic ranking error using per-player SHAP attribution — the model was
> penalizing established players for draft position years after their production made it
> irrelevant — and fixed it with an explicit evidence-weighted decay, improving accuracy
> while removing the bias.

> Built automated ingestion of five NFL seasons (30k+ weekly records) with schema-drift
> detection, and a shared feature contract validated at both training and serving to
> prevent skew; 152 tests across 8 CI jobs.
