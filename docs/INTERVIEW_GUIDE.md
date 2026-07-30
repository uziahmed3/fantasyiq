# FantasyIQ — how to talk about this project

`docs/INTERVIEW.md` is the component-by-component drill: "justify each index", "why 503
not 500". This document is the layer above it — how to open, the stories worth telling,
and how to handle the hard questions honestly.

Every number here was measured, not estimated. If you cannot reproduce a number live, do
not say it.

**If you only read one thing before an interview, read Part 2** — the script for "tell me
about your project", which you will be asked every single time.

1. [What the project actually is](#part-1--what-the-project-actually-is) — the domain and
   the problem, in language a non-player understands
2. [**"So, tell me about your project"**](#part-2--so-tell-me-about-your-project) — the
   script, the follow-up hooks, and how to adjust by audience
3. [The system in detail](#part-3--the-system-in-detail) — measured facts, request flow
4. [The five stories](#part-4--the-five-stories) — specifics that win interviews
5. [Defending the decisions](#part-5--defending-the-decisions) — tradeoffs and limitations
6. [Q&A, prep, and bullets](#part-6--qa-prep-and-bullets)

---

# Part 1 — What the project actually is

## The domain, for someone who has never played

Do not assume the interviewer knows fantasy football. Plenty of engineers don't, and if
you open with "target share" and "PPR scoring" you lose them in the first ten seconds.
Two sentences is all it takes:

> In fantasy football you draft real NFL players onto an imaginary team, and they score
> you points based on what they actually do in real games that week — catches, yards,
> touchdowns. So the whole game is a prediction problem: you're trying to pick the players
> who will score the most over the season.

That's enough. If they nod like they know it, skip straight to the problem. If they look
blank, one more line lands it:

> You draft a roster before the season, and then every week you choose which of your
> players to actually start, because you have more players than starting slots.

## The problem it solves — two decisions, not one

This is the framing to lead with, because the architecture falls straight out of it. A
fantasy player makes exactly two kinds of decision, and they are genuinely different
problems:

**1. Draft day, once a year — "who do I take?"**
You pick your roster in August and you're mostly stuck with it, so a bad pick costs you
for four months. Nobody has played a game yet.

**2. Every week of the season — "who do I start and who do I bench?"**
You have, say, five receivers and can only start three. Every week you pick. Now you *do*
have current-season data — you can see who's been getting the ball lately.

The app does both. It opens as a **draft board** before the season — every skill-position
player ranked, with a value score comparable across positions — and becomes a **weekly
start/sit tool** once real games exist, projecting each player's points for the upcoming
week against that week's opponent.

**The switch is automatic.** It routes on whether the player has games yet, not on the
calendar. That detail matters more than it sounds: week 1 for a veteran, and week 6 for
someone just back from injury, are the same problem — no recent form to read. A
date-based rule gets that wrong.

### Why either one is hard

- **Last season lies.** The obvious draft approach is to rank by what everyone scored last
  year. But one injury or one down year buries a genuinely great player. Justin Jefferson
  scored 19.5, 21.5, 20.4, 18.6, then 11.9 — ranking on that last number puts him 40th.
  He's not the 40th best receiver.
- **Rookies have no data at all.** Every year a chunk of the draft pool has never played
  an NFL game, and those are often the picks people most want help with.
- **Week to week is noisy.** A receiver can score 4 points and then 28 with no change in
  his role, because one ball happened to reach the end zone. That's why the weekly model
  carries volume — targets, receptions, yards over the last three games — alongside points
  rather than points alone. Volume is what actually persists; touchdowns are close to
  coin-flips at this sample size.
- **Points aren't comparable across positions.** A 15-point tight end is worth more than
  a 15-point running back, because the *next best available* tight end is so much worse.
  Raw projections answer "who scores more", not "who do I take".
- **A number with no reason behind it is useless.** If the tool says start player A over
  player B and won't say why, nobody trusts it — and they shouldn't.

## Why it's worth building as a system

The honest framing, and it plays well: **the model is the small part.** Predicting fantasy
points is a regression problem you could prototype in a notebook in an afternoon. What
makes it a real project is everything around it:

- Data has to arrive automatically and survive the source changing its schema on you.
- The model has to be *served* — versioned, rollback-able, behind an API, with the same
  feature code at training and serving time or it silently drifts.
- Projections have to be fast enough to page through, which means precomputing in batch
  rather than calling a model per row.
- It has to explain itself, or nobody uses it.

That's the sentence to have ready: *"the part I'd defend in a code review isn't the model,
it's the system around it."*

---

# Part 2 — "So, tell me about your project"

You will get this every single time. It's the most rehearsable question in the interview
and most people waste it by either rambling for four minutes or giving a one-line answer
that kills the conversation.

**The shape that works:** domain → problem → what you built → the one interesting thing.
Sixty to ninety seconds, then stop and let them steer. You are not trying to say
everything. You are trying to leave three or four hooks they'll want to pull on.

### The script

> *(domain)* It's a fantasy football projection platform. In fantasy football you draft
> real NFL players and score points based on what they actually do in real games.
>
> *(problem)* There are two decisions you make. In August you draft a roster, and then
> every week you decide which of your players to start, because you have more players than
> starting slots. So the app does both — it's a draft board before the season, and a
> weekly start/sit tool once the season begins.
>
> *(what you built)* It's three services. A pipeline that ingests five seasons of NFL data
> into Postgres — about 30,000 weekly stat lines — a FastAPI backend that serves
> projections over a REST API with Redis caching, and a separate ML service that loads
> versioned XGBoost models off disk. The whole thing runs on Docker Compose.
>
> *(the interesting thing)* Those two decisions turn out to be genuinely different
> prediction problems. Week to week you have recent form — what did he do the last three
> games, who's he playing. In August none of that exists, and for a rookie it never
> exists, so you're working from career history, draft position and where he sits on the
> depth chart. Different features, different targets. So it's two models behind one
> endpoint, and the router picks based on whether the player has games yet rather than
> what week it is — because week 1 for a veteran and week 6 for someone back from injury
> are actually the same problem.

Then **stop**. The silence is doing work — it hands them the wheel.

**Why this version is better than leading with the draft board.** "Two decisions, so two
models" makes the architecture sound like it followed the product, which is what good
design looks like. Leading with only the draft board makes the second model sound like an
afterthought you bolted on.

### The hooks you just planted

Each of these is something they can pull, and you have a real answer for all of them:

| If they ask… | Go to |
|---|---|
| "Why two services?" | Different scaling signals, independent deploys, model rollback without an API deploy — *and* the honest caveat that a monolith would be fine at this traffic |
| "How accurate is it?" | 29.9% better than carry-forward on a fully held-out season, RMSE 4.04 → 2.83 |
| "How do you handle rookies?" | Draft capital and depth-chart position — the only real signal in August — and the decay story |
| "What was hard?" | The Nacua bug. This is the one you want. |
| "Why XGBoost?" | Tabular data, small dataset, and it supports monotonic constraints, which I needed |
| "How does the weekly one work?" | 10 features: 3-game rolling targets/receptions/yards/TDs/points, last game, season average, games played, opponent rank, home/away |
| "Is it deployed?" | Written and validated in CI, not continuously running. Say it plainly. |

If they ask an open "what was the hardest part" — **always go to the Nacua bug**. It's the
strongest thing you have.

### Reading the room

**Recruiter or non-technical screen.** Stop after the problem and one sentence on what you
built. Say "fantasy football app that predicts how many points a player will score — so
it helps you draft, and then helps you pick who to start each week. Full-stack, with a
machine learning model behind it." Don't say Postgres. They're checking it's real and that
you can explain things to normal people — the second part is actually the test.

**Engineer or technical screen.** Full script. Lead with the architecture, keep the domain
to one sentence, and get to the two-model routing fast — that's the part that reads as
design thinking rather than tutorial-following.

**Hiring manager.** Emphasize *why* over *what*. The interesting version for them is that
you changed the model's target once you understood how it would actually be used: the
original predicted the first four weeks of the season, which was wrong for a draft board,
because someone drafting is buying a whole season, not September. Changing the target to
the full regular season was the single biggest accuracy improvement in the project — RMSE
3.56 → 2.89. That's a story about understanding the user, and it's rare in a new grad.

### Two things not to do

**Don't lead with the ML.** Every new grad project leads with "I trained a model." Leading
with the system is what differentiates you for a backend role, and it's also just true
here — the model is a few hundred lines and the system is ten thousand.

**Don't oversell it as production.** "It's deployed and serving live traffic" is a claim you
cannot back up and it takes about one follow-up question to unravel. "It runs on Docker
Compose and CI boots the full stack and exercises the API on every push" is honest, still
impressive, and unfalsifiable because it's true. If they ask how you'd deploy it, answer as
a design question — you have opinions about scaling the API on request count and the ML
service on CPU, and framing it as a plan rather than a claim is the correct move.

---

# Part 3 — The system in detail

## What actually exists

Measured on the repo as of the last commit:

| | |
|---|---|
| Python | 86 files, ~11,000 lines |
| Tests | 152 passing, ~2,400 lines across 15 files |
| API | 14 REST endpoints, versioned under `/api/v1` |
| Data | 30,710 weekly stat rows, 1,627 players, 2021–2025 |
| Migrations | 6 Alembic revisions |
| CI | 7 jobs — lint, 3 test suites, integration, docker build, compose smoke |
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

### The two paths through the system

Worth drawing separately, because they have different performance characteristics and
interviewers like the distinction:

```
DRAFT BOARD (preseason)              START/SIT (in-season)
one batch job, offseason             batch job, every week
    |                                    |
career history, draft capital,       last 3 games, opponent rank,
depth chart, team situation          home/away, season average
    |                                    |
preseason model (31 features)        in-season model (10 features)
    |                                    |
predictions @ week 0 ------------> predictions @ week N
    |                                    |
    +----------- both read by ------------+
                GET /rankings
```

Both write to the same append-only `predictions` table; week 0 is a sentinel meaning "the
season as a whole". That's why one leaderboard endpoint serves both, and why comparing a
projection to what actually happened works identically for either model.

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

# Part 4 — The five stories


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

# Part 5 — Defending the decisions

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
- **It is not deployed anywhere.** It runs on Docker Compose, and CI boots the full stack
  on every push. Say that plainly if asked — claiming production uptime you don't have is
  the fastest way to lose the room. "How would you deploy it" is a fair question and you
  should have an answer; "it is deployed" is a claim you'd have to walk back.
- **Model rollback is solved, image rollback isn't.** A bad model is an env var and a
  restart because artifacts are versioned on a volume. A bad build is a manual redeploy.

---

# Part 6 — Q&A, prep, and bullets

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

## Resume bullets

Moved to **`RESUME_BULLETS.md`** — four-bullet and three-bullet versions, variants for
ML-leaning and infra-leaning roles, the claims to avoid, and a table for reproducing every
number before you cite it.
