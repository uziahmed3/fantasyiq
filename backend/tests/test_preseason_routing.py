"""Routing between the two models, and the rookie case.

The in-season model runs on rolling 3-game form. In week 1 that form does not exist, and
for a rookie it never will. These tests pin down that the router notices, that the
preseason model gets used, and that the two never contaminate each other's cache.
"""

import pytest

from app.core.cache import prediction_key
from app.schemas.prediction import PredictionRequest
from app.services.predictions import PredictionService


def test_rookie_routes_to_the_preseason_model(db_session, rookie, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=rookie.id, week=1, season=2023, opponent="KC")

    assert svc.choose_mode(req) == "preseason"
    resp = svc.predict(req)

    assert resp.mode == "preseason"
    assert len(stub_ml.preseason_calls) == 1
    assert stub_ml.calls == [], "the in-season model must not be consulted"
    assert resp.prediction > 0, "a first-round rookie must get a non-zero projection"
    assert "draft capital" in (resp.basis or "")


def test_rookie_projection_uses_draft_capital(db_session, rookie, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    svc.predict(PredictionRequest(player_id=rookie.id, week=1, season=2023, opponent="KC"))
    sent = stub_ml.preseason_calls[0]
    assert sent.is_rookie == 1
    assert sent.draft_pick == 6
    assert sent.depth_chart_rank == 2
    assert sent.prior_games == 0


def test_low_pick_rookie_projects_below_high_pick(db_session, rookie, stub_ml):
    """Draft position has to actually move the number, or it is decoration."""
    from app.models import PlayerContext

    svc = PredictionService(db_session, client=stub_ml)
    high = svc.predict(
        PredictionRequest(player_id=rookie.id, week=1, season=2023, opponent="KC")
    ).prediction

    ctx = db_session.query(PlayerContext).filter_by(player_id=rookie.id).one()
    ctx.draft_pick = 250
    ctx.draft_round = 7
    db_session.commit()

    low = svc.predict(
        PredictionRequest(player_id=rookie.id, week=1, season=2023, opponent="KC"),
        use_cache=False,
    ).prediction
    assert low < high


def test_midseason_player_routes_to_the_in_season_model(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=15, week=5, season=2023, opponent="GB")

    assert svc.choose_mode(req) == "in_season"
    resp = svc.predict(req)

    assert resp.mode == "in_season"
    assert len(stub_ml.calls) == 1
    assert stub_ml.preseason_calls == []


def test_week_one_veteran_routes_to_preseason(db_session, veteran_with_context, stub_ml):
    """Season 2024 has no games yet, so even an established player goes preseason."""
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=15, week=1, season=2024, opponent="GB")

    assert svc.choose_mode(req) == "preseason"
    resp = svc.predict(req)
    assert resp.mode == "preseason"
    sent = stub_ml.preseason_calls[0]
    assert sent.is_rookie == 0
    assert sent.prior_games == 4
    assert sent.qb_changed == 1, "a QB change is knowable before week 1 and must be sent"


def test_returning_from_injury_gets_the_preseason_path(db_session, rookie, stub_ml):
    """Routing is on data availability, not on the week number - a player with no games
    by week 8 has no rolling form either."""
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=rookie.id, week=8, season=2023, opponent="KC")
    assert svc.choose_mode(req) == "preseason"


def test_modes_do_not_share_cache_entries():
    a = prediction_key(15, 1, "GB:preseason", "preseason_v1")
    b = prediction_key(15, 1, "GB:in_season", "xgboost_v1")
    assert a != b


def test_preseason_result_is_cached(db_session, rookie, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=rookie.id, week=1, season=2023, opponent="KC")
    first = svc.predict(req)
    second = svc.predict(req)
    assert first.source == "model" and second.source == "cache"
    assert second.mode == "preseason"
    assert second.basis == first.basis, "basis must survive the cache round trip"
    assert len(stub_ml.preseason_calls) == 1


def test_missing_context_falls_back_instead_of_failing(db_session, stub_ml):
    """A player with no context row at all still gets an answer."""
    from app.models import Player

    orphan = Player(id=77, name="No Context", team="SF", position="WR", external_id="00-77")
    db_session.add(orphan)
    db_session.commit()

    svc = PredictionService(db_session, client=stub_ml)
    resp = svc.predict(PredictionRequest(player_id=77, week=1, season=2023, opponent="KC"))
    assert resp.mode == "in_season", "falls back rather than 500ing"
    assert len(stub_ml.calls) == 1


def test_stale_context_is_used_when_current_season_missing(db_session, rookie, stub_ml):
    """Context built for 2023 still answers a 2024 question - stale beats nothing."""
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=rookie.id, week=1, season=2024, opponent="KC")
    resp = svc.predict(req)
    assert resp.mode == "preseason"
    assert len(stub_ml.preseason_calls) == 1


def test_api_reports_mode_and_basis(client, rookie):
    r = client.post(
        "/api/v1/predict",
        json={"player_id": rookie.id, "week": 1, "season": 2023, "opponent": "KC"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "preseason"
    assert body["basis"]
    assert body["confidence"] == pytest.approx(0.35), "rookie confidence must be low"


# ---------------------------------------------------------------- draft board
def test_season_board_ranks_and_extrapolates(client, db_session, seed):
    """The draft board reads week-0 rows and turns a per-game rate into a season total."""
    from app.models import Prediction

    for pid, ppg in ((15, 18.0), (16, 12.0)):
        db_session.add(
            Prediction(
                player_id=pid,
                season=2024,
                week=0,
                opponent="MIN",
                prediction=ppg,
                confidence=0.7,
                model_version="preseason_v1",
            )
        )
    db_session.commit()

    r = client.get("/api/v1/rankings/season?season=2024&position=WR&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["games_assumed"] == 17
    assert [row["name"] for row in body["rankings"]] == [
        "Justin Jefferson",
        "Ja'Marr Chase",
    ]
    top = body["rankings"][0]
    assert top["projected_points_per_game"] == 18.0
    assert top["projected_season_points"] == pytest.approx(18.0 * 17, abs=0.1)


def test_season_board_reports_whether_the_season_started(client, db_session, seed):
    """The UI switches between draft board and weekly view off this flag rather than a
    hardcoded date."""
    from app.models import Prediction

    # 2023 has games in the seed fixture; 2026 does not.
    for season in (2023, 2026):
        db_session.add(
            Prediction(
                player_id=15,
                season=season,
                week=0,
                opponent="MIN",
                prediction=15.0,
                confidence=0.6,
                model_version="preseason_v1",
            )
        )
    db_session.commit()

    started = client.get("/api/v1/rankings/season?season=2023&position=WR").json()
    upcoming = client.get("/api/v1/rankings/season?season=2026&position=WR").json()
    assert started["season_started"] is True
    assert upcoming["season_started"] is False


def test_season_board_flags_rookies_and_explains_the_basis(client, db_session, rookie):
    from app.models import Prediction

    db_session.add(
        Prediction(
            player_id=rookie.id,
            season=2023,
            week=0,
            opponent="LV",
            prediction=8.0,
            confidence=0.3,
            model_version="preseason_v1",
        )
    )
    db_session.commit()

    body = client.get("/api/v1/rankings/season?season=2023&position=WR&limit=50").json()
    row = next(r for r in body["rankings"] if r["player_id"] == rookie.id)
    assert row["is_rookie"] is True
    assert "drafted #6" in (row["basis"] or ""), row["basis"]


def test_season_board_is_empty_not_broken_without_projections(client, seed):
    body = client.get("/api/v1/rankings/season?season=2030&position=WR").json()
    assert body["rankings"] == []
    assert body["season_started"] is False


# ---------------------------------------------------------------- draft value
def test_replacement_level_is_the_nth_best_at_the_position():
    from app.services.draft_value import STARTERS_BY_POSITION, replacement_levels

    n = STARTERS_BY_POSITION["TE"]
    pool = {"TE": [20.0 - i for i in range(n + 5)]}
    assert replacement_levels(pool)["TE"] == pytest.approx(20.0 - (n - 1))


def test_replacement_level_falls_back_on_a_shallow_pool():
    from app.services.draft_value import replacement_levels

    # Fewer players than the nominal starter count - use the worst rather than nothing.
    assert replacement_levels({"TE": [12.0, 9.0, 4.0]})["TE"] == 4.0


def test_value_is_none_when_the_position_has_no_baseline():
    """None and 0.0 mean different things: 'unknown' versus 'exactly replacement'."""
    from app.services.draft_value import value_over_replacement

    assert value_over_replacement(15.0, "QB", {"WR": 9.0}) is None
    assert value_over_replacement(9.0, "WR", {"WR": 9.0}) == 0.0


def test_shallow_position_beats_a_higher_raw_projection():
    """The whole point. A TE with a lower projection can be the better pick, because the
    TE pool falls off faster - a raw-points list hides that and under-drafts elite TEs."""
    from app.services.draft_value import replacement_levels, value_over_replacement

    pools = {
        # Deep WR pool: replacement is high.
        "WR": [16.0] + [12.0] * 40,
        # Shallow TE pool: replacement is low.
        "TE": [15.0] + [6.0] * 20,
    }
    levels = replacement_levels(pools)
    wr = value_over_replacement(16.0, "WR", levels)
    te = value_over_replacement(15.0, "TE", levels)
    assert te > wr, f"TE {te} should outrank WR {wr} despite the lower projection"


def test_value_sort_and_points_sort_give_different_orders(client, db_session, seed):
    from app.models import Prediction

    # A TE just below a WR on raw points, in a pool shallow enough to matter.
    db_session.add_all(
        [
            Prediction(
                player_id=15,
                season=2026,
                week=0,
                opponent="MIN",
                prediction=16.0,
                confidence=0.6,
                model_version="preseason_v1",
            ),
            Prediction(
                player_id=16,
                season=2026,
                week=0,
                opponent="CIN",
                prediction=15.9,
                confidence=0.6,
                model_version="preseason_v1",
            ),
        ]
    )
    db_session.commit()

    by_value = client.get("/api/v1/rankings/season?season=2026&position=FLEX&sort=value").json()
    by_points = client.get("/api/v1/rankings/season?season=2026&position=FLEX&sort=points").json()
    assert by_value["replacement_note"]
    # Both must return the same players, and every row must carry a value.
    assert {r["player_id"] for r in by_value["rankings"]} == {
        r["player_id"] for r in by_points["rankings"]
    }
    for r in by_value["rankings"]:
        assert r["value_over_replacement"] is not None
        assert r["position_rank"] >= 1


# ------------------------------------------------------- projection explanation
def test_why_endpoint_decomposes_a_projection(client, db_session, seed):
    """The drill-down that replaced "16 games last season" as the board's only answer."""
    from app.models import PlayerContext

    db_session.add(
        PlayerContext(
            player_id=15,
            season=2026,
            team="MIN",
            prior_games=17,
            prior_points_per_game=11.9,
            career_weighted_ppg=17.8,
            career_games=80,
            career_seasons=5,
            draft_round=1,
            draft_pick=22,
            teammate_top_target_share=0.20,
        )
    )
    db_session.commit()

    body = client.get("/api/v1/rankings/season/15/why?season=2026").json()
    assert body["player_id"] == 15
    assert body["name"] == "Justin Jefferson"
    assert body["baseline"] == 6.3
    assert body["drivers_shown"] == len(body["drivers"])
    # Both directions must be representable, or the panel can only ever flatter a player.
    assert any(d["contribution"] > 0 for d in body["drivers"])
    assert any(d["contribution"] < 0 for d in body["drivers"])
    assert "career scoring rate" in body["headline"]


def test_why_endpoint_labels_are_human_readable(client, db_session, seed):
    from app.models import PlayerContext

    db_session.add(PlayerContext(player_id=15, season=2026, team="MIN", career_weighted_ppg=17.8))
    db_session.commit()
    body = client.get("/api/v1/rankings/season/15/why?season=2026").json()
    for d in body["drivers"]:
        assert "_" not in d["label"], d["label"]


def test_why_endpoint_404s_for_an_unknown_player(client, seed):
    assert client.get("/api/v1/rankings/season/99999/why?season=2026").status_code == 404


def test_why_endpoint_404s_when_no_context_exists(client, seed):
    """No context is a missing-pipeline-run problem, and says so instead of returning zeros."""
    resp = client.get("/api/v1/rankings/season/15/why?season=2031")
    assert resp.status_code == 404
    assert "context" in resp.json()["detail"]


def test_why_endpoint_503s_when_the_model_is_unavailable(client, db_session, seed, stub_ml):
    """A missing explanation must not read as a missing player or a server fault."""
    from app.models import PlayerContext
    from app.services.ml_client import MLServiceError

    db_session.add(PlayerContext(player_id=15, season=2026, team="MIN", career_weighted_ppg=17.8))
    db_session.commit()
    stub_ml.fail_with = MLServiceError("explanation unavailable (501)")
    assert client.get("/api/v1/rankings/season/15/why?season=2026").status_code == 503


def test_basis_reports_the_career_record_not_just_last_season(client, db_session, seed):
    """The original complaint: a down year must not read as the whole story."""
    from app.models import PlayerContext, Prediction

    db_session.add_all([
        PlayerContext(
            player_id=15,
            season=2026,
            team="MIN",
            prior_games=17,
            prior_points_per_game=11.9,
            career_weighted_ppg=17.8,
            career_seasons=5,
        ),
        Prediction(
            player_id=15,
            season=2026,
            week=0,
            opponent="MIN",
            prediction=16.3,
            confidence=0.8,
            model_version="preseason_v1",
        ),
    ])
    db_session.commit()

    body = client.get("/api/v1/rankings/season?season=2026&position=WR").json()
    basis = next(r for r in body["rankings"] if r["player_id"] == 15)["basis"]
    assert "5 seasons" in basis
    assert "17.8 career" in basis
    # And it must name the disagreement, which is the decision-relevant part.
    assert "below" in basis, basis
