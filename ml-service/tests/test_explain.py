"""Tests for draft-capital decay, monotonic constraints and per-feature attribution.

These cover the three changes that fixed a wrong ranking rather than a crash, which is
the harder kind of bug to keep fixed: nothing throws when a model quietly penalises a
player for a three-year-old draft position, so only an assertion about the ordering
itself will catch it coming back.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.explain import LABELS, describe, format_value, headline, label
from app.features import (
    DRAFT_EVIDENCE_GAMES,
    NEUTRAL_DRAFT_PICK,
    NEUTRAL_DRAFT_ROUND,
    PRESEASON_DEFAULTS,
    PRESEASON_FEATURE_ORDER,
    decay_draft_capital,
    draft_capital_weight,
    preseason_row,
)


# ------------------------------------------------------------------ draft decay
def test_rookie_keeps_full_draft_capital():
    """With no games played, draft position is the only evidence there is."""
    assert draft_capital_weight(0) == 1.0
    assert decay_draft_capital(1, 5, 0) == (1.0, 5.0)


def test_draft_capital_fully_gone_after_two_seasons():
    assert draft_capital_weight(DRAFT_EVIDENCE_GAMES) == 0.0
    assert draft_capital_weight(DRAFT_EVIDENCE_GAMES + 50) == 0.0
    rnd, pick = decay_draft_capital(7, 250, 44)
    assert (rnd, pick) == (NEUTRAL_DRAFT_ROUND, NEUTRAL_DRAFT_PICK)


def test_established_late_pick_is_not_penalised_relative_to_early_pick():
    """The Nacua case, reduced to its essentials.

    Two players with 44 career games, one a first-round pick and one a fifth-rounder.
    After decay they must look identical to the model on draft capital, because three
    seasons of tape has answered the question draft position was standing in for.
    """
    early = decay_draft_capital(1, 5, 44)
    late = decay_draft_capital(5, 177, 44)
    assert early == late


def test_decay_is_gradual_not_a_cliff():
    halfway = draft_capital_weight(DRAFT_EVIDENCE_GAMES / 2)
    assert halfway == pytest.approx(0.5)
    rnd, _ = decay_draft_capital(1, 1, DRAFT_EVIDENCE_GAMES / 2)
    # Halfway between an actual first-rounder and neutral.
    assert rnd == pytest.approx(0.5 * 1 + 0.5 * NEUTRAL_DRAFT_ROUND)


def test_missing_career_games_is_treated_as_no_evidence():
    """Absent history must not silently grant a veteran discount to an unknown player."""
    assert draft_capital_weight(None) == 1.0
    assert draft_capital_weight(float("nan")) == 1.0


def test_serving_path_applies_the_decay():
    """preseason_row must not hand raw draft position to a model trained on decayed."""
    feats = dict.fromkeys(PRESEASON_FEATURE_ORDER, 0.0)
    feats.update(draft_round=7, draft_pick=250, career_games=60)
    row = preseason_row(feats)[0]
    idx = {n: i for i, n in enumerate(PRESEASON_FEATURE_ORDER)}
    assert row[idx["draft_round"]] == pytest.approx(NEUTRAL_DRAFT_ROUND)
    assert row[idx["draft_pick"]] == pytest.approx(NEUTRAL_DRAFT_PICK)


def test_serving_path_leaves_a_rookies_draft_position_alone():
    feats = dict.fromkeys(PRESEASON_FEATURE_ORDER, 0.0)
    feats.update(draft_round=1, draft_pick=3, career_games=0, is_rookie=1)
    row = preseason_row(feats)[0]
    idx = {n: i for i, n in enumerate(PRESEASON_FEATURE_ORDER)}
    assert row[idx["draft_round"]] == pytest.approx(1.0)
    assert row[idx["draft_pick"]] == pytest.approx(3.0)


# ------------------------------------------------------- monotonic constraints
def test_monotone_map_covers_only_real_features():
    from train.train_preseason import MONOTONE, monotone_constraints

    assert set(MONOTONE) <= set(PRESEASON_FEATURE_ORDER)
    assert set(MONOTONE.values()) <= {-1, 1}
    parsed = monotone_constraints().strip("()").split(",")
    assert len(parsed) == len(PRESEASON_FEATURE_ORDER)
    # Positional: the string must line up with feature order or it constrains the wrong
    # columns, which would be silently wrong rather than an error.
    for pos, name in enumerate(PRESEASON_FEATURE_ORDER):
        assert int(parsed[pos]) == MONOTONE.get(name, 0)


def test_production_features_are_constrained_upward():
    from train.train_preseason import MONOTONE

    for name in ("career_weighted_ppg", "prior_points_per_game", "career_best_ppg"):
        assert MONOTONE[name] == 1, f"{name} must never lower a projection"


def test_cost_and_competition_features_are_constrained_downward():
    from train.train_preseason import MONOTONE

    for name in ("draft_pick", "depth_chart_rank", "teammate_top_target_share"):
        assert MONOTONE[name] == -1


def test_age_and_efficiency_delta_are_left_unconstrained():
    """Both are genuinely non-monotonic; forcing a direction would encode a wrong belief."""
    from train.train_preseason import MONOTONE

    assert "age" not in MONOTONE
    assert "efficiency_delta" not in MONOTONE


def test_more_career_production_never_lowers_the_projection(trained_preseason_model):
    """The property the constraint exists to guarantee, checked against a real artifact.

    Sweeps career scoring rate across the whole plausible range with everything else
    pinned. Before the constraint this curve could fall; now it cannot.
    """
    model = trained_preseason_model
    idx = PRESEASON_FEATURE_ORDER.index("career_weighted_ppg")
    base = np.array(
        [[PRESEASON_DEFAULTS[n] for n in PRESEASON_FEATURE_ORDER]], dtype="float32"
    ).repeat(30, axis=0)
    base[:, idx] = np.linspace(0, 29, 30)
    preds = model.predict(base)
    assert np.all(np.diff(preds) >= -1e-6), "projection fell as career production rose"


def test_worse_draft_position_never_raises_the_projection(trained_preseason_model):
    model = trained_preseason_model
    idx = PRESEASON_FEATURE_ORDER.index("draft_pick")
    base = np.array(
        [[PRESEASON_DEFAULTS[n] for n in PRESEASON_FEATURE_ORDER]], dtype="float32"
    ).repeat(25, axis=0)
    base[:, idx] = np.linspace(1, 250, 25)
    preds = model.predict(base)
    assert np.all(np.diff(preds) <= 1e-6)


# ------------------------------------------------------------------ attribution
def test_every_feature_has_a_human_label():
    """An unlabelled feature would surface as raw snake_case in the UI."""
    missing = [f for f in PRESEASON_FEATURE_ORDER if f not in LABELS]
    assert not missing, f"features with no plain-English label: {missing}"


def test_labels_do_not_leak_snake_case():
    for name in PRESEASON_FEATURE_ORDER:
        assert "_" not in label(name)


def test_shares_render_as_percentages_not_decimals():
    assert format_value("prior_target_share", 0.2857) == "29%"
    assert format_value("prior_carry_share", 0.42) == "42%"
    # Exact .5 rounds to even, per Python's default. Noted rather than fought: a snap
    # share displayed as 70% instead of 71% is not worth a custom rounding helper.
    assert format_value("prior_snap_share", 0.705) == "70%"


def test_flags_render_as_words():
    assert format_value("is_rookie", 1.0) == "yes"
    assert format_value("qb_changed", 0.0) == "no"


def test_counts_render_without_decimals():
    assert format_value("career_games", 44.0) == "44"
    assert format_value("depth_chart_rank", 1.0) == "1"


def test_decayed_draft_features_are_labelled_as_decayed():
    """They must not read as the player's actual round or pick - they are not.

    Ashton Jeanty went sixth overall but reaches the model at an effective round of about
    2.4 after one season. Showing "draft round 2" would be a false statement about a real
    fact, which is worse than showing nothing.
    """
    for name in ("draft_round", "draft_pick"):
        assert "faded" in label(name)


def test_describe_states_direction_and_size():
    up = describe("career_weighted_ppg", 24.2, 5.76)
    assert "raises" in up and "5.8" in up and "24.2" in up
    down = describe("teammate_top_target_share", 0.30, -1.2)
    assert "lowers" in down and "30%" in down


def test_headline_names_the_biggest_reason_each_way():
    drivers = [
        ("career_weighted_ppg", 24.2, 5.8),
        ("career_games", 51.0, -1.0),
        ("prior_target_share", 0.29, 0.8),
    ]
    line = headline(drivers)
    assert "career scoring rate" in line
    assert "held back by" in line and "career games" in line


def test_headline_survives_all_positive_drivers():
    line = headline([("career_weighted_ppg", 24.0, 5.0)])
    assert "held back by" not in line
    assert line


def test_headline_survives_no_drivers():
    assert headline([]) == "no single dominant factor"


def test_contributions_plus_baseline_equal_the_prediction(preseason_registry):
    """The arithmetic guarantee that makes this an explanation rather than a story."""
    registry, version = preseason_registry
    feats = {
        "prior_points_per_game": 23.6,
        "career_weighted_ppg": 24.2,
        "career_games": 44,
        "prior_games": 16,
        "draft_round": 5,
        "draft_pick": 177,
        "age": 24,
        "prior_target_share": 0.286,
    }
    full = dict.fromkeys(PRESEASON_FEATURE_ORDER, None)
    full.update(feats)
    baseline, total, drivers, _ = registry.explain_preseason(version, full, top=99)
    assert total == pytest.approx(baseline + sum(c for _, _, c in drivers), abs=1e-3)
    predicted, _, _ = registry.predict_preseason(version, full)
    assert total == pytest.approx(predicted, abs=1e-3)


def test_drivers_come_back_ordered_by_magnitude(preseason_registry):
    registry, version = preseason_registry
    full = dict.fromkeys(PRESEASON_FEATURE_ORDER, None)
    full.update(career_weighted_ppg=22.0, prior_points_per_game=20.0, career_games=40)
    _, _, drivers, _ = registry.explain_preseason(version, full, top=8)
    sizes = [abs(c) for _, _, c in drivers]
    assert sizes == sorted(sizes, reverse=True)


def test_top_limits_the_number_of_drivers(preseason_registry):
    registry, version = preseason_registry
    full = dict.fromkeys(PRESEASON_FEATURE_ORDER, None)
    full.update(career_weighted_ppg=18.0, career_games=30)
    _, _, drivers, _ = registry.explain_preseason(version, full, top=3)
    assert len(drivers) == 3


def test_explaining_a_heuristic_model_is_refused_not_faked():
    """No trees to read means no explanation. Inventing one would be the real failure."""
    from pathlib import Path

    from app.registry import PRESEASON_FALLBACK_VERSION, ExplainUnsupported, ModelRegistry

    registry = ModelRegistry(model_dir=Path("/nonexistent"))
    with pytest.raises(ExplainUnsupported):
        registry.explain_preseason(
            PRESEASON_FALLBACK_VERSION,
            dict.fromkeys(PRESEASON_FEATURE_ORDER, None),
        )
