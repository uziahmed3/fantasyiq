"""Turning feature contributions into sentences a person can act on.

The model's own vocabulary is useless at a draft table. "career_weighted_target_share
+0.46" is correct and unreadable; "commanded 27% of his team's targets across his career"
is the same fact in a form that helps somebody decide who to pick.

Two things live here: a label for each feature, and a formatter for its value. They are
deliberately separate from the numbers - the contribution is what the model actually did,
and this module only changes how it reads. Nothing here can alter a projection.
"""

from __future__ import annotations

LABELS: dict[str, str] = {
    "prior_points_per_game": "last season's points per game",
    "prior_last4_points_per_game": "how he finished last season",
    "prior_targets_per_game": "targets per game last season",
    "prior_target_share": "share of his team's targets",
    "prior_carries_per_game": "carries per game last season",
    "prior_carry_share": "share of his team's carries",
    "prior_yards_per_game": "yards per game last season",
    "prior_games": "games played last season",
    "prior_snap_share": "share of snaps played",
    "depth_chart_rank": "depth chart position",
    # These two reach the model already blended toward league-average in proportion to
    # games played (see features.decay_draft_capital), so the number shown is an index and
    # not the player's actual round or pick. Labelled to say so - Ashton Jeanty would
    # otherwise be reported as a third-rounder when he went sixth overall.
    "draft_round": "draft capital, faded by games played",
    "draft_pick": "draft slot, faded by games played",
    "years_experience": "years in the league",
    "is_rookie": "rookie",
    "age": "age",
    "team_pass_attempts_prior": "how much his team throws",
    "qb_changed": "new quarterback",
    "career_weighted_ppg": "career scoring rate",
    "career_weighted_targets_per_game": "career targets per game",
    "career_weighted_carries_per_game": "career carries per game",
    "career_weighted_target_share": "career target share",
    "career_best_ppg": "his best season",
    "career_seasons": "seasons played",
    "career_games": "career games",
    "prior_points_per_target": "points per target last season",
    "career_points_per_target": "career points per target",
    "efficiency_delta": "efficiency vs his own career norm",
    "team_departed_target_share": "targets vacated by departed teammates",
    "team_departed_carry_share": "carries vacated by departed teammates",
    "teammate_top_target_share": "competition for targets",
    "teammate_top_carry_share": "competition for carries",
}

# Features whose raw number means nothing without a unit.
SHARES = {
    "prior_target_share",
    "prior_carry_share",
    "prior_snap_share",
    "career_weighted_target_share",
    "team_departed_target_share",
    "team_departed_carry_share",
    "teammate_top_target_share",
    "teammate_top_carry_share",
}
FLAGS = {"is_rookie", "qb_changed"}
COUNTS = {
    "prior_games",
    "career_games",
    "career_seasons",
    "years_experience",
    "depth_chart_rank",
}


def label(feature: str) -> str:
    return LABELS.get(feature, feature.replace("_", " "))


def format_value(feature: str, value: float) -> str:
    if feature in FLAGS:
        return "yes" if value > 0.5 else "no"
    if feature in SHARES:
        return f"{value * 100:.0f}%"
    if feature in COUNTS:
        return f"{value:.0f}"
    if feature == "age":
        return f"{value:.0f}"
    return f"{value:.1f}"


def describe(feature: str, value: float, contribution: float) -> str:
    """One line: what the model looked at, what it saw, and which way it pushed."""
    direction = "raises" if contribution >= 0 else "lowers"
    return (
        f"{label(feature)} ({format_value(feature, value)}) "
        f"{direction} the projection by {abs(contribution):.1f}"
    )


def headline(drivers: list[tuple[str, float, float]]) -> str:
    """The single most compact honest summary: biggest reason up, biggest reason down.

    This replaces what the draft board used to show, which was "16 games last season" for
    almost everybody - true, and no help at all in deciding between two players.
    """
    up = next((d for d in drivers if d[2] > 0), None)
    down = next((d for d in drivers if d[2] < 0), None)
    parts = []
    if up:
        parts.append(f"{label(up[0])} {format_value(up[0], up[1])}")
    if down:
        parts.append(f"held back by {label(down[0])} {format_value(down[0], down[1])}")
    return "; ".join(parts) if parts else "no single dominant factor"
