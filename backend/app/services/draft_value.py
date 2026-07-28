"""Value over replacement, so a cross-position list is actually rankable.

The problem this solves: a FLEX board sorted by projected points is not draft advice.
Points are not comparable across positions, because what you give up by *not* taking a
player depends on how quickly his position falls off.

A worked example from real projections. Suppose a TE projects 15.7 points per game and a
WR projects 15.9. Sorted by points the WR goes first. But the 12th-best TE projects
around 7, while the 12th-best WR projects around 11 - so taking the TE gains you roughly
8.7 points a week over the TE you would otherwise end up with, and taking the WR gains
about 4.9. The TE is the better pick despite the lower raw number, and a points-sorted
board hides that completely. This is why raw-points FLEX lists systematically under-draft
elite tight ends.

Replacement level here is the projection of the Nth-best player at a position, where N is
roughly how many of that position get started across a league. Those counts are the
convention for a 12-team league with one starting TE and a flex spot; they are the one
genuinely arbitrary choice in this module, so they live in one dict with the reasoning
attached rather than being scattered as magic numbers.
"""

from __future__ import annotations

# Roughly how many of each position are starting in a 12-team league at any moment:
# 12 teams x 2 WR + flex, x 1 RB + flex, x 1 TE. The exact numbers move with league
# settings; what matters is the shape - the TE pool is shallow, so its baseline is low.
STARTERS_BY_POSITION: dict[str, int] = {
    "WR": 36,
    "RB": 30,
    "TE": 14,
}

# If a position has fewer projections than its starter count, fall back to the worst
# available rather than reporting no baseline.
_MIN_POOL = 3


def replacement_levels(pools: dict[str, list[float]]) -> dict[str, float]:
    """Baseline projection per position: the Nth-best player at that position.

    `pools` maps position to its projections. Order does not matter - sorted here so a
    caller cannot break this by passing them unsorted.
    """
    levels: dict[str, float] = {}
    for position, values in pools.items():
        if not values:
            continue
        ranked = sorted(values, reverse=True)
        n = STARTERS_BY_POSITION.get(position, 24)
        if len(ranked) >= n:
            levels[position] = round(ranked[n - 1], 2)
        elif len(ranked) >= _MIN_POOL:
            # Not enough players to reach the nominal baseline; use the last one, which is
            # the most pessimistic honest answer.
            levels[position] = round(ranked[-1], 2)
        else:
            levels[position] = 0.0
    return levels


def value_over_replacement(
    projection: float, position: str, levels: dict[str, float]
) -> float | None:
    """Points per game above a replacement starter at the same position.

    None when there is no baseline for the position, rather than 0.0 - "we do not know"
    and "exactly replacement level" are different statements and should not look alike.
    """
    baseline = levels.get(position)
    if baseline is None:
        return None
    return round(projection - baseline, 2)


def replacement_note(levels: dict[str, float]) -> str:
    """One line explaining what the numbers are measured against."""
    if not levels:
        return "No replacement baseline available."
    parts = [
        f"{pos} {levels[pos]:.1f} (#{STARTERS_BY_POSITION.get(pos, 24)})" for pos in sorted(levels)
    ]
    return (
        "Value is points per game above a replacement starter at the same position: "
        + ", ".join(parts)
        + ". Assumes a 12-team league."
    )
