from pydantic import BaseModel, ConfigDict, Field


class PlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    team: str | None = None
    position: str
    age: int | None = None


class PlayerDetailOut(PlayerOut):
    height_inches: int | None = None
    weight_lbs: int | None = None
    external_id: str | None = None


class PlayerStatsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    season: int
    week: int
    opponent: str | None = None
    is_home: bool
    targets: int
    receptions: int
    yards: float
    touchdowns: int
    fantasy_points: float


class RankingRow(BaseModel):
    rank: int
    player_id: int
    name: str
    team: str | None = None
    position: str
    projected_points: float = Field(..., description="Model projection for the requested week")
    confidence: float | None = None


class RankingsOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    season: int
    week: int
    position: str
    model_version: str
    rankings: list[RankingRow]


class PaginatedPlayers(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[PlayerOut]


class SeasonRankingRow(BaseModel):
    """One row of the draft board."""

    rank: int
    player_id: int
    name: str
    team: str | None = None
    position: str
    projected_points_per_game: float
    projected_season_points: float
    confidence: float | None = None
    is_rookie: bool = False
    # Why this number exists at all - prior production, or only draft capital.
    basis: str | None = None
    # Points per game above a replacement-level starter at the SAME position. This is the
    # number that answers "who should I take next", because raw points are not comparable
    # across positions - see the replacement_level note on the response.
    value_over_replacement: float | None = None
    position_rank: int | None = None


class SeasonRankingsOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    season: int
    position: str
    model_version: str
    games_assumed: int
    # True once the season has started, at which point weekly projections are the
    # better view and the UI should switch.
    season_started: bool
    # The per-position baseline each player is measured against, and how it was chosen.
    replacement_level: dict[str, float] = {}
    replacement_note: str = ""
    # Paging: the board runs to hundreds of players, and 20 at a time is what a draft
    # actually needs.
    page: int
    per_page: int
    total: int
    rankings: list[SeasonRankingRow]


class SeasonLeaderRow(BaseModel):
    """What a player actually did in a completed season - no model involved."""

    rank: int
    player_id: int
    name: str
    team: str | None = None
    position: str
    games: int
    total_points: float
    points_per_game: float
    targets: int
    receptions: int
    yards: float
    touchdowns: int


class SeasonLeadersOut(BaseModel):
    season: int
    position: str
    scoring: str
    regular_season_only: bool
    page: int
    per_page: int
    total: int
    leaders: list[SeasonLeaderRow]


class ProjectionDriver(BaseModel):
    """One feature's measured effect on a single player's projection."""

    feature: str
    label: str
    display_value: str
    contribution: float
    explanation: str


class ProjectionWhy(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    """The full decomposition of one player's season projection.

    baseline is where the model puts a player with no distinguishing features - roughly
    the average of everyone it trained on. Every driver is a signed move away from that.
    Only the largest few are returned, so they will not add up to the projection on their
    own; drivers_shown and total_features say so plainly rather than implying completeness.
    """

    player_id: int
    name: str
    position: str
    season: int
    projected_points_per_game: float
    baseline: float
    model_version: str
    headline: str
    drivers: list[ProjectionDriver]
    drivers_shown: int
    total_features: int
