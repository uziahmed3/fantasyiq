from pydantic import BaseModel, ConfigDict, Field


class FeaturePayload(BaseModel):
    targets_last_3: float = 0.0
    receptions_last_3: float = 0.0
    yards_last_3: float = 0.0
    touchdowns_last_3: float = 0.0
    fantasy_points_last_3: float = 0.0
    fantasy_points_last_1: float = 0.0
    season_avg_points: float = 0.0
    games_played: int = 0
    opponent_rank: int = Field(16, ge=1, le=32)
    is_home: int = Field(1, ge=0, le=1)


class PredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    features: FeaturePayload
    model_version: str | None = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: float
    confidence: float
    model_version: str
    framework: str


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[FeaturePayload] = Field(..., min_length=1, max_length=500)
    model_version: str | None = None


class BatchPredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    predictions: list[float]


# ------------------------------------------------------------------ preseason
class PreseasonFeatures(BaseModel):
    """Everything knowable before a season starts. All optional: a rookie has almost
    none of it, and the optional upstream feeds can be unavailable. The service
    substitutes documented defaults (see features.PRESEASON_DEFAULTS)."""

    prior_points_per_game: float | None = None
    prior_last4_points_per_game: float | None = None
    prior_targets_per_game: float | None = None
    prior_target_share: float | None = Field(None, ge=0, le=1)
    prior_carries_per_game: float | None = None
    prior_carry_share: float | None = Field(None, ge=0, le=1)
    prior_yards_per_game: float | None = None
    prior_games: int | None = Field(None, ge=0, le=25)
    prior_snap_share: float | None = Field(None, ge=0, le=1)
    depth_chart_rank: int | None = Field(None, ge=1, le=10)
    draft_round: int | None = Field(None, ge=1, le=8)
    draft_pick: int | None = Field(None, ge=1, le=300)
    years_experience: int | None = Field(None, ge=0, le=25)
    is_rookie: int | None = Field(None, ge=0, le=1)
    age: float | None = Field(None, ge=18, le=50)
    team_pass_attempts_prior: float | None = None
    qb_changed: int | None = Field(None, ge=0, le=1)
    # Multi-season history, the volume/efficiency split, and situation.
    career_weighted_ppg: float | None = None
    career_weighted_targets_per_game: float | None = None
    career_weighted_carries_per_game: float | None = None
    career_weighted_target_share: float | None = None
    career_best_ppg: float | None = None
    career_seasons: int | None = None
    career_games: int | None = None
    prior_points_per_target: float | None = None
    career_points_per_target: float | None = None
    efficiency_delta: float | None = None
    team_departed_target_share: float | None = None
    team_departed_carry_share: float | None = None
    teammate_top_target_share: float | None = None
    teammate_top_carry_share: float | None = None


class PreseasonRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    features: PreseasonFeatures
    model_version: str | None = None


class PreseasonResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: float
    confidence: float
    model_version: str
    framework: str
    basis: str  # what the number actually rests on - "draft capital", "prior season", ...


class PreseasonBatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    items: list[PreseasonFeatures] = Field(..., min_length=1, max_length=500)
    model_version: str | None = None


class PreseasonBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    predictions: list[float]
    # Per-item, because a draft board showing a rookie's number next to a veteran's
    # without any indication of how much is actually known would be misleading.
    confidences: list[float]


class PreseasonExplainRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    """Same inputs as a preseason prediction, plus how many drivers to return."""

    features: "PreseasonFeatures"
    model_version: str | None = None
    top: int = Field(6, ge=1, le=20)


class Driver(BaseModel):
    """One feature's exact effect on this player's projection."""

    feature: str
    label: str
    value: float
    display_value: str
    contribution: float
    explanation: str


class PreseasonExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    """A projection decomposed into terms that sum to it.

    baseline is the model's average output across its training data - where a player with
    no distinguishing features would land. Every driver is a signed departure from that,
    so baseline + sum(all contributions) == prediction exactly. Only the top N drivers are
    returned, so the listed ones will not sum to the total on their own; that is stated
    rather than hidden, via drivers_shown and total_features.
    """

    prediction: float
    baseline: float
    model_version: str
    headline: str
    drivers: list[Driver]
    drivers_shown: int
    total_features: int
