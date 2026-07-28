from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    player_id: int = Field(..., ge=1, examples=[15])
    week: int = Field(..., ge=1, le=22, examples=[6])
    season: int = Field(2023, ge=1999, le=2100)
    opponent: str = Field("UNK", min_length=2, max_length=8, examples=["GB"])
    is_home: bool = True


class PredictionResponse(BaseModel):
    # "model_version" collides with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    player_id: int
    player: str
    season: int
    week: int
    opponent: str
    prediction: float
    confidence: float | None = None
    model_version: str
    source: Literal["cache", "model"] = "model"
    # Which of the two models answered, and why. A week-1 or rookie projection rests on
    # completely different evidence than a mid-season one, and the caller should be able
    # to tell them apart without inspecting the model version string.
    mode: Literal["in_season", "preseason"] = "in_season"
    basis: str | None = None


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    week: int
    season: int
    prediction: float
    confidence: float | None
    model_version: str
    created_at: datetime


class PreseasonFeatureVector(BaseModel):
    """Preseason contract. Mirrors ml-service/app/features.PRESEASON_FEATURE_ORDER.

    Every field is optional because absence is meaningful and expected here: a rookie has
    no prior production, and the snap-count / depth-chart feeds are best-effort.
    """

    prior_points_per_game: float | None = None
    prior_last4_points_per_game: float | None = None
    prior_targets_per_game: float | None = None
    prior_target_share: float | None = None
    prior_carries_per_game: float | None = None
    prior_carry_share: float | None = None
    prior_yards_per_game: float | None = None
    prior_games: int | None = None
    prior_snap_share: float | None = None
    depth_chart_rank: int | None = None
    draft_round: int | None = None
    draft_pick: int | None = None
    years_experience: int | None = None
    is_rookie: int | None = None
    age: float | None = None
    team_pass_attempts_prior: float | None = None
    qb_changed: int | None = None
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
    qb_quality: float | None = None
    team_departed_target_share: float | None = None
    team_departed_carry_share: float | None = None
    teammate_top_target_share: float | None = None
    teammate_top_carry_share: float | None = None


class FeatureVector(BaseModel):
    """The contract between the backend and the ML service.

    Kept explicit (not **kwargs) so a training/serving skew shows up as a 422
    at the service boundary instead of a silently wrong prediction.
    """

    targets_last_3: float = 0.0
    receptions_last_3: float = 0.0
    yards_last_3: float = 0.0
    touchdowns_last_3: float = 0.0
    fantasy_points_last_3: float = 0.0
    fantasy_points_last_1: float = 0.0
    season_avg_points: float = 0.0
    games_played: int = 0
    opponent_rank: int = 16
    is_home: int = 1
