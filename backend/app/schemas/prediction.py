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


class PredictionRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    week: int
    season: int
    prediction: float
    confidence: float | None
    model_version: str
    created_at: datetime


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
