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
