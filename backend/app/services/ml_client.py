"""HTTP client for the ML prediction service.

The ML service is a separate deployable: it scales on CPU, the API scales on IO, and a
bad model rollout can be reverted without redeploying the API. The cost of that split is
a network hop, so this client owns the timeout/retry/failure semantics.
"""

import time

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import ML_LATENCY, PREDICTION_FAILURES
from app.schemas.prediction import FeatureVector


class MLServiceError(RuntimeError):
    pass


class MLClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.ml_service_url).rstrip("/")
        self.timeout = timeout or settings.ml_service_timeout_seconds

    def predict(self, features: FeatureVector, model_version: str | None = None) -> dict:
        payload = {
            "features": features.model_dump(),
            "model_version": model_version or settings.active_model_version,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(f"{self.base_url}/predict", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            PREDICTION_FAILURES.labels(reason="ml_timeout").inc()
            raise MLServiceError("ML service timed out") from exc
        except httpx.HTTPStatusError as exc:
            PREDICTION_FAILURES.labels(reason=f"ml_http_{exc.response.status_code}").inc()
            raise MLServiceError(f"ML service returned {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            PREDICTION_FAILURES.labels(reason="ml_unreachable").inc()
            raise MLServiceError("ML service unreachable") from exc
        finally:
            elapsed = time.perf_counter() - started
            ML_LATENCY.observe(elapsed)
            logger.debug("ml_call", elapsed_ms=round(elapsed * 1000, 2))

    def health(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                return client.get(f"{self.base_url}/health").status_code == 200
        except httpx.HTTPError:
            return False


ml_client = MLClient()
