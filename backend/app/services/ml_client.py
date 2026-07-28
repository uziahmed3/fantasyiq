"""HTTP client for the ML prediction service.

The ML service is a separate deployable: it scales on CPU, the API scales on IO, and a
bad model rollout can be reverted without redeploying the API. The cost of that split is
a network hop, so this client owns the timeout/retry/failure semantics.

Note on `trust_env=False`: httpx reads HTTP_PROXY / HTTPS_PROXY / ALL_PROXY from the
environment by default. This call is always internal - localhost in local mode, private
VPC DNS in AWS - so routing it through an outbound proxy is never correct and, on a
corporate machine that sets those variables, breaks the API outright. Disabling env
trust here is the fix; external calls (which this service does not make) would opt in.
"""

import time

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import ML_LATENCY, PREDICTION_FAILURES
from app.schemas.prediction import FeatureVector, PreseasonFeatureVector


class MLServiceError(RuntimeError):
    pass


class MLClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.base_url = (base_url or settings.ml_service_url).rstrip("/")
        self.timeout = timeout or settings.ml_service_timeout_seconds
        self.trust_env = settings.ml_service_trust_env

    def predict(self, features: FeatureVector, model_version: str | None = None) -> dict:
        payload = {
            "features": features.model_dump(),
            "model_version": model_version or settings.active_model_version,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
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
        except Exception as exc:  # noqa: BLE001
            # Misconfiguration (bad proxy scheme, malformed URL) must still surface as a
            # dependency failure - a 503 - rather than a 500 blamed on this service.
            PREDICTION_FAILURES.labels(reason="ml_client_error").inc()
            raise MLServiceError(f"ML client error: {exc}") from exc
        finally:
            elapsed = time.perf_counter() - started
            ML_LATENCY.observe(elapsed)
            logger.debug("ml_call", elapsed_ms=round(elapsed * 1000, 2))

    def predict_preseason(
        self, features: PreseasonFeatureVector, model_version: str | None = None
    ) -> dict:
        """Call the preseason endpoint. Nulls are sent as nulls, not zeros.

        exclude_none would drop them, which is the same thing on the wire, but sending an
        explicit null makes the request self-describing when you read it in a log: the
        service can tell "we do not know his snap share" from "his snap share was 0".
        """
        payload = {
            "features": features.model_dump(),
            "model_version": model_version or settings.active_preseason_model_version,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
                resp = client.post(f"{self.base_url}/predict/preseason", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.TimeoutException as exc:
            PREDICTION_FAILURES.labels(reason="preseason_timeout").inc()
            raise MLServiceError("ML service timed out") from exc
        except httpx.HTTPStatusError as exc:
            PREDICTION_FAILURES.labels(reason=f"preseason_http_{exc.response.status_code}").inc()
            raise MLServiceError(f"ML service returned {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            PREDICTION_FAILURES.labels(reason="preseason_unreachable").inc()
            raise MLServiceError("ML service unreachable") from exc
        except Exception as exc:  # noqa: BLE001
            PREDICTION_FAILURES.labels(reason="preseason_client_error").inc()
            raise MLServiceError(f"ML client error: {exc}") from exc
        finally:
            ML_LATENCY.observe(time.perf_counter() - started)

    def explain_preseason(
        self, features: PreseasonFeatureVector, model_version: str | None = None, top: int = 6
    ) -> dict:
        """Ask the model why it produced the number it did.

        A separate call rather than data folded into the prediction response, because the
        cost is not the same: attribution walks every tree for every feature, so making
        the hot leaderboard path pay for it to serve an explanation nobody has asked to
        see would be the wrong trade. This is invoked when a user opens one player.

        A failure here is not a prediction failure - the projection is already valid and
        on screen. So it is counted under its own label instead of inflating the
        prediction-failure metric that alerting watches.
        """
        payload = {
            "features": features.model_dump(),
            "model_version": model_version or settings.active_preseason_model_version,
            "top": top,
        }
        started = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout, trust_env=self.trust_env) as client:
                resp = client.post(f"{self.base_url}/predict/preseason/explain", json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            PREDICTION_FAILURES.labels(reason=f"explain_http_{exc.response.status_code}").inc()
            # 501 means the active model has no tree structure to read - a real answer,
            # not an outage, so it is passed through with its own status.
            raise MLServiceError(
                f"explanation unavailable ({exc.response.status_code})",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            PREDICTION_FAILURES.labels(reason="explain_error").inc()
            raise MLServiceError(f"explanation unavailable: {exc}") from exc
        finally:
            ML_LATENCY.observe(time.perf_counter() - started)

    def health(self) -> bool:
        """Never raises. /ready calls this, and a readiness probe that 500s is useless."""
        try:
            with httpx.Client(timeout=2.0, trust_env=self.trust_env) as client:
                return client.get(f"{self.base_url}/health").status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("ml_health_failed", error=str(exc))
            return False


ml_client = MLClient()
