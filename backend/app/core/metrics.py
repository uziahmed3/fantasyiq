from prometheus_client import Counter, Histogram

CACHE_EVENTS = Counter("fantasyiq_cache_events_total", "Cache lookups by outcome", ["result"])
PREDICTIONS = Counter(
    "fantasyiq_predictions_total", "Predictions served", ["model_version", "source"]
)
PREDICTION_FAILURES = Counter(
    "fantasyiq_prediction_failures_total", "Failed prediction attempts", ["reason"]
)
ML_LATENCY = Histogram(
    "fantasyiq_ml_service_latency_seconds",
    "Latency of downstream ML service calls",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PIPELINE_ROWS = Counter(
    "fantasyiq_pipeline_rows_total", "Rows written by the ETL pipeline", ["table"]
)
