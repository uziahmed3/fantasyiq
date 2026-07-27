from app.core.cache import cache_get, cache_invalidate, cache_set, prediction_key
from app.schemas.prediction import PredictionRequest
from app.services.predictions import PredictionService


def test_cache_roundtrip():
    cache_set("k1", {"a": 1}, ttl=60)
    assert cache_get("k1") == {"a": 1}
    assert cache_get("missing") is None


def test_cache_invalidate_by_pattern():
    cache_set("pred:v1:m:1:5:GB", {"prediction": 1}, ttl=60)
    cache_set("pred:v1:m:2:5:GB", {"prediction": 2}, ttl=60)
    cache_set("rank:v1:WR:5:25", {"x": 1}, ttl=60)
    assert cache_invalidate("pred:v1:*") == 2
    assert cache_get("rank:v1:WR:5:25") is not None


def test_second_call_hits_cache_and_skips_model(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=15, week=6, season=2023, opponent="GB")

    first = svc.predict(req)
    assert first.source == "model"
    assert len(stub_ml.calls) == 1

    second = svc.predict(req)
    assert second.source == "cache"
    assert second.prediction == first.prediction
    assert len(stub_ml.calls) == 1, "cache hit must not invoke the model"


def test_refresh_flag_bypasses_cache(db_session, seed, stub_ml):
    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=15, week=6, season=2023, opponent="GB")
    svc.predict(req)
    svc.predict(req, use_cache=False)
    assert len(stub_ml.calls) == 2


def test_cache_key_is_model_version_scoped():
    a = prediction_key(15, 6, "GB", "xgboost_v1")
    b = prediction_key(15, 6, "GB", "xgboost_v2")
    assert a != b, "a new model version must not read the old model's cached values"


def test_redis_outage_degrades_to_miss(monkeypatch):
    import redis

    from app.core import cache as cache_module

    class Broken:
        def get(self, *_a, **_k):
            raise redis.RedisError("down")

        def setex(self, *_a, **_k):
            raise redis.RedisError("down")

    cache_module.set_redis(Broken())
    assert cache_get("anything") is None  # no exception raised
    cache_set("anything", {"a": 1}, ttl=10)  # no exception raised


# ---------------------------------------------------------------- memory backend


def test_memory_backend_roundtrip_and_ttl():
    from app.core import cache as cache_module

    mem = cache_module.MemoryCache()
    cache_module.set_redis(mem)
    assert cache_module.backend_name() == "memory"

    cache_set("m1", {"a": 1}, ttl=60)
    assert cache_get("m1") == {"a": 1}

    cache_set("m2", {"b": 2}, ttl=0)  # already expired on read
    assert cache_get("m2") is None


def test_memory_backend_glob_invalidation():
    from app.core import cache as cache_module

    cache_module.set_redis(cache_module.MemoryCache())
    cache_set("pred:v1:m:1:5:GB", {"p": 1}, ttl=60)
    cache_set("pred:v1:m:2:5:GB", {"p": 2}, ttl=60)
    cache_set("rank:v1:WR:5:25", {"r": 1}, ttl=60)

    assert cache_invalidate("pred:v1:*") == 2
    assert cache_get("pred:v1:m:1:5:GB") is None
    assert cache_get("rank:v1:WR:5:25") is not None


def test_memory_backend_bounds_its_size():
    from app.core import cache as cache_module

    mem = cache_module.MemoryCache()
    mem.MAX_KEYS = 10
    cache_module.set_redis(mem)
    for i in range(50):
        cache_set(f"k{i}", {"i": i}, ttl=60)
    assert len(mem._store) <= 10, "unbounded memory cache would leak on a long-running process"


def test_memory_backend_serves_cache_hits_end_to_end(db_session, seed, stub_ml, monkeypatch):
    """The whole point: with no Redis, the cache-aside path still works."""
    from app.core import cache as cache_module

    cache_module.set_redis(cache_module.MemoryCache())

    svc = PredictionService(db_session, client=stub_ml)
    req = PredictionRequest(player_id=15, week=7, season=2023, opponent="GB")

    assert svc.predict(req).source == "model"
    assert svc.predict(req).source == "cache"
    assert len(stub_ml.calls) == 1


def test_empty_redis_url_selects_memory_backend(monkeypatch):
    from app.core import cache as cache_module
    from app.core.config import settings as app_settings

    monkeypatch.setattr(app_settings, "redis_url", "memory://", raising=False)
    cache_module.set_redis(None)
    assert cache_module.backend_name() == "memory"
    cache_module.set_redis(None)
