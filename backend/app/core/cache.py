"""Cache-aside layer.

Design notes:
  * cache-aside (lazy loading): read-through on miss, write on compute.
  * TTL-bounded so a stale prediction ages out rather than being invalidated by hand.
  * every failure path degrades to "cache miss" - Redis being down must never 500 the API.

Two backends behind one interface:

  redis://...    Redis / ElastiCache. What production runs.
  memory://      Process-local TTL dict. What `run-local.ps1` uses when there is no
                 Docker to run Redis in.

The in-memory backend is not a substitute for Redis - it is per-process, so it does not
survive a restart and two API workers would each keep their own copy. It exists so the
cache-aside code path is exercised and observable (`source: "cache"`, the hit/miss
metrics) on a laptop with nothing installed. Both backends implement the same four
methods, so nothing above this module knows which one is active.
"""

import fnmatch
import json
import threading
import time
from typing import Any, Protocol

import redis

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import CACHE_EVENTS


class CacheBackend(Protocol):
    def get(self, key: str) -> str | None: ...
    def setex(self, key: str, ttl: int, value: str) -> None: ...
    def scan_iter(self, match: str, count: int = 500): ...
    def delete(self, key: str) -> int: ...


class MemoryCache:
    """Thread-safe TTL dict. Expiry is lazy (checked on read) plus a bounded sweep on
    write, which is enough for a single-process dev server and avoids a background task."""

    MAX_KEYS = 10_000

    def __init__(self) -> None:
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._store[key]
                return None
            return value

    def setex(self, key: str, ttl: int, value: str) -> None:
        with self._lock:
            if len(self._store) >= self.MAX_KEYS:
                self._evict_expired_locked()
            if len(self._store) >= self.MAX_KEYS:
                # Still full: drop the soonest-to-expire key. Crude approximation of
                # Redis's allkeys-lru; good enough for a dev cache.
                oldest = min(self._store, key=lambda k: self._store[k][0])
                del self._store[oldest]
            self._store[key] = (time.monotonic() + ttl, value)

    def _evict_expired_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, (exp, _) in self._store.items() if exp < now]:
            del self._store[key]

    def scan_iter(self, match: str, count: int = 500):
        with self._lock:
            keys = list(self._store)
        # fnmatch implements the same glob semantics as Redis SCAN MATCH.
        yield from (k for k in keys if fnmatch.fnmatch(k, match))

    def delete(self, key: str) -> int:
        with self._lock:
            return 1 if self._store.pop(key, None) is not None else 0


_client: CacheBackend | None = None


def _build_client() -> CacheBackend:
    url = settings.redis_url.strip()
    if not url or url.startswith("memory://"):
        logger.info("cache_backend", backend="memory", note="no Redis; process-local TTL cache")
        return MemoryCache()
    return redis.from_url(url, decode_responses=True, socket_timeout=1, socket_connect_timeout=1)


def get_cache() -> CacheBackend:
    global _client
    if _client is None:
        _client = _build_client()
    return _client


# Kept as an alias so existing call sites and tests read naturally.
get_redis = get_cache


def set_redis(client: CacheBackend | None) -> None:
    """Test seam: inject fakeredis, a MemoryCache, or None to reset."""
    global _client
    _client = client


def backend_name() -> str:
    return "memory" if isinstance(get_cache(), MemoryCache) else "redis"


def prediction_key(player_id: int, week: int, opponent: str, model_version: str) -> str:
    return f"pred:v1:{model_version}:{player_id}:{week}:{opponent.upper()}"


def rankings_key(position: str, week: int, limit: int) -> str:
    return f"rank:v1:{position.upper()}:{week}:{limit}"


def cache_get(key: str) -> Any | None:
    try:
        raw = get_cache().get(key)
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", op="get", key=key, error=str(exc))
        CACHE_EVENTS.labels(result="error").inc()
        return None
    if raw is None:
        CACHE_EVENTS.labels(result="miss").inc()
        return None
    CACHE_EVENTS.labels(result="hit").inc()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        get_cache().setex(key, ttl, json.dumps(value, default=str))
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", op="set", key=key, error=str(exc))
        CACHE_EVENTS.labels(result="error").inc()


def cache_invalidate(pattern: str) -> int:
    """SCAN-based invalidation (never KEYS - it blocks the event loop server-side)."""
    deleted = 0
    try:
        client = get_cache()
        for key in client.scan_iter(match=pattern, count=500):
            deleted += client.delete(key)
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", op="invalidate", pattern=pattern, error=str(exc))
    return deleted
