"""Thin Redis wrapper.

Design notes for interviews:
  * cache-aside (lazy loading): read-through on miss, write on compute.
  * TTL-bounded so a stale prediction ages out rather than being invalidated by hand.
  * every failure path degrades to "cache miss" - Redis being down must never 500 the API.
"""

import json
from typing import Any

import redis

from app.core.config import settings
from app.core.logging import logger
from app.core.metrics import CACHE_EVENTS

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(
            settings.redis_url, decode_responses=True, socket_timeout=1, socket_connect_timeout=1
        )
    return _client


def set_redis(client: redis.Redis | None) -> None:
    """Test seam: inject fakeredis (or None to reset)."""
    global _client
    _client = client


def prediction_key(player_id: int, week: int, opponent: str, model_version: str) -> str:
    return f"pred:v1:{model_version}:{player_id}:{week}:{opponent.upper()}"


def rankings_key(position: str, week: int, limit: int) -> str:
    return f"rank:v1:{position.upper()}:{week}:{limit}"


def cache_get(key: str) -> Any | None:
    try:
        raw = get_redis().get(key)
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
        get_redis().setex(key, ttl, json.dumps(value, default=str))
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", op="set", key=key, error=str(exc))
        CACHE_EVENTS.labels(result="error").inc()


def cache_invalidate(pattern: str) -> int:
    """SCAN-based invalidation (never KEYS - it blocks the event loop server-side)."""
    deleted = 0
    try:
        client = get_redis()
        for key in client.scan_iter(match=pattern, count=500):
            deleted += client.delete(key)
    except redis.RedisError as exc:
        logger.warning("cache_unavailable", op="invalidate", pattern=pattern, error=str(exc))
    return deleted
