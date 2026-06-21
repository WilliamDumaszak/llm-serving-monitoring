"""
Semantic response cache backed by Redis.

Caching strategy:
  Layer 1 — Exact-match: SHA-256 of the normalized query string.
             Fast, zero-cost. Handles repeated identical queries (FAQ patterns).
             Typical hit latency: <5 ms vs 2-30 s for LLM inference.
  Layer 2 — For production, swap _cache_key() for cosine similarity over
             query embeddings stored in a Redis vector index (RediSearch /
             redis-vl). Set a similarity threshold (e.g. 0.92) to catch
             near-duplicates such as "What is RAG?" vs "What does RAG mean?".

Cache-aside pattern (read-through on /query):
  1. Normalize & hash the incoming query → look up in Redis.
  2. Cache HIT  → return stored response immediately, skip RAG + LLM entirely.
  3. Cache MISS → run RAG + LLM → store result with TTL → return response.

Notes:
  - /query/stream bypasses the cache (streaming responses are incompatible
    with blob storage; the client already expects a live stream).
  - All Redis errors are non-fatal: the request falls through to the LLM.
  - TTL is controlled by the CACHE_TTL_SECONDS env var (default 3600 s).
"""

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default

_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        import redis

        host = os.getenv("REDIS_HOST", "redis")
        port = int(os.getenv("REDIS_PORT", "6379"))
        _redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def _cache_key(query: str) -> str:
    """Normalize query and produce a deterministic cache key."""
    normalized = " ".join(query.strip().lower().split())
    return f"llm:cache:{hashlib.sha256(normalized.encode()).hexdigest()}"


def get_cached(query: str) -> dict | None:
    """Return cached response dict, or None on cache miss or Redis error."""
    try:
        r = _get_redis()
        value = r.get(_cache_key(query))
        if value:
            logger.info("Cache HIT — skipping RAG + LLM.")
            return json.loads(value)
    except Exception as exc:
        logger.warning(f"Cache read failed (non-fatal, falling through): {exc}")
    return None


def set_cached(query: str, response: dict, ttl: int = _CACHE_TTL) -> None:
    """Persist response dict in Redis with a TTL (seconds)."""
    try:
        r = _get_redis()
        r.setex(_cache_key(query), ttl, json.dumps(response))
    except Exception as exc:
        logger.warning(f"Cache write failed (non-fatal): {exc}")


def cache_ready() -> bool:
    """Liveness check — returns True if Redis is reachable."""
    try:
        return bool(_get_redis().ping())
    except Exception:
        return False
