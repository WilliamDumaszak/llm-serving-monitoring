"""
Prometheus metrics for LLM serving monitoring.

Exposes:
  - llm_request_total          (counter)   — total requests by status
  - llm_response_time_seconds  (histogram) — end-to-end latency
  - llm_hit_rate               (gauge)     — RAG retrieval hit rate (rolling)
  - llm_mrr                    (gauge)     — Mean Reciprocal Rank (rolling)
  - llm_precision_at_k         (gauge)     — Precision@K (rolling), label k
  - llm_tokens_used_total      (counter)   — cumulative token spend, label type=input|output
  - llm_cache_hits_total       (counter)   — requests served from Redis cache
"""

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "llm_request_total",
    "Total number of LLM requests",
    ["status"],
)

RESPONSE_TIME = Histogram(
    "llm_response_time_seconds",
    "LLM response time in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

HIT_RATE = Gauge(
    "llm_hit_rate",
    "RAG retrieval hit rate (rolling)",
)

MRR_GAUGE = Gauge(
    "llm_mrr",
    "RAG Mean Reciprocal Rank (rolling)",
)

PRECISION_AT_K = Gauge(
    "llm_precision_at_k",
    "RAG Precision@K (rolling average)",
    ["k"],
)

TOKEN_COUNT = Counter(
    "llm_tokens_used_total",
    "Cumulative LLM tokens consumed",
    ["type"],  # "input" | "output"
)

CACHE_HIT = Counter(
    "llm_cache_hits_total",
    "Requests served directly from Redis cache (no RAG or LLM call)",
)
