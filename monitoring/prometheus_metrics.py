"""
Prometheus metrics for LLM serving monitoring.

Exposes:
  - llm_request_total          (counter)
  - llm_response_time_seconds  (histogram)
  - llm_hit_rate               (gauge)
  - llm_mrr                    (gauge)
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
