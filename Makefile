.PHONY: up down build restart test logs shell health metrics eval ingest clean

# ── Stack lifecycle ────────────────────────────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --no-cache

restart:
	docker compose restart api

# ── Development ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

lint:
	python -m py_compile api/main.py api/schemas.py llm/ollama_client.py rag/cache.py monitoring/prometheus_metrics.py monitoring/rag_metrics.py

# ── Observability ──────────────────────────────────────────────────────────────
logs:
	docker compose logs -f api

health:
	@curl -s http://localhost:8000/health | python3 -m json.tool

metrics:
	@curl -s http://localhost:8000/metrics | grep "^llm_"

# ── LLM queries ───────────────────────────────────────────────────────────────
query:
	@curl -s -X POST http://localhost:8000/query \
	  -H "Content-Type: application/json" \
	  -d '{"query": "$(Q)"}' | python3 -m json.tool

stream:
	@curl -s -X POST http://localhost:8000/query/stream \
	  -H "Content-Type: application/json" \
	  -d '{"query": "$(Q)"}'

# ── Evaluation ─────────────────────────────────────────────────────────────────
eval:
	@curl -s -X POST "http://localhost:8000/evaluate/ragas/db?limit=20" | python3 -m json.tool

# ── Data ingestion ────────────────────────────────────────────────────────────
ingest:
	docker compose exec airflow airflow dags trigger llm_rag_ingest_pipeline

# ── Shells ────────────────────────────────────────────────────────────────────
shell:
	docker compose exec api bash

redis-cli:
	docker compose exec redis redis-cli

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	docker compose down -v

help:
	@echo "Available targets:"
	@echo "  up          Start all services"
	@echo "  down        Stop all services"
	@echo "  build       Rebuild images (no cache)"
	@echo "  restart     Restart only the API container"
	@echo "  test        Run pytest test suite"
	@echo "  logs        Follow API logs"
	@echo "  health      Check /health endpoint"
	@echo "  metrics     Show LLM Prometheus metrics"
	@echo "  query Q=... POST a query (e.g. make query Q='What is RAG?')"
	@echo "  stream Q=.. Streaming query via SSE"
	@echo "  eval        Run RAGAS eval on last 20 DB interactions"
	@echo "  ingest      Trigger Airflow ingestion DAG"
	@echo "  shell       Open shell in API container"
	@echo "  redis-cli   Open Redis CLI"
	@echo "  clean       Remove all containers and volumes"
