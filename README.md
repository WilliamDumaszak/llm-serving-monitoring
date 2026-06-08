# LLM Serving and Monitoring

Production-style LLM serving stack with RAG retrieval, observability, storage, scheduled ingestion, and quality evaluation.

## Overview

This project demonstrates an end-to-end serving environment for LLM-powered RAG systems:
- Retrieval via Elasticsearch (local) or Azure AI Search (cloud)
- Generation via Ollama/HuggingFace/Azure OpenAI/vLLM
- Interaction persistence in PostgreSQL
- Real-time metrics with Prometheus and Grafana
- Scheduled ingestion with Airflow
- RAG quality checks with RAGAS

## Architecture

```text
New documents
  -> Airflow DAG
  -> Search index refresh

User query
  -> /query API
  -> retrieval (BM25 / cloud search)
  -> LLM generation
  -> save interaction (PostgreSQL)
  -> expose metrics (/metrics)
  -> dashboards (Prometheus/Grafana)

Evaluation
  -> /evaluate/ragas
  -> /evaluate/ragas/db
```

## Repository Structure

```text
llm-serving-monitoring/
  config/
    config.yaml
  rag/
    elasticsearch_rag.py
  llm/
    ollama_client.py
  database/
    postgres.py
  monitoring/
    prometheus_metrics.py
    rag_metrics.py
    prometheus.yml
  evaluation/
    ragas_eval.py
  airflow/
    dags/
      ingest_pipeline.py
  api/
    main.py
    schemas.py
  tests/
    test_api.py
  docker-compose.yml
  Dockerfile
  requirements.txt
  .env.example
```

## Local Run

### 1) Setup

```bash
cd llm-serving-monitoring
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start full stack

```bash
docker compose up -d
```

### 3) Send a query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is supply chain management?"}'
```

### 4) Submit feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"<doc_id>","rating":5,"comment":"Great answer"}'
```

### 5) Check metrics

```bash
curl -s http://localhost:8000/metrics
```

Useful UIs:
- Grafana: `http://localhost:3000` (`admin/admin`)
- Airflow: `http://localhost:8080` (`admin/admin`)

## Configuration Highlights

See `config/config.yaml` for:
- LLM provider and model routing
- Search provider selection
- Database connection settings
- API and evaluation defaults

## CI/CD

Workflow: `.github/workflows/pipeline.yml`
- Test job on push/PR
- Build and push images on `main`
- Supports Docker Hub and Azure Container Registry

## Notes

- For cloud mode, set Azure variables from `.env.example`.
- For local-first development, keep defaults (`ollama` + `elasticsearch`).

## License

This project follows the repository root license.
