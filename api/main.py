"""
FastAPI serving application with full LLM monitoring.

Endpoints:
  POST /query       — RAG + LLM inference, logs metrics
  POST /feedback    — store user feedback
  GET  /health      — liveness check (ES + DB)
  GET  /metrics     — Prometheus metrics scrape endpoint
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import FeedbackRequest, HealthResponse, QueryRequest, QueryResponse, RagasSample, RagasEvalResponse
from database.audit import create_audit_table, get_audit_trail, log_audit_event
from database.postgres import save_feedback, save_interaction, setup_tables
from llm.ollama_client import query_llm
from monitoring.prometheus_metrics import (
    CACHE_HIT, HIT_RATE, MRR_GAUGE, PRECISION_AT_K, REQUEST_COUNT, RESPONSE_TIME, TOKEN_COUNT,
)
from monitoring.rag_metrics import evaluate_search
from rag.cache import cache_ready, get_cached, set_cached
from rag.elasticsearch_rag import ensure_index, get_client, generate_doc_id, search

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        setup_tables()
    except Exception as exc:
        logger.warning(f"DB setup skipped: {exc}")
    try:
        create_audit_table()
    except Exception as exc:
        logger.warning(f"Audit table setup skipped: {exc}")
    try:
        es = get_client()
        ensure_index(es)
    except Exception as exc:
        logger.warning(f"ES setup skipped: {exc}")
    yield


app = FastAPI(
    title="LLM Serving & Monitoring",
    description=(
        "Production LLM serving with ElasticSearch RAG, Ollama, "
        "Prometheus metrics, Grafana dashboards, and user feedback loop."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


def _es_ready() -> bool:
    try:
        return get_client().ping()
    except Exception:
        return False


def _db_ready() -> bool:
    try:
        from database.postgres import get_connection
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        elasticsearch_ready=_es_ready(),
        database_ready=_db_ready(),
        cache_ready=cache_ready(),
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/query", response_model=QueryResponse, tags=["inference"])
@limiter.limit("30/minute")
def query(request: Request, body: QueryRequest):
    # ── Cache check ────────────────────────────────────────────────────────────
    cached = get_cached(body.query)
    if cached is not None:
        CACHE_HIT.inc()
        REQUEST_COUNT.labels(status="success").inc()
        cached["cache_hit"] = True
        return JSONResponse(content=cached)

    try:
        es = get_client()
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=f"ElasticSearch unavailable: {exc}")

    # retrieve context
    rag_results = search(es, body.query)
    context = " ".join(r.get("answer", r.get("text", "")) for r in rag_results)

    # evaluate retrieval quality on-the-fly
    relevance_list = [r.get("doc_id") is not None for r in rag_results]
    relevance = [relevance_list]
    from monitoring.rag_metrics import hit_rate, mrr, precision_at_k
    hr = hit_rate(relevance)
    mrr_score = mrr(relevance)
    pk = precision_at_k(relevance_list, k=5)

    # call LLM
    answer, response_time, in_tokens, out_tokens = query_llm(body.query, context)
    response_ms = response_time * 1000

    # generate doc id
    doc_id = generate_doc_id(body.query, answer)

    # update Prometheus metrics
    REQUEST_COUNT.labels(status="success").inc()
    RESPONSE_TIME.observe(response_time)
    HIT_RATE.set(hr)
    MRR_GAUGE.set(mrr_score)
    PRECISION_AT_K.labels(k="5").set(pk)
    if in_tokens:
        TOKEN_COUNT.labels(type="input").inc(in_tokens)
    if out_tokens:
        TOKEN_COUNT.labels(type="output").inc(out_tokens)

    # persist interaction
    try:
        save_interaction(
            doc_id=doc_id,
            query=body.query,
            answer=answer,
            llm_score=0.0,
            response_ms=response_ms,
            hit_rate=hr,
            mrr=mrr_score,
        )
    except Exception as exc:
        logger.warning(f"Failed to persist interaction: {exc}")

    # audit trail: log evidence chain + prompt hash
    request_id = None
    try:
        from database.audit import hash_prompt
        confidence = 0.5  # ES-based RAG does not produce a direct confidence score;
                          # replace with model logit confidence when available
        routed_to_hitl = False

        request_id = log_audit_event(
            question=body.query,
            answer=answer,
            confidence=confidence,
            docs=[],           # ES results are dicts, not LangChain docs; pass empty for now
            prompt_hash=hash_prompt(context),
            prompt_version="n/a",
            model_provider=CONFIG.get("llm", {}).get("provider", "unknown"),
            model_name=CONFIG.get("llm", {}).get("model", "unknown"),
            routed_to_hitl=routed_to_hitl,
            response_ms=response_ms,
        )
    except Exception as exc:
        logger.warning(f"Audit logging failed (non-fatal): {exc}")

    resp_data = {
        "doc_id": doc_id,
        "answer": answer,
        "response_time_ms": round(response_ms, 2),
        "hit_rate": round(hr, 4),
        "mrr": round(mrr_score, 4),
        "cache_hit": False,
    }
    if request_id:
        resp_data["request_id"] = request_id

    # store result in cache only for successful LLM responses
    if not answer.startswith("Error:"):
        set_cached(body.query, resp_data)

    return JSONResponse(content=resp_data)


@app.post("/query/stream", tags=["inference"])
@limiter.limit("20/minute")
async def query_stream(request: Request, body: QueryRequest):
    """
    Streaming LLM response via Server-Sent Events (SSE).

    Each event:   data: {"chunk": "...text..."}
    Final event:  data: [DONE]

    Connect with EventSource (browser) or any SSE client library.
    Cache is bypassed — the client receives a live token stream.
    """
    try:
        es = get_client()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"ElasticSearch unavailable: {exc}")

    rag_results = search(es, body.query)
    context = " ".join(r.get("answer", r.get("text", "")) for r in rag_results)

    from llm.ollama_client import astream_llm

    async def _generate():
        try:
            async for chunk in astream_llm(body.query, context):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as exc:
            logger.error(f"Streaming error: {exc}")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    REQUEST_COUNT.labels(status="success").inc()
    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/audit/{request_id}", tags=["audit"])
def audit_trail(request_id: str):
    """
    Retrieve the full evidence chain for a specific request.

    Returns the audit record including:
      - prompt_hash and prompt_version
      - evidence_chain: [{source, excerpt, score}] for each retrieved document
      - routed_to_hitl: whether the response was flagged for human review
      - model_provider, model_name, response_ms

    Use request_id from the 'request_id' field in /query responses.
    """
    records = get_audit_trail(request_id)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No audit trail found for request_id={request_id}",
        )
    return records


@app.post("/feedback", tags=["monitoring"])
@limiter.limit("60/minute")
def feedback(request: Request, body: FeedbackRequest):
    try:
        save_feedback(body.doc_id, body.rating, body.comment)
        return {"message": "Feedback saved."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate/ragas", response_model=RagasEvalResponse, tags=["evaluation"])
@limiter.limit("5/minute")
def evaluate_ragas(request: Request, samples: list[RagasSample]):
    """
    Run RAGAS evaluation (faithfulness, answer_relevancy, context_precision).
    Send a list of {question, answer, contexts, ground_truth?} objects.
    """
    if not samples:
        raise HTTPException(status_code=422, detail="Provide at least one sample.")
    try:
        from evaluation.ragas_eval import evaluate_with_ragas
        result = evaluate_with_ragas([s.model_dump() for s in samples])
        return RagasEvalResponse(**result)
    except Exception as exc:
        logger.error(f"RAGAS eval failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate/ragas/db", response_model=RagasEvalResponse, tags=["evaluation"])
@limiter.limit("5/minute")
def evaluate_ragas_from_db(request: Request, limit: int = 20):
    """
    Pull the last N interactions from PostgreSQL and run RAGAS on them.
    Enables continuous quality monitoring without manual input.
    """
    try:
        from evaluation.ragas_eval import evaluate_from_db
        result = evaluate_from_db(limit=limit)
        return RagasEvalResponse(**result)
    except Exception as exc:
        logger.error(f"RAGAS DB eval failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
