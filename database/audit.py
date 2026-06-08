"""
Audit trail with evidence chain — PostgreSQL-backed.

Why an audit trail?
  For regulated industries (finance, healthcare, legal), every AI-generated response
  must be traceable back to:
    1. Exactly which documents were used to compose the answer (evidence chain)
    2. Which version of the prompt template was active
    3. Whether the response was flagged for human review
    4. The confidence score at the time of generation

  This enables post-hoc auditing ("why did the system say X on date Y?"),
  debugging regressions, and compliance reporting.

Schema:
  audit_log table:
    id                SERIAL PRIMARY KEY
    request_id        UUID     — ties together parallel audit records for one request
    prompt_hash       VARCHAR  — SHA-256 of the prompt template file used
    prompt_version    VARCHAR  — e.g. "v2"
    question          TEXT     — user query (PII-sanitized before storage)
    answer            TEXT     — LLM response
    confidence        FLOAT    — model confidence score
    doc_ids_used      JSONB    — list of document source identifiers retrieved
    evidence_chain    JSONB    — [{source, excerpt, score}, ...] for top retrieved docs
    routed_to_hitl    BOOLEAN  — was this flagged for human review?
    model_provider    VARCHAR  — "ollama" | "azure_openai" | "vllm" | "huggingface"
    model_name        VARCHAR  — specific model identifier
    response_ms       FLOAT    — end-to-end latency in milliseconds
    created_at        TIMESTAMPTZ DEFAULT NOW()
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone

import psycopg2
import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


# ── Connection ────────────────────────────────────────────────────────────────

def get_connection():
    db = CONFIG["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        user=db["user"],
        password=db["password"],
        dbname=db["dbname"],
    )


# ── Schema ────────────────────────────────────────────────────────────────────

def create_audit_table() -> None:
    """Create audit_log table if it does not exist."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id             SERIAL PRIMARY KEY,
                request_id     UUID         NOT NULL,
                prompt_hash    VARCHAR(64)  NOT NULL DEFAULT '',
                prompt_version VARCHAR(16)  NOT NULL DEFAULT 'unknown',
                question       TEXT         NOT NULL,
                answer         TEXT,
                confidence     FLOAT,
                doc_ids_used   JSONB        DEFAULT '[]',
                evidence_chain JSONB        DEFAULT '[]',
                routed_to_hitl BOOLEAN      DEFAULT FALSE,
                model_provider VARCHAR(32),
                model_name     VARCHAR(128),
                response_ms    FLOAT,
                created_at     TIMESTAMPTZ  DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_request_id
            ON audit_log (request_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_created_at
            ON audit_log (created_at)
        """)
        conn.commit()
        logger.info("Audit log table ready.")
    except Exception as exc:
        conn.rollback()
        logger.error(f"Audit table setup failed: {exc}")
    finally:
        cur.close()
        conn.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_prompt(prompt_text: str) -> str:
    """Return SHA-256 hash of a prompt string. Used as a stable identifier."""
    return hashlib.sha256(prompt_text.encode()).hexdigest()


def build_evidence_chain(docs: list) -> list[dict]:
    """
    Build an evidence chain from retrieved LangChain Documents.

    Each entry captures:
      source  — document filename or identifier
      excerpt — first 200 chars of the chunk (enough for audit, not too verbose)
      score   — retrieval similarity score (if available in metadata)

    This lets auditors trace exactly which text passages informed the answer.
    """
    chain = []
    for doc in docs:
        chain.append({
            "source": doc.metadata.get("source", "unknown"),
            "excerpt": doc.page_content[:200],
            "score": doc.metadata.get("score", None),
            "page": doc.metadata.get("page", None),
        })
    return chain


# ── Write ─────────────────────────────────────────────────────────────────────

def log_audit_event(
    question: str,
    answer: str,
    confidence: float,
    docs: list,
    prompt_hash: str,
    prompt_version: str,
    model_provider: str,
    model_name: str,
    routed_to_hitl: bool = False,
    response_ms: float = 0.0,
    request_id: str | None = None,
) -> str:
    """
    Record a complete audit trail entry for one LLM request.

    Args:
        question:       User query (should be PII-sanitized before calling this).
        answer:         LLM-generated answer.
        confidence:     Model confidence score.
        docs:           List of LangChain Documents used as retrieval context.
        prompt_hash:    SHA-256 of the prompt template file.
        prompt_version: Version string of the prompt (e.g. "v2").
        model_provider: LLM provider name.
        model_name:     Specific model identifier.
        routed_to_hitl: Whether this response was sent to the HITL queue.
        response_ms:    End-to-end latency in milliseconds.
        request_id:     Optional UUID to correlate logs. Auto-generated if not provided.

    Returns:
        request_id string (UUID) for downstream use / response header injection.
    """
    req_id = request_id or str(uuid.uuid4())
    evidence_chain = build_evidence_chain(docs)
    doc_ids = [e["source"] for e in evidence_chain]

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO audit_log (
                request_id, prompt_hash, prompt_version, question, answer,
                confidence, doc_ids_used, evidence_chain, routed_to_hitl,
                model_provider, model_name, response_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                req_id,
                prompt_hash,
                prompt_version,
                question,
                answer,
                confidence,
                json.dumps(doc_ids),
                json.dumps(evidence_chain),
                routed_to_hitl,
                model_provider,
                model_name,
                response_ms,
            ),
        )
        conn.commit()
        logger.info(f"Audit event logged: request_id={req_id}")
    except Exception as exc:
        conn.rollback()
        logger.error(f"Failed to log audit event: {exc}")
    finally:
        cur.close()
        conn.close()

    return req_id


# ── Read ──────────────────────────────────────────────────────────────────────

def get_audit_trail(request_id: str) -> list[dict]:
    """
    Retrieve the full audit trail for a specific request ID.

    Returns a list of audit records (usually one, but can be multiple if
    the request spawned sub-calls, e.g. in a multi-agent setup).
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT id, request_id, prompt_hash, prompt_version, question, answer,
                      confidence, doc_ids_used, evidence_chain, routed_to_hitl,
                      model_provider, model_name, response_ms, created_at
               FROM audit_log
               WHERE request_id = %s
               ORDER BY created_at ASC""",
            (request_id,),
        )
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        result = []
        for row in rows:
            record = dict(zip(columns, row))
            # psycopg2 returns JSONB as strings; parse them
            for field in ("doc_ids_used", "evidence_chain"):
                if isinstance(record.get(field), str):
                    record[field] = json.loads(record[field])
            # Serialize datetime to ISO string
            if hasattr(record.get("created_at"), "isoformat"):
                record["created_at"] = record["created_at"].isoformat()
            result.append(record)
        return result
    except Exception as exc:
        logger.error(f"Failed to get audit trail: {exc}")
        return []
    finally:
        cur.close()
        conn.close()


def get_recent_audit_events(limit: int = 50) -> list[dict]:
    """Return the most recent audit log entries for monitoring dashboards."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT id, request_id, prompt_version, confidence, routed_to_hitl,
                      model_provider, model_name, response_ms, created_at
               FROM audit_log
               ORDER BY created_at DESC
               LIMIT %s""",
            (limit,),
        )
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        result = []
        for row in rows:
            record = dict(zip(columns, row))
            if hasattr(record.get("created_at"), "isoformat"):
                record["created_at"] = record["created_at"].isoformat()
            result.append(record)
        return result
    except Exception as exc:
        logger.error(f"Failed to get recent audit events: {exc}")
        return []
    finally:
        cur.close()
        conn.close()
