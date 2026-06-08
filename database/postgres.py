"""
PostgreSQL storage for interactions, feedback, and evaluation results.
"""

import logging

import psycopg2
import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def get_connection():
    db = CONFIG["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        user=db["user"],
        password=db["password"],
        dbname=db["dbname"],
    )


def setup_tables() -> None:
    """Create tables if they do not exist."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id          SERIAL PRIMARY KEY,
                doc_id      VARCHAR(16),
                query       TEXT NOT NULL,
                answer      TEXT,
                llm_score   FLOAT,
                response_ms FLOAT,
                hit_rate    FLOAT,
                mrr         FLOAT,
                created_at  TIMESTAMP DEFAULT NOW()
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id             SERIAL PRIMARY KEY,
                doc_id         VARCHAR(16),
                rating         SMALLINT CHECK (rating BETWEEN 1 AND 5),
                comment        TEXT,
                created_at     TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        logger.info("Database tables ready.")
    except Exception as exc:
        conn.rollback()
        logger.error(f"Table setup failed: {exc}")
    finally:
        cur.close()
        conn.close()


def save_interaction(
    doc_id: str,
    query: str,
    answer: str,
    llm_score: float,
    response_ms: float,
    hit_rate: float,
    mrr: float,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """INSERT INTO interactions
               (doc_id, query, answer, llm_score, response_ms, hit_rate, mrr)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (doc_id, query, answer, llm_score, response_ms, hit_rate, mrr),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Failed to save interaction: {exc}")
    finally:
        cur.close()
        conn.close()


def save_feedback(doc_id: str, rating: int, comment: str = "") -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO feedback (doc_id, rating, comment) VALUES (%s, %s, %s)",
            (doc_id, rating, comment),
        )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error(f"Failed to save feedback: {exc}")
    finally:
        cur.close()
        conn.close()
