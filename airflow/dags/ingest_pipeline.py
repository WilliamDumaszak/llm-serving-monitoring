"""
Airflow DAG — daily document ingestion pipeline for the LLM RAG index.

Flow:
  check_elasticsearch → load_documents → index_to_elasticsearch → done
"""

import json
import os
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

DEFAULT_ARGS = {
    "owner": "llm-serving-monitoring",
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

DOCS_DIR = os.getenv("DOCS_DIR", "/opt/airflow/documents")
ES_HOST = os.getenv("ES_HOST", "http://elasticsearch:9200")
INDEX_NAME = os.getenv("ES_INDEX", "rag_documents")


def _check_elasticsearch():
    import requests
    resp = requests.get(ES_HOST, timeout=10)
    resp.raise_for_status()
    print(f"ElasticSearch is up: {resp.json()}")


def _load_and_index_documents():
    import hashlib
    import requests

    if not os.path.isdir(DOCS_DIR):
        raise FileNotFoundError(f"Documents directory not found: {DOCS_DIR}")

    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files to index.")

    indexed = 0
    for fname in files:
        fpath = os.path.join(DOCS_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)

        # support list of docs or single doc
        docs = data if isinstance(data, list) else [data]

        for doc in docs:
            text = " ".join(f"{k}: {v}" for k, v in doc.items())
            question = doc.get("question", doc.get("title", fname))
            answer = doc.get("answer", doc.get("content", text[:200]))
            doc_id = hashlib.md5(text[:20].encode()).hexdigest()[:8]

            payload = {
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "text": text,
            }
            url = f"{ES_HOST}/{INDEX_NAME}/_doc/{doc_id}"
            requests.put(url, json=payload, timeout=10)
            indexed += 1

    print(f"Indexed {indexed} documents into '{INDEX_NAME}'.")


with DAG(
    dag_id="llm_rag_ingest_pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",   # daily at 02:00
    description="Daily document ingestion into ElasticSearch RAG index",
    catchup=False,
    tags=["llm", "rag", "ingestion"],
) as dag:

    check_es = PythonOperator(
        task_id="check_elasticsearch",
        python_callable=_check_elasticsearch,
    )

    index_docs = PythonOperator(
        task_id="index_documents",
        python_callable=_load_and_index_documents,
    )

    check_es >> index_docs
