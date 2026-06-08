"""
Search backend — supports Elasticsearch (local) and Azure AI Search (cloud).
Provider is chosen via config.yaml (search.provider).

Public API (same regardless of provider):
  get_client()                   → opaque client object
  search(client, query)          → list[dict]
  index_document(client, doc)    → None
  ensure_index(client)           → None
  generate_doc_id(q, a)          → str
"""

import hashlib
import logging
import os

import yaml

logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

_PROVIDER = CONFIG.get("search", {}).get("provider", "elasticsearch")


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_doc_id(question: str, answer: str) -> str:
    combined = f"{question[:10]}-{answer[:10]}"
    return hashlib.md5(combined.encode()).hexdigest()[:8]


# ── Azure AI Search wrapper ───────────────────────────────────────────────────

class _AzureSearchWrapper:
    """Wraps azure.search.documents.SearchClient with an ES-compatible ping()."""

    def __init__(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        from azure.search.documents.indexes import SearchIndexClient

        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", CONFIG["search"].get("azure_search_endpoint", ""))
        api_key = os.getenv("AZURE_SEARCH_API_KEY", "")
        index_name = CONFIG["search"]["index_name"]
        credential = AzureKeyCredential(api_key)

        self._client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        self._index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
        self._index_name = index_name
        self._endpoint = endpoint
        self._credential = credential

    def ping(self) -> bool:
        try:
            self._index_client.get_index(self._index_name)
            return True
        except Exception:
            return False


# ── Factory ───────────────────────────────────────────────────────────────────

def get_client():
    """Return the active search client (Elasticsearch or Azure AI Search wrapper)."""
    if _PROVIDER == "azure_search":
        return _AzureSearchWrapper()
    from elasticsearch import Elasticsearch
    return Elasticsearch(CONFIG["search"]["elasticsearch_host"])


# ── Public operations ─────────────────────────────────────────────────────────

def ensure_index(client) -> None:
    """Create the index if it does not exist."""
    index = CONFIG["search"]["index_name"]

    if _PROVIDER == "azure_search":
        from azure.search.documents.indexes.models import (
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SimpleField,
            SearchableField,
        )
        fields = [
            SimpleField(name="doc_id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="question", type=SearchFieldDataType.String),
            SearchableField(name="answer", type=SearchFieldDataType.String),
            SearchableField(name="text", type=SearchFieldDataType.String),
        ]
        idx = SearchIndex(name=index, fields=fields)
        client._index_client.create_or_update_index(idx)
        logger.info(f"Azure AI Search index '{index}' ready.")
        return

    # Elasticsearch
    if not client.indices.exists(index=index):
        client.indices.create(
            index=index,
            body={
                "mappings": {
                    "properties": {
                        "doc_id": {"type": "keyword"},
                        "question": {"type": "text"},
                        "answer": {"type": "text"},
                        "text": {"type": "text"},
                    }
                }
            },
        )
        logger.info(f"Created ElasticSearch index '{index}'.")


def index_document(client, doc: dict) -> None:
    """Index a single document. doc must have: doc_id, question, answer, text."""
    if _PROVIDER == "azure_search":
        batch = [{"@search.action": "upload", "doc_id": doc["doc_id"], **doc}]
        client._client.upload_documents(documents=batch)
        return

    # Elasticsearch
    client.index(index=CONFIG["search"]["index_name"], id=doc["doc_id"], body=doc)


def search(client, query: str) -> list[dict]:
    """Search for documents matching query. Returns list of source dicts."""
    top_k = CONFIG["search"]["top_k"]

    if _PROVIDER == "azure_search":
        results = client._client.search(search_text=query, top=top_k)
        return [dict(r) for r in results]

    # Elasticsearch BM25
    index = CONFIG["search"]["index_name"]
    body = {
        "size": top_k,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^2", "text", "answer"],
                        "type": "best_fields",
                    }
                }
            }
        },
    }
    response = client.search(index=index, body=body)
    return [hit["_source"] for hit in response["hits"]["hits"]]

