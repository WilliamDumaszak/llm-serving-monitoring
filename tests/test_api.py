import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("api.main.get_client")
@patch("api.main.search")
@patch("api.main.query_llm", return_value=("Test answer", 0.5))
@patch("api.main.save_interaction")
def test_query(mock_save, mock_llm, mock_search, mock_es, client):
    mock_search.return_value = [{"doc_id": "abc123", "answer": "some context"}]
    payload = {"query": "What is supply chain?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "response_time_ms" in data


def test_query_too_short(client):
    response = client.post("/query", json={"query": "hi"})
    assert response.status_code == 422


@patch("api.main.save_feedback")
def test_feedback(mock_save, client):
    payload = {"doc_id": "abc123", "rating": 5, "comment": "Great!"}
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "llm_request_total" in response.text
