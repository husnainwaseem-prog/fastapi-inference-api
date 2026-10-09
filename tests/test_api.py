"""Tests for the inference & embeddings API.

Covers: health/info endpoints, prediction correctness, input validation,
embeddings, similarity scoring, and the sub-200ms latency budget.
"""

import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

CLASSES = ["Complaint", "Feedback", "Inquiry", "Other"]


@pytest.fixture(scope="module")
def client():
    # The context manager runs the app lifespan (model loads once here).
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_info(client):
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["classes"] == CLASSES
    assert body["embedding_dim"] > 0
    assert "TF-IDF" in body["vectorizer"]


def test_predict_complaint(client):
    response = client.post("/predict", json={"text": "My order arrived broken and nobody replied."})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01
    assert body["latency_ms"] < config.LATENCY_BUDGET_MS


def test_predict_inquiry(client):
    response = client.post("/predict", json={"text": "Do you deliver on Sundays?"})
    assert response.status_code == 200
    assert response.json()["label"] == "Inquiry"


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_too_long_text(client):
    response = client.post("/predict", json={"text": "x" * (config.MAX_TEXT_LENGTH + 1)})
    assert response.status_code == 422


def test_predict_batch(client):
    texts = [
        "The refund I was promised never showed up.",
        "Do you deliver on Sundays?",
        "The service was very helpful.",
    ]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [p["label"] for p in body["predictions"]] == ["Complaint", "Inquiry", "Feedback"]
    assert body["latency_ms"] < config.LATENCY_BUDGET_MS


def test_predict_batch_rejects_empty_list(client):
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_batch_rejects_empty_or_oversized_item(client):
    assert client.post("/predict/batch", json={"texts": ["ok", ""]}).status_code == 422
    too_long = "x" * (config.MAX_TEXT_LENGTH + 1)
    assert client.post("/predict/batch", json={"texts": [too_long]}).status_code == 422


def test_embed_shape_and_norm(client):
    response = client.post("/embed", json={"text": "hello world, this is a test"})
    assert response.status_code == 200
    body = response.json()
    info = client.get("/info").json()
    assert body["dim"] == info["embedding_dim"]
    assert len(body["embedding"]) == body["dim"]
    norm = sum(x * x for x in body["embedding"]) ** 0.5
    assert abs(norm - 1.0) < 1e-6  # L2-normalised unit vector


def test_similarity_identical_texts(client):
    response = client.post(
        "/similarity",
        json={"text_a": "the quick brown fox", "text_b": "the quick brown fox"},
    )
    assert response.status_code == 200
    assert abs(response.json()["similarity"] - 1.0) < 1e-6


def test_similarity_unrelated_texts_lower(client):
    response = client.post(
        "/similarity",
        json={
            "text_a": "my refund never arrived and support ignores me",
            "text_b": "what are your opening hours on weekends",
        },
    )
    assert response.status_code == 200
    assert 0.0 <= response.json()["similarity"] < 0.9


def test_latency_budget_over_many_requests(client):
    """Every single prediction must stay under the 200 ms budget."""
    for _ in range(15):
        response = client.post("/predict", json={"text": "Where is my parcel?"})
        assert response.status_code == 200
        assert response.json()["latency_ms"] < config.LATENCY_BUDGET_MS
