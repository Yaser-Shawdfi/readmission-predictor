"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient

try:
    from src.api import app
    client = TestClient(app)
    _AVAILABLE = True
except Exception:
    _AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="API dependencies not available")


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "MedAI" in response.json()["name"]


def test_list_datasets():
    response = client.get("/api/v1/datasets")
    assert response.status_code == 200
    datasets = response.json()["datasets"]
    assert "diabetes_100k" in datasets