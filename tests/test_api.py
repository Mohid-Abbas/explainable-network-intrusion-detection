import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_model_info():
    resp = client.get("/model/info")
    assert resp.status_code == 200
    assert "model_version" in resp.json()


def test_predict_missing_model():
    resp = client.post("/predict", json={"features": {}})
    assert resp.status_code in (200, 500)
