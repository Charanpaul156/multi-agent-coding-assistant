"""Health endpoint tests."""

from fastapi.testclient import TestClient

from backend.main import app


def test_health_check() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

