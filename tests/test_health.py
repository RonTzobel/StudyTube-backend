from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """The /health endpoint should return 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    """The root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "StudyTube" in response.json()["message"]
