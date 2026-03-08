import io

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def make_video_file(content: bytes = b"fake video content", filename: str = "test.mp4"):
    """Helper that builds a fake in-memory file for upload tests."""
    return ("file", (filename, io.BytesIO(content), "video/mp4"))


def test_upload_video_success():
    """A valid video file should return 200 with status 'uploaded'."""
    response = client.post(
        "/api/v1/videos/upload",
        files=[make_video_file()],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "uploaded"
    assert body["original_filename"] == "test.mp4"
    assert "filename" in body  # the unique saved name


def test_upload_video_wrong_type():
    """A non-video file (e.g. plain text) should be rejected with 400."""
    response = client.post(
        "/api/v1/videos/upload",
        files=[("file", ("notes.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
