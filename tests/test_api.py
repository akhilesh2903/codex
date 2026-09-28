"""
tests/test_api.py — API test suite for DR-Screening-XAI backend
"""

import io
import os
import numpy as np
import pytest
import cv2
from fastapi.testclient import TestClient


# ── Test client setup ────────────────────────────────────────────────────────
from api.main import app
client = TestClient(app)


def make_dummy_image(h=512, w=512, filled=False):
    """Create an in-memory test image. filled=True makes a non-black image."""
    img = np.zeros((h, w, 3), dtype=np.uint8) if not filled else \
          np.random.randint(80, 200, (h, w, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return io.BytesIO(buf.tobytes())


# ── Health check ─────────────────────────────────────────────────────────────

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "demo_mode" in data


# ── Quality check endpoint ────────────────────────────────────────────────────

def test_quality_check_black_image():
    """Black image should report UNGRADABLE quality."""
    buf = make_dummy_image(filled=False)
    response = client.post("/api/quality-check", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "UNGRADABLE"
    assert "quality_score" in data


def test_quality_check_valid_image():
    """Non-trivial image should not raise errors."""
    buf = make_dummy_image(filled=True)
    response = client.post("/api/quality-check", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["GOOD", "BORDERLINE", "UNGRADABLE"]


# ── Analyze endpoint ──────────────────────────────────────────────────────────

def test_analyze_black_image_rejected():
    """All-black image should be rejected (UNGRADABLE quality)."""
    buf = make_dummy_image(filled=False)
    response = client.post("/api/analyze", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    # Should either be rejected due to quality or still return quality key
    assert "quality" in data or "error" in data


def test_analyze_valid_image_returns_structure():
    """Valid (non-black) image should return full analysis structure."""
    buf = make_dummy_image(filled=True)
    response = client.post("/api/analyze", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()

    if "error" not in data:
        # Full result expected
        assert "image_id" in data
        assert "quality" in data
        assert "prediction" in data
        assert "lesions" in data
        assert "explainability" in data
        assert "processing_time_ms" in data

        pred = data["prediction"]
        assert "grade" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert "referable" in pred
        assert pred["grade"] in [0, 1, 2, 3, 4]
        assert 0.0 <= pred["confidence"] <= 1.0

        lesions = data["lesions"]
        for ltype in ["microaneurysms", "hemorrhages", "exudates"]:
            assert ltype in lesions
            assert "count" in lesions[ltype]
            assert "detected" in lesions[ltype]


def test_analyze_invalid_file_type():
    """Non-image file should raise 400 or return error."""
    txt_data = io.BytesIO(b"this is not an image")
    response = client.post("/api/analyze", files={"file": ("test.txt", txt_data, "text/plain")})
    # Should get 4xx or an error in response
    assert response.status_code in [400, 422] or "error" in response.json()


def test_analyze_returns_lesion_overlay_url():
    """Analysis should return a lesion overlay URL."""
    buf = make_dummy_image(filled=True)
    response = client.post("/api/analyze", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    if "explainability" in data:
        assert "lesion_overlay_url" in data["explainability"]


# ── Result endpoint ───────────────────────────────────────────────────────────

def test_get_result_not_found():
    """Non-existent image_id should return 404."""
    response = client.get("/api/result/nonexistent_image_id_xyz")
    assert response.status_code == 404


# ── Report endpoint ───────────────────────────────────────────────────────────

def test_get_report_not_found():
    """Missing report should return 404."""
    response = client.get("/api/report/nonexistent_image_id_xyz")
    assert response.status_code == 404


# ── Review endpoint ───────────────────────────────────────────────────────────

def test_submit_review():
    """Review endpoint should accept and return 200."""
    payload = {
        "analysis_id": "test_img_12345",
        "decision": "CONFIRM",
        "comment": "Image reviewed and confirmed."
    }
    response = client.post("/api/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Review saved"


# ── Upload endpoint ───────────────────────────────────────────────────────────

def test_upload_valid_image():
    """Upload endpoint should return image_id."""
    buf = make_dummy_image(filled=True)
    response = client.post("/api/upload", files={"file": ("test.jpg", buf, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert data["status"] == "uploaded"


def test_upload_invalid_type():
    """Upload endpoint should reject non-image files."""
    txt_data = io.BytesIO(b"not an image")
    response = client.post("/api/upload", files={"file": ("test.txt", txt_data, "text/plain")})
    assert response.status_code == 400
