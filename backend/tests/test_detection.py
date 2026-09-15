"""Fake services throughout - no trained model, checkpoint, or GPU needed."""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.api.v1.endpoints.detection import get_gradcam_service, get_image_size, get_inference_service


class FakeGradCAMService:
    def predict_and_explain(self, rgb_tensor, freq_tensor, original_image):
        return {
            "label": "fake",
            "confidence": 0.91,
            "probabilities": {"real": 0.09, "fake": 0.91},
            "heatmap_image": "data:image/png;base64,iVBORw0KGgo=",
        }


class FakeInferenceService:
    def predict(self, rgb_array, freq_array):
        return {"label": "real", "confidence": 0.87, "probabilities": {"real": 0.87, "fake": 0.13}}


def _sample_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), color=(128, 64, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


app.dependency_overrides[get_gradcam_service] = lambda: FakeGradCAMService()
app.dependency_overrides[get_inference_service] = lambda: FakeInferenceService()
app.dependency_overrides[get_image_size] = lambda: 224

client = TestClient(app)  # no `with` block needed - nothing here depends on lifespan anymore


def test_health_check() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_heatmap_and_confidence() -> None:
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "fake"
    assert body["heatmap_image"].startswith("data:image/png;base64,")


def test_predict_fast_returns_classification_only() -> None:
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    response = client.post("/api/v1/predict/fast", files=files)

    assert response.status_code == 200
    assert "heatmap_image" not in response.json()


def test_predict_rejects_non_image_file() -> None:
    files = {"file": ("not_an_image.pdf", b"%PDF-1.4 fake content", "application/pdf")}
    response = client.post("/api/v1/predict", files=files)

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_image"
