from fastapi.testclient import TestClient

from app.main import app
from app.inference.contracts import PredictionResult, PredictorMetadata, TopPrediction
from app.inference.predictor import prediction_service


client = TestClient(app)


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint_returns_prometheus_payload() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "vision_classifier_requests_total" in response.text


def test_predict_endpoint_returns_top_k(monkeypatch) -> None:
    def fake_predict(_: bytes) -> PredictionResult:
        return PredictionResult(
            predicted_class="class_a",
            confidence=0.97,
            top_k=[
                TopPrediction(class_name="class_a", score=0.97),
                TopPrediction(class_name="class_b", score=0.03),
            ],
        )

    monkeypatch.setattr(prediction_service, "predict_bytes", fake_predict)
    monkeypatch.setattr(
        prediction_service,
        "metadata",
        lambda: PredictorMetadata(
            backend="torch",
            loaded=True,
            class_names=("class_a", "class_b"),
            model_version="test-model",
        ),
    )

    response = client.post(
        "/predict",
        files={"file": ("image.jpg", b"binary-image", "image/jpeg")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "class_a"
    assert len(body["top_k"]) == 2


def test_batch_predict_endpoint_returns_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        prediction_service,
        "predict_batch",
        lambda payloads: [
            PredictionResult(
                predicted_class="class_a",
                confidence=0.9,
                top_k=[TopPrediction(class_name="class_a", score=0.9)],
            )
            for _ in payloads
        ],
    )
    monkeypatch.setattr(
        prediction_service,
        "metadata",
        lambda: PredictorMetadata(
            backend="torch",
            loaded=True,
            class_names=("class_a", "class_b"),
            model_version="test-model",
        ),
    )

    response = client.post(
        "/predict/batch",
        files=[
            ("files", ("one.jpg", b"a", "image/jpeg")),
            ("files", ("two.jpg", b"b", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 2