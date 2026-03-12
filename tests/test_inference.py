import torch

from app.inference.postprocess import extract_top_k
from app.inference.predictor import TorchImagePredictor


def test_extract_top_k_returns_ranked_predictions() -> None:
    logits = torch.tensor([[3.0, 1.0, 0.5]])
    results = extract_top_k(logits, ["cat", "dog", "bird"], top_k=2)

    assert results[0].class_name == "cat"
    assert len(results) == 2


def test_torch_predictor_metadata_defaults() -> None:
    predictor = TorchImagePredictor()

    metadata = predictor.metadata()

    assert metadata.backend == "torch"
    assert metadata.loaded is False