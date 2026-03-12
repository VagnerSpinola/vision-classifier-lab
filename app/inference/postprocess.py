from __future__ import annotations

import torch

from app.inference.contracts import TopPrediction


def extract_prediction(logits: torch.Tensor, class_names: list[str]) -> tuple[str, float]:
    probabilities = torch.softmax(logits, dim=1)
    confidence, predicted_index = probabilities.max(dim=1)
    class_name = class_names[predicted_index.item()]
    return class_name, float(confidence.item())


def extract_top_k(logits: torch.Tensor, class_names: list[str], top_k: int) -> list[TopPrediction]:
    probabilities = torch.softmax(logits, dim=1)
    scores, indices = torch.topk(probabilities, k=min(top_k, len(class_names)), dim=1)
    return [
        TopPrediction(
            class_name=class_names[index.item()],
            score=float(score.item()),
        )
        for score, index in zip(scores[0], indices[0])
    ]