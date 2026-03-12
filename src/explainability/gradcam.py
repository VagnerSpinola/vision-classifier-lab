from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from app.models.classifier import get_gradcam_target_layer


class GradCAM:
    def __init__(self, model: nn.Module, architecture: str) -> None:
        self.model = model
        self.target_layer = get_gradcam_target_layer(model, architecture)
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        logits = self.model(input_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())

        self.model.zero_grad(set_to_none=True)
        logits[:, class_index].backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = (pooled_gradients * self.activations).sum(dim=1).squeeze(0)
        heatmap = torch.relu(weighted_activations)
        heatmap /= heatmap.max().clamp(min=1e-8)
        return heatmap.cpu().numpy()


def save_gradcam_overlay(image: Image.Image, heatmap: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image)
    axis.imshow(heatmap, cmap="jet", alpha=0.4)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output