from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image
import torch
from torch import Tensor

from src.data.transforms import build_eval_transform


def load_image_from_bytes(payload: bytes) -> Image.Image:
    image = Image.open(BytesIO(payload)).convert("RGB")
    return image


def load_image_from_path(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def preprocess_image(image: Image.Image, image_size: int) -> Tensor:
    tensor = build_eval_transform(image_size)(image)
    return tensor.unsqueeze(0)


def preprocess_images(images: list[Image.Image], image_size: int) -> Tensor:
    tensors = [build_eval_transform(image_size)(image) for image in images]
    return torch.stack(tensors)