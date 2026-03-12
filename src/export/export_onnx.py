from __future__ import annotations

import argparse

import onnx
import torch

from app.core.config import PROJECT_ROOT, load_experiment_config
from app.core.logging import get_logger, setup_logging
from app.models.classifier import load_model_from_checkpoint
from src.utils.device import get_device


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a trained classifier to ONNX format.")
    parser.add_argument("--config", default="experiments/configs/resnet18.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment_config(PROJECT_ROOT / args.config)
    device = get_device()
    model, _ = load_model_from_checkpoint(str(PROJECT_ROOT / config.model.checkpoint_path), device=device)
    model.eval()

    export_path = PROJECT_ROOT / config.export.onnx_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, config.training.image_size, config.training.image_size, device=device)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if config.export.dynamic_axes else None

    torch.onnx.export(
        model,
        dummy_input,
        export_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=config.export.opset_version,
    )

    onnx_model = onnx.load(str(export_path))
    onnx.checker.check_model(onnx_model)
    LOGGER.info("exported ONNX model to %s", export_path)


if __name__ == "__main__":
    main()