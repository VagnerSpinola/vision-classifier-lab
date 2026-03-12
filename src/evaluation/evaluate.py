from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.core.config import PROJECT_ROOT, load_experiment_config
from app.core.logging import get_logger, setup_logging
from app.models.classifier import load_model_from_checkpoint
from mlops.mlflow.tracking import MLflowTracker
from src.data.dataset import create_dataloaders
from src.evaluation.classification_report import save_classification_report
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.error_analysis import save_error_analysis
from src.evaluation.metrics import compute_classification_metrics
from src.utils.device import get_device
from src.utils.io import write_json


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classification model.")
    parser.add_argument("--config", default="experiments/configs/resnet18.yaml")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment_config(PROJECT_ROOT / args.config)
    device = get_device()

    dataloaders = create_dataloaders(
        processed_dir=PROJECT_ROOT / config.data.processed_dir,
        image_size=config.training.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        augmentation=config.data.augmentation,
        imbalance_strategy=config.data.imbalance_strategy,
    )

    checkpoint_path = PROJECT_ROOT / config.model.checkpoint_path
    model, checkpoint = load_model_from_checkpoint(str(checkpoint_path), device=device)

    targets: list[int] = []
    predictions: list[int] = []
    confidences: list[float] = []
    sample_paths: list[str] = []

    test_dataset = dataloaders.test.dataset
    dataset_samples = getattr(test_dataset, "samples", [])
    seen_examples = 0

    with torch.inference_mode():
        for inputs, labels in dataloaders.test:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            confidence_values, predicted_indices = probabilities.max(dim=1)
            predicted = predicted_indices.cpu().tolist()
            predictions.extend(predicted)
            targets.extend(labels.tolist())
            confidences.extend(confidence_values.cpu().tolist())
            batch_size = len(labels)
            batch_paths = dataset_samples[seen_examples : seen_examples + batch_size]
            sample_paths.extend([str(path) for path, _ in batch_paths])
            seen_examples += batch_size

    metrics = compute_classification_metrics(targets, predictions)
    output_dir = PROJECT_ROOT / config.evaluation.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = write_json(metrics, output_dir / f"{config.experiment_name}_metrics.json")

    confusion_matrix_path = output_dir / f"{config.experiment_name}_confusion_matrix.png"
    class_names = checkpoint.get("class_names", dataloaders.class_names)
    save_confusion_matrix(
        y_true=targets,
        y_pred=predictions,
        class_names=list(class_names),
        output_path=confusion_matrix_path,
    )

    report_json, report_csv = save_classification_report(
        y_true=targets,
        y_pred=predictions,
        class_names=list(class_names),
        output_dir=output_dir,
        experiment_name=config.experiment_name,
    )

    error_analysis_path = save_error_analysis(
        sample_paths=sample_paths,
        y_true=targets,
        y_pred=predictions,
        confidences=confidences,
        class_names=list(class_names),
        output_path=output_dir / f"{config.experiment_name}_error_analysis.csv",
        limit=config.evaluation.max_error_samples,
    )

    with MLflowTracker(config) as tracker:
        tracker.log_metrics({f"eval_{key}": value for key, value in metrics.items()})
        tracker.log_artifacts([metrics_path, confusion_matrix_path, report_json, report_csv, error_analysis_path])

    LOGGER.info("evaluation complete. metrics=%s", metrics)
    LOGGER.info("saved metrics to %s", metrics_path)
    LOGGER.info("saved confusion matrix to %s", confusion_matrix_path)
    LOGGER.info("saved classification report to %s", report_json)
    LOGGER.info("saved error analysis to %s", error_analysis_path)


if __name__ == "__main__":
    main()