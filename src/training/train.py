from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.core.config import PROJECT_ROOT, load_experiment_config
from app.core.logging import get_logger, setup_logging
from app.models.classifier import build_classifier
from mlops.mlflow.tracking import MLflowTracker
from src.data.dataset import create_dataloaders
from src.training.loss import build_loss
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.utils.io import write_json
from src.utils.seed import seed_everything


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument(
        "--config",
        default="experiments/configs/resnet18.yaml",
        help="Path to the experiment YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_experiment_config(PROJECT_ROOT / args.config)
    seed_everything(config.seed)
    device = get_device()

    dataloaders = create_dataloaders(
        processed_dir=PROJECT_ROOT / config.data.processed_dir,
        image_size=config.training.image_size,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        augmentation=config.data.augmentation,
        imbalance_strategy=config.data.imbalance_strategy,
    )

    model = build_classifier(
        architecture=config.model.architecture,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        freeze_backbone=config.model.freeze_backbone,
        dropout=config.model.dropout,
    )
    model.to(device)

    criterion = build_loss(
        class_weights=dataloaders.class_weights if config.data.imbalance_strategy == "weighted_loss" else None,
        label_smoothing=config.training.label_smoothing,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        params=(parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    with MLflowTracker(config) as tracker:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            class_names=dataloaders.class_names,
            checkpoint_path=PROJECT_ROOT / config.model.checkpoint_path,
            registry_path=PROJECT_ROOT / config.model.registry_path,
            architecture=config.model.architecture,
            image_size=config.training.image_size,
            best_metric=config.training.best_metric,
            early_stopping_patience=config.training.early_stopping_patience,
            tracker=tracker,
        )

        history = trainer.fit(
            train_loader=dataloaders.train,
            val_loader=dataloaders.val,
            epochs=config.training.epochs,
        )
        test_metrics = trainer.evaluate(dataloaders.test, split_name="test")

        history_path = Path(PROJECT_ROOT / config.model.checkpoint_path).with_suffix(".history.json")
        write_json(
            {
                "history": history.epochs,
                "best_checkpoint": history.best_checkpoint,
                "best_metric": history.best_metric,
                "test_metrics": test_metrics,
            },
            history_path,
        )
        tracker.log_artifact(history_path)
        LOGGER.info("training complete on device=%s; history saved to %s", device, history_path)


if __name__ == "__main__":
    main()