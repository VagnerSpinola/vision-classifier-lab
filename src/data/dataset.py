from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from app.core.config import AugmentationConfig
from src.data.transforms import build_eval_transform, build_train_transform


SUPPORTED_IMBALANCE_STRATEGIES = frozenset({"none", "weighted_loss", "weighted_sampler"})


@dataclass(slots=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    class_weights: torch.Tensor | None
    imbalance_strategy: str


def create_dataloaders(
    processed_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augmentation: AugmentationConfig | None = None,
    imbalance_strategy: str = "none",
) -> DataLoaders:
    processed_path = Path(processed_dir)
    _validate_imbalance_strategy(imbalance_strategy)

    train_dataset = ImageFolder(
        root=processed_path / "train",
        transform=build_train_transform(image_size, augmentation=augmentation),
    )
    val_dataset = ImageFolder(
        root=processed_path / "val",
        transform=build_eval_transform(image_size),
    )
    test_dataset = ImageFolder(
        root=processed_path / "test",
        transform=build_eval_transform(image_size),
    )

    class_weights = _compute_class_weights(train_dataset.targets)
    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if imbalance_strategy == "weighted_sampler":
        sampler = _build_weighted_sampler(train_dataset.targets, class_weights)
        shuffle = False

    return DataLoaders(
        train=_build_dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
        ),
        val=_build_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=None,
        ),
        test=_build_dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=None,
        ),
        class_names=train_dataset.classes,
        class_weights=class_weights if imbalance_strategy in {"weighted_loss", "weighted_sampler"} else None,
        imbalance_strategy=imbalance_strategy,
    )


def _build_dataloader(
    dataset: ImageFolder,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _validate_imbalance_strategy(strategy: str) -> None:
    if strategy not in SUPPORTED_IMBALANCE_STRATEGIES:
        raise ValueError(
            f"Unsupported imbalance strategy '{strategy}'. Expected one of {sorted(SUPPORTED_IMBALANCE_STRATEGIES)}"
        )


def _compute_class_weights(targets: list[int]) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(targets), minlength=max(targets, default=0) + 1).float()
    counts = torch.where(counts == 0, torch.ones_like(counts), counts)
    weights = counts.sum() / counts
    return weights / weights.sum() * len(weights)


def _build_weighted_sampler(targets: list[int], class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = class_weights[torch.tensor(targets)].double()
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)