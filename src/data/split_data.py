from __future__ import annotations

import argparse
import random
import shutil
from collections.abc import Iterable
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    clear_output: bool = False,
) -> None:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be less than 1.0")

    if clear_output and output_path.exists():
        shutil.rmtree(output_path)

    rng = random.Random(seed)

    for class_dir in sorted(entry for entry in raw_path.iterdir() if entry.is_dir()):
        images = list(_iter_images(class_dir))
        rng.shuffle(images)

        total = len(images)
        train_cutoff = int(total * train_ratio)
        val_cutoff = train_cutoff + int(total * val_ratio)

        partitions = {
            "train": images[:train_cutoff],
            "val": images[train_cutoff:val_cutoff],
            "test": images[val_cutoff:],
        }

        for split_name, split_images in partitions.items():
            destination = output_path / split_name / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, destination / image_path.name)


def _iter_images(class_dir: Path) -> Iterable[Path]:
    for candidate in class_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            yield candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split raw image folders into train/val/test datasets.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    split_dataset(
        raw_dir=arguments.raw_dir,
        output_dir=arguments.output_dir,
        train_ratio=arguments.train_ratio,
        val_ratio=arguments.val_ratio,
        seed=arguments.seed,
        clear_output=arguments.clear_output,
    )