from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_SPLITS = ("train", "val", "test")


def summarize_dataset(processed_dir: str | Path) -> pd.DataFrame:
    processed_path = Path(processed_dir)
    rows: list[dict[str, object]] = []

    for split in EXPECTED_SPLITS:
        split_dir = processed_path / split
        for class_dir in sorted(entry for entry in split_dir.iterdir() if entry.is_dir()):
            rows.append(
                {
                    "split": split,
                    "class_name": class_dir.name,
                    "image_count": len([path for path in class_dir.iterdir() if path.is_file()]),
                }
            )

    return pd.DataFrame(rows)


def validate_dataset_structure(processed_dir: str | Path) -> dict[str, object]:
    processed_path = Path(processed_dir)
    missing_splits = [split for split in EXPECTED_SPLITS if not (processed_path / split).exists()]
    if missing_splits:
        raise FileNotFoundError(f"Missing dataset splits: {missing_splits}")

    summary = summarize_dataset(processed_path)
    class_sets = {
        split: set(summary.loc[summary["split"] == split, "class_name"].tolist())
        for split in EXPECTED_SPLITS
    }
    consistent_classes = class_sets["train"] == class_sets["val"] == class_sets["test"]

    return {
        "processed_dir": str(processed_path),
        "total_images": int(summary["image_count"].sum()),
        "class_consistency": consistent_classes,
        "classes": sorted(class_sets["train"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed dataset structure and counts.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output", default="data/processed/dataset_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.processed_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    report = validate_dataset_structure(args.processed_dir)
    print(report)


if __name__ == "__main__":
    main()