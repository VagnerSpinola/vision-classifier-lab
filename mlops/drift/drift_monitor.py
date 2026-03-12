from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def compute_distribution_shift(reference: pd.Series, current: pd.Series) -> float:
    reference_distribution = reference.value_counts(normalize=True)
    current_distribution = current.value_counts(normalize=True)
    aligned = pd.concat([reference_distribution, current_distribution], axis=1).fillna(0.0)
    aligned.columns = ["reference", "current"]
    return float((aligned["reference"] - aligned["current"]).abs().sum())


def run_drift_monitor(reference_csv: str | Path, current_csv: str | Path) -> dict[str, float | str]:
    reference = pd.read_csv(reference_csv)
    current = pd.read_csv(current_csv)
    shift_score = compute_distribution_shift(reference["predicted_class"], current["predicted_class"])
    return {
        "reference_path": str(reference_csv),
        "current_path": str(current_csv),
        "prediction_distribution_shift": shift_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prediction distributions between reference and current inference windows.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(run_drift_monitor(arguments.reference, arguments.current))