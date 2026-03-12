from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import write_dataframe


def save_error_analysis(
    sample_paths: list[str],
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    class_names: list[str],
    output_path: str | Path,
    limit: int = 25,
) -> Path:
    rows: list[dict[str, object]] = []
    for sample_path, true_index, pred_index, confidence in zip(sample_paths, y_true, y_pred, confidences):
        if true_index == pred_index:
            continue
        rows.append(
            {
                "sample_path": sample_path,
                "true_class": class_names[true_index],
                "predicted_class": class_names[pred_index],
                "confidence": confidence,
            }
        )

    if not rows:
        dataframe = pd.DataFrame(columns=["sample_path", "true_class", "predicted_class", "confidence"])
    else:
        dataframe = pd.DataFrame(rows).sort_values(by="confidence", ascending=False).head(limit)
    return write_dataframe(dataframe, output_path)