from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from src.utils.io import write_dataframe, write_json


def save_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_dir: str | Path,
    experiment_name: str,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    json_path = write_json(report, output_path / f"{experiment_name}_classification_report.json")
    csv_path = write_dataframe(
        pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"}),
        output_path / f"{experiment_name}_classification_report.csv",
    )
    return json_path, csv_path