from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: str | Path,
    normalize: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    figure, axis = plt.subplots(figsize=(8, 8))
    display.plot(ax=axis, xticks_rotation=45, colorbar=False)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output