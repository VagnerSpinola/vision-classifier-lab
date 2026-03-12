from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(payload: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def write_dataframe(dataframe: pd.DataFrame, path: str | Path) -> Path:
    target = ensure_parent_dir(path)
    dataframe.to_csv(target, index=False)
    return target