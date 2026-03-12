from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.inference.predictor import prediction_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch inference on a folder of images.")
    parser.add_argument("--input-dir", default="data/samples")
    parser.add_argument("--output", default="data/processed/batch_predictions.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_service.load()
    input_dir = Path(args.input_dir)
    image_paths = [path for path in sorted(input_dir.iterdir()) if path.is_file()]
    payloads = [path.read_bytes() for path in image_paths]
    results = prediction_service.predict_batch(payloads)
    serialized = [
        {
            "filename": path.name,
            "predicted_class": result.predicted_class,
            "confidence": result.confidence,
            "top_k": result.top_k,
        }
        for path, result in zip(image_paths, results)
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()