from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.config import MODEL_ARTIFACT_PATH
from viralscope.synthetic import generate_synthetic_posts
from viralscope.training import save_model_artifact, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a ViralScope model.")
    parser.add_argument("--input", type=Path, help="CSV file. If omitted, synthetic data is generated.")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--model", default="gradient_boosting", choices=["logistic", "random_forest", "gradient_boosting"])
    parser.add_argument("--output", type=Path, default=MODEL_ARTIFACT_PATH)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()

    if args.input:
        data = pd.read_csv(args.input)
    else:
        data = generate_synthetic_posts(n_rows=args.rows, seed=42)

    result = train_model(data, model_name=args.model, calibrate=not args.no_calibration)
    path = save_model_artifact(result, args.output)
    print(f"Saved model to {path}")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
