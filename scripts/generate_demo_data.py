from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.config import DEMO_DATA_PATH
from viralscope.synthetic import generate_synthetic_posts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic ViralScope demo data.")
    parser.add_argument("--rows", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEMO_DATA_PATH)
    args = parser.parse_args()

    data = generate_synthetic_posts(n_rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data):,} rows to {args.output}")
    print(f"Viral positive rate: {data['is_viral'].mean():.3f}")


if __name__ == "__main__":
    main()
