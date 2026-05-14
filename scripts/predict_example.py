from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.prediction import predict_from_saved_artifact


EXAMPLE_POST = {
    "platform": "YouTube",
    "niche": "productivity",
    "caption": "I tried studying 12 hours using AI tools",
    "hashtags": "#study #aitools #productivity",
    "media_type": "short_video",
    "post_time": "2026-05-11T20:00:00+00:00",
    "account_follower_count": 25_000,
    "account_age_days": 730,
    "early_window_hours": 2,
    "early_views": 18_000,
    "early_likes": 2_100,
    "early_comments": 320,
    "early_shares": 450,
}


def main() -> None:
    result = predict_from_saved_artifact(EXAMPLE_POST)
    print(f"Virality probability: {result.score}% ({result.bucket})")
    print(result.reasoning_summary)
    print("Positive factors:")
    for factor in result.positive_factors:
        print(f"- {factor['name']}: {factor['detail']}")
    print("Recommendations:")
    for item in result.recommendations:
        print(f"- {item}")


if __name__ == "__main__":
    main()
