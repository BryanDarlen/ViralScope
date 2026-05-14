from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.api_adapters import ApiAdapterError, YouTubeOfficialApiAdapter


def _save_or_print(frame, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
        print(f"Saved {len(frame)} rows to {output}")
        return
    if frame.empty:
        print("No rows returned.")
        return
    print(frame.head(min(len(frame), 10)).to_string(index=False))
    print(f"\nRows returned: {len(frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch live YouTube data into ViralScope schema.")
    parser.add_argument("--api-key", help="YouTube Data API key. Falls back to YOUTUBE_API_KEY.")
    parser.add_argument("--output", type=Path, help="Optional CSV output path.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum rows to fetch.")
    parser.add_argument("--region-code", default="US", help="Region code for mostPopular benchmark pulls.")
    parser.add_argument("--video-category-id", help="Optional YouTube video category id for benchmark pulls.")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Search YouTube videos by query.")
    mode.add_argument("--video", help="Fetch a single video by YouTube URL or ID.")
    mode.add_argument("--benchmark", action="store_true", help="Fetch a mostPopular benchmark set.")
    args = parser.parse_args()

    adapter = YouTubeOfficialApiAdapter(api_key=args.api_key, region_code=args.region_code)

    try:
        if args.query:
            frame = adapter.fetch_posts(args.query, limit=args.limit)
        elif args.video:
            frame = adapter.fetch_video(args.video)
        else:
            frame = adapter.fetch_benchmark(
                limit=args.limit,
                region_code=args.region_code,
                video_category_id=args.video_category_id,
            )
    except ApiAdapterError as exc:
        raise SystemExit(str(exc)) from exc

    _save_or_print(frame, args.output)


if __name__ == "__main__":
    main()
