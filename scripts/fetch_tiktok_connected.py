from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from viralscope.api_adapters import ApiAdapterError, TikTokConnectedAccountAdapter


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
    parser = argparse.ArgumentParser(description="Work with TikTok connected-account data for ViralScope.")
    parser.add_argument("--access-token", help="TikTok user access token. Falls back to TIKTOK_ACCESS_TOKEN.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    auth_parser = subparsers.add_parser("auth-url", help="Build the TikTok authorization URL.")
    auth_parser.add_argument("--client-key", required=True)
    auth_parser.add_argument("--redirect-uri", required=True)
    auth_parser.add_argument("--state", required=True)
    auth_parser.add_argument("--disable-auto-auth", action="store_true")

    videos_parser = subparsers.add_parser("videos", help="Fetch recent connected-account videos.")
    videos_parser.add_argument("--limit", type=int, default=10)
    videos_parser.add_argument("--cursor", type=int)
    videos_parser.add_argument("--output", type=Path)

    profile_parser = subparsers.add_parser("profile", help="Fetch connected-account profile metadata.")

    query_parser = subparsers.add_parser("query", help="Fetch connected-account videos by TikTok video IDs.")
    query_parser.add_argument("video_ids", nargs="+")
    query_parser.add_argument("--output", type=Path)

    exchange_parser = subparsers.add_parser("exchange-code", help="Exchange an auth code for user tokens.")
    exchange_parser.add_argument("--client-key", required=True)
    exchange_parser.add_argument("--client-secret", required=True)
    exchange_parser.add_argument("--redirect-uri", required=True)
    exchange_parser.add_argument("--code", required=True)
    exchange_parser.add_argument("--code-verifier")

    refresh_parser = subparsers.add_parser("refresh-token", help="Refresh a TikTok user access token.")
    refresh_parser.add_argument("--client-key", required=True)
    refresh_parser.add_argument("--client-secret", required=True)
    refresh_parser.add_argument("--refresh-token", required=True)

    args = parser.parse_args()
    adapter = TikTokConnectedAccountAdapter(access_token=args.access_token)

    try:
        if args.command == "auth-url":
            print(
                adapter.build_authorization_url(
                    client_key=args.client_key,
                    redirect_uri=args.redirect_uri,
                    state=args.state,
                    disable_auto_auth=args.disable_auto_auth,
                )
            )
            return

        if args.command == "profile":
            print(json.dumps(adapter.fetch_connected_profile(), indent=2))
            return

        if args.command == "videos":
            frame = adapter.fetch_connected_videos(limit=args.limit, cursor=args.cursor)
            _save_or_print(frame, args.output)
            return

        if args.command == "query":
            frame = adapter.fetch_videos_by_ids(args.video_ids)
            _save_or_print(frame, args.output)
            return

        if args.command == "exchange-code":
            payload = adapter.exchange_code_for_tokens(
                code=args.code,
                client_key=args.client_key,
                client_secret=args.client_secret,
                redirect_uri=args.redirect_uri,
                code_verifier=args.code_verifier,
            )
            print(json.dumps(payload, indent=2))
            return

        if args.command == "refresh-token":
            payload = adapter.refresh_access_token(
                refresh_token=args.refresh_token,
                client_key=args.client_key,
                client_secret=args.client_secret,
            )
            print(json.dumps(payload, indent=2))
            return
    except ApiAdapterError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
