from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from viralscope.api_adapters import (
    TikTokConnectedAccountAdapter,
    YouTubeOfficialApiAdapter,
    extract_youtube_video_id,
)


YOUTUBE_ID = "dQw4w9WgXcQ"


class FakeYouTubeAdapter(YouTubeOfficialApiAdapter):
    def __init__(self) -> None:
        super().__init__(api_key="demo-key")

    def _youtube_get(self, path: str, params: dict):  # type: ignore[override]
        if path == "/videos":
            return {
                "items": [
                    {
                        "id": YOUTUBE_ID,
                        "snippet": {
                            "publishedAt": "2026-05-10T18:00:00Z",
                            "channelId": "UCDEMO123",
                            "channelTitle": "Demo Channel",
                            "title": "AI coding sprint #coding",
                            "description": "Building a Python tool with #ai",
                            "thumbnails": {"high": {"url": "https://img.youtube.test/high.jpg"}},
                            "tags": ["Coding", "AI Tools"],
                        },
                        "statistics": {
                            "viewCount": "15000",
                            "likeCount": "1800",
                            "commentCount": "215",
                        },
                        "contentDetails": {"duration": "PT1M25S"},
                    }
                ]
            }
        if path == "/channels":
            return {
                "items": [
                    {
                        "id": "UCDEMO123",
                        "snippet": {"publishedAt": "2020-01-05T00:00:00Z"},
                        "statistics": {"subscriberCount": "42000"},
                    }
                ]
            }
        if path == "/search":
            return {"items": [{"id": {"videoId": YOUTUBE_ID}}]}
        raise AssertionError(f"Unexpected path: {path}")


class FakeTikTokAdapter(TikTokConnectedAccountAdapter):
    def __init__(self) -> None:
        super().__init__(access_token="act.demo")

    def _tiktok_request(self, path: str, *, method: str, query_params=None, json_body=None):  # type: ignore[override]
        if path == "/v2/user/info/":
            return {
                "data": {
                    "user": {
                        "open_id": "open-demo",
                        "username": "demo_creator",
                        "display_name": "Demo Creator",
                        "follower_count": 7800,
                    }
                },
                "error": {"code": "ok", "message": "", "log_id": "demo"},
            }
        if path == "/v2/video/list/":
            return {
                "data": {
                    "videos": [
                        {
                            "id": "7260000000000000001",
                            "title": "POV your code finally works",
                            "video_description": "Late-night fix #coding #python",
                            "create_time": 1715600000,
                            "duration": 42,
                            "view_count": 9800,
                            "like_count": 1400,
                            "comment_count": 185,
                            "share_count": 290,
                            "share_url": "https://www.tiktok.com/@demo/video/7260000000000000001",
                            "cover_image_url": "https://img.tiktok.test/cover.jpg",
                        }
                    ]
                },
                "error": {"code": "ok", "message": "", "log_id": "demo"},
            }
        if path == "/v2/video/query/":
            return self._tiktok_request("/v2/video/list/", method=method, query_params=query_params, json_body=json_body)
        raise AssertionError(f"Unexpected path: {path}")


def test_extract_youtube_video_id_from_common_forms() -> None:
    assert extract_youtube_video_id(YOUTUBE_ID) == YOUTUBE_ID
    assert extract_youtube_video_id(f"https://www.youtube.com/watch?v={YOUTUBE_ID}") == YOUTUBE_ID
    assert extract_youtube_video_id(f"https://youtu.be/{YOUTUBE_ID}") == YOUTUBE_ID
    assert extract_youtube_video_id(f"https://www.youtube.com/shorts/{YOUTUBE_ID}") == YOUTUBE_ID


def test_youtube_adapter_normalizes_video_row() -> None:
    adapter = FakeYouTubeAdapter()
    frame = adapter.fetch_video(YOUTUBE_ID)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["platform"] == "YouTube"
    assert row["platform_post_id"] == YOUTUBE_ID
    assert row["media_type"] == "short_video"
    assert row["account_follower_count"] == 42000
    assert row["early_views"] == 15000
    assert "#coding" in row["hashtags"].lower()
    assert row["adapter_source_mode"] == "video_lookup"


def test_tiktok_adapter_builds_auth_url_and_normalizes_connected_videos() -> None:
    adapter = FakeTikTokAdapter()
    url = adapter.build_authorization_url(
        client_key="client-key",
        redirect_uri="https://example.com/callback",
        state="csrf-token",
        disable_auto_auth=True,
    )
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert params["client_key"][0] == "client-key"
    assert params["response_type"][0] == "code"
    assert params["state"][0] == "csrf-token"
    scopes = set(params["scope"][0].split(","))
    assert {"user.info.basic", "video.list"}.issubset(scopes)

    frame = adapter.fetch_connected_videos(limit=1)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["platform"] == "TikTok"
    assert row["media_type"] == "short_video"
    assert row["account_follower_count"] == 7800
    assert row["early_shares"] == 290
    assert row["account_username"] == "demo_creator"
