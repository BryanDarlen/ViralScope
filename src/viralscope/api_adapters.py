"""Official API adapters for live ViralScope data ingestion.

These adapters intentionally avoid scraping. They normalize official-platform
responses into the same tabular schema used by the CSV and synthetic flows.

Notes:
- YouTube public APIs expose current cumulative metrics, not exact first-hour
  snapshots. ViralScope therefore treats "early_*" metrics as "current metrics
  observed after X hours since publish" when a live post is queried.
- TikTok connected-account access requires explicit user authorization and the
  relevant approved scopes. This adapter is designed for creator-owned account
  connections, not broad public-data collection.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd


HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9_]+")
YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
ISO_8601_DURATION_RE = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
TIKTOK_API_BASE = "https://open.tiktokapis.com"
TIKTOK_AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"

TIKTOK_CONNECTED_SCOPES = (
    "user.info.basic",
    "user.info.profile",
    "user.info.stats",
    "video.list",
)

NICHE_KEYWORDS = {
    "ai": ("ai", "chatgpt", "automation", "llm", "prompt"),
    "beauty": ("beauty", "makeup", "skincare", "routine", "glow"),
    "business": ("business", "startup", "sales", "founder", "marketing"),
    "coding": ("coding", "code", "programming", "python", "javascript", "debug"),
    "education": ("study", "learning", "lesson", "exam", "tutorial", "school"),
    "entertainment": ("reaction", "challenge", "funny", "pov", "meme"),
    "fitness": ("fitness", "workout", "gym", "protein", "training"),
    "food": ("food", "recipe", "cook", "restaurant", "taste"),
    "gaming": ("gaming", "game", "speedrun", "ranked", "stream"),
    "productivity": ("productivity", "focus", "deep work", "planner", "workflow"),
}


class ApiAdapterError(RuntimeError):
    """Raised when an official platform adapter cannot complete a request."""


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return _utc_now()
    return pd.Timestamp(timestamp)


def _hours_since(timestamp: pd.Timestamp, minimum: float = 1.0) -> float:
    delta = _utc_now() - _to_timestamp(timestamp)
    hours = max(delta.total_seconds() / 3600.0, minimum)
    return round(hours, 2)


def _days_since(timestamp: pd.Timestamp) -> int:
    delta = _utc_now() - _to_timestamp(timestamp)
    return max(int(delta // timedelta(days=1)), 0)


def _chunked(values: Iterable[str], size: int) -> list[list[str]]:
    rows = list(values)
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _extract_hashtags(*parts: Any) -> str:
    found: list[str] = []
    for part in parts:
        matches = HASHTAG_RE.findall(str(part or ""))
        for match in matches:
            if match not in found:
                found.append(match)
    return " ".join(found)


def _tags_to_hashtags(tags: Iterable[str] | None) -> str:
    if not tags:
        return ""
    found: list[str] = []
    for tag in tags:
        cleaned = SAFE_TAG_RE.sub("", str(tag).strip().replace(" ", "_"))
        if not cleaned:
            continue
        value = f"#{cleaned.lower()}"
        if value not in found:
            found.append(value)
    return " ".join(found[:8])


def _infer_niche(*parts: Any) -> str:
    text = " ".join(str(part or "") for part in parts).lower()
    for niche, keywords in NICHE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return niche
    return "general"


def _media_type_from_duration(duration_seconds: int, platform: str) -> str:
    if platform == "TikTok":
        return "short_video" if duration_seconds <= 180 else "video"
    return "short_video" if 0 < duration_seconds <= 180 else "video"


def _parse_iso_duration(duration_text: str | None) -> int:
    if not duration_text:
        return 0
    match = ISO_8601_DURATION_RE.match(duration_text)
    if not match:
        return 0
    parts = {key: _coerce_int(value) for key, value in match.groupdict().items()}
    return (
        parts.get("days", 0) * 86400
        + parts.get("hours", 0) * 3600
        + parts.get("minutes", 0) * 60
        + parts.get("seconds", 0)
    )


def extract_youtube_video_id(video_id_or_url: str) -> str:
    value = str(video_id_or_url).strip()
    if YOUTUBE_VIDEO_ID_RE.match(value):
        return value

    parsed = urlparse(value)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        candidate = parsed.path.strip("/").split("/")[0]
        if YOUTUBE_VIDEO_ID_RE.match(candidate):
            return candidate

    query_params = parse_qs(parsed.query)
    candidate = query_params.get("v", [None])[0]
    if candidate and YOUTUBE_VIDEO_ID_RE.match(candidate):
        return candidate

    path_parts = [part for part in parsed.path.split("/") if part]
    if "shorts" in path_parts:
        index = path_parts.index("shorts")
        if index + 1 < len(path_parts):
            candidate = path_parts[index + 1]
            if YOUTUBE_VIDEO_ID_RE.match(candidate):
                return candidate

    raise ApiAdapterError(f"Could not extract a YouTube video ID from: {video_id_or_url}")


@dataclass
class ApiAdapter:
    platform: str
    request_timeout: float = 20.0

    def fetch_posts(self, query: str, limit: int = 100) -> pd.DataFrame:
        raise NotImplementedError

    def _request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        query_params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_headers = {"Accept": "application/json"}
        if headers:
            request_headers.update(headers)

        encoded_url = url
        if query_params:
            query_string = urlencode(query_params, doseq=True, safe=",")
            encoded_url = f"{url}?{query_string}"

        data: bytes | None = None
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        elif form_body is not None:
            data = urlencode(form_body).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")

        request = Request(encoded_url, data=data, headers=request_headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise ApiAdapterError(f"{self.platform} API HTTP {exc.code}: {message}") from exc
        except URLError as exc:
            raise ApiAdapterError(f"{self.platform} API request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ApiAdapterError(f"{self.platform} API returned non-JSON data.") from exc

        self._raise_if_api_error(parsed)
        return parsed

    def _raise_if_api_error(self, payload: dict[str, Any]) -> None:
        error = payload.get("error")
        if not error:
            return
        if isinstance(error, dict):
            code = error.get("code")
            if code in (None, 0, "0", "ok", "OK"):
                return
            message = error.get("message") or error.get("error_description") or str(error)
            raise ApiAdapterError(f"{self.platform} API error ({code}): {message}")
        raise ApiAdapterError(f"{self.platform} API error: {error}")


class YouTubeOfficialApiAdapter(ApiAdapter):
    """Official YouTube Data API adapter.

    Uses public YouTube Data API endpoints:
    - search.list
    - videos.list
    - channels.list
    """

    def __init__(self, api_key: str | None = None, region_code: str = "US", request_timeout: float = 20.0) -> None:
        super().__init__(platform="YouTube", request_timeout=request_timeout)
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.region_code = region_code

    def fetch_posts(self, query: str, limit: int = 25) -> pd.DataFrame:
        if not query.strip():
            raise ApiAdapterError("YouTube search requires a non-empty query.")
        limit = max(1, min(limit, 50))
        payload = self._youtube_get(
            "/search",
            {
                "part": "snippet",
                "type": "video",
                "q": query,
                "maxResults": limit,
                "order": "relevance",
            },
        )
        video_ids = [item.get("id", {}).get("videoId") for item in payload.get("items", [])]
        video_ids = [video_id for video_id in video_ids if video_id]
        return self.fetch_videos_by_ids(video_ids, niche_hint=_infer_niche(query), source_mode="query_search")

    def fetch_video(self, video_id_or_url: str) -> pd.DataFrame:
        video_id = extract_youtube_video_id(video_id_or_url)
        return self.fetch_videos_by_ids([video_id], source_mode="video_lookup")

    def fetch_benchmark(
        self,
        limit: int = 25,
        *,
        region_code: str | None = None,
        video_category_id: str | None = None,
    ) -> pd.DataFrame:
        params: dict[str, Any] = {
            "part": "snippet,statistics,contentDetails",
            "chart": "mostPopular",
            "maxResults": max(1, min(limit, 50)),
            "regionCode": region_code or self.region_code,
        }
        if video_category_id:
            params["videoCategoryId"] = video_category_id
        payload = self._youtube_get("/videos", params)
        items = payload.get("items", [])
        channel_map = self._fetch_channel_map(item.get("snippet", {}).get("channelId") for item in items)
        rows = [self._normalize_video_item(item, channel_map, source_mode="most_popular") for item in items]
        return pd.DataFrame(rows)

    def fetch_videos_by_ids(
        self,
        video_ids: Iterable[str],
        *,
        niche_hint: str = "general",
        source_mode: str = "video_lookup",
    ) -> pd.DataFrame:
        unique_ids = [video_id for video_id in dict.fromkeys(video_ids) if video_id]
        if not unique_ids:
            return pd.DataFrame()

        items: list[dict[str, Any]] = []
        for chunk in _chunked(unique_ids, 50):
            payload = self._youtube_get(
                "/videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(chunk),
                    "maxResults": len(chunk),
                },
            )
            items.extend(payload.get("items", []))

        channel_ids = [item.get("snippet", {}).get("channelId") for item in items]
        channel_map = self._fetch_channel_map(channel_ids)
        rows = [
            self._normalize_video_item(item, channel_map, niche_hint=niche_hint, source_mode=source_mode)
            for item in items
        ]
        return pd.DataFrame(rows)

    def _youtube_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ApiAdapterError("Missing YouTube API key. Set YOUTUBE_API_KEY before using this adapter.")
        merged = dict(params)
        merged["key"] = self.api_key
        return self._request_json(f"{YOUTUBE_API_BASE}{path}", query_params=merged)

    def _fetch_channel_map(self, channel_ids: Iterable[str | None]) -> dict[str, dict[str, Any]]:
        cleaned = [channel_id for channel_id in dict.fromkeys(channel_ids) if channel_id]
        if not cleaned:
            return {}

        channel_map: dict[str, dict[str, Any]] = {}
        for chunk in _chunked(cleaned, 50):
            payload = self._youtube_get(
                "/channels",
                {
                    "part": "snippet,statistics",
                    "id": ",".join(chunk),
                    "maxResults": len(chunk),
                },
            )
            for item in payload.get("items", []):
                channel_map[item.get("id", "")] = item
        return channel_map

    def _normalize_video_item(
        self,
        item: dict[str, Any],
        channel_map: dict[str, dict[str, Any]],
        *,
        niche_hint: str = "general",
        source_mode: str,
    ) -> dict[str, Any]:
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})
        channel = channel_map.get(snippet.get("channelId", ""), {})
        channel_stats = channel.get("statistics", {})
        channel_snippet = channel.get("snippet", {})

        post_time = _to_timestamp(snippet.get("publishedAt"))
        duration_seconds = _parse_iso_duration(content.get("duration"))
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        tag_hashtags = _tags_to_hashtags(snippet.get("tags"))
        caption_hashtags = _extract_hashtags(title, description)
        hashtags = " ".join(part for part in (caption_hashtags, tag_hashtags) if part).strip()

        return {
            "platform": "YouTube",
            "platform_post_id": item.get("id"),
            "niche": niche_hint if niche_hint != "general" else _infer_niche(title, description),
            "caption": title or description,
            "hashtags": hashtags,
            "media_type": _media_type_from_duration(duration_seconds, "YouTube"),
            "post_time": post_time.isoformat(),
            "account_follower_count": _coerce_int(channel_stats.get("subscriberCount")),
            "account_age_days": _days_since(_to_timestamp(channel_snippet.get("publishedAt"))),
            "early_window_hours": _hours_since(post_time),
            "early_views": _coerce_int(stats.get("viewCount")),
            "early_likes": _coerce_int(stats.get("likeCount")),
            "early_comments": _coerce_int(stats.get("commentCount")),
            "early_shares": 0,
            "source_url": f"https://www.youtube.com/watch?v={item.get('id', '')}",
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "thumbnail_url": ((snippet.get("thumbnails") or {}).get("high") or {}).get("url"),
            "duration_seconds": duration_seconds,
            "adapter_source_mode": source_mode,
            "adapter_notes": (
                "YouTube public API provides cumulative public metrics. Share count is not exposed here."
            ),
        }


class TikTokConnectedAccountAdapter(ApiAdapter):
    """Official TikTok Display API adapter for connected creator accounts."""

    def __init__(self, access_token: str | None = None, request_timeout: float = 20.0) -> None:
        super().__init__(platform="TikTok", request_timeout=request_timeout)
        self.access_token = access_token or os.getenv("TIKTOK_ACCESS_TOKEN")

    def fetch_posts(self, query: str = "", limit: int = 20) -> pd.DataFrame:
        return self.fetch_connected_videos(limit=limit)

    def fetch_connected_videos(self, limit: int = 20, cursor: int | None = None) -> pd.DataFrame:
        limit = max(1, min(limit, 20))
        profile = self.fetch_connected_profile()
        fields = (
            "id,create_time,cover_image_url,share_url,video_description,"
            "duration,height,width,title,like_count,comment_count,share_count,view_count"
        )
        body: dict[str, Any] = {"max_count": limit}
        if cursor is not None:
            body["cursor"] = int(cursor)
        payload = self._tiktok_request(
            "/v2/video/list/",
            method="POST",
            query_params={"fields": fields},
            json_body=body,
        )
        rows = [
            self._normalize_video_item(item, profile=profile, source_mode="connected_video_list")
            for item in payload.get("data", {}).get("videos", [])
        ]
        return pd.DataFrame(rows)

    def fetch_videos_by_ids(self, video_ids: Iterable[str]) -> pd.DataFrame:
        ids = [str(video_id).strip() for video_id in dict.fromkeys(video_ids) if str(video_id).strip()]
        if not ids:
            return pd.DataFrame()
        if len(ids) > 20:
            raise ApiAdapterError("TikTok video.query supports at most 20 video IDs per request.")

        profile = self.fetch_connected_profile()
        fields = (
            "id,create_time,cover_image_url,share_url,video_description,"
            "duration,height,width,title,like_count,comment_count,share_count,view_count"
        )
        payload = self._tiktok_request(
            "/v2/video/query/",
            method="POST",
            query_params={"fields": fields},
            json_body={"filters": {"video_ids": ids}},
        )
        rows = [
            self._normalize_video_item(item, profile=profile, source_mode="connected_video_query")
            for item in payload.get("data", {}).get("videos", [])
        ]
        return pd.DataFrame(rows)

    def fetch_connected_profile(self) -> dict[str, Any]:
        fields = (
            "open_id,display_name,username,profile_deep_link,avatar_url,"
            "follower_count,following_count,likes_count,video_count"
        )
        payload = self._tiktok_request("/v2/user/info/", method="GET", query_params={"fields": fields})
        return payload.get("data", {}).get("user", {})

    def build_authorization_url(
        self,
        *,
        client_key: str,
        redirect_uri: str,
        state: str,
        scopes: Iterable[str] = TIKTOK_CONNECTED_SCOPES,
        disable_auto_auth: bool = False,
    ) -> str:
        params = {
            "client_key": client_key,
            "response_type": "code",
            "scope": ",".join(scopes),
            "redirect_uri": redirect_uri,
            "state": state,
        }
        if disable_auto_auth:
            params["disable_auto_auth"] = 1
        return f"{TIKTOK_AUTHORIZE_URL}?{urlencode(params, safe=',:/')}"

    def exchange_code_for_tokens(
        self,
        *,
        code: str,
        client_key: str,
        client_secret: str,
        redirect_uri: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any]:
        form = {
            "client_key": client_key,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        }
        if code_verifier:
            form["code_verifier"] = code_verifier
        return self._request_json(
            f"{TIKTOK_API_BASE}/v2/oauth/token/",
            method="POST",
            form_body=form,
        )

    def refresh_access_token(
        self,
        *,
        refresh_token: str,
        client_key: str,
        client_secret: str,
    ) -> dict[str, Any]:
        form = {
            "client_key": client_key,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        return self._request_json(
            f"{TIKTOK_API_BASE}/v2/oauth/token/",
            method="POST",
            form_body=form,
        )

    def _tiktok_request(
        self,
        path: str,
        *,
        method: str,
        query_params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.access_token:
            raise ApiAdapterError(
                "Missing TikTok access token. Set TIKTOK_ACCESS_TOKEN or provide access_token explicitly."
            )
        return self._request_json(
            f"{TIKTOK_API_BASE}{path}",
            method=method,
            query_params=query_params,
            json_body=json_body,
            headers={"Authorization": f"Bearer {self.access_token}"},
        )

    def _normalize_video_item(
        self,
        item: dict[str, Any],
        *,
        profile: dict[str, Any],
        source_mode: str,
    ) -> dict[str, Any]:
        title = item.get("title", "")
        description = item.get("video_description", "")
        caption = title or description
        hashtags = _extract_hashtags(title, description)
        post_time = pd.to_datetime(_coerce_int(item.get("create_time")), unit="s", utc=True)
        duration_seconds = _coerce_int(item.get("duration"))

        return {
            "platform": "TikTok",
            "platform_post_id": item.get("id"),
            "niche": _infer_niche(caption),
            "caption": caption,
            "hashtags": hashtags,
            "media_type": _media_type_from_duration(duration_seconds, "TikTok"),
            "post_time": _to_timestamp(post_time).isoformat(),
            "account_follower_count": _coerce_int(profile.get("follower_count")),
            "account_age_days": 0,
            "early_window_hours": _hours_since(post_time),
            "early_views": _coerce_int(item.get("view_count")),
            "early_likes": _coerce_int(item.get("like_count")),
            "early_comments": _coerce_int(item.get("comment_count")),
            "early_shares": _coerce_int(item.get("share_count")),
            "source_url": item.get("share_url"),
            "account_open_id": profile.get("open_id"),
            "account_username": profile.get("username"),
            "account_display_name": profile.get("display_name"),
            "thumbnail_url": item.get("cover_image_url"),
            "duration_seconds": duration_seconds,
            "adapter_source_mode": source_mode,
            "adapter_notes": (
                "TikTok Display API is using connected-account public video metadata. "
                "Account age is not exposed here and is left as 0."
            ),
        }
