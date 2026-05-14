from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from viralscope.api_adapters import ApiAdapterError
from webapp.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_score() -> None:
    payload = {
        "platform": "YouTube",
        "niche": "productivity",
        "caption": "I tried studying 12 hours using AI tools",
        "hashtags": "#study #aitools #productivity",
        "media_type": "short_video",
        "post_time": "2026-05-12T20:00:00+00:00",
        "account_follower_count": 25000,
        "account_age_days": 730,
        "early_window_hours": 2,
        "early_views": 18000,
        "early_likes": 2100,
        "early_comments": 320,
        "early_shares": 450,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["score"] <= 100
    assert "signals" in body


def test_bootstrap_exposes_evaluation_diagnostics() -> None:
    response = client.get("/api/bootstrap")
    assert response.status_code == 200
    body = response.json()
    assert "validation" in body["model"]
    assert "calibration_curve" in body["evaluation"]
    assert "precision_recall_curve" in body["evaluation"]
    assert "slice_metrics" in body["evaluation"]


def test_live_youtube_video_endpoint_returns_normalized_post_and_prediction(monkeypatch) -> None:
    def fake_fetch_video(self, video_url_or_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "platform": "YouTube",
                    "platform_post_id": "demo1234567a",
                    "niche": "coding",
                    "caption": "AI coding sprint #coding",
                    "hashtags": "#coding #ai",
                    "media_type": "short_video",
                    "post_time": "2026-05-12T18:00:00+00:00",
                    "account_follower_count": 42000,
                    "account_age_days": 1200,
                    "early_window_hours": 24,
                    "early_views": 15000,
                    "early_likes": 1800,
                    "early_comments": 215,
                    "early_shares": 0,
                    "source_url": "https://www.youtube.com/watch?v=demo1234567a",
                    "channel_id": "UCDEMO123",
                    "channel_title": "Demo Channel",
                    "thumbnail_url": "https://img.youtube.test/high.jpg",
                    "duration_seconds": 85,
                    "adapter_source_mode": "video_lookup",
                    "adapter_notes": "demo",
                }
            ]
        )

    monkeypatch.setattr("webapp.main.YouTubeOfficialApiAdapter.fetch_video", fake_fetch_video)
    response = client.post("/api/live/youtube/video", json={"video_url_or_id": "demo1234567a"})
    assert response.status_code == 200
    body = response.json()
    assert body["post"]["platform"] == "YouTube"
    assert body["post"]["platform_post_id"] == "demo1234567a"
    assert 0 <= body["prediction"]["score"] <= 100
    assert "signals" in body["prediction"]


def test_live_youtube_video_endpoint_returns_422_for_invalid_video_id(monkeypatch) -> None:
    def fake_fetch_video(self, video_url_or_id: str) -> pd.DataFrame:
        raise ApiAdapterError(f"Could not extract a YouTube video ID from: {video_url_or_id}")

    monkeypatch.setattr("webapp.main.YouTubeOfficialApiAdapter.fetch_video", fake_fetch_video)
    response = client.post("/api/live/youtube/video", json={"video_url_or_id": "not-a-youtube-id"})
    assert response.status_code == 422
    assert "Could not extract a YouTube video ID" in response.json()["detail"]


def test_live_youtube_video_endpoint_returns_503_for_missing_api_key(monkeypatch) -> None:
    def fake_fetch_video(self, video_url_or_id: str) -> pd.DataFrame:
        raise ApiAdapterError("Missing YouTube API key. Set YOUTUBE_API_KEY before using this adapter.")

    monkeypatch.setattr("webapp.main.YouTubeOfficialApiAdapter.fetch_video", fake_fetch_video)
    response = client.post("/api/live/youtube/video", json={"video_url_or_id": "dQw4w9WgXcQ"})
    assert response.status_code == 503
    assert "Missing YouTube API key" in response.json()["detail"]
