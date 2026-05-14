from __future__ import annotations

from viralscope.prediction import predict_virality
from viralscope.synthetic import generate_synthetic_posts
from viralscope.training import train_model


def test_train_model_returns_metrics() -> None:
    data = generate_synthetic_posts(n_rows=500, seed=3)
    result = train_model(data, model_name="logistic", calibrate=False)
    assert result.metrics["rows"] > 0
    assert "roc_auc" in result.metrics
    assert 0 <= result.metrics["positive_rate"] <= 1
    assert result.metadata["validation"]["strategy"] == "temporal_holdout"
    assert result.metadata["validation"]["test_rows"] == result.metrics["rows"]
    assert result.metrics["calibration_curve"]
    assert result.metrics["precision_recall_curve"]
    assert result.metrics["slice_metrics"]["platform"]
    assert result.metrics["slice_metrics"]["niche"]


def test_high_signal_post_scores_above_low_signal_post() -> None:
    data = generate_synthetic_posts(n_rows=900, seed=4)
    result = train_model(data, model_name="random_forest", calibrate=False)

    high_signal = {
        "platform": "TikTok",
        "niche": "coding",
        "caption": "POV: your code finally works at 3AM",
        "hashtags": "#coding #programming #studentlife",
        "media_type": "short_video",
        "post_time": "2026-05-11T20:00:00+00:00",
        "account_follower_count": 3_000,
        "account_age_days": 365,
        "early_window_hours": 1,
        "early_views": 9_500,
        "early_likes": 1_400,
        "early_comments": 180,
        "early_shares": 300,
    }
    low_signal = {
        "platform": "Reddit",
        "niche": "productivity",
        "caption": "My thoughts on productivity",
        "hashtags": "",
        "media_type": "text",
        "post_time": "2026-05-11T03:00:00+00:00",
        "account_follower_count": 60,
        "account_age_days": 20,
        "early_window_hours": 2,
        "early_views": 40,
        "early_likes": 4,
        "early_comments": 0,
        "early_shares": 0,
    }

    high = predict_virality(result.pipeline, high_signal, metadata=result.metadata)
    low = predict_virality(result.pipeline, low_signal, metadata=result.metadata)
    assert 0 <= high.score <= 100
    assert 0 <= low.score <= 100
    assert high.probability > low.probability
