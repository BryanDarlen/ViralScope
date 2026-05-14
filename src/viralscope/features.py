"""Feature engineering for post-time and early-post virality signals."""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .cleaning import clean_post_dataframe


HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")

TREND_KEYWORDS = {
    "ai",
    "chatgpt",
    "pov",
    "hack",
    "challenge",
    "study",
    "productivity",
    "coding",
    "programming",
    "student",
    "viral",
    "trend",
    "tools",
    "finally",
    "secret",
    "mistakes",
    "tutorial",
}

CATEGORICAL_FEATURES = ["platform", "niche", "media_type"]

NUMERIC_FEATURES = [
    "caption_length",
    "word_count",
    "hashtag_count",
    "mention_count",
    "trend_keyword_count",
    "posting_hour",
    "posting_hour_sin",
    "posting_hour_cos",
    "posting_dayofweek",
    "posting_day_sin",
    "posting_day_cos",
    "is_weekend",
    "account_follower_count_log",
    "account_age_days_log",
    "early_window_hours",
    "early_views_log",
    "early_likes_log",
    "early_comments_log",
    "early_shares_log",
    "early_engagement_rate",
    "likes_per_view",
    "comments_per_view",
    "shares_per_view",
    "views_per_hour_log",
    "likes_per_hour_log",
    "comments_per_hour_log",
    "shares_per_hour_log",
    "engagement_per_follower",
    "views_per_follower",
    "engagement_rate_vs_platform",
    "views_per_hour_vs_platform",
    "likes_per_hour_vs_platform",
    "comments_per_hour_vs_platform",
    "shares_per_hour_vs_platform",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

REFERENCE_COLUMNS = [
    "early_engagement_rate",
    "views_per_hour",
    "likes_per_hour",
    "comments_per_hour",
    "shares_per_hour",
]


def _count_hashtags(caption: str, hashtags: str) -> int:
    combined = f"{caption or ''} {hashtags or ''}"
    return len(HASHTAG_RE.findall(combined))


def _count_mentions(caption: str) -> int:
    return len(MENTION_RE.findall(caption or ""))


def _count_trend_keywords(caption: str, hashtags: str) -> int:
    text = f"{caption or ''} {hashtags or ''}".lower()
    tokens = set(re.findall(r"[a-z0-9_]+", text))
    return len(tokens.intersection(TREND_KEYWORDS))


def _safe_ratio(numerator: pd.Series, denominator: pd.Series | float, default: float = 0.0) -> pd.Series:
    result = numerator / pd.Series(denominator, index=numerator.index).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _cyclic(value: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * math.pi * value / period
    return np.sin(radians), np.cos(radians)


def _reference_value(
    platform: str,
    column: str,
    platform_reference: dict[str, dict[str, float]] | None,
    global_reference: dict[str, float] | None,
) -> float:
    platform_reference = platform_reference or {}
    global_reference = global_reference or {}
    value = platform_reference.get(platform, {}).get(column)
    if value is None or pd.isna(value) or value <= 0:
        value = global_reference.get(column, 1.0)
    if value is None or pd.isna(value) or value <= 0:
        return 1.0
    return float(value)


def build_features(
    df: pd.DataFrame,
    platform_reference: dict[str, dict[str, float]] | None = None,
    global_reference: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Create model-ready feature columns from raw post rows."""
    data = clean_post_dataframe(df)
    features = data.copy()

    features["caption_length"] = features["caption"].str.len()
    features["word_count"] = features["caption"].str.findall(r"\b\w+\b").str.len().fillna(0)
    features["hashtag_count"] = [
        _count_hashtags(caption, hashtags)
        for caption, hashtags in zip(features["caption"], features["hashtags"], strict=False)
    ]
    features["mention_count"] = features["caption"].map(_count_mentions)
    features["trend_keyword_count"] = [
        _count_trend_keywords(caption, hashtags)
        for caption, hashtags in zip(features["caption"], features["hashtags"], strict=False)
    ]

    features["posting_hour"] = features["post_time"].dt.hour
    features["posting_dayofweek"] = features["post_time"].dt.dayofweek
    features["posting_hour_sin"], features["posting_hour_cos"] = _cyclic(features["posting_hour"], 24)
    features["posting_day_sin"], features["posting_day_cos"] = _cyclic(features["posting_dayofweek"], 7)
    features["is_weekend"] = features["posting_dayofweek"].isin([5, 6]).astype(int)

    engagement = features["early_likes"] + features["early_comments"] + features["early_shares"]
    views = features["early_views"].clip(lower=0)
    followers = features["account_follower_count"].clip(lower=0)
    hours = features["early_window_hours"].clip(lower=0.25)

    features["account_follower_count_log"] = np.log1p(followers)
    features["account_age_days_log"] = np.log1p(features["account_age_days"].clip(lower=0))
    features["early_views_log"] = np.log1p(views)
    features["early_likes_log"] = np.log1p(features["early_likes"])
    features["early_comments_log"] = np.log1p(features["early_comments"])
    features["early_shares_log"] = np.log1p(features["early_shares"])

    features["early_engagement_rate"] = _safe_ratio(engagement, views)
    features["likes_per_view"] = _safe_ratio(features["early_likes"], views)
    features["comments_per_view"] = _safe_ratio(features["early_comments"], views)
    features["shares_per_view"] = _safe_ratio(features["early_shares"], views)

    features["views_per_hour"] = _safe_ratio(views, hours)
    features["likes_per_hour"] = _safe_ratio(features["early_likes"], hours)
    features["comments_per_hour"] = _safe_ratio(features["early_comments"], hours)
    features["shares_per_hour"] = _safe_ratio(features["early_shares"], hours)

    features["views_per_hour_log"] = np.log1p(features["views_per_hour"])
    features["likes_per_hour_log"] = np.log1p(features["likes_per_hour"])
    features["comments_per_hour_log"] = np.log1p(features["comments_per_hour"])
    features["shares_per_hour_log"] = np.log1p(features["shares_per_hour"])
    features["engagement_per_follower"] = _safe_ratio(engagement, followers)
    features["views_per_follower"] = _safe_ratio(views, followers)

    for raw_column, feature_column in [
        ("early_engagement_rate", "engagement_rate_vs_platform"),
        ("views_per_hour", "views_per_hour_vs_platform"),
        ("likes_per_hour", "likes_per_hour_vs_platform"),
        ("comments_per_hour", "comments_per_hour_vs_platform"),
        ("shares_per_hour", "shares_per_hour_vs_platform"),
    ]:
        ratios = []
        for platform, value in zip(features["platform"], features[raw_column], strict=False):
            ref = _reference_value(platform, raw_column, platform_reference, global_reference)
            ratios.append(float(value) / ref if ref else 0.0)
        features[feature_column] = np.clip(ratios, 0, 100)

    for column in FEATURE_COLUMNS:
        if column not in features.columns:
            features[column] = 0

    return features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that learns platform baselines in fit."""

    def __init__(self) -> None:
        self.platform_reference_: dict[str, dict[str, float]] = {}
        self.global_reference_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureEngineer":
        base = build_features(X)
        working = clean_post_dataframe(X)

        reference_frame = pd.DataFrame(index=working.index)
        reference_frame["platform"] = working["platform"]
        reference_frame["early_engagement_rate"] = base["early_engagement_rate"]
        reference_frame["views_per_hour"] = np.expm1(base["views_per_hour_log"])
        reference_frame["likes_per_hour"] = np.expm1(base["likes_per_hour_log"])
        reference_frame["comments_per_hour"] = np.expm1(base["comments_per_hour_log"])
        reference_frame["shares_per_hour"] = np.expm1(base["shares_per_hour_log"])

        self.platform_reference_ = (
            reference_frame.groupby("platform")[REFERENCE_COLUMNS].median().fillna(1.0).to_dict(orient="index")
        )
        self.global_reference_ = reference_frame[REFERENCE_COLUMNS].median().fillna(1.0).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(
            X,
            platform_reference=self.platform_reference_,
            global_reference=self.global_reference_,
        )


def feature_reference_from_transformer(transformer: FeatureEngineer) -> dict[str, dict[str, float] | float]:
    return {
        "by_platform": transformer.platform_reference_,
        "global": transformer.global_reference_,
    }
