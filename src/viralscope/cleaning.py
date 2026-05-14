"""Data cleaning and target-label utilities."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .config import (
    FINAL_METRIC_COLUMNS,
    PLATFORM_ALIASES,
    RAW_INPUT_COLUMNS,
    SUPPORTED_PLATFORMS,
    TARGET_COLUMN,
    VIRALITY_QUANTILE,
)


COLUMN_ALIASES = {
    "title": "caption",
    "text": "caption",
    "description": "caption",
    "body": "caption",
    "tags": "hashtags",
    "tag_list": "hashtags",
    "type": "media_type",
    "published_at": "post_time",
    "created_at": "post_time",
    "timestamp": "post_time",
    "followers": "account_follower_count",
    "subscriber_count": "account_follower_count",
    "subscribers": "account_follower_count",
    "account_karma": "account_follower_count",
    "subreddit_members": "account_follower_count",
    "account_age": "account_age_days",
    "views_early": "early_views",
    "likes_early": "early_likes",
    "comments_early": "early_comments",
    "shares_early": "early_shares",
    "retweets_early": "early_shares",
    "upvotes_early": "early_likes",
    "window_hours": "early_window_hours",
    "hours_since_post": "early_window_hours",
    "viral": TARGET_COLUMN,
    "label": TARGET_COLUMN,
    "target": TARGET_COLUMN,
}


def _snake_case(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case columns and known aliases mapped."""
    renamed = {}
    for column in df.columns:
        normalized = _snake_case(column)
        renamed[column] = COLUMN_ALIASES.get(normalized, normalized)
    return df.rename(columns=renamed).copy()


def normalize_platform(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    cleaned = str(value).strip()
    alias = PLATFORM_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    for platform in SUPPORTED_PLATFORMS:
        if cleaned.lower() == platform.lower():
            return platform
    return cleaned.title() if cleaned else "Unknown"


def _clean_media_type(value: object) -> str:
    if pd.isna(value):
        return "text"
    cleaned = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if cleaned in {"short", "shorts", "reel", "reels"}:
        return "short_video"
    if cleaned in {"photo", "picture"}:
        return "image"
    return cleaned or "text"


def _coerce_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def clean_post_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw post rows while keeping the schema friendly for CSV uploads."""
    data = normalize_column_names(df)

    defaults = {
        "platform": "Unknown",
        "niche": "general",
        "caption": "",
        "hashtags": "",
        "media_type": "text",
        "post_time": pd.Timestamp.utcnow(),
        "account_follower_count": 0,
        "account_age_days": 30,
        "early_window_hours": 1,
        "early_views": 0,
        "early_likes": 0,
        "early_comments": 0,
        "early_shares": 0,
    }
    for column, default in defaults.items():
        if column not in data.columns:
            data[column] = default

    data["platform"] = data["platform"].map(normalize_platform)
    data["niche"] = data["niche"].fillna("general").astype(str).str.strip().str.lower().replace("", "general")
    data["caption"] = data["caption"].fillna("").astype(str)
    data["hashtags"] = data["hashtags"].fillna("").astype(str)
    data["media_type"] = data["media_type"].map(_clean_media_type)
    data["post_time"] = pd.to_datetime(data["post_time"], errors="coerce", utc=True).fillna(pd.Timestamp.utcnow())

    numeric_defaults = {
        "account_follower_count": 0,
        "account_age_days": 30,
        "early_window_hours": 1,
        "early_views": 0,
        "early_likes": 0,
        "early_comments": 0,
        "early_shares": 0,
    }
    for column, default in numeric_defaults.items():
        data[column] = _coerce_numeric(data[column], default=default).clip(lower=0)

    data["early_window_hours"] = data["early_window_hours"].replace(0, 1).clip(lower=0.25)

    for column in FINAL_METRIC_COLUMNS:
        if column in data.columns:
            data[column] = _coerce_numeric(data[column], default=np.nan)

    if TARGET_COLUMN in data.columns:
        data[TARGET_COLUMN] = _coerce_numeric(data[TARGET_COLUMN], default=0).round().clip(0, 1).astype(int)

    return data


def compute_five_day_engagement_rate(df: pd.DataFrame) -> pd.Series:
    """Compute a unified five-day engagement rate when final metrics exist."""
    data = clean_post_dataframe(df)
    if "five_day_engagement_rate" in data.columns and data["five_day_engagement_rate"].notna().any():
        return _coerce_numeric(data["five_day_engagement_rate"], default=np.nan)

    final_like = data.get("five_day_likes")
    final_comment = data.get("five_day_comments")
    final_share = data.get("five_day_shares")
    final_views = data.get("five_day_views")

    if final_like is None or final_comment is None or final_share is None:
        raise ValueError(
            "Training data needs either is_viral, five_day_engagement_rate, "
            "or five_day_likes/five_day_comments/five_day_shares columns."
        )

    engagement = _coerce_numeric(final_like) + _coerce_numeric(final_comment) + _coerce_numeric(final_share)
    denominator = _coerce_numeric(final_views, default=0)
    fallback_denominator = data["account_follower_count"].replace(0, np.nan)
    denominator = denominator.where(denominator > 0, fallback_denominator).fillna(1)
    return (engagement / denominator).replace([np.inf, -np.inf], np.nan).fillna(0)


def ensure_virality_label(
    df: pd.DataFrame,
    quantile: float = VIRALITY_QUANTILE,
    group_columns: Iterable[str] = ("platform", "niche"),
) -> pd.DataFrame:
    """Add is_viral using the top quantile within platform/niche groups.

    If a CSV already contains an is_viral column, this function keeps it.
    Otherwise it builds the label from five-day engagement rate.
    """
    data = clean_post_dataframe(df)
    if TARGET_COLUMN in data.columns and data[TARGET_COLUMN].notna().any():
        data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int).clip(0, 1)
        return data

    data["five_day_engagement_rate"] = compute_five_day_engagement_rate(data)
    group_columns = [column for column in group_columns if column in data.columns]

    if group_columns:
        thresholds = data.groupby(group_columns)["five_day_engagement_rate"].transform(
            lambda values: values.quantile(quantile) if len(values) >= 20 else np.nan
        )
    else:
        thresholds = pd.Series(data["five_day_engagement_rate"].quantile(quantile), index=data.index)

    if "platform" in data.columns:
        platform_thresholds = data.groupby("platform")["five_day_engagement_rate"].transform(
            lambda values: values.quantile(quantile) if len(values) >= 20 else np.nan
        )
        thresholds = thresholds.fillna(platform_thresholds)

    global_threshold = data["five_day_engagement_rate"].quantile(quantile)
    thresholds = thresholds.fillna(global_threshold)
    data[TARGET_COLUMN] = (data["five_day_engagement_rate"] >= thresholds).astype(int)
    return data


def required_training_columns() -> list[str]:
    return RAW_INPUT_COLUMNS + [TARGET_COLUMN]
