"""Synthetic data generator for API-free demos and tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .cleaning import ensure_virality_label
from .config import MEDIA_TYPES, NICHES, SUPPORTED_PLATFORMS


PLATFORM_DISCOVERY = {
    "TikTok": 0.34,
    "YouTube": 0.16,
    "Reddit": 0.08,
    "Twitter/X": 0.11,
}

PLATFORM_ENGAGEMENT = {
    "TikTok": {"like": 0.064, "comment": 0.007, "share": 0.010},
    "YouTube": {"like": 0.040, "comment": 0.004, "share": 0.007},
    "Reddit": {"like": 0.090, "comment": 0.018, "share": 0.004},
    "Twitter/X": {"like": 0.034, "comment": 0.007, "share": 0.012},
}

NICHE_TRENDINESS = {
    "ai": 0.40,
    "beauty": 0.22,
    "business": 0.05,
    "coding": 0.20,
    "education": 0.08,
    "entertainment": 0.30,
    "fitness": 0.15,
    "food": 0.18,
    "gaming": 0.24,
    "productivity": 0.16,
}

TREND_TERMS = {
    "ai": ["AI tools", "ChatGPT", "automation"],
    "beauty": ["routine", "glow up", "dupe"],
    "business": ["startup", "side hustle", "growth"],
    "coding": ["code finally works", "programming", "debugging"],
    "education": ["study", "exam", "learning"],
    "entertainment": ["POV", "challenge", "reaction"],
    "fitness": ["workout", "protein", "transformation"],
    "food": ["recipe", "street food", "taste test"],
    "gaming": ["ranked", "speedrun", "patch"],
    "productivity": ["productivity", "focus", "deep work"],
}


def _choose_post_time(rng: np.random.Generator) -> datetime:
    days_back = int(rng.integers(0, 120))
    if rng.random() < 0.65:
        hour = int(rng.choice([18, 19, 20, 21, 22, 11, 12]))
    else:
        hour = int(rng.integers(0, 24))
    minute = int(rng.integers(0, 60))
    return datetime.now(timezone.utc).replace(minute=minute, second=0, microsecond=0) - timedelta(
        days=days_back,
        hours=(datetime.now(timezone.utc).hour - hour) % 24,
    )


def _caption_for(niche: str, platform: str, rng: np.random.Generator, hook_strength: float) -> str:
    term = rng.choice(TREND_TERMS[niche])
    templates = [
        f"I tried {term} for a week",
        f"{term}: what nobody tells beginners",
        f"POV: {term} finally makes sense",
        f"3 mistakes I made with {term}",
        f"Testing a viral {term} hack",
        f"My honest thoughts on {term}",
    ]
    caption = str(rng.choice(templates))
    if hook_strength > 1.0:
        caption = f"{caption} - results surprised me"
    if platform == "Reddit" and hook_strength < 0:
        caption = f"My thoughts on {niche}"
    return caption


def _hashtags_for(niche: str, rng: np.random.Generator, hook_strength: float) -> str:
    count = int(np.clip(rng.poisson(2 + max(hook_strength, 0)), 0, 8))
    tags = [f"#{niche}", "#viral" if hook_strength > 1.3 else "#tips", "#learn", "#creator"]
    rng.shuffle(tags)
    return " ".join(tags[:count])


def _media_type_for(platform: str, rng: np.random.Generator) -> str:
    if platform in {"TikTok", "YouTube"}:
        return str(rng.choice(["short_video", "video", "image"], p=[0.70, 0.25, 0.05]))
    if platform == "Reddit":
        return str(rng.choice(["text", "image", "link", "video"], p=[0.42, 0.25, 0.23, 0.10]))
    return str(rng.choice(MEDIA_TYPES, p=[0.38, 0.27, 0.12, 0.10, 0.13]))


def generate_synthetic_posts(n_rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-enough social post rows with five-day labels."""
    rng = np.random.default_rng(seed)
    rows = []

    for idx in range(n_rows):
        platform = str(rng.choice(SUPPORTED_PLATFORMS, p=[0.30, 0.27, 0.23, 0.20]))
        niche = str(rng.choice(NICHES))
        media_type = _media_type_for(platform, rng)
        account_follower_count = int(np.clip(rng.lognormal(mean=9.1, sigma=1.35), 25, 8_000_000))
        account_age_days = int(rng.integers(10, 4200))
        early_window_hours = float(rng.choice([1, 2, 3, 6], p=[0.30, 0.42, 0.18, 0.10]))
        post_time = _choose_post_time(rng)
        hour = post_time.hour

        hook_strength = rng.normal(0, 0.85) + NICHE_TRENDINESS[niche]
        peak_hour_bonus = 0.32 if hour in {18, 19, 20, 21, 22} else 0.0
        format_bonus = 0.0
        if platform in {"TikTok", "YouTube"} and media_type == "short_video":
            format_bonus += 0.46
        if platform == "Reddit" and media_type in {"text", "link"}:
            format_bonus += 0.20
        if platform == "Twitter/X" and media_type in {"text", "image"}:
            format_bonus += 0.14

        account_size_effect = min(np.log1p(account_follower_count) / 11, 1.45)
        quality = hook_strength + peak_hour_bonus + format_bonus + rng.normal(0, 0.45)
        reach_multiplier = np.exp(quality) * account_size_effect
        early_views = int(
            np.clip(
                account_follower_count * PLATFORM_DISCOVERY[platform] * reach_multiplier * rng.lognormal(0, 0.45),
                0,
                60_000_000,
            )
        )
        if platform == "Reddit":
            early_views = int(max(early_views, rng.poisson(max(account_follower_count * 0.015, 5))))

        rates = PLATFORM_ENGAGEMENT[platform]
        rate_boost = np.clip(np.exp(quality * 0.28), 0.35, 4.8)
        like_rate = np.clip(rates["like"] * rate_boost * rng.lognormal(0, 0.16), 0.001, 0.65)
        comment_rate = np.clip(rates["comment"] * rate_boost * rng.lognormal(0, 0.22), 0.0001, 0.22)
        share_rate = np.clip(rates["share"] * rate_boost * rng.lognormal(0, 0.25), 0.0001, 0.32)

        early_likes = int(rng.poisson(max(early_views * like_rate, 0)))
        early_comments = int(rng.poisson(max(early_views * comment_rate, 0)))
        early_shares = int(rng.poisson(max(early_views * share_rate, 0)))

        growth = 1 + rng.lognormal(mean=1.15 + max(quality, -1.5) * 0.38, sigma=0.55)
        five_day_views = int(np.clip(early_views * growth, early_views, 300_000_000))
        five_day_likes = int(np.clip(early_likes * growth * rng.lognormal(0, 0.18), early_likes, None))
        five_day_comments = int(np.clip(early_comments * growth * rng.lognormal(0, 0.22), early_comments, None))
        five_day_shares = int(np.clip(early_shares * growth * rng.lognormal(0, 0.24), early_shares, None))

        caption = _caption_for(niche, platform, rng, hook_strength)
        hashtags = _hashtags_for(niche, rng, hook_strength)

        rows.append(
            {
                "post_id": f"demo_{idx:05d}",
                "platform": platform,
                "niche": niche,
                "caption": caption,
                "hashtags": hashtags,
                "media_type": media_type,
                "post_time": post_time.isoformat(),
                "account_follower_count": account_follower_count,
                "account_age_days": account_age_days,
                "early_window_hours": early_window_hours,
                "early_views": early_views,
                "early_likes": early_likes,
                "early_comments": early_comments,
                "early_shares": early_shares,
                "five_day_views": five_day_views,
                "five_day_likes": five_day_likes,
                "five_day_comments": five_day_comments,
                "five_day_shares": five_day_shares,
                "five_day_engagement_rate": (five_day_likes + five_day_comments + five_day_shares)
                / max(five_day_views, 1),
            }
        )

    return ensure_virality_label(pd.DataFrame(rows))
