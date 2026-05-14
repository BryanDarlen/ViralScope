"""Shared configuration for ViralScope."""

from pathlib import Path

RANDOM_STATE = 42
TARGET_COLUMN = "is_viral"
VIRALITY_QUANTILE = 0.95

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_ARTIFACT_PATH = MODEL_DIR / "viralscope_model.joblib"
DEMO_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_viralscope_posts.csv"

SUPPORTED_PLATFORMS = ["YouTube", "TikTok", "Reddit", "Twitter/X"]
MEDIA_TYPES = ["text", "image", "video", "short_video", "link"]
NICHES = [
    "ai",
    "beauty",
    "business",
    "coding",
    "education",
    "entertainment",
    "fitness",
    "food",
    "gaming",
    "productivity",
]

PLATFORM_ALIASES = {
    "youtube": "YouTube",
    "yt": "YouTube",
    "tiktok": "TikTok",
    "tik tok": "TikTok",
    "reddit": "Reddit",
    "twitter": "Twitter/X",
    "twitter/x": "Twitter/X",
    "x": "Twitter/X",
    "x/twitter": "Twitter/X",
}

RAW_INPUT_COLUMNS = [
    "platform",
    "niche",
    "caption",
    "hashtags",
    "media_type",
    "post_time",
    "account_follower_count",
    "account_age_days",
    "early_window_hours",
    "early_views",
    "early_likes",
    "early_comments",
    "early_shares",
]

FINAL_METRIC_COLUMNS = [
    "five_day_views",
    "five_day_likes",
    "five_day_comments",
    "five_day_shares",
    "five_day_engagement_rate",
]
