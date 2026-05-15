"""Human-readable prediction explanations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ExplanationFactor:
    name: str
    impact: float
    direction: str
    detail: str


def _factor(name: str, impact: float, direction: str, detail: str) -> ExplanationFactor:
    return ExplanationFactor(name=name, impact=float(abs(impact)), direction=direction, detail=detail)


def explain_prediction(model, raw_post: dict, metadata: dict | None = None) -> tuple[list[dict], list[dict], dict]:
    """Create concise local explanations from engineered signal strength.

    This is intentionally model-adjacent rather than pretending every estimator has
    exact SHAP values available. It compares the post's engineered signals against
    platform baselines learned during training.
    """
    feature_step = model.named_steps["features"]
    engineered = feature_step.transform(pd.DataFrame([raw_post])).iloc[0].to_dict()
    platform = str(raw_post.get("platform", "Unknown"))
    media_type = str(raw_post.get("media_type", "text")).lower()

    factors: list[ExplanationFactor] = []

    engagement_ratio = engineered["engagement_rate_vs_platform"]
    if engagement_ratio >= 2.5:
        factors.append(_factor("Early engagement rate", engagement_ratio, "positive", "Engagement rate is far above the platform baseline."))
    elif engagement_ratio < 0.75:
        factors.append(_factor("Early engagement rate", 1 - engagement_ratio, "negative", "Engagement rate is below the platform baseline."))

    views_ratio = engineered["views_per_hour_vs_platform"]
    if views_ratio >= 2.0:
        factors.append(_factor("View velocity", views_ratio, "positive", "Views per hour are outperforming comparable posts."))
    elif views_ratio < 0.60:
        factors.append(_factor("View velocity", 1 - views_ratio, "negative", "Views per hour are still weak for this platform."))

    comments_ratio = engineered["comments_per_hour_vs_platform"]
    if comments_ratio >= 2.0:
        factors.append(_factor("Comment velocity", comments_ratio, "positive", "Comment activity suggests strong discussion momentum."))
    elif comments_ratio < 0.50:
        factors.append(_factor("Comment velocity", 1 - comments_ratio, "negative", "Low comment velocity may limit ranking and sharing loops."))

    shares_ratio = engineered["shares_per_hour_vs_platform"]
    if shares_ratio >= 2.0:
        factors.append(_factor("Share velocity", shares_ratio, "positive", "Share behavior is a strong early virality signal."))
    elif shares_ratio < 0.50:
        factors.append(_factor("Share velocity", 1 - shares_ratio, "negative", "Share velocity is not yet strong."))

    views_per_follower = engineered["views_per_follower"]
    if views_per_follower >= 0.50:
        factors.append(_factor("Audience overperformance", views_per_follower, "positive", "The post is reaching beyond the account's normal audience size."))
    elif views_per_follower < 0.03:
        factors.append(_factor("Audience overperformance", 1 - views_per_follower, "negative", "Early reach is small relative to account size."))

    hashtag_count = engineered["hashtag_count"]
    if 2 <= hashtag_count <= 5:
        factors.append(_factor("Hashtag focus", 1.0, "positive", "Hashtag count is focused enough for discoverability without looking spammy."))
    elif hashtag_count == 0:
        factors.append(_factor("Hashtag focus", 0.8, "negative", "No hashtags or topic tags were detected."))
    elif hashtag_count > 8:
        factors.append(_factor("Hashtag focus", 0.6, "negative", "Hashtag count is high and may dilute the content's topic focus."))

    if engineered["trend_keyword_count"] > 0:
        factors.append(_factor("Trend language", engineered["trend_keyword_count"], "positive", "Caption or tags contain recognizable trend/community language."))

    hour = int(engineered["posting_hour"])
    if hour in {18, 19, 20, 21, 22}:
        factors.append(_factor("Posting time", 0.8, "positive", "Posting time falls in a common high-activity evening window."))
    elif hour in {2, 3, 4, 5}:
        factors.append(_factor("Posting time", 0.6, "negative", "Posting time is in a low-activity overnight window for many audiences."))

    if platform in {"TikTok", "YouTube"} and media_type == "short_video":
        factors.append(_factor("Format fit", 1.1, "positive", "Short-form video fits discovery-heavy platforms well."))
    elif platform == "Reddit" and media_type in {"text", "link"}:
        factors.append(_factor("Format fit", 0.7, "positive", "Text/link formats can work well for community discussion."))

    positives = [item for item in factors if item.direction == "positive"]
    negatives = [item for item in factors if item.direction == "negative"]
    positives = sorted(positives, key=lambda item: item.impact, reverse=True)[:5]
    negatives = sorted(negatives, key=lambda item: item.impact, reverse=True)[:5]
    return (
        [item.__dict__ for item in positives],
        [item.__dict__ for item in negatives],
        engineered,
    )


def recommendations_from_factors(negative_factors: list[dict], engineered: dict) -> list[str]:
    names = {factor["name"] for factor in negative_factors}
    recommendations = []
    if "View velocity" in names:
        recommendations.append(
            "Improve the opening hook or distribution timing to lift views per hour. "
            "Example: start with a clear payoff like 'I tested this focus playlist for 7 days. Here is what changed.'"
        )
    if "Comment velocity" in names:
        recommendations.append(
            "Add a concrete question or opinion hook that invites replies. "
            "Example: ask 'Which version would you use while studying: calm piano or full orchestra?'"
        )
    if "Share velocity" in names:
        recommendations.append(
            "Make the payoff more saveable or shareable: checklist, surprise, template, or relatable moment. "
            "Example: frame it as '3 tracks to save for deep work, reading, and exam prep.'"
        )
    if "Hashtag focus" in names and engineered.get("hashtag_count", 0) == 0:
        recommendations.append(
            "Add 2-4 niche-specific hashtags or topic tags. "
            "Example: use focused tags like #study, #focusmusic, and #productivity instead of broad generic tags."
        )
    if "Posting time" in names:
        recommendations.append(
            "Test posting closer to your audience's active window, often evening or lunch hours. "
            "Example: compare the same format at 12 PM and 8 PM for two weeks."
        )
    if not recommendations:
        recommendations.append(
            "Keep monitoring early velocity; the strongest next improvement is testing clearer hooks and thumbnails/titles. "
            "Example: try two title styles, one curiosity-led and one benefit-led, then compare the first 2-hour view rate."
        )
    return recommendations[:4]


def reasoning_summary(probability: float, positives: list[dict], negatives: list[dict]) -> str:
    level = "low"
    if probability >= 0.75:
        level = "high"
    elif probability >= 0.55:
        level = "medium-high"
    elif probability >= 0.30:
        level = "medium"

    positive_text = ", ".join(factor["name"].lower() for factor in positives[:3]) or "no strong positive signal"
    negative_text = ", ".join(factor["name"].lower() for factor in negatives[:2]) or "no major weakness detected"
    return f"The model sees {level} viral potential, mainly driven by {positive_text}. Main watchout: {negative_text}."
