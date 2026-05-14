# ViralScope Model Card

**Last updated:** May 14, 2026

## Summary

ViralScope is a local-first machine learning project that predicts whether a social media post is likely to become viral within five days.

The project currently uses tabular features derived from post metadata and early engagement signals. It supports:

- synthetic benchmark data,
- uploaded CSV datasets,
- official live-data adapters for YouTube and TikTok connected accounts.

## Prediction Target

Default target:

> A post is viral if its five-day engagement rate is in the top 5% within the same platform and niche.

If the source dataset already contains `is_viral`, the training pipeline uses that label directly.

## Inputs

Core raw inputs include:

- `platform`
- `niche`
- `caption`
- `hashtags`
- `media_type`
- `post_time`
- `account_follower_count`
- `account_age_days`
- `early_window_hours`
- `early_views`
- `early_likes`
- `early_comments`
- `early_shares`

## Engineered Features

The project derives additional features such as:

- caption length and word count
- hashtag and mention counts
- posting hour and day cyclical features
- early engagement rate
- likes, comments, shares per view
- views, likes, comments, shares per hour
- engagement per follower
- platform-normalized velocity ratios

## Models

Supported model families:

- Logistic Regression
- Random Forest
- Gradient Boosting
- optional `xgboost`
- optional `lightgbm`

The default training flow applies probability calibration when class coverage allows it.

## Validation

The current training pipeline uses a **temporal holdout** by default:

- earlier posts are used for training
- later posts are used for evaluation

If the time-ordered split cannot keep both classes in train and test, the pipeline falls back to a random holdout and records that fallback in metadata.

Evaluation surfaces include:

- ROC AUC
- average precision
- F1
- Brier score
- confusion matrix
- calibration curve
- precision-recall curve
- platform slice metrics
- niche slice metrics

## Intended Use

ViralScope is intended for:

- portfolio demonstrations
- local experimentation
- creator analytics prototypes
- benchmarking early-engagement-based virality scoring

It is not intended for:

- guaranteed reach forecasting
- ad spend allocation without human review
- political manipulation
- harassment or spam workflows

## Limitations

- Cross-platform virality behavior is not identical, so performance can vary by platform and niche.
- Synthetic data is useful for UX and architecture validation, but it is not a substitute for real labeled outcomes.
- Public live APIs often expose cumulative metrics, not true first-hour snapshots.
- YouTube does not expose share count through the public Data API endpoints used here.
- TikTok connected-account mode only works for creator-authorized public account data.
- Reddit and some other platform datasets may have explicit restrictions on ML training use.

## Privacy and Compliance Notes

- ViralScope does not scrape restricted platforms.
- Live integrations should only use official APIs.
- Training data should only be used when the user has the right to store and model it.
- See `privacy.md` and `DATASET_CARD.md` for the current data-handling policy.
