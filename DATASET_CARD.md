# ViralScope Dataset Card

**Last updated:** May 14, 2026

## Overview

ViralScope is designed to work with multiple dataset modes:

1. synthetic demo data generated locally
2. user-supplied CSV files
3. official live API data where platform terms and credentials allow it

The project is intentionally CSV-first and API-safe.

## Dataset Modes

## 1. Synthetic Demo Data

The synthetic generator creates plausible but artificial rows for:

- YouTube
- TikTok
- Reddit
- Twitter/X

Synthetic rows simulate:

- account size
- posting time
- media type
- early views, likes, comments, shares
- five-day engagement outcomes

Use cases:

- frontend demos
- model pipeline smoke tests
- hackathon or portfolio demonstrations

Limitations:

- not suitable as evidence of real-world predictive accuracy
- may encode assumptions from the generator rather than platform reality

## 2. User CSV Data

Users can upload historical CSVs containing:

- early-post features
- optional five-day final metrics
- optional `is_viral` labels

Recommended minimum fields:

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

Recommended training labels:

- `is_viral`
- or `five_day_engagement_rate`
- or `five_day_views`, `five_day_likes`, `five_day_comments`, `five_day_shares`

## 3. Official Live Data Adapters

Current adapter direction:

- **YouTube official adapter**
  - public video lookup and search
  - public benchmark pulls using `mostPopular`
- **TikTok connected-account adapter**
  - creator-authorized profile and public video metadata

Important:

- live API pulls usually return **current cumulative metrics**
- they do not automatically give exact five-day labels
- historical benchmarking still requires repeated collection over time or owned historical exports

## Label Definition

Default derived label:

> `is_viral = 1` when five-day engagement rate is in the top 5% within the same platform and niche.

If `is_viral` already exists in the uploaded dataset, that label is preserved.

## Known Biases and Risks

- synthetic data can overfit to generator assumptions
- creator-owned datasets may overrepresent a narrow niche or audience size
- public platform metrics often omit hidden variables such as retention, impressions, or recommendation traffic sources
- account-size effects can dominate weak feature sets if not normalized carefully

## Collection and Storage Guidance

- keep raw live pulls under `data/raw/`
- keep processed benchmarking data under `data/processed/`
- do not commit private creator exports or access tokens to git
- document source, pull date, and terms for each real dataset

## Recommended Metadata for Real Datasets

For each real dataset, store:

- source platform
- collection method
- API endpoint or export source
- collection timestamp
- time zone assumptions
- labeling method
- known missing fields
- usage permission notes

## Compliance Notes

- do not scrape restricted platforms
- do not train on data when the platform terms or data owner permissions do not allow it
- prefer creator-owned, consented, or platform-permitted historical data
