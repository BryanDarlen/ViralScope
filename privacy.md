# ViralScope Privacy and Data Use Policy

**Last updated: May 14, 2026**

## Overview

ViralScope is a local, portfolio-oriented machine learning project for studying social media virality. The MVP is designed to run without scraping restricted platforms.

The project supports:

- synthetic demo data,
- manually uploaded CSV files supplied by the user,
- official API integrations only when platform terms and credentials allow them.

## Data Sources

ViralScope does not require live platform API access for the MVP. By default, it trains and predicts using synthetic data or local CSV files.

Current live-data helper direction:

- YouTube official Data API for public video metadata and statistics
- TikTok connected-account Display API for creator-authorized profile and public video metadata

Official API adapters should only use approved APIs and should follow each platform's developer terms, rate limits, data retention rules, privacy rules, and restrictions on machine learning use.

## Data Accessed

Typical local CSV fields may include:

- platform and niche,
- caption/title text,
- hashtags or topic tags,
- media type,
- posting timestamp,
- account size or audience size,
- account age,
- early views, likes/upvotes, comments/replies, and shares/retweets,
- optional five-day engagement metrics or an `is_viral` label.

The MVP does not require private messages, private profiles, passwords, cookies, browser sessions, or non-public user data.

For live helper scripts, ViralScope may access:

- public YouTube video metadata and statistics,
- public YouTube channel metadata and subscriber counts,
- TikTok connected-account profile fields authorized by the creator,
- TikTok connected-account public video metrics authorized by the creator.

## Data Storage

- Synthetic datasets may be saved locally under `data/processed/`.
- Live API pulls may be saved locally under `data/raw/` when the user chooses an output path.
- Trained model artifacts may be saved locally under `models/`.
- Uploaded CSV data in the web dashboard is processed in memory unless the user explicitly saves derived files.
- Raw data and model artifacts are gitignored by default.
- Access tokens and client secrets should never be committed to git.

## Machine Learning Use

ViralScope can train machine learning models on synthetic data or CSV data that the user has the right to use for that purpose.

Do not train on platform data if the source terms prohibit machine learning use, long-term storage, redistribution, or derived datasets. When in doubt, use synthetic data or manually curated data with clear permission.

In particular:

- YouTube public metadata may be used only in ways consistent with YouTube API terms and quotas.
- TikTok connected-account data should only be used for the creator who explicitly authorized the application.
- Platform-specific restrictions can change over time, so verify terms before building production workflows.

## Responsible Use

ViralScope is intended for educational, portfolio, and product-prototyping use. It should not be used to:

- identify, profile, or target private individuals,
- infer sensitive personal attributes,
- bypass platform access controls,
- scrape restricted content,
- redistribute platform data without permission,
- automate spam or manipulation.

## Contact

Bryan Quinn Darlen - darlen.bryan77@gmail.com
