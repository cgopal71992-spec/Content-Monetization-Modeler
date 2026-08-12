# Content Monetization Modeler

A machine learning project that predicts YouTube ad revenue based on video performance metrics and contextual features.

## Project Objective

The goal is to estimate `ad_revenue_usd` for YouTube videos using features such as views, likes, comments, watch time, subscribers, category, device, country, and date-related information.

## Dataset

- Dataset: YouTube Ad Revenue Dataset
- Total records: 122,400
- Target variable: `ad_revenue_usd`
- Missing values handled in likes, comments, and watch time
- Duplicate rows removed: 2,400

## Data Preprocessing

- Removed duplicate records
- Filled missing numeric values using the median
- Converted date into month and day-of-week features
- Created an engagement-rate feature:

```text
engagement_rate = (likes + comments) / views
