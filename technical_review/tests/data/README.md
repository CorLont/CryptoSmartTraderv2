# Test Data

This directory contains small test datasets for use in unit and integration tests.

## Guidelines

- Keep files small (< 1MB each)
- Use compressed formats (Parquet, CSV.gz) when possible
- No large artifacts or real production data
- Generate synthetic data that mimics real patterns
- Document data schema and purpose

## Files

- `sample_market_data.parquet`: Sample OHLCV data for testing
- `sample_portfolio.json`: Sample portfolio positions
- `sample_signals.csv`: Sample trading signals for backtesting
- `sample_news.json`: Sample news articles for sentiment analysis
- `sample_features.parquet`: Sample engineered features

## Data Generation

Test data is generated using deterministic seeds to ensure reproducible tests.
See `tests/fixtures/data_generators.py` for generation scripts.