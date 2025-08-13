#!/usr/bin/env python3
"""
Complete data scraping pipeline for production
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
import ccxt
import os
import sys
import time
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_kraken_client():
    """Setup Kraken client with API credentials"""
    api_key = os.getenv("KRAKEN_API_KEY")
    secret = os.getenv("KRAKEN_SECRET")

    if api_key and secret:
        client = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": secret,
                "sandbox": False,
                "enableRateLimit": True,
                "timeout": 30000,
            }
        )
        logger.info("Kraken client initialized with API credentials")
    else:
        client = ccxt.kraken({"enableRateLimit": True, "timeout": 30000})
        logger.warning("Kraken client initialized without API credentials (public data only)")

    return client


def scrape_kraken_tickers(client) -> Dict:
    """Scrape all Kraken tickers"""
    logger.info("Scraping Kraken tickers...")

    try:
        # Get all markets
        markets = client.load_markets()
        logger.info(f"Loaded {len(markets)} Kraken markets")

        # Get tickers for USD pairs
        tickers = client.fetch_tickers()

        # Filter for USD pairs
        usd_tickers = {}
        for symbol, ticker in tickers.items():
            if "/USD" in symbol and ticker["last"] is not None:
                usd_tickers[symbol] = ticker

        logger.info(f"Collected {len(usd_tickers)} USD tickers")

        # Save raw data
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        ticker_data = {
            "timestamp": datetime.now().isoformat(),
            "exchange": "kraken",
            "tickers": usd_tickers,
        }

        ticker_file = raw_dir / "kraken_tickers.json"
        with open(ticker_file, "w") as f:
            json.dump(ticker_data, f, indent=2, default=str)

        logger.info(f"Saved ticker data to {ticker_file}")
        return usd_tickers

    except Exception as e:
        logger.error(f"Failed to scrape Kraken tickers: {e}")
        return {}


def scrape_historical_data(client, symbols: List[str], timeframe="1d", limit=100) -> Dict:
    """Scrape historical OHLCV data"""
    logger.info(f"Scraping historical data for {len(symbols)} symbols...")

    historical_data = {}

    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"Fetching {symbol} ({i + 1}/{len(symbols)})")

            # Fetch OHLCV data
            ohlcv = client.fetch_ohlcv(symbol, timeframe, limit=limit)

            if ohlcv:
                df = pd.DataFrame(
                    ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

                historical_data[symbol] = df.to_dict("records")
                logger.info(f"Collected {len(df)} candles for {symbol}")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            continue

    # Save historical data
    raw_dir = Path("data/raw")
    hist_file = raw_dir / "historical_data.json"

    with open(hist_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "data": historical_data,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Saved historical data to {hist_file}")
    return historical_data


def create_features(ticker_data: Dict, historical_data: Dict) -> pd.DataFrame:
    """Create features from scraped data"""
    logger.info("Creating features...")

    features_data = []

    for symbol, ticker in ticker_data.items():
        try:
            coin = symbol.split("/")[0]

            # Basic ticker features
            features = {
                "coin": coin,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "volume_24h": ticker["baseVolume"],
                "price_change_24h": ticker["percentage"],
                "high_24h": ticker["high"],
                "low_24h": ticker["low"],
            }

            # Calculate additional features
            if ticker["bid"] and ticker["ask"]:
                features["spread"] = (ticker["ask"] - ticker["bid"]) / ticker["ask"]
            else:
                features["spread"] = 0

            # Historical features if available
            if symbol in historical_data:
                hist_df = pd.DataFrame(historical_data[symbol])

                if len(hist_df) > 1:
                    # Price momentum features
                    hist_df["returns"] = hist_df["close"].pct_change()
                    features["volatility_7d"] = hist_df["returns"].tail(7).std()
                    features["momentum_3d"] = (
                        (hist_df["close"].iloc[-1] / hist_df["close"].iloc[-4] - 1)
                        if len(hist_df) >= 4
                        else 0
                    )
                    features["momentum_7d"] = (
                        (hist_df["close"].iloc[-1] / hist_df["close"].iloc[-8] - 1)
                        if len(hist_df) >= 8
                        else 0
                    )

                    # Volume features
                    features["volume_trend_7d"] = (
                        hist_df["volume"].tail(7).mean() / hist_df["volume"].mean()
                        if hist_df["volume"].mean() > 0
                        else 1
                    )

                    # Technical indicators (simplified)
                    if len(hist_df) >= 20:
                        sma_20 = hist_df["close"].tail(20).mean()
                        features["price_vs_sma20"] = (ticker["last"] - sma_20) / sma_20
                    else:
                        features["price_vs_sma20"] = 0

            # Market cap proxy (volume * price)
            if ticker["baseVolume"] and ticker["last"]:
                features["market_activity"] = ticker["baseVolume"] * ticker["last"]
            else:
                features["market_activity"] = 0

            # Risk features
            features["price_volatility"] = abs(features.get("price_change_24h", 0)) / 100
            features["liquidity_score"] = (
                min(ticker["baseVolume"] / 1000000, 1.0) if ticker["baseVolume"] else 0
            )

            features_data.append(features)

        except Exception as e:
            logger.warning(f"Failed to create features for {symbol}: {e}")
            continue

    features_df = pd.DataFrame(features_data)
    logger.info(f"Created features for {len(features_df)} coins")

    # Save processed features
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    features_file = processed_dir / "features.csv"
    features_df.to_csv(features_file, index=False)
    logger.info(f"Saved features to {features_file}")

    return features_df


def main():
    """Main scraping pipeline"""
    logger.info("Starting complete data scraping pipeline")

    # Setup client
    client = setup_kraken_client()

    # Scrape ticker data
    ticker_data = scrape_kraken_tickers(client)

    if not ticker_data:
        logger.error("Failed to scrape ticker data")
        sys.exit(1)

    # Get top symbols by volume for historical data
    top_symbols = sorted(ticker_data.items(), key=lambda x: x[1]["baseVolume"] or 0, reverse=True)[
        :50
    ]  # Top 50 by volume

    top_symbol_names = [symbol for symbol, _ in top_symbols]
    logger.info(f"Selected top {len(top_symbol_names)} symbols for historical data")

    # Scrape historical data
    historical_data = scrape_historical_data(client, top_symbol_names)

    # Create features
    features_df = create_features(ticker_data, historical_data)

    if features_df.empty:
        logger.error("Failed to create features")
        sys.exit(1)

    # Summary
    logger.info("=== Scraping Summary ===")
    logger.info(f"Tickers collected: {len(ticker_data)}")
    logger.info(f"Historical data: {len(historical_data)} symbols")
    logger.info(f"Features created: {len(features_df)} coins")
    logger.info(
        f"Average volume: ${features_df['volume_24h'].mean():,.0f}"
        if "volume_24h" in features_df.columns
        else "Volume data unavailable"
    )

    logger.info("Complete data scraping pipeline finished successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
