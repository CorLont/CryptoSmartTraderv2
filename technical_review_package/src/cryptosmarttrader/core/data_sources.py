#!/usr/bin/env python3
"""
Data Sources - Centralized data acquisition with resilience patterns
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from ..core.http_client import http_client


class DataSourceManager:
    """
    Centralized data source management

    Handles:
    - Exchange API data (Kraken, Binance)
    - Market data aggregators (CoinGecko, CoinMarketCap)
    - Social sentiment (Reddit, Twitter)
    - News sources
    - Rate limiting and error handling
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Exchange endpoints
        self.exchange_endpoints = {
            "kraken": {
                "base_url": "https://api.kraken.com/0",
                "ticker": "/public/Ticker",
                "orderbook": "/public/Depth",
                "trades": "/public/Trades",
                "ohlc": "/public/OHLC",
            },
            "binance": {
                "base_url": "https://api.binance.com/api/v3",
                "ticker": "/ticker/24hr",
                "orderbook": "/depth",
                "trades": "/trades",
                "klines": "/klines",
            },
        }

        # Market data endpoints
        self.market_endpoints = {
            "coingecko": {
                "base_url": "https://api.coingecko.com/api/v3",
                "coins": "/coins/markets",
                "price": "/simple/price",
                "trending": "/search/trending",
            },
            "coinmarketcap": {
                "base_url": "https://pro-api.coinmarketcap.com/v1",
                "listings": "/cryptocurrency/listings/latest",
                "quotes": "/cryptocurrency/quotes/latest",
                "metadata": "/cryptocurrency/info",
            },
        }

        # Social/news endpoints
        self.social_endpoints = {
            "reddit": {
                "base_url": "https://www.reddit.com/r",
                "search": "/search.json",
                "hot": "/hot.json",
            },
            "newsapi": {
                "base_url": "https://newsapi.org/v2",
                "everything": "/everything",
                "headlines": "/top-headlines",
            },
        }

    async def get_exchange_data(
        self, exchange: str, endpoint_type: str, symbol: Optional[str] = None, **params
    ) -> Dict[str, Any]:
        """
        Get data from cryptocurrency exchange

        Args:
            exchange: Exchange name (kraken, binance)
            endpoint_type: Type of data (ticker, orderbook, trades, ohlc)
            symbol: Trading pair symbol
            **params: Additional parameters

        Returns:
            Exchange API response
        """

        if exchange not in self.exchange_endpoints:
            raise ValueError(f"Unsupported exchange: {exchange}")

        config = self.exchange_endpoints[exchange]

        if endpoint_type not in config:
            raise ValueError(f"Unsupported endpoint type: {endpoint_type}")

        url = config["base_url"] + config[endpoint_type]

        # Add symbol to parameters
        if symbol:
            if exchange == "kraken":
                params["pair"] = symbol
            elif exchange == "binance":
                params["symbol"] = symbol

        try:
            data = await http_client.get(
                service=exchange,
                url=url,
                params=params,
                data_type="price" if endpoint_type == "ticker" else "market",
            )

            self.logger.debug(f"Retrieved {endpoint_type} data from {exchange}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to get {endpoint_type} from {exchange}: {e}")
            raise

    async def get_market_overview(self, source: str = "coingecko", limit: int = 100) -> List[Dict]:
        """
        Get market overview data

        Args:
            source: Data source (coingecko, coinmarketcap)
            limit: Number of coins to fetch

        Returns:
            List of coin market data
        """

        if source == "coingecko":
            url = (
                self.market_endpoints["coingecko"]["base_url"]
                + self.market_endpoints["coingecko"]["coins"]
            )

            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": limit,
                "page": 1,
                "sparkline": "false",
            }

        elif source == "coinmarketcap":
            url = (
                self.market_endpoints["coinmarketcap"]["base_url"]
                + self.market_endpoints["coinmarketcap"]["listings"]
            )

            params = {"start": 1, "limit": limit, "convert": "USD"}

        else:
            raise ValueError(f"Unsupported market data source: {source}")

        try:
            data = await http_client.get(
                service=source,
                url=url,
                params=params,
                data_type="market",
                cache_ttl=300,  # 5 minute cache
            )

            self.logger.info(f"Retrieved market overview from {source} ({limit} coins)")
            return data

        except Exception as e:
            self.logger.error(f"Failed to get market overview from {source}: {e}")
            raise

    async def get_price_data(
        self, symbols: List[str], source: str = "coingecko"
    ) -> Dict[str, Dict]:
        """
        Get current price data for multiple symbols

        Args:
            symbols: List of coin IDs or symbols
            source: Data source

        Returns:
            Price data by symbol
        """

        if source == "coingecko":
            url = (
                self.market_endpoints["coingecko"]["base_url"]
                + self.market_endpoints["coingecko"]["price"]
            )

            params = {
                "ids": ",".join(symbols),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
            }

        else:
            raise ValueError(f"Unsupported price data source: {source}")

        try:
            data = await http_client.get(
                service=source,
                url=url,
                params=params,
                data_type="price",
                cache_ttl=30,  # 30 second cache for prices
            )

            self.logger.debug(f"Retrieved price data for {len(symbols)} symbols from {source}")
            return data

        except Exception as e:
            self.logger.error(f"Failed to get price data from {source}: {e}")
            raise

    async def get_social_sentiment(
        self, query: str, source: str = "reddit", limit: int = 25
    ) -> List[Dict]:
        """
        Get social sentiment data

        Args:
            query: Search query
            source: Social media source
            limit: Number of posts to analyze

        Returns:
            Social media posts for sentiment analysis
        """

        if source == "reddit":
            # Search across crypto subreddits
            subreddits = ["cryptocurrency", "bitcoin", "ethereum", "altcoin"]
            all_posts = []

            for subreddit in subreddits:
                try:
                    url = f"{self.social_endpoints['reddit']['base_url']}/{subreddit}/search.json"

                    params = {
                        "q": query,
                        "sort": "relevance",
                        "t": "day",  # Last 24 hours
                        "limit": limit // len(subreddits),
                    }

                    data = await http_client.get(
                        service="reddit",
                        url=url,
                        params=params,
                        data_type="social",
                        cache_ttl=180,  # 3 minute cache
                    )

                    if "data" in data and "children" in data["data"]:
                        posts = [post["data"] for post in data["data"]["children"]]
                        all_posts.extend(posts)

                except Exception as e:
                    self.logger.warning(f"Failed to get Reddit data from r/{subreddit}: {e}")
                    continue

                # Small delay between subreddit requests
                await asyncio.sleep(0.5)

            self.logger.info(f"Retrieved {len(all_posts)} Reddit posts for '{query}'")
            return all_posts

        else:
            raise ValueError(f"Unsupported social source: {source}")

    async def get_news_data(
        self, query: str = "cryptocurrency", source: str = "newsapi", limit: int = 50
    ) -> List[Dict]:
        """
        Get news articles for sentiment analysis

        Args:
            query: Search query
            source: News source
            limit: Number of articles

        Returns:
            News articles
        """

        if source == "newsapi":
            url = (
                self.social_endpoints["newsapi"]["base_url"]
                + self.social_endpoints["newsapi"]["everything"]
            )

            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": limit,
                "from": (datetime.utcnow() - timedelta(days=1)).isoformat(),
            }

        else:
            raise ValueError(f"Unsupported news source: {source}")

        try:
            # Note: NewsAPI requires API key
            headers = {}
            # headers["X-API-Key"] = settings.NEWS_API_KEY  # Add when available

            data = await http_client.get(
                service=source,
                url=url,
                params=params,
                headers=headers,
                data_type="news",
                cache_ttl=600,  # 10 minute cache
            )

            articles = data.get("articles", []) if isinstance(data, dict) else []
            self.logger.info(f"Retrieved {len(articles)} news articles for '{query}'")
            return articles

        except Exception as e:
            self.logger.error(f"Failed to get news data from {source}: {e}")
            return []  # Return empty list instead of raising

    async def get_trending_coins(self, source: str = "coingecko") -> List[Dict]:
        """
        Get trending/popular coins

        Args:
            source: Data source

        Returns:
            Trending coins data
        """

        if source == "coingecko":
            url = (
                self.market_endpoints["coingecko"]["base_url"]
                + self.market_endpoints["coingecko"]["trending"]
            )

        else:
            raise ValueError(f"Unsupported trending source: {source}")

        try:
            data = await http_client.get(
                service=source,
                url=url,
                data_type="market",
                cache_ttl=900,  # 15 minute cache
            )

            trending = data.get("coins", []) if isinstance(data, dict) else []
            self.logger.info(f"Retrieved {len(trending)} trending coins from {source}")
            return trending

        except Exception as e:
            self.logger.error(f"Failed to get trending data from {source}: {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all data sources

        Returns:
            Health status of each data source
        """

        health_status = {}

        # Check circuit breaker status
        circuit_status = http_client.get_circuit_breaker_status()

        # Test connectivity to key endpoints
        test_endpoints = [
            ("kraken", "https://api.kraken.com/0/public/Time"),
            ("binance", "https://api.binance.com/api/v3/ping"),
            ("coingecko", "https://api.coingecko.com/api/v3/ping"),
        ]

        for service, test_url in test_endpoints:
            try:
                start_time = datetime.utcnow()
                await http_client.get(service=service, url=test_url, use_cache=False)
                response_time = (datetime.utcnow() - start_time).total_seconds()

                health_status[service] = {
                    "status": "healthy",
                    "response_time_ms": round(response_time * 1000, 2),
                    "circuit_breaker": circuit_status.get(service, {}),
                }

            except Exception as e:
                health_status[service] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "circuit_breaker": circuit_status.get(service, {}),
                }

        # Add cache statistics
        health_status["cache"] = http_client.get_cache_stats()

        return health_status

    async def close(self):
        """Close HTTP client connections"""
        await http_client.close()


# Global data source manager
data_sources = DataSourceManager()
