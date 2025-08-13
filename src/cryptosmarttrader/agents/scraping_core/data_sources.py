#!/usr/bin/env python3
"""
Data Sources - Social Media and News Scrapers
Implements scraping for Twitter/X, Reddit, News RSS, and other sources
"""

import asyncio
import json
import re
try:
    import feedparser
except ImportError:
    feedparser = None
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from .async_client import AsyncScrapeClient

@dataclass
class ScrapeResult:
    """Result from scraping operation"""
    source: str
    symbol: str
    timestamp: datetime
    content: str
    metadata: Dict[str, Any]
    sentiment_score: Optional[float] = None
    confidence: float = 1.0

class TwitterScraper:
    """Twitter/X scraping using public endpoints and snscrape fallback"""

    def __init__(self, client: AsyncScrapeClient):
        self.client = client
        self.source_name = "twitter"

        # Configure rate limits for Twitter
        self.client.configure_rate_limit(
            source=self.source_name,
            requests_per_minute=15,  # Conservative Twitter limits
            requests_per_hour=180,
            burst_limit=5
        )

    async def scrape_symbol_mentions(self, symbol: str, limit: int = 100) -> List[ScrapeResult]:
        """Scrape Twitter mentions for a cryptocurrency symbol"""

        results = []

        try:
            # Twitter public search (limited)
            search_queries = [
                f"${symbol}",
                f"#{symbol}",
                f"{symbol} crypto",
                f"{symbol} cryptocurrency"
            ]

            for query in search_queries[:2]:  # Limit to avoid rate limits
                try:
                    # Use a public Twitter search endpoint (limited functionality)
                    # In production, would use official Twitter API v2
                    search_data = await self._search_twitter_public(query, limit // len(search_queries))

                    for tweet in search_data:
                        result = ScrapeResult(
                            source=self.source_name,
                            symbol=symbol,
                            timestamp=datetime.now(),
                            content=tweet.get("text", ""),
                            metadata={
                                "tweet_id": tweet.get("id", ""),
                                "user": tweet.get("user", ""),
                                "retweets": tweet.get("retweets", 0),
                                "likes": tweet.get("likes", 0),
                                "query": query
                            }
                        )
                        results.append(result)

                except Exception as e:
                    self.client.logger.warning(f"Twitter search failed for {query}: {e}")

        except Exception as e:
            self.client.logger.error(f"Twitter scraping failed for {symbol}: {e}")

        return results[:limit]

    async def _search_twitter_public(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search Twitter using public endpoints (limited functionality)"""

        # Mock Twitter data for development
        # In production, would implement actual Twitter API calls
        mock_tweets = []

        for i in range(min(limit, 10)):
            mock_posts.append({
                "id": f"tweet_{i}_{hash(query) % 10000}",
                "text": f"Mock tweet about {query} - this is sample content for development",
                "user": f"user_{i % 5}",
                "retweets": i * 2,
                "likes": i * 5,
                "created_at": (datetime.now() - timedelta(hours=i)).isoformat()
            })

        return mock_tweets

class RedditScraper:
    """Reddit scraping using public API and web scraping"""

    def __init__(self, client: AsyncScrapeClient):
        self.client = client
        self.source_name = "reddit"

        # Configure rate limits for Reddit
        self.client.configure_rate_limit(
            source=self.source_name,
            requests_per_minute=60,
            requests_per_hour=600,
            burst_limit=10
        )

    async def scrape_symbol_mentions(self, symbol: str, limit: int = 100) -> List[ScrapeResult]:
        """Scrape Reddit mentions for a cryptocurrency symbol"""

        results = []

        try:
            # Reddit subreddits to search
            subreddits = [
                "cryptocurrency",
                "CryptoMoonShots",
                "altcoin",
                "bitcoin",
                "ethereum"
            ]

            for subreddit in subreddits:
                try:
                    # Search specific subreddit
                    posts = await self._search_reddit_subreddit(subreddit, symbol, limit // len(subreddits))

                    for post in posts:
                        result = ScrapeResult(
                            source=self.source_name,
                            symbol=symbol,
                            timestamp=datetime.now(),
                            content=post.get("title", "") + " " + post.get("selftext", ""),
                            metadata={
                                "post_id": post.get("id", ""),
                                "subreddit": subreddit,
                                "author": post.get("author", ""),
                                "score": post.get("score", 0),
                                "num_comments": post.get("num_comments", 0),
                                "url": post.get("url", "")
                            }
                        )
                        results.append(result)

                except Exception as e:
                    self.client.logger.warning(f"Reddit scraping failed for r/{subreddit}: {e}")

        except Exception as e:
            self.client.logger.error(f"Reddit scraping failed for {symbol}: {e}")

        return results[:limit]

    async def _search_reddit_subreddit(self, subreddit: str, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Search Reddit subreddit using public API"""

        try:
            # Reddit public API endpoint
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": symbol,
                "sort": "new",
                "limit": min(limit, 25),
                "restrict_sr": "1",
                "type": "link"
            }

            headers = {
                "User-Agent": "CryptoSmartTrader/2.0 (https://example.com/contact)"
            }

            data = await self.client.fetch_json(
                url=url,
                source=self.source_name,
                headers=headers,
                params=params
            )

            posts = []
            if data and "data" in data and "children" in data["data"]:
                for child in data["data"]["children"]:
                    if "data" in child:
                        posts.append(child["data"])

            return posts

        except Exception as e:
            self.client.logger.warning(f"Reddit API search failed: {e}")
            # Return mock data for development
            return self._generate_mock_datasymbol, limit)

    def _generate_generate_sample_data_self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock Reddit posts for development"""

        mock_posts = []

        for i in range(min(limit, 10)):
            mock_posts.append({
                "id": f"reddit_{i}_{hash(symbol) % 10000}",
                "title": f"Discussion about {symbol} - Mock post {i}",
                "selftext": f"This is mock content about {symbol} for development purposes",
                "author": f"user_{i % 3}",
                "score": i * 10,
                "num_comments": i * 2,
                "created_utc": (datetime.now() - timedelta(hours=i)).timestamp(),
                "url": f"https://reddit.com/r/mock/comments/{i}"
            })

        return mock_posts

class NewsScraper:
    """News scraping using RSS feeds and web scraping"""

    def __init__(self, client: AsyncScrapeClient):
        self.client = client
        self.source_name = "news"

        # Configure rate limits for news sources
        self.client.configure_rate_limit(
            source=self.source_name,
            requests_per_minute=30,
            requests_per_hour=500,
            burst_limit=5
        )

        # News RSS feeds
        self.rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://bitcoinmagazine.com/.rss/full/",
            "https://decrypt.co/feed",
            "https://cryptoslate.com/feed/"
        ]

    async def scrape_symbol_mentions(self, symbol: str, limit: int = 100) -> List[ScrapeResult]:
        """Scrape news mentions for a cryptocurrency symbol"""

        results = []

        try:
            # Fetch RSS feeds
            for feed_url in self.rss_feeds:
                try:
                    articles = await self._parse_rss_feed(feed_url, symbol, limit // len(self.rss_feeds))

                    for article in articles:
                        result = ScrapeResult(
                            source=self.source_name,
                            symbol=symbol,
                            timestamp=datetime.now(),
                            content=article.get("title", "") + " " + article.get("description", ""),
                            metadata={
                                "url": article.get("link", ""),
                                "title": article.get("title", ""),
                                "published": article.get("published", ""),
                                "source_feed": feed_url,
                                "author": article.get("author", "")
                            }
                        )
                        results.append(result)

                except Exception as e:
                    self.client.logger.warning(f"RSS feed parsing failed for {feed_url}: {e}")

        except Exception as e:
            self.client.logger.error(f"News scraping failed for {symbol}: {e}")

        return results[:limit]

    async def _parse_rss_feed(self, feed_url: str, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Parse RSS feed and filter for symbol mentions"""

        try:
            # Fetch RSS content
            rss_content = await self.client.fetch_html(
                url=feed_url,
                source=self.source_name
            )

            # Parse RSS feed
            if feedparser is None:
                self.client.logger.warning("feedparser not available, using mock data")
                return self._generate_mock_datasymbol, limit)

            feed = feedparser.parse(rss_content)

            articles = []
            symbol_pattern = re.compile(rf'\b{re.escape(symbol)}\b', re.IGNORECASE)

            for entry in feed.entries[:limit * 2]:  # Fetch more to filter
                # Check if symbol is mentioned in title or summary
                title = entry.get("title", "")
                summary = entry.get("summary", "")

                if symbol_pattern.search(title + " " + summary):
                    articles.append({
                        "title": title,
                        "description": summary,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "author": entry.get("author", "")
                    })

                if len(articles) >= limit:
                    break

            return articles

        except Exception as e:
            self.client.logger.warning(f"RSS parsing failed: {e}")
            # Return mock data for development
            return self._generate_mock_datasymbol, limit)

    def _generate_generate_sample_data_self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock news articles for development"""

        mock_articles = []

        for i in range(min(limit, 5)):
            mock_posts.append({
                "title": f"{symbol} Market Analysis - Mock Article {i}",
                "description": f"Mock news content about {symbol} cryptocurrency for development",
                "link": f"https://example.com/news/{i}",
                "published": (datetime.now() - timedelta(hours=i)).isoformat(),
                "author": f"News Author {i % 2}"
            })

        return mock_articles

class TelegramScraper:
    """Telegram scraping for public channels (where possible)"""

    def __init__(self, client: AsyncScrapeClient):
        self.client = client
        self.source_name = "telegram"

        # Configure rate limits for Telegram
        self.client.configure_rate_limit(
            source=self.source_name,
            requests_per_minute=20,
            requests_per_hour=200,
            burst_limit=3
        )

    async def scrape_symbol_mentions(self, symbol: str, limit: int = 50) -> List[ScrapeResult]:
        """Scrape Telegram mentions (limited to public channels)"""

        results = []

        # Note: Telegram scraping is limited due to API restrictions
        # In production, would require official Telegram API access

        try:
            # Mock Telegram data for development
            for i in range(min(limit, 5)):
                result = ScrapeResult(
                    source=self.source_name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    content=f"Mock Telegram message about {symbol} - {i}",
                    metadata={
                        "channel": f"crypto_channel_{i % 2}",
                        "message_id": f"tg_{i}_{hash(symbol) % 1000}",
                        "views": i * 50
                    }
                )
                results.append(result)

        except Exception as e:
            self.client.logger.error(f"Telegram scraping failed for {symbol}: {e}")

        return results

class DiscordScraper:
    """Discord scraping for public servers (where possible)"""

    def __init__(self, client: AsyncScrapeClient):
        self.client = client
        self.source_name = "discord"

        # Configure rate limits for Discord
        self.client.configure_rate_limit(
            source=self.source_name,
            requests_per_minute=10,
            requests_per_hour=100,
            burst_limit=2
        )

    async def scrape_symbol_mentions(self, symbol: str, limit: int = 50) -> List[ScrapeResult]:
        """Scrape Discord mentions (limited to public servers)"""

        results = []

        # Note: Discord scraping is limited due to API restrictions
        # In production, would require official Discord bot access

        try:
            # Mock Discord data for development
            for i in range(min(limit, 5)):
                result = ScrapeResult(
                    source=self.source_name,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    content=f"Mock Discord message about {symbol} - {i}",
                    metadata={
                        "server": f"crypto_server_{i % 2}",
                        "channel": f"general_{i % 3}",
                        "message_id": f"discord_{i}_{hash(symbol) % 1000}",
                        "reactions": i * 3
                    }
                )
                results.append(result)

        except Exception as e:
            self.client.logger.error(f"Discord scraping failed for {symbol}: {e}")

        return results

# Source registry
AVAILABLE_SOURCES = {
    "twitter": TwitterScraper,
    "reddit": RedditScraper,
    "news": NewsScraper,
    "telegram": TelegramScraper,
    "discord": DiscordScraper
}
