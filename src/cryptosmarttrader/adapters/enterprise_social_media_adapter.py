#!/usr/bin/env python3
"""
Enterprise Social Media Adapter with Centralized Throttling & Error Handling
TOS-compliant social media data collection met unified infrastructure
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from ..infrastructure.centralized_throttling import ServiceType, throttled
from ..infrastructure.unified_error_handler import unified_error_handling

logger = logging.getLogger(__name__)


@dataclass
class SocialMediaPost:
    """Standardized social media post structure"""
    platform: str
    post_id: str
    author_id: str
    author_username: str
    content: str
    timestamp: datetime
    engagement_metrics: Dict[str, int]
    crypto_keywords: List[str]
    sentiment_indicators: List[str]
    metadata: Dict[str, Any]
    collection_timestamp: datetime


class BaseSocialMediaAdapter(ABC):
    """Base class voor social media adapters"""
    
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.cache_dir = Path(f"cache/social_media/{platform_name}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # TOS compliance tracking
        self.rate_limit_history = []
        self.last_request_time = 0.0
        self.banned_until = None
        
        logger.info(f"🐦 {platform_name} adapter initialized with enterprise features")
    
    @abstractmethod
    async def fetch_posts_async(
        self,
        query: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SocialMediaPost]:
        """Fetch posts from platform"""
        pass
    
    def _is_banned(self) -> bool:
        """Check if currently banned"""
        if self.banned_until and datetime.now() < self.banned_until:
            return True
        return False
    
    def _detect_ban_signals(self, response_text: str, status_code: int) -> bool:
        """Detect ban/block signals"""
        ban_indicators = [
            "rate limit",
            "too many requests", 
            "suspended",
            "blocked",
            "forbidden",
            "captcha",
            "bot detection",
            "access denied"
        ]
        
        if status_code in [403, 429, 503]:
            return True
            
        return any(indicator in response_text.lower() for indicator in ban_indicators)
    
    def _extract_crypto_keywords(self, text: str) -> List[str]:
        """Extract cryptocurrency keywords from text"""
        crypto_keywords = [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "blockchain", "defi", "nft", "altcoin", "hodl", "pump", "dump",
            "bull", "bear", "moon", "dip", "whale", "satoshi", "gwei",
            "staking", "yield", "farming", "liquidity", "swap", "dex"
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in crypto_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_sentiment_indicators(self, text: str) -> List[str]:
        """Extract sentiment indicators from text"""
        positive_indicators = ["bullish", "moon", "pump", "buy", "hodl", "diamond hands"]
        negative_indicators = ["bearish", "dump", "sell", "crash", "rekt", "paper hands"]
        
        text_lower = text.lower()
        indicators = []
        
        for indicator in positive_indicators:
            if indicator in text_lower:
                indicators.append(f"positive:{indicator}")
        
        for indicator in negative_indicators:
            if indicator in text_lower:
                indicators.append(f"negative:{indicator}")
        
        return indicators


class EnterpriseRedditAdapter(BaseSocialMediaAdapter):
    """Enterprise Reddit adapter met throttling en error handling"""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        super().__init__("reddit")
        
        import os
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit credentials not found, using public API only")
    
    @throttled(ServiceType.SOCIAL_MEDIA, endpoint="reddit_posts")
    @unified_error_handling("social_media", endpoint="reddit_posts") 
    async def fetch_posts_async(
        self,
        subreddit: str = "cryptocurrency",
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SocialMediaPost]:
        """
        Fetch posts from Reddit subreddit
        
        Args:
            subreddit: Subreddit name
            limit: Number of posts to fetch
            since: Only posts after this date
            
        Returns:
            List of standardized social media posts
        """
        if self._is_banned():
            logger.warning(f"Reddit adapter is banned until {self.banned_until}")
            return []
        
        try:
            import praw
            
            # Initialize Reddit client
            reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent="CryptoSmartTrader/2.0"
            )
            
            posts = []
            subreddit_obj = reddit.subreddit(subreddit)
            
            # Fetch hot posts
            for submission in subreddit_obj.hot(limit=limit):
                # Skip if too old
                post_time = datetime.fromtimestamp(submission.created_utc)
                if since and post_time < since:
                    continue
                
                # Extract crypto keywords and sentiment
                full_text = f"{submission.title} {submission.selftext}"
                crypto_keywords = self._extract_crypto_keywords(full_text)
                sentiment_indicators = self._extract_sentiment_indicators(full_text)
                
                # Only include crypto-relevant posts
                if not crypto_keywords:
                    continue
                
                post = SocialMediaPost(
                    platform="reddit",
                    post_id=submission.id,
                    author_id=str(submission.author) if submission.author else "deleted",
                    author_username=str(submission.author) if submission.author else "deleted",
                    content=full_text,
                    timestamp=post_time,
                    engagement_metrics={
                        "upvotes": submission.ups,
                        "downvotes": submission.downs,
                        "score": submission.score,
                        "comments": submission.num_comments,
                        "awards": submission.total_awards_received
                    },
                    crypto_keywords=crypto_keywords,
                    sentiment_indicators=sentiment_indicators,
                    metadata={
                        "subreddit": subreddit,
                        "url": submission.url,
                        "permalink": submission.permalink,
                        "is_self": submission.is_self,
                        "flair": submission.link_flair_text
                    },
                    collection_timestamp=datetime.now()
                )
                
                posts.append(post)
            
            logger.info(f"Fetched {len(posts)} crypto-relevant posts from r/{subreddit}")
            return posts
            
        except Exception as e:
            # Check for ban signals
            if self._detect_ban_signals(str(e), 0):
                self.banned_until = datetime.now() + timedelta(hours=2)
                logger.error(f"Reddit ban detected, suspended until {self.banned_until}")
            
            logger.error(f"Failed to fetch Reddit posts: {e}")
            return []


class EnterpriseTwitterAdapter(BaseSocialMediaAdapter):
    """Enterprise Twitter adapter met throttling en error handling"""
    
    def __init__(self, bearer_token: Optional[str] = None):
        super().__init__("twitter")
        
        import os
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        if not self.bearer_token:
            logger.warning("Twitter bearer token not found")
    
    @throttled(ServiceType.SOCIAL_MEDIA, endpoint="twitter_search")
    @unified_error_handling("social_media", endpoint="twitter_search")
    async def fetch_posts_async(
        self,
        query: str = "crypto OR bitcoin OR ethereum",
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[SocialMediaPost]:
        """
        Fetch tweets using Twitter API v2
        
        Args:
            query: Search query
            limit: Number of tweets to fetch
            since: Only tweets after this date
            
        Returns:
            List of standardized social media posts
        """
        if self._is_banned():
            logger.warning(f"Twitter adapter is banned until {self.banned_until}")
            return []
        
        if not self.bearer_token:
            logger.error("Twitter bearer token required")
            return []
        
        try:
            import aiohttp
            
            # Prepare API request
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "CryptoSmartTrader/2.0"
            }
            
            params = {
                "query": f"{query} -is:retweet lang:en",
                "max_results": min(limit, 100),  # API limit
                "tweet.fields": "created_at,author_id,public_metrics,context_annotations",
                "user.fields": "username,verified",
                "expansions": "author_id"
            }
            
            if since:
                params["start_time"] = since.isoformat()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        
                        # Check for ban signals
                        if self._detect_ban_signals(error_text, response.status):
                            self.banned_until = datetime.now() + timedelta(hours=4)
                            logger.error(f"Twitter ban detected, suspended until {self.banned_until}")
                        
                        raise Exception(f"Twitter API error {response.status}: {error_text}")
                    
                    data = await response.json()
            
            posts = []
            tweets = data.get("data", [])
            users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
            
            for tweet in tweets:
                # Extract crypto keywords and sentiment
                crypto_keywords = self._extract_crypto_keywords(tweet["text"])
                sentiment_indicators = self._extract_sentiment_indicators(tweet["text"])
                
                # Only include crypto-relevant tweets
                if not crypto_keywords:
                    continue
                
                author = users.get(tweet["author_id"], {})
                metrics = tweet.get("public_metrics", {})
                
                post = SocialMediaPost(
                    platform="twitter",
                    post_id=tweet["id"],
                    author_id=tweet["author_id"],
                    author_username=author.get("username", "unknown"),
                    content=tweet["text"],
                    timestamp=datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00")),
                    engagement_metrics={
                        "retweets": metrics.get("retweet_count", 0),
                        "likes": metrics.get("like_count", 0),
                        "replies": metrics.get("reply_count", 0),
                        "quotes": metrics.get("quote_count", 0)
                    },
                    crypto_keywords=crypto_keywords,
                    sentiment_indicators=sentiment_indicators,
                    metadata={
                        "verified": author.get("verified", False),
                        "context_annotations": tweet.get("context_annotations", []),
                        "source": "twitter_api_v2"
                    },
                    collection_timestamp=datetime.now()
                )
                
                posts.append(post)
            
            logger.info(f"Fetched {len(posts)} crypto-relevant tweets")
            return posts
            
        except Exception as e:
            logger.error(f"Failed to fetch Twitter posts: {e}")
            return []


class EnterpriseSocialMediaManager:
    """
    Centralized manager voor all social media adapters
    KRITIEK: Single point of control voor social media data collection
    """
    
    def __init__(self):
        """Initialize social media manager"""
        self.adapters = {}
        self.collection_stats = {
            "total_posts": 0,
            "posts_by_platform": {},
            "last_collection": None,
            "errors": []
        }
        
        # Initialize adapters
        self._initialize_adapters()
        
        logger.info("🌐 EnterpriseSocialMediaManager initialized")
    
    def _initialize_adapters(self) -> None:
        """Initialize available social media adapters"""
        try:
            self.adapters["reddit"] = EnterpriseRedditAdapter()
        except Exception as e:
            logger.warning(f"Failed to initialize Reddit adapter: {e}")
        
        try:
            self.adapters["twitter"] = EnterpriseTwitterAdapter()
        except Exception as e:
            logger.warning(f"Failed to initialize Twitter adapter: {e}")
        
        logger.info(f"Initialized {len(self.adapters)} social media adapters")
    
    async def collect_all_posts(
        self,
        platforms: Optional[List[str]] = None,
        limit_per_platform: int = 50,
        since: Optional[datetime] = None
    ) -> List[SocialMediaPost]:
        """
        Collect posts from all available platforms
        
        Args:
            platforms: Specific platforms to collect from
            limit_per_platform: Posts per platform
            since: Only posts after this date
            
        Returns:
            Combined list of posts from all platforms
        """
        if platforms is None:
            platforms = list(self.adapters.keys())
        
        all_posts = []
        platform_results = {}
        
        # Collect from each platform concurrently
        tasks = []
        for platform in platforms:
            if platform in self.adapters:
                adapter = self.adapters[platform]
                
                if platform == "reddit":
                    task = adapter.fetch_posts_async(
                        subreddit="cryptocurrency",
                        limit=limit_per_platform,
                        since=since
                    )
                elif platform == "twitter":
                    task = adapter.fetch_posts_async(
                        query="crypto OR bitcoin OR ethereum",
                        limit=limit_per_platform,
                        since=since
                    )
                else:
                    continue
                
                tasks.append((platform, task))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            )
            
            for i, (platform, _) in enumerate(tasks):
                result = results[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Failed to collect from {platform}: {result}")
                    self.collection_stats["errors"].append({
                        "platform": platform,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    platform_results[platform] = []
                else:
                    platform_results[platform] = result
                    all_posts.extend(result)
        
        # Update statistics
        self.collection_stats["total_posts"] += len(all_posts)
        self.collection_stats["last_collection"] = datetime.now().isoformat()
        
        for platform, posts in platform_results.items():
            if platform not in self.collection_stats["posts_by_platform"]:
                self.collection_stats["posts_by_platform"][platform] = 0
            self.collection_stats["posts_by_platform"][platform] += len(posts)
        
        # Sort by timestamp descending
        all_posts.sort(key=lambda x: x.timestamp, reverse=True)
        
        logger.info(
            f"Collected {len(all_posts)} total posts from {len(platform_results)} platforms"
        )
        
        return all_posts
    
    def collect_posts_sync(self, *args, **kwargs) -> List[SocialMediaPost]:
        """Sync wrapper voor post collection"""
        return asyncio.run(self.collect_all_posts(*args, **kwargs))
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.collection_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset collection statistics"""
        self.collection_stats = {
            "total_posts": 0,
            "posts_by_platform": {},
            "last_collection": None,
            "errors": []
        }
        logger.info("Collection statistics reset")


# Export public interface
__all__ = [
    'EnterpriseSocialMediaManager',
    'EnterpriseRedditAdapter', 
    'EnterpriseTwitterAdapter',
    'SocialMediaPost'
]