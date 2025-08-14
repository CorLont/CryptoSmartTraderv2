#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise Social Media Data Ingestion
TOS-compliant, ban-protected social media data collection with enterprise guardrails
"""

import asyncio
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Set
import threading
from collections import defaultdict, deque
import aiohttp
import logging

from core.structured_logger import get_structured_logger


class PlatformType(Enum):
    """Supported social media platforms"""
    REDDIT = "reddit"
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    DISCORD = "discord"


class ContentType(Enum):
    """Types of social media content"""
    POST = "post"
    COMMENT = "comment"
    REPLY = "reply"
    THREAD = "thread"
    MESSAGE = "message"


class DataQualityLevel(Enum):
    """Quality levels for scraped data"""
    HIGH = "high"           # Official API, full metadata
    MEDIUM = "medium"       # Semi-official, limited metadata
    LOW = "low"            # Public scraping, minimal metadata
    BLOCKED = "blocked"     # TOS violation or banned


@dataclass
class SocialMediaPost:
    """Standardized social media post structure"""
    platform: PlatformType
    post_id: str
    author_id: str
    author_username: str
    content: str
    timestamp: datetime
    content_type: ContentType
    engagement_metrics: Dict[str, int]  # likes, shares, comments, etc.
    metadata: Dict[str, Any]
    crypto_keywords: List[str]
    sentiment_indicators: List[str]
    quality_level: DataQualityLevel
    source_url: Optional[str] = None
    parent_post_id: Optional[str] = None
    collection_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "platform": self.platform.value,
            "post_id": self.post_id,
            "author_id": self.author_id,
            "author_username": self.author_username,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "content_type": self.content_type.value,
            "engagement_metrics": self.engagement_metrics,
            "metadata": self.metadata,
            "crypto_keywords": self.crypto_keywords,
            "sentiment_indicators": self.sentiment_indicators,
            "quality_level": self.quality_level.value,
            "source_url": self.source_url,
            "parent_post_id": self.parent_post_id,
            "collection_timestamp": self.collection_timestamp.isoformat()
        }


@dataclass
class TOSCompliance:
    """Terms of Service compliance configuration"""
    platform: PlatformType
    api_required: bool = True
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 1000
    requires_auth: bool = True
    allowed_endpoints: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    user_agent_required: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def is_compliant_action(self, action: str) -> bool:
        """Check if action is TOS compliant"""
        return action not in self.forbidden_actions


@dataclass
class RateLimitState:
    """Rate limiting state tracking"""
    requests_this_hour: int = 0
    requests_this_day: int = 0
    last_request_time: Optional[datetime] = None
    hour_reset_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))
    day_reset_time: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=1))
    backoff_until: Optional[datetime] = None
    consecutive_failures: int = 0


class BanProtection:
    """Advanced ban protection and detection"""
    
    def __init__(self, platform: PlatformType):
        self.platform = platform
        self.logger = get_structured_logger(f"BanProtection_{platform.value}")
        
        # Ban detection patterns
        self.ban_indicators = {
            "http_codes": [429, 403, 503, 520, 521, 522, 523, 524],
            "response_patterns": [
                "rate limit", "too many requests", "banned", "suspended",
                "unauthorized", "forbidden", "temporarily unavailable"
            ],
            "timing_patterns": {
                "response_time_spike": 10.0,  # seconds
                "consecutive_errors": 5,
                "error_rate_threshold": 0.5   # 50% error rate
            }
        }
        
        # Adaptive backoff strategy
        self.backoff_strategy = {
            "initial_delay": 60,      # 1 minute
            "max_delay": 3600,        # 1 hour
            "multiplier": 2.0,        # exponential backoff
            "jitter": 0.1             # 10% random jitter
        }
        
        self.detection_history = deque(maxlen=100)
        self.last_ban_check = datetime.now()
    
    def detect_potential_ban(self, 
                           response_code: int,
                           response_text: str,
                           response_time: float) -> bool:
        """Detect potential ban conditions"""
        
        ban_detected = False
        reasons = []
        
        # HTTP code analysis
        if response_code in self.ban_indicators["http_codes"]:
            ban_detected = True
            reasons.append(f"Suspicious HTTP code: {response_code}")
        
        # Response text analysis
        response_lower = response_text.lower()
        for pattern in self.ban_indicators["response_patterns"]:
            if pattern in response_lower:
                ban_detected = True
                reasons.append(f"Ban pattern detected: {pattern}")
        
        # Timing analysis
        if response_time > self.ban_indicators["timing_patterns"]["response_time_spike"]:
            reasons.append(f"Response time spike: {response_time:.2f}s")
        
        # Record detection event
        detection_event = {
            "timestamp": datetime.now(),
            "response_code": response_code,
            "response_time": response_time,
            "ban_detected": ban_detected,
            "reasons": reasons
        }
        
        self.detection_history.append(detection_event)
        
        if ban_detected:
            self.logger.warning(f"Potential ban detected for {self.platform.value}: {reasons}")
        
        return ban_detected
    
    def calculate_backoff_delay(self, failure_count: int) -> int:
        """Calculate adaptive backoff delay"""
        import random
        
        delay = min(
            self.backoff_strategy["initial_delay"] * 
            (self.backoff_strategy["multiplier"] ** failure_count),
            self.backoff_strategy["max_delay"]
        )
        
        # Add jitter
        jitter = delay * self.backoff_strategy["jitter"] * random.random()
        return int(delay + jitter)
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get ban detection summary"""
        if not self.detection_history:
            return {"status": "no_data"}
        
        recent_events = [e for e in self.detection_history 
                        if (datetime.now() - e["timestamp"]).total_seconds() < 3600]
        
        ban_events = [e for e in recent_events if e["ban_detected"]]
        
        return {
            "total_events": len(self.detection_history),
            "recent_events_1h": len(recent_events),
            "ban_events_1h": len(ban_events),
            "ban_rate_1h": len(ban_events) / max(1, len(recent_events)),
            "last_ban_time": ban_events[-1]["timestamp"].isoformat() if ban_events else None,
            "current_risk_level": self._assess_risk_level(recent_events, ban_events)
        }
    
    def _assess_risk_level(self, recent_events: List, ban_events: List) -> str:
        """Assess current ban risk level"""
        if not recent_events:
            return "unknown"
        
        ban_rate = len(ban_events) / len(recent_events)
        
        if ban_rate > 0.5:
            return "critical"
        elif ban_rate > 0.3:
            return "high"
        elif ban_rate > 0.1:
            return "medium"
        else:
            return "low"


class TOSComplianceManager:
    """TOS compliance enforcement and monitoring"""
    
    def __init__(self):
        self.logger = get_structured_logger("TOSComplianceManager")
        
        # Platform-specific TOS configurations
        self.compliance_configs = {
            PlatformType.REDDIT: TOSCompliance(
                platform=PlatformType.REDDIT,
                api_required=True,
                rate_limit_per_hour=100,
                rate_limit_per_day=1000,
                requires_auth=True,
                allowed_endpoints=[
                    "/r/{subreddit}/hot",
                    "/r/{subreddit}/new", 
                    "/r/{subreddit}/top",
                    "/api/info"
                ],
                forbidden_actions=[
                    "mass_download",
                    "automated_posting",
                    "vote_manipulation",
                    "user_stalking"
                ]
            ),
            
            PlatformType.TWITTER: TOSCompliance(
                platform=PlatformType.TWITTER,
                api_required=True,
                rate_limit_per_hour=300,
                rate_limit_per_day=1500,
                requires_auth=True,
                allowed_endpoints=[
                    "/2/tweets/search/recent",
                    "/2/users/by/username",
                    "/2/tweets"
                ],
                forbidden_actions=[
                    "automated_following",
                    "spam_detection_evasion",
                    "fake_engagement",
                    "data_resale"
                ]
            )
        }
        
        self.rate_limits = {platform: RateLimitState() 
                          for platform in PlatformType}
        self.lock = threading.Lock()
    
    def check_rate_limit(self, platform: PlatformType) -> bool:
        """Check if request is within rate limits"""
        with self.lock:
            config = self.compliance_configs.get(platform)
            if not config:
                self.logger.warning(f"No TOS config for platform: {platform}")
                return False
            
            state = self.rate_limits[platform]
            now = datetime.now()
            
            # Reset counters if needed
            if now >= state.hour_reset_time:
                state.requests_this_hour = 0
                state.hour_reset_time = now + timedelta(hours=1)
            
            if now >= state.day_reset_time:
                state.requests_this_day = 0
                state.day_reset_time = now + timedelta(days=1)
            
            # Check backoff period
            if state.backoff_until and now < state.backoff_until:
                return False
            
            # Check rate limits
            if state.requests_this_hour >= config.rate_limit_per_hour:
                self.logger.warning(f"Hourly rate limit exceeded for {platform.value}")
                return False
            
            if state.requests_this_day >= config.rate_limit_per_day:
                self.logger.warning(f"Daily rate limit exceeded for {platform.value}")
                return False
            
            return True
    
    def record_request(self, platform: PlatformType, success: bool = True):
        """Record API request for rate limiting"""
        with self.lock:
            state = self.rate_limits[platform]
            state.requests_this_hour += 1
            state.requests_this_day += 1
            state.last_request_time = datetime.now()
            
            if not success:
                state.consecutive_failures += 1
            else:
                state.consecutive_failures = 0
    
    def apply_backoff(self, platform: PlatformType, failure_count: int):
        """Apply backoff period for platform"""
        with self.lock:
            ban_protection = BanProtection(platform)
            backoff_seconds = ban_protection.calculate_backoff_delay(failure_count)
            
            state = self.rate_limits[platform]
            state.backoff_until = datetime.now() + timedelta(seconds=backoff_seconds)
            
            self.logger.info(f"Applied {backoff_seconds}s backoff for {platform.value}")
    
    def validate_action(self, platform: PlatformType, action: str) -> bool:
        """Validate if action is TOS compliant"""
        config = self.compliance_configs.get(platform)
        if not config:
            return False
        
        return config.is_compliant_action(action)
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get comprehensive compliance status"""
        with self.lock:
            status = {
                "timestamp": datetime.now().isoformat(),
                "platforms": {}
            }
            
            for platform, state in self.rate_limits.items():
                config = self.compliance_configs.get(platform)
                
                status["platforms"][platform.value] = {
                    "rate_limit_status": {
                        "hourly": f"{state.requests_this_hour}/{config.rate_limit_per_hour if config else 'unknown'}",
                        "daily": f"{state.requests_this_day}/{config.rate_limit_per_day if config else 'unknown'}",
                        "backoff_until": state.backoff_until.isoformat() if state.backoff_until else None,
                        "consecutive_failures": state.consecutive_failures
                    },
                    "tos_compliance": {
                        "api_required": config.api_required if config else True,
                        "auth_required": config.requires_auth if config else True,
                        "allowed_endpoints": len(config.allowed_endpoints) if config else 0,
                        "forbidden_actions": len(config.forbidden_actions) if config else 0
                    }
                }
            
            return status


class BaseSocialMediaCollector(ABC):
    """Base class for social media data collectors"""
    
    def __init__(self, platform: PlatformType, compliance_manager: TOSComplianceManager):
        self.platform = platform
        self.compliance_manager = compliance_manager
        self.ban_protection = BanProtection(platform)
        self.logger = get_structured_logger(f"SocialCollector_{platform.value}")
        
        # Crypto-specific keywords for filtering
        self.crypto_keywords = {
            "coins": ["bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency"],
            "topics": ["defi", "nft", "web3", "blockchain", "altcoin", "trading"],
            "sentiment": ["moon", "dump", "hodl", "fud", "fomo", "diamond hands"]
        }
        
        self.collection_metrics = defaultdict(int)
    
    @abstractmethod
    async def collect_posts(self, 
                          keywords: List[str], 
                          limit: int = 100,
                          time_range: timedelta = timedelta(hours=24)) -> List[SocialMediaPost]:
        """Collect posts from platform"""
        pass
    
    def _extract_crypto_keywords(self, text: str) -> List[str]:
        """Extract crypto keywords from text"""
        text_lower = text.lower()
        found_keywords = []
        
        for category, keywords in self.crypto_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        return list(set(found_keywords))
    
    def _extract_sentiment_indicators(self, text: str) -> List[str]:
        """Extract sentiment indicators from text"""
        sentiment_patterns = {
            "positive": ["bullish", "moon", "pump", "rally", "green", "up"],
            "negative": ["bearish", "dump", "crash", "red", "down", "rekt"],
            "neutral": ["hold", "wait", "watch", "analysis"]
        }
        
        text_lower = text.lower()
        indicators = []
        
        for sentiment, patterns in sentiment_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    indicators.append(f"{sentiment}:{pattern}")
        
        return indicators
    
    async def _make_request(self, url: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Make TOS-compliant HTTP request"""
        
        # Check rate limits
        if not self.compliance_manager.check_rate_limit(self.platform):
            self.logger.warning(f"Rate limit exceeded for {self.platform.value}")
            return None
        
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, **kwargs) as response:
                    response_time = time.time() - start_time
                    response_text = await response.text()
                    
                    # Record request
                    success = response.status == 200
                    self.compliance_manager.record_request(self.platform, success)
                    
                    # Check for ban indicators
                    ban_detected = self.ban_protection.detect_potential_ban(
                        response.status, response_text, response_time
                    )
                    
                    if ban_detected:
                        # Apply backoff
                        state = self.compliance_manager.rate_limits[self.platform]
                        self.compliance_manager.apply_backoff(
                            self.platform, state.consecutive_failures
                        )
                        return None
                    
                    if success:
                        return await response.json()
                    else:
                        self.logger.error(f"Request failed: {response.status} - {response_text}")
                        return None
        
        except Exception as e:
            self.logger.error(f"Request error: {e}")
            self.compliance_manager.record_request(self.platform, False)
            return None


class RedditCollector(BaseSocialMediaCollector):
    """TOS-compliant Reddit data collector"""
    
    def __init__(self, compliance_manager: TOSComplianceManager, 
                 client_id: str = None, client_secret: str = None):
        super().__init__(PlatformType.REDDIT, compliance_manager)
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        
        # Reddit-specific subreddits for crypto content
        self.crypto_subreddits = [
            "cryptocurrency", "bitcoin", "ethereum", "cryptomarkets",
            "defi", "altcoin", "cryptotechnology", "bitcoinmarkets"
        ]
    
    async def authenticate(self) -> bool:
        """Authenticate with Reddit API"""
        if not self.client_id or not self.client_secret:
            self.logger.error("Reddit API credentials not provided")
            return False
        
        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth_data = {
            "grant_type": "client_credentials"
        }
        
        import aiohttp
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        headers = {"User-Agent": "CryptoSmartTrader/2.0"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(auth_url, data=auth_data, 
                                      auth=auth, headers=headers) as response:
                    if response.status == 200:
                        token_data = await response.json()
                        self.access_token = token_data.get("access_token")
                        self.logger.info("Reddit authentication successful")
                        return True
                    else:
                        self.logger.error(f"Reddit auth failed: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Reddit authentication error: {e}")
            return False
    
    async def collect_posts(self, 
                          keywords: List[str], 
                          limit: int = 100,
                          time_range: timedelta = timedelta(hours=24)) -> List[SocialMediaPost]:
        """Collect posts from Reddit subreddits"""
        
        if not await self.authenticate():
            return []
        
        posts = []
        
        for subreddit in self.crypto_subreddits:
            try:
                subreddit_posts = await self._collect_subreddit_posts(
                    subreddit, keywords, limit // len(self.crypto_subreddits)
                )
                posts.extend(subreddit_posts)
                
                # Respect rate limits between subreddits
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error collecting from r/{subreddit}: {e}")
                continue
        
        self.logger.info(f"Collected {len(posts)} posts from Reddit")
        return posts
    
    async def _collect_subreddit_posts(self, 
                                     subreddit: str, 
                                     keywords: List[str], 
                                     limit: int) -> List[SocialMediaPost]:
        """Collect posts from specific subreddit"""
        
        url = f"https://oauth.reddit.com/r/{subreddit}/hot"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": "CryptoSmartTrader/2.0"
        }
        
        params = {"limit": min(limit, 100)}  # Reddit API limit
        
        response_data = await self._make_request(url, headers=headers, params=params)
        if not response_data:
            return []
        
        posts = []
        
        try:
            for item in response_data.get("data", {}).get("children", []):
                post_data = item.get("data", {})
                
                # Filter by keywords
                title = post_data.get("title", "").lower()
                content = post_data.get("selftext", "").lower()
                
                if not any(keyword.lower() in title or keyword.lower() in content 
                          for keyword in keywords):
                    continue
                
                # Create standardized post
                post = SocialMediaPost(
                    platform=PlatformType.REDDIT,
                    post_id=post_data.get("id"),
                    author_id=post_data.get("author_fullname"),
                    author_username=post_data.get("author"),
                    content=f"{post_data.get('title', '')} {post_data.get('selftext', '')}",
                    timestamp=datetime.fromtimestamp(post_data.get("created_utc", 0)),
                    content_type=ContentType.POST,
                    engagement_metrics={
                        "upvotes": post_data.get("ups", 0),
                        "downvotes": post_data.get("downs", 0),
                        "comments": post_data.get("num_comments", 0),
                        "score": post_data.get("score", 0)
                    },
                    metadata={
                        "subreddit": subreddit,
                        "url": post_data.get("url"),
                        "permalink": post_data.get("permalink"),
                        "flair": post_data.get("link_flair_text")
                    },
                    crypto_keywords=self._extract_crypto_keywords(
                        f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                    ),
                    sentiment_indicators=self._extract_sentiment_indicators(
                        f"{post_data.get('title', '')} {post_data.get('selftext', '')}"
                    ),
                    quality_level=DataQualityLevel.HIGH,
                    source_url=f"https://reddit.com{post_data.get('permalink', '')}"
                )
                
                posts.append(post)
                
        except Exception as e:
            self.logger.error(f"Error parsing Reddit posts: {e}")
        
        return posts


class TwitterCollector(BaseSocialMediaCollector):
    """TOS-compliant Twitter data collector"""
    
    def __init__(self, compliance_manager: TOSComplianceManager,
                 bearer_token: str = None):
        super().__init__(PlatformType.TWITTER, compliance_manager)
        self.bearer_token = bearer_token
    
    async def collect_posts(self, 
                          keywords: List[str], 
                          limit: int = 100,
                          time_range: timedelta = timedelta(hours=24)) -> List[SocialMediaPost]:
        """Collect tweets using Twitter API v2"""
        
        if not self.bearer_token:
            self.logger.error("Twitter Bearer Token not provided")
            return []
        
        # Construct search query
        query = " OR ".join(keywords) + " lang:en -is:retweet"
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "User-Agent": "CryptoSmartTrader/2.0"
        }
        
        params = {
            "query": query,
            "max_results": min(limit, 100),  # Twitter API limit
            "tweet.fields": "created_at,author_id,public_metrics,context_annotations",
            "user.fields": "username,public_metrics",
            "expansions": "author_id"
        }
        
        response_data = await self._make_request(url, headers=headers, params=params)
        if not response_data:
            return []
        
        posts = []
        
        try:
            tweets = response_data.get("data", [])
            users = {user["id"]: user for user in response_data.get("includes", {}).get("users", [])}
            
            for tweet in tweets:
                author_id = tweet.get("author_id")
                author = users.get(author_id, {})
                
                post = SocialMediaPost(
                    platform=PlatformType.TWITTER,
                    post_id=tweet.get("id"),
                    author_id=author_id,
                    author_username=author.get("username", "unknown"),
                    content=tweet.get("text", ""),
                    timestamp=datetime.fromisoformat(
                        tweet.get("created_at", "").replace("Z", "+00:00")
                    ),
                    content_type=ContentType.POST,
                    engagement_metrics={
                        "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                        "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                        "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
                        "quotes": tweet.get("public_metrics", {}).get("quote_count", 0)
                    },
                    metadata={
                        "context_annotations": tweet.get("context_annotations", []),
                        "author_metrics": author.get("public_metrics", {}),
                        "lang": tweet.get("lang", "en")
                    },
                    crypto_keywords=self._extract_crypto_keywords(tweet.get("text", "")),
                    sentiment_indicators=self._extract_sentiment_indicators(tweet.get("text", "")),
                    quality_level=DataQualityLevel.HIGH,
                    source_url=f"https://twitter.com/user/status/{tweet.get('id')}"
                )
                
                posts.append(post)
                
        except Exception as e:
            self.logger.error(f"Error parsing Twitter data: {e}")
        
        self.logger.info(f"Collected {len(posts)} tweets")
        return posts


class EnterpriseSocialMediaManager:
    """Enterprise social media data ingestion coordinator"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_structured_logger("EnterpriseSocialMediaManager")
        
        # Initialize compliance manager
        self.compliance_manager = TOSComplianceManager()
        
        # Initialize collectors
        self.collectors = {}
        self._initialize_collectors()
        
        # Collection metrics
        self.metrics = defaultdict(lambda: defaultdict(int))
        self.last_collection = {}
        
        self.logger.info("Enterprise Social Media Manager initialized")
    
    def _initialize_collectors(self):
        """Initialize platform collectors"""
        
        # Reddit collector
        reddit_credentials = self.config.get("reddit", {})
        if reddit_credentials.get("client_id") and reddit_credentials.get("client_secret"):
            self.collectors[PlatformType.REDDIT] = RedditCollector(
                self.compliance_manager,
                reddit_credentials["client_id"],
                reddit_credentials["client_secret"]
            )
            self.logger.info("Reddit collector initialized")
        
        # Twitter collector
        twitter_credentials = self.config.get("twitter", {})
        if twitter_credentials.get("bearer_token"):
            self.collectors[PlatformType.TWITTER] = TwitterCollector(
                self.compliance_manager,
                twitter_credentials["bearer_token"]
            )
            self.logger.info("Twitter collector initialized")
        
        if not self.collectors:
            self.logger.warning("No social media collectors initialized - missing credentials")
    
    async def collect_social_data(self,
                                keywords: List[str],
                                platforms: Optional[List[PlatformType]] = None,
                                limit_per_platform: int = 100,
                                time_range: timedelta = timedelta(hours=24)) -> Dict[str, List[SocialMediaPost]]:
        """Collect data from multiple social media platforms"""
        
        if platforms is None:
            platforms = list(self.collectors.keys())
        
        results = {}
        
        for platform in platforms:
            if platform not in self.collectors:
                self.logger.warning(f"No collector available for {platform.value}")
                continue
            
            try:
                self.logger.info(f"Collecting from {platform.value}...")
                
                collector = self.collectors[platform]
                posts = await collector.collect_posts(
                    keywords, limit_per_platform, time_range
                )
                
                results[platform.value] = posts
                
                # Record metrics
                self.metrics[platform.value]["total_posts"] += len(posts)
                self.metrics[platform.value]["collections"] += 1
                self.last_collection[platform.value] = datetime.now()
                
                self.logger.info(f"Collected {len(posts)} posts from {platform.value}")
                
            except Exception as e:
                self.logger.error(f"Collection failed for {platform.value}: {e}")
                results[platform.value] = []
        
        return results
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get comprehensive compliance and metrics status"""
        
        compliance_status = self.compliance_manager.get_compliance_status()
        
        # Add collection metrics
        collection_metrics = {}
        for platform, metrics in self.metrics.items():
            collection_metrics[platform] = {
                "total_posts_collected": metrics["total_posts"],
                "total_collections": metrics["collections"],
                "last_collection": self.last_collection.get(platform, {}).isoformat() if self.last_collection.get(platform) else None,
                "avg_posts_per_collection": metrics["total_posts"] / max(1, metrics["collections"])
            }
        
        # Add ban protection status
        ban_protection_status = {}
        for platform, collector in self.collectors.items():
            ban_protection_status[platform.value] = collector.ban_protection.get_detection_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "compliance_status": compliance_status,
            "collection_metrics": collection_metrics,
            "ban_protection": ban_protection_status,
            "available_collectors": list(self.collectors.keys()),
            "total_platforms": len(self.collectors)
        }


# Global singleton
_social_media_manager = None

def get_social_media_manager(config: Dict[str, Any] = None) -> EnterpriseSocialMediaManager:
    """Get singleton social media manager"""
    global _social_media_manager
    if _social_media_manager is None:
        _social_media_manager = EnterpriseSocialMediaManager(config)
    return _social_media_manager


if __name__ == "__main__":
    # Basic validation
    import asyncio
    
    async def test_social_media():
        # Test with demo config
        config = {
            "reddit": {
                "client_id": "demo_client_id",
                "client_secret": "demo_secret"
            },
            "twitter": {
                "bearer_token": "demo_bearer_token"
            }
        }
        
        manager = get_social_media_manager(config)
        
        # Test compliance status
        status = manager.get_compliance_status()
        print(f"Compliance Status: {status}")
        
        # Note: Actual collection would require real API credentials
        print("Social Media Manager: Initialized and ready")
    
    asyncio.run(test_social_media())