#!/usr/bin/env python3
"""
Robust OpenAI Adapter - Enterprise-grade LLM integration
Complete implementation with rate limiting, caching, fallbacks, and validation
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json  # SECURITY: Replaced pickle with json
from pathlib import Path

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

try:
    import backoff
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

    # Define dummy decorators for when tenacity is not available
    def retry(**kwargs):
        def decorator(func):
            return func

        return decorator

    def stop_after_attempt(attempts):
        return None

    def wait_exponential(**kwargs):
        return None

    def retry_if_exception_type(exception_types):
        return None


from ..core.structured_logger import get_logger


class LLMTaskType(Enum):
    """Types of LLM tasks with different requirements"""

    NEWS_ANALYSIS = "news_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MARKET_ANALYSIS = "market_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class LLMConfig:
    """Configuration for LLM requests"""

    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 1000
    timeout: int = 30
    max_retries: int = 3
    cache_ttl_hours: int = 24
    cost_limit_per_hour: float = 10.0  # USD


class NewsImpactSchema(BaseModel):
    """Validated schema for news impact analysis"""

    sentiment: str  # "bullish", "bearish", "neutral"
    impact_magnitude: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    half_life_hours: float  # Expected impact duration
    affected_symbols: List[str]  # Specific coins affected
    key_factors: List[str]  # Main impact drivers
    impact_timeline: str  # "immediate", "short_term", "medium_term", "long_term"
    reasoning: str  # Explanation of analysis


class SentimentSchema(BaseModel):
    """Validated schema for sentiment analysis"""

    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    emotions: List[str]  # ["fear", "greed", "optimism", etc.]
    key_phrases: List[str]  # Important phrases that drove sentiment
    market_relevance: float  # 0.0 to 1.0


class MarketAnalysisSchema(BaseModel):
    """Validated schema for market analysis"""

    market_regime: str  # "bull", "bear", "sideways", "volatile"
    trend_strength: float  # 0.0 to 1.0
    volatility_assessment: str  # "low", "medium", "high"
    key_levels: List[float]  # Important support/resistance levels
    outlook: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    reasoning: str


class CircuitBreaker:
    """Circuit breaker for API failures"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""

        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN - API calls blocked")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        """Handle successful API call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed API call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_refill = time.time()

    def acquire(self) -> bool:
        """Try to acquire a token for API call"""
        current_time = time.time()

        # Refill tokens based on time passed
        time_passed = current_time - self.last_refill
        tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
        self.tokens = min(self.burst_limit, self.tokens + tokens_to_add)
        self.last_refill = current_time

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def wait_time(self) -> float:
        """Calculate wait time until next token is available"""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) * (60.0 / self.requests_per_minute)


class LLMCache:
    """Persistent cache for LLM responses"""

    def __init__(self, cache_dir: str = "cache/llm_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("LLMCache")

    def _hash_request(self, prompt: str, config: LLMConfig) -> str:
        """Create hash key for request"""
        request_data = f"{prompt}:{config.model}:{config.temperature}"
        return hashlib.sha256(request_data.encode()).hexdigest()

    def get(self, prompt: str, config: LLMConfig) -> Optional[Dict[str, Any]]:
        """Get cached response if available and fresh"""
        try:
            cache_key = self._hash_request(prompt, config)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            if not cache_file.exists():
                return None

            with open(cache_file, "rb") as f:
                cached_data = json.load(f)

            # Check if cache is still fresh
            cache_age = datetime.utcnow() - cached_data["timestamp"]
            if cache_age.total_seconds() > config.cache_ttl_hours * 3600:
                cache_file.unlink()  # Delete stale cache
                return None

            self.logger.info(f"Cache hit for request (age: {cache_age})")
            return cached_data["response"]

        except Exception as e:
            self.logger.error(f"Cache retrieval failed: {e}")
            return None

    def set(self, prompt: str, config: LLMConfig, response: Dict[str, Any]):
        """Cache response"""
        try:
            cache_key = self._hash_request(prompt, config)
            cache_file = self.cache_dir / f"{cache_key}.pkl"

            cache_data = {
                "timestamp": datetime.utcnow(),
                "response": response,
                "prompt_length": len(prompt),
                "model": config.model,
            }

            with open(cache_file, "wb") as f:
                json.dump(cache_data, f)

            self.logger.info(f"Cached response for {cache_key[:8]}...")

        except Exception as e:
            self.logger.error(f"Cache storage failed: {e}")


class CostTracker:
    """Track and limit API costs"""

    def __init__(self, hourly_limit: float = 10.0):
        self.hourly_limit = hourly_limit
        self.usage_file = Path("cache/openai_usage.json")
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)

    def estimate_cost(self, prompt: str, response: str, model: str = "gpt-4o") -> float:
        """Estimate cost of API call"""
        # Rough cost estimation (adjust based on actual pricing)
        input_tokens = len(prompt.split()) * 1.3  # Rough token estimation
        output_tokens = len(response.split()) * 1.3

        if model == "gpt-4o":
            input_cost = input_tokens * 0.00005  # $0.05 per 1K tokens
            output_cost = output_tokens * 0.00015  # $0.15 per 1K tokens
        else:
            input_cost = input_tokens * 0.00001  # Fallback pricing
            output_cost = output_tokens * 0.00003

        return input_cost + output_cost

    def add_usage(self, cost: float):
        """Add usage cost"""
        try:
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

            # Load existing usage
            usage_data = {}
            if self.usage_file.exists():
                with open(self.usage_file, "r") as f:
                    usage_data = json.load(f)

            hour_key = current_hour.isoformat()
            usage_data[hour_key] = usage_data.get(hour_key, 0.0) + cost

            # Clean old data (keep only last 24 hours)
            cutoff_time = current_hour - timedelta(hours=24)
            usage_data = {
                k: v for k, v in usage_data.items() if datetime.fromisoformat(k) >= cutoff_time
            }

            with open(self.usage_file, "w") as f:
                json.dump(usage_data, f, indent=2)

        except Exception as e:
            print(f"Cost tracking error: {e}")

    def check_limit(self) -> bool:
        """Check if we're within cost limits"""
        try:
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

            if not self.usage_file.exists():
                return True

            with open(self.usage_file, "r") as f:
                usage_data = json.load(f)

            hour_key = current_hour.isoformat()
            current_usage = usage_data.get(hour_key, 0.0)

            return current_usage < self.hourly_limit

        except Exception:
            return True  # Default to allowing if check fails


class PurePythonFallback:
    """Pure Python fallback for basic analysis when OpenAI fails"""

    def __init__(self):
        self.logger = get_logger("PythonFallback")

        # Simple sentiment keywords
        self.positive_words = {
            "bullish",
            "bull",
            "moon",
            "pump",
            "surge",
            "rally",
            "breakout",
            "breakthrough",
            "adoption",
            "partnership",
            "upgrade",
            "successful",
        }
        self.negative_words = {
            "bearish",
            "bear",
            "dump",
            "crash",
            "drop",
            "fall",
            "regulation",
            "ban",
            "hack",
            "exploit",
            "concern",
            "worry",
            "fear",
            "volatile",
        }

    def analyze_news_sentiment(self, text: str) -> NewsImpactSchema:
        """Fallback news analysis using keyword matching"""

        self.logger.warning("Using fallback sentiment analysis")

        text_lower = text.lower()
        words = set(text_lower.split())

        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))

        if positive_count > negative_count:
            sentiment = "bullish"
            impact_magnitude = min(0.7, positive_count * 0.1)
        elif negative_count > positive_count:
            sentiment = "bearish"
            impact_magnitude = min(0.7, negative_count * 0.1)
        else:
            sentiment = "neutral"
            impact_magnitude = 0.1

        return NewsImpactSchema(
            sentiment=sentiment,
            impact_magnitude=impact_magnitude,
            confidence=0.4,  # Low confidence for fallback
            half_life_hours=12.0,
            affected_symbols=["BTC", "ETH"],  # Default assumption
            key_factors=["keyword_analysis"],
            impact_timeline="short_term",
            reasoning="Fallback analysis based on keyword matching",
        )

    def analyze_basic_sentiment(self, text: str) -> SentimentSchema:
        """Basic sentiment analysis fallback"""

        text_lower = text.lower()
        words = set(text_lower.split())

        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words

        return SentimentSchema(
            sentiment_score=sentiment_score,
            confidence=0.3,
            emotions=["neutral"],
            key_phrases=[],
            market_relevance=0.5,
        )


class RobustOpenAIAdapter:
    """Enterprise-grade OpenAI adapter with all safety features"""

    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None):
        self.logger = get_logger("RobustOpenAIAdapter")

        # Initialize OpenAI client
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.logger.warning("No OpenAI API key found - only fallback will be available")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        # Configuration
        self.config = config or LLMConfig()

        # Safety components
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter()
        self.cache = LLMCache()
        self.cost_tracker = CostTracker(self.config.cost_limit_per_hour)
        self.fallback = PurePythonFallback()

        # Few-shot examples for consistent JSON output
        self.few_shot_examples = self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> Dict[LLMTaskType, List[Dict[str, str]]]:
        """Load few-shot examples for each task type"""

        return {
            LLMTaskType.NEWS_ANALYSIS: [
                {
                    "input": "Bitcoin ETF approved by SEC, expected to drive institutional adoption",
                    "output": json.dumps(
                        {
                            "sentiment": "bullish",
                            "impact_magnitude": 0.9,
                            "confidence": 0.95,
                            "half_life_hours": 72.0,
                            "affected_symbols": ["BTC", "ETH"],
                            "key_factors": ["institutional_adoption", "regulatory_approval"],
                            "impact_timeline": "medium_term",
                            "reasoning": "SEC approval removes major regulatory uncertainty and opens institutional access",
                        }
                    ),
                }
            ],
            LLMTaskType.SENTIMENT_ANALYSIS: [
                {
                    "input": "The market is showing strong bullish momentum with widespread green candles",
                    "output": json.dumps(
                        {
                            "sentiment_score": 0.8,
                            "confidence": 0.9,
                            "emotions": ["optimism", "greed"],
                            "key_phrases": ["strong bullish momentum", "widespread green"],
                            "market_relevance": 0.95,
                        }
                    ),
                }
            ],
        }

    @(
        retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
        )
        if TENACITY_AVAILABLE
        else lambda x: x
    )
    async def _make_openai_request(self, prompt: str, task_type: LLMTaskType) -> Dict[str, Any]:
        """Make OpenAI API request with retries and rate limiting"""

        if not self.client:
            raise Exception("OpenAI client not available")

        # Rate limiting
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            self.logger.info(f"Rate limited - waiting {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
            if not self.rate_limiter.acquire():
                raise Exception("Rate limit exceeded")

        # Cost checking
        if not self.cost_tracker.check_limit():
            raise Exception("Hourly cost limit exceeded")

        # Create system prompt with few-shot examples
        system_prompt = self._create_system_prompt(task_type)

        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )

        response_text = response.choices[0].message.content

        # Track cost
        estimated_cost = self.cost_tracker.estimate_cost(prompt, response_text, self.config.model)
        self.cost_tracker.add_usage(estimated_cost)

        return {
            "content": response_text,
            "usage": response.usage.dict() if response.usage else {},
            "estimated_cost": estimated_cost,
        }

    def _create_system_prompt(self, task_type: LLMTaskType) -> str:
        """Create system prompt with few-shot examples"""

        base_prompt = """You are an expert cryptocurrency market analyst. Analyze the provided input and return a structured JSON response exactly matching the required schema. Be precise and objective."""

        if task_type == LLMTaskType.NEWS_ANALYSIS:
            schema_example = """
Required JSON schema:
{
    "sentiment": "bullish/bearish/neutral",
    "impact_magnitude": 0.0-1.0,
    "confidence": 0.0-1.0,
    "half_life_hours": float,
    "affected_symbols": ["BTC", "ETH", ...],
    "key_factors": ["factor1", "factor2", ...],
    "impact_timeline": "immediate/short_term/medium_term/long_term",
    "reasoning": "explanation"
}"""
        elif task_type == LLMTaskType.SENTIMENT_ANALYSIS:
            schema_example = """
Required JSON schema:
{
    "sentiment_score": -1.0 to 1.0,
    "confidence": 0.0-1.0,
    "emotions": ["fear", "greed", "optimism", ...],
    "key_phrases": ["phrase1", "phrase2", ...],
    "market_relevance": 0.0-1.0
}"""
        else:
            schema_example = "Analyze the input and provide structured JSON output."

        # Add few-shot examples
        examples = self.few_shot_examples.get(task_type, [])
        example_text = ""
        for i, example in enumerate(examples[:2]):  # Limit to 2 examples
            example_text += (
                f"\nExample {i + 1}:\nInput: {example['input']}\nOutput: {example['output']}\n"
            )

        return f"{base_prompt}\n\n{schema_example}\n{example_text}\nNow analyze the provided input:"

    async def analyze_news_impact(self, news_text: str) -> NewsImpactSchema:
        """Analyze news impact with full safety features"""

        try:
            # Check cache first
            cached_response = self.cache.get(news_text, self.config)
            if cached_response:
                return NewsImpactSchema(**json.loads(cached_response["content"]))

            # Make API request with circuit breaker
            response = self.circuit_breaker.call(
                self._make_openai_request, news_text, LLMTaskType.NEWS_ANALYSIS
            )

            if asyncio.iscoroutine(response):
                response = await response

            # Parse and validate response
            parsed_response = json.loads(response["content"])
            validated_result = NewsImpactSchema(**parsed_response)

            # Cache successful response
            self.cache.set(news_text, self.config, response)

            self.logger.info(f"News analysis completed: {validated_result.sentiment} sentiment")
            return validated_result

        except Exception as e:
            self.logger.error(f"OpenAI news analysis failed: {e}")
            self.logger.info("Falling back to pure Python analysis")
            return self.fallback.analyze_news_sentiment(news_text)

    async def analyze_sentiment(self, text: str) -> SentimentSchema:
        """Analyze sentiment with fallback"""

        try:
            # Check cache
            cached_response = self.cache.get(text, self.config)
            if cached_response:
                return SentimentSchema(**json.loads(cached_response["content"]))

            # API request
            response = self.circuit_breaker.call(
                self._make_openai_request, text, LLMTaskType.SENTIMENT_ANALYSIS
            )

            if asyncio.iscoroutine(response):
                response = await response

            parsed_response = json.loads(response["content"])
            validated_result = SentimentSchema(**parsed_response)

            self.cache.set(text, self.config, response)

            return validated_result

        except Exception as e:
            self.logger.error(f"OpenAI sentiment analysis failed: {e}")
            return self.fallback.analyze_basic_sentiment(text)

    def get_status(self) -> Dict[str, Any]:
        """Get adapter status and health"""

        return {
            "api_available": self.client is not None,
            "circuit_breaker_state": self.circuit_breaker.state,
            "failure_count": self.circuit_breaker.failure_count,
            "rate_limit_tokens": self.rate_limiter.tokens,
            "cost_limit_ok": self.cost_tracker.check_limit(),
            "cache_dir_exists": self.cache.cache_dir.exists(),
            "config": {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "cost_limit": self.config.cost_limit_per_hour,
            },
        }


# Global adapter instance
_global_adapter: Optional[RobustOpenAIAdapter] = None


def get_openai_adapter() -> RobustOpenAIAdapter:
    """Get or create global OpenAI adapter instance"""
    global _global_adapter

    if _global_adapter is None:
        _global_adapter = RobustOpenAIAdapter()

    return _global_adapter


# Convenience functions for easy use
async def analyze_news_impact(news_text: str) -> NewsImpactSchema:
    """Convenience function for news impact analysis"""
    adapter = get_openai_adapter()
    return await adapter.analyze_news_impact(news_text)


async def analyze_sentiment(text: str) -> SentimentSchema:
    """Convenience function for sentiment analysis"""
    adapter = get_openai_adapter()
    return await adapter.analyze_sentiment(text)
