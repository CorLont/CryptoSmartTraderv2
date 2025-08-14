#!/usr/bin/env python3
"""
Enterprise OpenAI Adapter with Centralized Throttling & Error Handling
Vervangt scattered OpenAI calls met uniforme infrastructure
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from ..infrastructure.centralized_throttling import ServiceType, throttled
from ..infrastructure.unified_error_handler import unified_error_handling

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None
    OpenAI = None

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    model: str
    tokens_used: int
    cost_estimate: float
    response_time: float
    cached: bool = False
    metadata: Dict[str, Any] = None


class EnterpriseOpenAIAdapter:
    """
    Enterprise OpenAI adapter met centralized throttling en error handling
    KRITIEK: Single point of control voor ALL OpenAI calls
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize enterprise OpenAI adapter"""
        import os
        
        if not openai or not OpenAI:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable or api_key parameter required")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Enterprise caching
        self.cache_dir = Path("cache/openai_enterprise")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cost tracking
        self.cost_log = Path("logs/openai_enterprise_costs.json")
        self.costs = self._load_cost_log()
        
        # Model pricing (per 1K tokens) - Updated for current models
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        logger.info("🤖 EnterpriseOpenAIAdapter initialized with throttling & error handling")
    
    def _load_cost_log(self) -> Dict[str, Any]:
        """Load cost tracking data"""
        if self.cost_log.exists():
            try:
                with open(self.cost_log, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cost log: {e}")
                return {"total_cost": 0, "calls": []}
        return {"total_cost": 0, "calls": []}
    
    def _save_cost_log(self) -> None:
        """Save cost tracking data"""
        try:
            self.cost_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cost_log, "w") as f:
                json.dump(self.costs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cost log: {e}")
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call"""
        if model not in self.pricing:
            logger.warning(f"Unknown model {model}, using gpt-4o pricing")
            model = "gpt-4o"
        
        pricing = self.pricing[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _get_cache_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Generate cache key for request"""
        import hashlib
        
        cache_data = {
            "messages": messages,
            "model": model,
            "kwargs": {k: v for k, v in kwargs.items() if k not in ["stream"]}
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                
                # Check if cache is fresh (1 hour for LLM responses)
                if time.time() - cached_data["timestamp"] < 3600:
                    logger.info("Using cached OpenAI response")
                    
                    return LLMResponse(
                        content=cached_data["content"],
                        model=cached_data["model"],
                        tokens_used=cached_data["tokens_used"],
                        cost_estimate=cached_data["cost_estimate"],
                        response_time=cached_data["response_time"],
                        cached=True,
                        metadata=cached_data.get("metadata", {})
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached response: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Cache successful response"""
        try:
            cache_data = {
                "timestamp": time.time(),
                "content": response.content,
                "model": response.model,
                "tokens_used": response.tokens_used,
                "cost_estimate": response.cost_estimate,
                "response_time": response.response_time,
                "metadata": response.metadata or {}
            }
            
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def _log_cost(self, response: LLMResponse) -> None:
        """Log cost information"""
        cost_entry = {
            "timestamp": time.time(),
            "model": response.model,
            "tokens_used": response.tokens_used,
            "cost": response.cost_estimate,
            "cached": response.cached,
            "response_time": response.response_time
        }
        
        self.costs["calls"].append(cost_entry)
        self.costs["total_cost"] += response.cost_estimate
        
        # Keep only last 1000 calls
        if len(self.costs["calls"]) > 1000:
            self.costs["calls"] = self.costs["calls"][-1000:]
        
        self._save_cost_log()
    
    @throttled(ServiceType.LLM_API, endpoint="chat_completion")
    @unified_error_handling("llm_api", endpoint="chat_completion")
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Async chat completion met enterprise features
        
        Args:
            messages: Chat messages
            model: Model to use
            temperature: Response randomness
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use caching
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse: Standardized response
        """
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(messages, model, temperature=temperature, max_tokens=max_tokens, **kwargs)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self._log_cost(cached_response)
                return cached_response
        
        # Make API call
        completion_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens
        
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **completion_kwargs
        )
        
        # Extract response data
        content = response.choices[0].message.content
        usage = response.usage
        
        # Calculate cost
        cost_estimate = self._calculate_cost(
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens
        )
        
        response_time = time.time() - start_time
        
        # Create response object
        llm_response = LLMResponse(
            content=content,
            model=model,
            tokens_used=usage.total_tokens,
            cost_estimate=cost_estimate,
            response_time=response_time,
            cached=False,
            metadata={
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "temperature": temperature,
                "finish_reason": response.choices[0].finish_reason
            }
        )
        
        # Cache response
        if use_cache:
            self._cache_response(cache_key, llm_response)
        
        # Log cost
        self._log_cost(llm_response)
        
        logger.info(
            f"OpenAI completion: {usage.total_tokens} tokens, "
            f"${cost_estimate:.4f}, {response_time:.2f}s"
        )
        
        return llm_response
    
    def chat_completion(self, *args, **kwargs) -> LLMResponse:
        """Sync wrapper voor chat completion"""
        return asyncio.run(self.chat_completion_async(*args, **kwargs))
    
    @throttled(ServiceType.LLM_API, endpoint="sentiment_analysis")
    @unified_error_handling("llm_api", endpoint="sentiment_analysis")
    async def analyze_sentiment_async(
        self,
        text: str,
        context: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Sentiment analysis met enterprise guardrails
        
        Args:
            text: Text to analyze
            context: Optional context
            model: Model to use
            
        Returns:
            Dict with sentiment analysis results
        """
        system_prompt = """
        You are a cryptocurrency sentiment analysis expert. Analyze the sentiment of the given text.
        
        Respond with a JSON object containing:
        - sentiment: "positive", "negative", or "neutral"
        - confidence: float between 0 and 1
        - key_indicators: list of words/phrases that influenced the sentiment
        - crypto_relevance: float between 0 and 1 indicating crypto relevance
        - emotional_intensity: float between 0 and 1
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Text: {text}\n\nContext: {context or 'None'}"}
        ]
        
        response = await self.chat_completion_async(
            messages=messages,
            model=model,
            temperature=0.3,
            max_tokens=500
        )
        
        try:
            # Parse JSON response
            sentiment_data = json.loads(response.content)
            sentiment_data["_metadata"] = {
                "model": response.model,
                "tokens_used": response.tokens_used,
                "cost": response.cost_estimate,
                "response_time": response.response_time
            }
            return sentiment_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sentiment analysis response: {e}")
            # Fallback response
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "key_indicators": [],
                "crypto_relevance": 0.0,
                "emotional_intensity": 0.0,
                "error": "Failed to parse response",
                "_metadata": {
                    "model": response.model,
                    "tokens_used": response.tokens_used,
                    "cost": response.cost_estimate,
                    "response_time": response.response_time
                }
            }
    
    def analyze_sentiment(self, *args, **kwargs) -> Dict[str, Any]:
        """Sync wrapper voor sentiment analysis"""
        return asyncio.run(self.analyze_sentiment_async(*args, **kwargs))
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary and statistics"""
        if not self.costs["calls"]:
            return {"total_cost": 0, "call_count": 0, "average_cost": 0}
        
        calls = self.costs["calls"]
        recent_calls = [c for c in calls if time.time() - c["timestamp"] < 86400]  # Last 24h
        
        return {
            "total_cost": self.costs["total_cost"],
            "call_count": len(calls),
            "recent_calls_24h": len(recent_calls),
            "average_cost": self.costs["total_cost"] / len(calls),
            "recent_cost_24h": sum(c["cost"] for c in recent_calls),
            "models_used": list(set(c["model"] for c in calls)),
            "cached_responses": len([c for c in calls if c.get("cached", False)])
        }


# Export public interface
__all__ = [
    'EnterpriseOpenAIAdapter',
    'LLMResponse'
]