#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise AI Governance System
Comprehensive framework voor production-ready AI/LLM integration met enterprise guardrails
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path
import threading
from collections import defaultdict, deque
import hashlib

try:
    from pydantic import BaseModel, ValidationError, Field
    from openai import OpenAI, AsyncOpenAI
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback voor development
    class BaseModel:
        pass
    ValidationError = Exception

from core.structured_logger import get_structured_logger


class AITaskType(Enum):
    """Enterprise AI task categorization with different SLAs"""
    NEWS_ANALYSIS = "news_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis" 
    MARKET_PREDICTION = "market_prediction"
    RISK_ASSESSMENT = "risk_assessment"
    FEATURE_EXTRACTION = "feature_extraction"
    ANOMALY_DETECTION = "anomaly_detection"


class AIModelTier(Enum):
    """Model tiers with different cost/performance profiles"""
    FAST = "fast"        # gpt-4o-mini - low cost, fast response
    BALANCED = "balanced" # gpt-4o - balanced cost/performance
    PREMIUM = "premium"   # gpt-4o with higher context - highest quality


@dataclass
class AITaskConfig:
    """Configuration per AI task type"""
    model_tier: AIModelTier = AIModelTier.BALANCED
    max_tokens: int = 1000
    temperature: float = 0.1
    timeout_seconds: int = 30
    max_retries: int = 3
    cache_ttl_hours: int = 24
    cost_limit_per_hour: float = 5.0  # USD
    requests_per_minute: int = 10
    fallback_enabled: bool = True


@dataclass 
class AIUsageMetrics:
    """Real-time usage tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    fallback_requests: int = 0
    cache_hits: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)


class EnterpriseRateLimiter:
    """Multi-tier enterprise rate limiting with burst protection"""
    
    def __init__(self, config: AITaskConfig):
        self.config = config
        self.tokens = config.requests_per_minute
        self.max_tokens = config.requests_per_minute
        self.last_refill = time.time()
        self.burst_allowance = min(5, config.requests_per_minute // 2)
        self.lock = threading.Lock()
        
    def acquire(self, tokens_needed: int = 1) -> bool:
        """Acquire tokens for API call with burst protection"""
        with self.lock:
            current_time = time.time()
            
            # Refill tokens based on time passed
            time_passed = current_time - self.last_refill
            tokens_to_add = time_passed * (self.config.requests_per_minute / 60.0)
            self.tokens = min(self.max_tokens + self.burst_allowance, 
                            self.tokens + tokens_to_add)
            self.last_refill = current_time
            
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False
    
    def time_until_available(self) -> float:
        """Calculate seconds until tokens available"""
        with self.lock:
            if self.tokens >= 1:
                return 0.0
            return (1 - self.tokens) * (60.0 / self.config.requests_per_minute)


class EnterpriseCircuitBreaker:
    """Advanced circuit breaker met exponential backoff"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 300,
                 half_open_max_calls: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self.lock = threading.Lock()
        
        self.logger = get_structured_logger("EnterpriseCircuitBreaker")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise AICircuitBreakerError("Circuit breaker is OPEN - API calls blocked")
            
            elif self.state == "HALF_OPEN":
                if self.half_open_calls >= self.half_open_max_calls:
                    raise AICircuitBreakerError("Circuit breaker HALF_OPEN limit reached")
                self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        # Exponential backoff
        backoff_time = min(self.recovery_timeout * (2 ** (self.failure_count - self.failure_threshold)), 
                          3600)  # Max 1 hour
        return time.time() - self.last_failure_time > backoff_time
    
    def _on_success(self):
        """Handle successful API call"""
        with self.lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= 2:  # 2 successes to fully close
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.logger.info("Circuit breaker CLOSED after successful recovery")
            elif self.state == "CLOSED":
                self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
    
    def _on_failure(self, exception: Exception):
        """Handle failed API call"""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.logger.warning("Circuit breaker OPEN from HALF_OPEN due to failure")
            elif self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.error(f"Circuit breaker OPEN after {self.failure_count} failures")


class AICostController:
    """Real-time cost monitoring en budget enforcement"""
    
    def __init__(self):
        self.hourly_costs = defaultdict(float)
        self.daily_costs = defaultdict(float)
        self.model_pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
        }
        self.lock = threading.Lock()
        self.logger = get_structured_logger("AICostController")
    
    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given usage"""
        pricing = self.model_pricing.get(model, self.model_pricing["gpt-4o"])
        return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
    
    def check_budget(self, task_config: AITaskConfig, estimated_cost: float) -> bool:
        """Check if request is within budget limits"""
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        
        with self.lock:
            hourly_total = self.hourly_costs[current_hour] + estimated_cost
            
            if hourly_total > task_config.cost_limit_per_hour:
                self.logger.warning(
                    f"Cost limit exceeded: ${hourly_total:.4f} > ${task_config.cost_limit_per_hour}"
                )
                return False
            return True
    
    def record_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record actual usage and return cost"""
        cost = self.estimate_cost(model, input_tokens, output_tokens)
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        current_day = datetime.now().strftime("%Y-%m-%d")
        
        with self.lock:
            self.hourly_costs[current_hour] += cost
            self.daily_costs[current_day] += cost
        
        return cost


class AIOutputValidator:
    """Enterprise schema validation voor LLM outputs"""
    
    def __init__(self):
        self.logger = get_structured_logger("AIOutputValidator")
        self.validation_cache = {}
    
    def validate_response(self, response: str, task_type: AITaskType) -> Dict[str, Any]:
        """Validate and parse LLM response according to task schema"""
        try:
            # Parse JSON response
            parsed_response = json.loads(response)
            
            # Task-specific validation
            if task_type == AITaskType.NEWS_ANALYSIS:
                return self._validate_news_analysis(parsed_response)
            elif task_type == AITaskType.SENTIMENT_ANALYSIS:
                return self._validate_await get_sentiment_analyzer().analyze_text(parsed_response)
            elif task_type == AITaskType.MARKET_PREDICTION:
                return self._validate_market_prediction(parsed_response)
            else:
                return self._validate_generic(parsed_response)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON response: {e}")
            return {"error": "invalid_json", "raw_response": response}
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return {"error": "validation_failed", "raw_response": response}
    
    def _validate_news_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate news analysis response"""
        required_fields = ["sentiment", "impact_magnitude", "confidence", "reasoning"]
        
        for field in required_fields:
            if field not in data:
                return {"error": f"missing_field_{field}", "data": data}
        
        # Validate ranges
        if not (0.0 <= data.get("impact_magnitude", -1) <= 1.0):
            return {"error": "invalid_impact_magnitude", "data": data}
        
        if not (0.0 <= data.get("confidence", -1) <= 1.0):
            return {"error": "invalid_confidence", "data": data}
        
        return {"validated": True, "data": data}
    
    def _validate_await get_sentiment_analyzer().analyze_text(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate sentiment analysis response"""
        required_fields = ["sentiment_score", "confidence"]
        
        for field in required_fields:
            if field not in data:
                return {"error": f"missing_field_{field}", "data": data}
        
        if not (-1.0 <= data.get("sentiment_score", -2) <= 1.0):
            return {"error": "invalid_sentiment_score", "data": data}
        
        return {"validated": True, "data": data}
    
    def _validate_market_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate market prediction response"""
        required_fields = ["direction", "confidence", "time_horizon"]
        
        for field in required_fields:
            if field not in data:
                return {"error": f"missing_field_{field}", "data": data}
        
        valid_directions = ["bullish", "bearish", "neutral"]
        if data.get("direction") not in valid_directions:
            return {"error": "invalid_direction", "data": data}
        
        return {"validated": True, "data": data}
    
    def _validate_generic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generic validation for unknown task types"""
        if not isinstance(data, dict):
            return {"error": "response_not_dict", "data": data}
        
        return {"validated": True, "data": data}


class AIFallbackEngine:
    """Comprehensive fallback strategy voor LLM failures"""
    
    def __init__(self):
        self.logger = get_structured_logger("AIFallbackEngine")
        self.fallback_cache = {}
        
    async def execute_with_fallback(self, 
                                  primary_func: Callable,
                                  task_type: AITaskType,
                                  fallback_data: Optional[Dict] = None,
                                  *args, **kwargs) -> Dict[str, Any]:
        """Execute function with comprehensive fallback strategy"""
        try:
            # Primary execution
            result = await primary_func(*args, **kwargs)
            return {"source": "primary", "data": result}
            
        except AICircuitBreakerError:
            self.logger.warning("Circuit breaker open - using fallback")
            return await self._get_fallback_response(task_type, fallback_data)
            
        except AIRateLimitError:
            self.logger.warning("Rate limit exceeded - using fallback")
            return await self._get_fallback_response(task_type, fallback_data)
            
        except AICostLimitError:
            self.logger.warning("Cost limit exceeded - using fallback")
            return await self._get_fallback_response(task_type, fallback_data)
            
        except Exception as e:
            self.logger.error(f"AI function failed: {e}")
            return await self._get_fallback_response(task_type, fallback_data)
    
    async def _get_fallback_response(self, 
                                   task_type: AITaskType, 
                                   fallback_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate fallback response based on task type"""
        
        if task_type == AITaskType.NEWS_ANALYSIS:
            return {
                "source": "fallback",
                "data": {
                    "sentiment": "neutral",
                    "impact_magnitude": 0.0,
                    "confidence": 0.0,
                    "reasoning": "AI service unavailable - using neutral fallback"
                }
            }
        
        elif task_type == AITaskType.SENTIMENT_ANALYSIS:
            return {
                "source": "fallback",
                "data": {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "fallback_reason": "AI service unavailable"
                }
            }
        
        elif task_type == AITaskType.MARKET_PREDICTION:
            return {
                "source": "fallback", 
                "data": {
                    "direction": "neutral",
                    "confidence": 0.0,
                    "time_horizon": "unknown",
                    "fallback_reason": "AI service unavailable"
                }
            }
        
        else:
            return {
                "source": "fallback",
                "data": {"error": "AI service unavailable", "confidence": 0.0}
            }


# Custom Exceptions
class AICircuitBreakerError(Exception):
    """Circuit breaker is open"""
    pass

class AIRateLimitError(Exception):
    """Rate limit exceeded"""
    pass

class AICostLimitError(Exception):
    """Cost limit exceeded"""
    pass


class EnterpriseAIGovernance:
    """Main enterprise AI governance coordinator"""
    
    def __init__(self):
        self.logger = get_structured_logger("EnterpriseAIGovernance")
        
        # Core components
        self.task_configs = self._load_task_configs()
        self.rate_limiters = {}
        self.circuit_breakers = {}
        self.cost_controller = AICostController()
        self.output_validator = AIOutputValidator()
        self.fallback_engine = AIFallbackEngine()
        
        # Metrics tracking
        self.usage_metrics = defaultdict(AIUsageMetrics)
        
        # Initialize per-task components
        for task_type in AITaskType:
            config = self.task_configs[task_type]
            self.rate_limiters[task_type] = EnterpriseRateLimiter(config)
            self.circuit_breakers[task_type] = EnterpriseCircuitBreaker()
        
        self.logger.info("Enterprise AI Governance system initialized")
    
    def _load_task_configs(self) -> Dict[AITaskType, AITaskConfig]:
        """Load task-specific configurations"""
        configs = {}
        
        # Default configurations per task type
        for task_type in AITaskType:
            if task_type in [AITaskType.NEWS_ANALYSIS, AITaskType.SENTIMENT_ANALYSIS]:
                configs[task_type] = AITaskConfig(
                    model_tier=AIModelTier.FAST,
                    max_tokens=500,
                    cost_limit_per_hour=2.0,
                    requests_per_minute=15
                )
            elif task_type == AITaskType.MARKET_PREDICTION:
                configs[task_type] = AITaskConfig(
                    model_tier=AIModelTier.BALANCED,
                    max_tokens=1000,
                    cost_limit_per_hour=5.0,
                    requests_per_minute=10
                )
            else:
                configs[task_type] = AITaskConfig()  # Default config
        
        return configs
    
    async def execute_ai_task(self,
                            task_type: AITaskType,
                            ai_function: Callable,
                            *args, **kwargs) -> Dict[str, Any]:
        """Execute AI task with full governance"""
        start_time = time.time()
        config = self.task_configs[task_type]
        
        try:
            # 1. Rate limiting check
            rate_limiter = self.rate_limiters[task_type]
            if not rate_limiter.acquire():
                wait_time = rate_limiter.time_until_available()
                raise AIRateLimitError(f"Rate limit exceeded, wait {wait_time:.1f}s")
            
            # 2. Cost control check
            estimated_cost = self.cost_controller.estimate_cost("gpt-4o", 500, 200)  # Estimate
            if not self.cost_controller.check_budget(config, estimated_cost):
                raise AICostLimitError("Budget limit exceeded")
            
            # 3. Execute with circuit breaker and fallback
            circuit_breaker = self.circuit_breakers[task_type]
            result = await self.fallback_engine.execute_with_fallback(
                lambda: circuit_breaker.call(ai_function, *args, **kwargs),
                task_type
            )
            
            # 4. Validate output if from primary source
            if result.get("source") == "primary":
                validation_result = self.output_validator.validate_response(
                    json.dumps(result["data"]) if isinstance(result["data"], dict) else str(result["data"]),
                    task_type
                )
                if not validation_result.get("validated"):
                    self.logger.warning(f"AI output validation failed: {validation_result}")
                    # Continue with fallback
                    result = await self.fallback_engine._get_fallback_response(task_type)
            
            # 5. Record metrics
            self._record_metrics(task_type, time.time() - start_time, True, result.get("source", "unknown"))
            
            return result
            
        except Exception as e:
            self._record_metrics(task_type, time.time() - start_time, False, "error")
            self.logger.error(f"AI task {task_type.value} failed: {e}")
            
            # Final fallback
            return await self.fallback_engine._get_fallback_response(task_type)
    
    def _record_metrics(self, task_type: AITaskType, latency: float, success: bool, source: str):
        """Record execution metrics"""
        metrics = self.usage_metrics[task_type]
        
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        if source == "fallback":
            metrics.fallback_requests += 1
        elif source == "cache":
            metrics.cache_hits += 1
        
        # Update average latency
        total_completed = metrics.successful_requests + metrics.failed_requests
        if total_completed > 1:
            metrics.avg_latency_ms = (
                (metrics.avg_latency_ms * (total_completed - 1) + latency * 1000) / total_completed
            )
        else:
            metrics.avg_latency_ms = latency * 1000
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get comprehensive governance status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "task_metrics": {},
            "circuit_breaker_status": {},
            "rate_limiter_status": {},
            "cost_summary": {
                "hourly_costs": dict(self.cost_controller.hourly_costs),
                "daily_costs": dict(self.cost_controller.daily_costs)
            }
        }
        
        for task_type in AITaskType:
            # Metrics
            metrics = self.usage_metrics[task_type]
            status["task_metrics"][task_type.value] = {
                "total_requests": metrics.total_requests,
                "success_rate": (
                    metrics.successful_requests / max(1, metrics.total_requests) * 100
                ),
                "fallback_rate": (
                    metrics.fallback_requests / max(1, metrics.total_requests) * 100
                ),
                "cache_hit_rate": (
                    metrics.cache_hits / max(1, metrics.total_requests) * 100
                ),
                "avg_latency_ms": metrics.avg_latency_ms,
                "total_cost": metrics.total_cost_usd
            }
            
            # Circuit breaker status
            cb = self.circuit_breakers[task_type]
            status["circuit_breaker_status"][task_type.value] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            
            # Rate limiter status
            rl = self.rate_limiters[task_type]
            status["rate_limiter_status"][task_type.value] = {
                "tokens_available": rl.tokens,
                "max_tokens": rl.max_tokens,
                "time_until_available": rl.time_until_available()
            }
        
        return status


# Global singleton
_ai_governance_instance = None

def get_ai_governance() -> EnterpriseAIGovernance:
    """Get singleton AI governance instance"""
    global _ai_governance_instance
    if _ai_governance_instance is None:
        _ai_governance_instance = EnterpriseAIGovernance()
    return _ai_governance_instance


if __name__ == "__main__":
    # Basic validation
    governance = get_ai_governance()
    status = governance.get_governance_status()
    print(json.dumps(status, indent=2))