# core/openai_adapter.py - Enterprise OpenAI integration with cost tracking
import os
import json
import hashlib
import time
import requests
from functools import lru_cache
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OpenAIAdapter:
    """Robust OpenAI adapter with caching, cost tracking, and schema validation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.cache_dir = Path("cache/openai")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cost_log = Path("logs/openai_costs.json")
        self.costs = self._load_cost_log()
        
        # Model pricing (per 1K tokens)
        self.pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015}
        }
    
    def _load_cost_log(self):
        """Load existing cost tracking"""
        if self.cost_log.exists():
            try:
                with open(self.cost_log, 'r') as f:
                    return json.load(f)
            except:
                return {"total_cost": 0, "calls": []}
        return {"total_cost": 0, "calls": []}
    
    def _save_cost_log(self):
        """Save cost tracking"""
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cost_log, 'w') as f:
            json.dump(self.costs, f, indent=2)
    
    def _get_cache_key(self, text: str, schema: str = "event_impact") -> str:
        """Generate cache key for text"""
        return hashlib.md5(f"{schema}:{text}".encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str):
        """Get cached response if available and fresh"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                
                # Check if cache is fresh (24 hours)
                if time.time() - cached['timestamp'] < 86400:
                    logger.info(f"Using cached OpenAI response")
                    return cached['response']
            except:
                pass
        
        return None
    
    def _cache_response(self, cache_key: str, response: dict):
        """Cache successful response"""
        cache_data = {
            'timestamp': time.time(),
            'response': response
        }
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def _log_api_call(self, model: str, input_tokens: int, output_tokens: int, cost: float):
        """Log API call for cost tracking"""
        call_data = {
            'timestamp': time.time(),
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        }
        
        self.costs['calls'].append(call_data)
        self.costs['total_cost'] += cost
        
        # Keep only last 1000 calls
        self.costs['calls'] = self.costs['calls'][-1000:]
        
        self._save_cost_log()
        
        logger.info(f"OpenAI call: ${cost:.4f} (Total: ${self.costs['total_cost']:.2f})")
    
    def llm_event_impact(self, text: str, model: str = "gpt-4o-mini") -> dict:
        """
        Analyze news/event impact with structured JSON output
        
        Returns: {
            "direction": "bull|bear|neutral",
            "magnitude": 0.0-1.0,  # Impact strength
            "half_life_h": 1-168,   # Hours until impact fades
            "confidence": 0.0-1.0,  # LLM confidence in analysis
            "reasoning": "brief explanation"
        }
        """
        cache_key = self._get_cache_key(text, "event_impact")
        
        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # Prepare API request
        prompt = f"""Analyze the following cryptocurrency news/event and provide structured impact analysis.

Return ONLY valid JSON with this exact schema:
{{
    "direction": "bull|bear|neutral",
    "magnitude": 0.5,
    "half_life_h": 24,
    "confidence": 0.8,
    "reasoning": "Brief explanation"
}}

Guidelines:
- direction: "bull" (positive), "bear" (negative), "neutral" (minimal impact)  
- magnitude: 0.0 (no impact) to 1.0 (massive market-moving event)
- half_life_h: Hours until 50% of impact fades (1-168 hours)
- confidence: Your confidence in this analysis (0.0-1.0)
- reasoning: One sentence explanation

News/Event: {text[:1000]}"""  # Limit input to control costs

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,  # Lower temperature for more consistent analysis
            "max_tokens": 200    # Limit output tokens
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse and validate JSON response
            data = json.loads(content)
            
            # Schema validation
            required_fields = ["direction", "magnitude", "half_life_h", "confidence", "reasoning"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Value validation
            if data["direction"] not in ["bull", "bear", "neutral"]:
                raise ValueError(f"Invalid direction: {data['direction']}")
            
            if not (0 <= data["magnitude"] <= 1):
                raise ValueError(f"Invalid magnitude: {data['magnitude']}")
            
            if not (1 <= data["half_life_h"] <= 168):
                raise ValueError(f"Invalid half_life_h: {data['half_life_h']}")
            
            if not (0 <= data["confidence"] <= 1):
                raise ValueError(f"Invalid confidence: {data['confidence']}")
            
            # Log costs
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", self._estimate_tokens(prompt))
            output_tokens = usage.get("completion_tokens", self._estimate_tokens(content))
            
            cost = (
                (input_tokens / 1000) * self.pricing[model]["input"] +
                (output_tokens / 1000) * self.pricing[model]["output"]
            )
            
            self._log_api_call(model, input_tokens, output_tokens, cost)
            
            # Cache successful response
            self._cache_response(cache_key, data)
            
            return data
            
        except requests.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from OpenAI: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {e}")
            raise
    
    def get_cost_summary(self) -> dict:
        """Get cost tracking summary"""
        total_calls = len(self.costs['calls'])
        
        if total_calls == 0:
            return {
                'total_cost': 0,
                'total_calls': 0,
                'avg_cost_per_call': 0,
                'daily_cost': 0
            }
        
        # Calculate daily cost (last 24 hours)
        yesterday = time.time() - 86400
        daily_cost = sum(
            call['cost'] for call in self.costs['calls'] 
            if call['timestamp'] > yesterday
        )
        
        return {
            'total_cost': self.costs['total_cost'],
            'total_calls': total_calls,
            'avg_cost_per_call': self.costs['total_cost'] / total_calls,
            'daily_cost': daily_cost
        }

# Global adapter instance
_adapter = None

def get_openai_adapter() -> OpenAIAdapter:
    """Get global OpenAI adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = OpenAIAdapter()
    return _adapter

def llm_event_impact(text: str) -> dict:
    """Convenience function for event impact analysis"""
    return get_openai_adapter().llm_event_impact(text)