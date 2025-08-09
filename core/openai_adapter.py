#!/usr/bin/env python3
"""
OpenAI Adapter - Robuuste LLM adapter met schema validatie, caching en rate limits
Implementatie van alle requirements uit de review voor intelligent OpenAI gebruik
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class OpenAIAdapter:
    """Production-ready OpenAI adapter with caching, rate limits, and validation"""
    
    def __init__(self, cache_dir: str = "cache/openai", cost_limit_per_hour: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cost_limit_per_hour = cost_limit_per_hour
        self.usage_file = self.cache_dir / "usage_log.json"
        
        # Initialize usage tracking
        if not self.usage_file.exists():
            self._save_usage({
                "hourly_costs": {},
                "total_requests": 0,
                "total_cost": 0.0
            })
    
    def _get_cache_key(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Generate cache key from prompt and model"""
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached response if available and fresh"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            
            # Check if cache is fresh (24 hours)
            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=24):
                cache_file.unlink()  # Remove stale cache
                return None
            
            return cached["response"]
            
        except Exception:
            return None
    
    def _save_cache(self, cache_key: str, response: Dict):
        """Save response to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "response": response
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def _load_usage(self) -> Dict:
        """Load usage tracking data"""
        try:
            with open(self.usage_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "hourly_costs": {},
                "total_requests": 0,
                "total_cost": 0.0
            }
    
    def _save_usage(self, usage_data: Dict):
        """Save usage tracking data"""
        with open(self.usage_file, 'w') as f:
            json.dump(usage_data, f, indent=2)
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within hourly cost limits"""
        usage = self._load_usage()
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        
        hourly_cost = usage["hourly_costs"].get(current_hour, 0.0)
        return hourly_cost < self.cost_limit_per_hour
    
    def _update_usage(self, cost: float):
        """Update usage tracking"""
        usage = self._load_usage()
        current_hour = datetime.now().strftime("%Y-%m-%d-%H")
        
        usage["hourly_costs"][current_hour] = usage["hourly_costs"].get(current_hour, 0.0) + cost
        usage["total_requests"] += 1
        usage["total_cost"] += cost
        
        # Cleanup old hourly data (keep only last 48 hours)
        cutoff = (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%d-%H")
        usage["hourly_costs"] = {
            k: v for k, v in usage["hourly_costs"].items() 
            if k >= cutoff
        }
        
        self._save_usage(usage)
    
    def _estimate_cost(self, prompt: str, model: str = "gpt-4o-mini") -> float:
        """Estimate API call cost"""
        # Rough token estimation (4 chars = 1 token)
        input_tokens = len(prompt) / 4
        output_tokens = 200  # Estimated response size
        
        # Pricing (as of 2024) - adjust as needed
        pricing = {
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-4o": {"input": 0.005 / 1000, "output": 0.015 / 1000}
        }
        
        rates = pricing.get(model, pricing["gpt-4o-mini"])
        return input_tokens * rates["input"] + output_tokens * rates["output"]
    
    def _validate_schema(self, data: Dict, schema: Dict) -> bool:
        """Validate response against schema"""
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Schema validation failed: missing required field '{field}'")
        
        # Type validation (basic)
        field_types = schema.get("properties", {})
        for field, field_schema in field_types.items():
            if field in data:
                expected_type = field_schema.get("type")
                if expected_type:
                    value = data[field]
                    if expected_type == "number" and not isinstance(value, (int, float)):
                        raise ValueError(f"Schema validation failed: '{field}' should be number, got {type(value)}")
                    elif expected_type == "string" and not isinstance(value, str):
                        raise ValueError(f"Schema validation failed: '{field}' should be string, got {type(value)}")
        
        return True
    
    def call_llm(self, prompt: str, schema: Dict, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Call OpenAI API with robust error handling and validation
        
        Args:
            prompt: The prompt to send
            schema: JSON schema for response validation
            model: Model to use (default: gpt-4o-mini for cost efficiency)
        
        Returns:
            Validated JSON response
        """
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Check rate limits
        if not self._check_rate_limit():
            raise ValueError(f"Hourly cost limit ({self.cost_limit_per_hour}$) exceeded")
        
        # Check cache first
        cache_key = self._get_cache_key(prompt, model)
        cached_response = self._load_cache(cache_key)
        if cached_response:
            print(f"✅ Using cached OpenAI response (key: {cache_key[:8]}...)")
            return cached_response
        
        # Estimate cost
        estimated_cost = self._estimate_cost(prompt, model)
        print(f"📊 OpenAI call: ~${estimated_cost:.4f} estimated cost")
        
        # Make API call with simple retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,  # Lower temperature for more consistent results
                    "max_tokens": 500
                }
                
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                break
                
            except (requests.RequestException, requests.HTTPError) as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
        
        # Parse response
        response_data = response.json()
        content = response_data["choices"][0]["message"]["content"]
        
        try:
            parsed_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"OpenAI returned invalid JSON: {e}")
        
        # Validate against schema
        self._validate_schema(parsed_data, schema)
        
        # Update usage tracking
        actual_cost = estimated_cost  # Use estimate for now
        self._update_usage(actual_cost)
        
        # Cache response
        self._save_cache(cache_key, parsed_data)
        
        print(f"✅ OpenAI call successful (cached for 24h)")
        return parsed_data

# Global adapter instance
_adapter = None

def get_openai_adapter() -> OpenAIAdapter:
    """Get global OpenAI adapter instance"""
    global _adapter
    if _adapter is None:
        _adapter = OpenAIAdapter()
    return _adapter

def analyze_crypto_news_impact(news_text: str) -> Dict[str, Any]:
    """
    Analyze crypto news for market impact using OpenAI
    Returns structured impact assessment
    """
    
    adapter = get_openai_adapter()
    
    schema = {
        "required": ["direction", "magnitude", "half_life_h", "confidence"],
        "properties": {
            "direction": {"type": "string"},  # bull/bear/neutral
            "magnitude": {"type": "number"},  # 0.0-1.0
            "half_life_h": {"type": "number"},  # 1-168 hours
            "confidence": {"type": "number"},  # 0.0-1.0
            "reasoning": {"type": "string"}
        }
    }
    
    prompt = f"""Analyseer dit cryptocurrency nieuws voor marktimpact en geef een JSON response:

Nieuws: "{news_text}"

Geef een gestructureerde analyse in JSON format met:
- "direction": "bull" (positief), "bear" (negatief), of "neutral"  
- "magnitude": impact sterkte van 0.0 (geen impact) tot 1.0 (extreme impact)
- "half_life_h": hoe lang het effect zal duren in uren (1-168)
- "confidence": zekerheid van deze analyse (0.0-1.0)
- "reasoning": korte uitleg van de analyse

Focus op directe marktimpact, niet op langetermijn fundamentals."""

    return adapter.call_llm(prompt, schema)

def generate_trading_insights(coin_data: Dict) -> Dict[str, Any]:
    """Generate AI-powered trading insights for a specific coin"""
    
    adapter = get_openai_adapter()
    
    schema = {
        "required": ["trend_direction", "risk_assessment", "key_factors"],
        "properties": {
            "trend_direction": {"type": "string"},  # bullish/bearish/sideways
            "risk_assessment": {"type": "string"},  # low/medium/high
            "key_factors": {"type": "string"},
            "short_term_outlook": {"type": "string"},
            "confidence_score": {"type": "number"}
        }
    }
    
    prompt = f"""Analyseer deze cryptocurrency data en geef trading insights in JSON:

Coin: {coin_data.get('coin', 'Unknown')}
Huidige prijs: ${coin_data.get('price', 0):.4f}
24h verandering: {coin_data.get('change_24h', 0):.2f}%
Volume: ${coin_data.get('volume_24h', 0):,.0f}
RSI: {coin_data.get('rsi', 50):.1f}
MACD: {coin_data.get('macd', 0):.4f}

Geef analyse in JSON met:
- "trend_direction": "bullish", "bearish", of "sideways"
- "risk_assessment": "low", "medium", of "high" 
- "key_factors": belangrijkste technische/fundamentele factoren
- "short_term_outlook": korte termijn verwachting (1-7 dagen)
- "confidence_score": zekerheid van analyse (0.0-1.0)

Focus op technische analyse en marktsentiment."""

    return adapter.call_llm(prompt, schema)

if __name__ == "__main__":
    print("🧪 Testing OpenAI Adapter...")
    
    # Test basic functionality
    adapter = get_openai_adapter()
    
    # Test news analysis
    try:
        test_news = "Bitcoin reaches new all-time high of $75,000 as institutional adoption accelerates"
        result = analyze_crypto_news_impact(test_news)
        
        print(f"\n📰 News Impact Analysis:")
        print(f"Direction: {result['direction']}")
        print(f"Magnitude: {result['magnitude']:.2f}")
        print(f"Half-life: {result['half_life_h']}h")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Reasoning: {result['reasoning']}")
        
    except Exception as e:
        print(f"❌ News analysis failed: {e}")
    
    # Test trading insights
    try:
        test_coin_data = {
            "coin": "BTC",
            "price": 45000,
            "change_24h": 5.2,
            "volume_24h": 25000000000,
            "rsi": 65.5,
            "macd": 0.15
        }
        
        insights = generate_trading_insights(test_coin_data)
        
        print(f"\n🔍 Trading Insights:")
        print(f"Trend: {insights['trend_direction']}")
        print(f"Risk: {insights['risk_assessment']}")
        print(f"Confidence: {insights.get('confidence_score', 0):.2f}")
        
    except Exception as e:
        print(f"❌ Trading insights failed: {e}")
    
    # Show usage stats
    usage = adapter._load_usage()
    current_hour = datetime.now().strftime("%Y-%m-%d-%H")
    hourly_cost = usage["hourly_costs"].get(current_hour, 0.0)
    
    print(f"\n📊 Usage Statistics:")
    print(f"Total requests: {usage['total_requests']}")
    print(f"Total cost: ${usage['total_cost']:.4f}")
    print(f"Current hour cost: ${hourly_cost:.4f}")
    
    print(f"\n✅ OpenAI Adapter test complete!")