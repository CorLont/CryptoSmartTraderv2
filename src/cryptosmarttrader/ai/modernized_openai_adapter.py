#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Modernized OpenAI Adapter with Enterprise Governance
Replaces experimental implementations with production-ready AI integration
"""

import asyncio
import hashlib
import json
import os
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from core.structured_logger import get_structured_logger
from src.cryptosmarttrader.ai.enterprise_ai_governance import (
    get_ai_governance, 
    AITaskType, 
    AITaskConfig,
    AIModelTier
)
from src.cryptosmarttrader.ai.enterprise_ai_evaluator import get_ai_evaluator


@dataclass
class NewsAnalysisResult:
    """Structured result for news analysis"""
    sentiment: str  # "bullish", "bearish", "neutral"
    impact_magnitude: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    half_life_hours: float  # Expected impact duration
    affected_symbols: List[str]  # Specific coins affected
    key_factors: List[str]  # Main impact drivers
    reasoning: str  # Explanation
    source: str  # "primary" or "fallback"


@dataclass
class SentimentAnalysisResult:
    """Structured result for sentiment analysis"""
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    emotions: List[str]  # ["fear", "greed", "optimism", etc.]
    key_phrases: List[str]  # Important phrases
    source: str  # "primary" or "fallback"


class ModernizedOpenAIAdapter:
    """Production-ready OpenAI adapter with enterprise governance"""
    
    def __init__(self):
        self.logger = get_structured_logger("ModernizedOpenAIAdapter")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.governance = get_ai_governance()
        self.evaluator = get_ai_evaluator()
        
        # Cache for responses
        self.cache_dir = Path("cache/openai_v2")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Modernized OpenAI Adapter initialized with enterprise governance")
    
    async def analyze_news_impact(self, 
                                news_content: str,
                                symbols_context: Optional[List[str]] = None) -> NewsAnalysisResult:
        """Analyze news impact with full governance"""
        
        start_time = time.time()
        
        try:
            # Execute through governance system
            result = await self.governance.execute_ai_task(
                AITaskType.NEWS_ANALYSIS,
                self._execute_news_analysis,
                news_content,
                symbols_context or []
            )
            
            # Evaluate response
            response_time_ms = (time.time() - start_time) * 1000
            await self.evaluator.evaluate_ai_response(
                model_name="gpt-4o",
                response=json.dumps(result["data"]),
                expected_schema={
                    "sentiment": "str",
                    "impact_magnitude": "float", 
                    "confidence": "float",
                    "reasoning": "str"
                },
                task_type="news_analysis",
                response_time_ms=response_time_ms,
                cost_usd=0.01  # Estimated
            )
            
            # Convert to structured result
            data = result["data"]
            return NewsAnalysisResult(
                sentiment=data.get("sentiment", "neutral"),
                impact_magnitude=data.get("impact_magnitude", 0.0),
                confidence=data.get("confidence", 0.0),
                half_life_hours=data.get("half_life_hours", 24.0),
                affected_symbols=data.get("affected_symbols", symbols_context or []),
                key_factors=data.get("key_factors", []),
                reasoning=data.get("reasoning", ""),
                source=result.get("source", "primary")
            )
            
        except Exception as e:
            self.logger.error(f"News analysis failed: {e}")
            
            # Return safe fallback
            return NewsAnalysisResult(
                sentiment="neutral",
                impact_magnitude=0.0,
                confidence=0.0,
                half_life_hours=24.0,
                affected_symbols=symbols_context or [],
                key_factors=[],
                reasoning="Analysis unavailable due to service error",
                source="fallback"
            )
    
    async def _execute_news_analysis(self, news_content: str, symbols_context: List[str]) -> Dict[str, Any]:
        """Execute news analysis API call"""
        
        # Prepare symbols context
        symbols_text = f" Focus on these symbols: {', '.join(symbols_context)}" if symbols_context else ""
        
        prompt = f"""Analyze the following cryptocurrency news and provide structured impact analysis.{symbols_text}

News content: {news_content}

Provide response in JSON format with these exact fields:
{{
    "sentiment": "bullish|bearish|neutral",
    "impact_magnitude": 0.0-1.0,
    "confidence": 0.0-1.0,
    "half_life_hours": 1-168,
    "affected_symbols": ["BTC", "ETH", ...],
    "key_factors": ["regulatory news", "adoption", ...],
    "reasoning": "Brief explanation of analysis"
}}

Focus on actionable trading insights and be conservative in impact estimates."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency market analyst. Provide accurate, structured analysis in JSON format only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1,
                timeout=30.0
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            try:
                result = json.loads(content)
                
                # Validate critical fields
                if not isinstance(result.get("sentiment"), str):
                    result["sentiment"] = "neutral"
                if not isinstance(result.get("impact_magnitude"), (int, float)):
                    result["impact_magnitude"] = 0.0
                if not isinstance(result.get("confidence"), (int, float)):
                    result["confidence"] = 0.0
                
                # Clamp numeric values
                result["impact_magnitude"] = max(0.0, min(1.0, float(result["impact_magnitude"])))
                result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
                
                return result
                
            except json.JSONDecodeError:
                # Fallback parsing
                return {
                    "sentiment": "neutral",
                    "impact_magnitude": 0.0,
                    "confidence": 0.0,
                    "half_life_hours": 24.0,
                    "affected_symbols": symbols_context,
                    "key_factors": [],
                    "reasoning": "Failed to parse structured response"
                }
                
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise e
    
    async def await get_sentiment_analyzer().analyze_text(self, text: str) -> SentimentAnalysisResult:
        """Analyze sentiment with governance"""
        
        start_time = time.time()
        
        try:
            result = await self.governance.execute_ai_task(
                AITaskType.SENTIMENT_ANALYSIS,
                self._execute_sentiment_analysis,
                text
            )
            
            # Evaluate response
            response_time_ms = (time.time() - start_time) * 1000
            await self.evaluator.evaluate_ai_response(
                model_name="gpt-4o-mini",
                response=json.dumps(result["data"]),
                expected_schema={
                    "sentiment_score": "float",
                    "confidence": "float"
                },
                task_type="sentiment_analysis",
                response_time_ms=response_time_ms,
                cost_usd=0.001  # Estimated for mini model
            )
            
            data = result["data"]
            return SentimentAnalysisResult(
                sentiment_score=data.get("sentiment_score", 0.0),
                confidence=data.get("confidence", 0.0),
                emotions=data.get("emotions", []),
                key_phrases=data.get("key_phrases", []),
                source=result.get("source", "primary")
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            
            return SentimentAnalysisResult(
                sentiment_score=0.0,
                confidence=0.0,
                emotions=[],
                key_phrases=[],
                source="fallback"
            )
    
    async def _execute_await get_sentiment_analyzer().analyze_text(self, text: str) -> Dict[str, Any]:
        """Execute sentiment analysis API call"""
        
        prompt = f"""Analyze the sentiment of this cryptocurrency-related text:

Text: {text}

Provide response in JSON format:
{{
    "sentiment_score": -1.0 to 1.0,
    "confidence": 0.0 to 1.0,
    "emotions": ["fear", "greed", "optimism", ...],
    "key_phrases": ["specific phrases that influenced sentiment"]
}}

Be precise and objective in your analysis."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use faster/cheaper model for sentiment
                messages=[
                    {"role": "system", "content": "You are an expert sentiment analyst. Provide accurate JSON-formatted sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.0,
                timeout=20.0
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                
                # Validate and clamp values
                sentiment_score = result.get("sentiment_score", 0.0)
                result["sentiment_score"] = max(-1.0, min(1.0, float(sentiment_score)))
                
                confidence = result.get("confidence", 0.0)
                result["confidence"] = max(0.0, min(1.0, float(confidence)))
                
                return result
                
            except json.JSONDecodeError:
                return {
                    "sentiment_score": 0.0,
                    "confidence": 0.0,
                    "emotions": [],
                    "key_phrases": []
                }
                
        except Exception as e:
            self.logger.error(f"Sentiment API call failed: {e}")
            raise e
    
    def get_governance_status(self) -> Dict[str, Any]:
        """Get AI governance status"""
        return self.governance.get_governance_status()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get AI evaluation summary"""
        return self.evaluator.get_evaluation_summary()


# Global singleton
_openai_adapter_instance = None

def get_modernized_openai_adapter() -> ModernizedOpenAIAdapter:
    """Get singleton modernized OpenAI adapter"""
    global _openai_adapter_instance
    if _openai_adapter_instance is None:
        _openai_adapter_instance = ModernizedOpenAIAdapter()
    return _openai_adapter_instance


if __name__ == "__main__":
    # Basic validation
    async def test_adapter():
        adapter = get_modernized_openai_adapter()
        
        # Test news analysis
        news_result = await adapter.analyze_news_impact(
            "Bitcoin ETF approval expected next week, could drive significant price movement"
        )
        print(f"News Analysis: {news_result}")
        
        # Test sentiment analysis
        sentiment_result = await adapter.await get_sentiment_analyzer().analyze_text(
            "Crypto market looking very bullish with strong institutional adoption"
        )
        print(f"Sentiment Analysis: {sentiment_result}")
        
        # Show status
        print(f"Governance Status: {adapter.get_governance_status()}")
        print(f"Evaluation Summary: {adapter.get_evaluation_summary()}")
    
    asyncio.run(test_adapter())