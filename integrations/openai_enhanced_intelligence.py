#!/usr/bin/env python3
"""
OpenAI Enhanced Intelligence Integration
Authentic AI-powered analysis and insights for CryptoSmartTrader V2
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

# OpenAI imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from core.unified_structured_logger import get_unified_logger

@dataclass
class AIInsight:
    """AI-generated market insight"""
    insight_type: str
    confidence: float
    description: str
    reasoning: str
    actionability: str
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class SentimentAnalysis:
    """AI-powered sentiment analysis result"""
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    key_factors: List[str]
    market_impact: str
    reasoning: str

class OpenAIEnhancedIntelligence:
    """OpenAI integration for enhanced market intelligence"""
    
    def __init__(self):
        self.logger = get_unified_logger("OpenAIIntelligence")
        self.client = None
        self.model = "gpt-4o"  # Latest OpenAI model
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                    self.logger.info("OpenAI client initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.client = None
            else:
                self.logger.warning("OPENAI_API_KEY not found in environment")
        else:
            self.logger.warning("OpenAI library not available")
    
    def is_available(self) -> bool:
        """Check if OpenAI integration is available"""
        return self.client is not None
    
    async def analyze_market_sentiment(self, news_data: List[str], 
                                     market_data: pd.DataFrame) -> Optional[SentimentAnalysis]:
        """Analyze market sentiment using AI"""
        
        if not self.is_available():
            self.logger.warning("OpenAI not available for sentiment analysis")
            return None
        
        try:
            # Prepare market context
            latest_prices = market_data.tail(5) if not market_data.empty else pd.DataFrame()
            price_context = ""
            
            if not latest_prices.empty:
                btc_change = ((latest_prices['btc_price'].iloc[-1] - latest_prices['btc_price'].iloc[0]) 
                            / latest_prices['btc_price'].iloc[0] * 100) if 'btc_price' in latest_prices.columns else 0
                price_context = f"Recent BTC price change: {btc_change:.1f}%"
            
            # Combine news data
            news_text = "\n".join(news_data[:10])  # Limit to recent news
            
            # Create comprehensive prompt
            prompt = f"""
            Analyze the current cryptocurrency market sentiment based on the following data:

            NEWS DATA:
            {news_text}

            MARKET CONTEXT:
            {price_context}

            Provide your analysis in JSON format with:
            - overall_sentiment: "bullish", "bearish", or "neutral"
            - sentiment_score: float between -1 (very bearish) and 1 (very bullish)
            - confidence: float between 0 and 1
            - key_factors: list of 3-5 key factors influencing sentiment
            - market_impact: "high", "medium", or "low"
            - reasoning: detailed explanation of your analysis

            Focus on actionable insights for trading decisions.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert cryptocurrency market analyst with deep knowledge of market dynamics, sentiment analysis, and trading psychology."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return SentimentAnalysis(
                overall_sentiment=result.get("overall_sentiment", "neutral"),
                sentiment_score=float(result.get("sentiment_score", 0.0)),
                confidence=float(result.get("confidence", 0.5)),
                key_factors=result.get("key_factors", []),
                market_impact=result.get("market_impact", "medium"),
                reasoning=result.get("reasoning", "No reasoning provided")
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return None
    
    async def generate_market_insights(self, coin_data: Dict[str, Any], 
                                     predictions: pd.DataFrame) -> List[AIInsight]:
        """Generate AI-powered market insights"""
        
        if not self.is_available():
            self.logger.warning("OpenAI not available for insight generation")
            return []
        
        try:
            insights = []
            
            # Analyze top predictions
            top_coins = predictions.head(5) if not predictions.empty else pd.DataFrame()
            
            for _, row in top_coins.iterrows():
                coin = row.get('coin', 'Unknown')
                pred_30d = row.get('pred_30d', 0)
                confidence = row.get('conf_30d', 0)
                
                # Create insight generation prompt
                prompt = f"""
                Analyze the trading opportunity for {coin} with the following data:
                
                - 30-day prediction: {pred_30d:.1%}
                - Model confidence: {confidence:.1%}
                - Current market conditions: Mixed
                
                Generate a concise trading insight in JSON format:
                - insight_type: "opportunity", "risk", or "neutral"
                - confidence: float 0-1 representing your confidence in this insight
                - description: Brief description of the opportunity/risk
                - reasoning: Why this prediction is significant
                - actionability: "high", "medium", or "low" action priority
                
                Focus on practical trading implications.
                """
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a professional cryptocurrency trading analyst. Provide actionable insights based on ML predictions."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=500
                )
                
                result = json.loads(response.choices[0].message.content)
                
                insight = AIInsight(
                    insight_type=result.get("insight_type", "neutral"),
                    confidence=float(result.get("confidence", 0.5)),
                    description=result.get("description", "No description"),
                    reasoning=result.get("reasoning", "No reasoning"),
                    actionability=result.get("actionability", "medium"),
                    timestamp=datetime.now(),
                    metadata={"coin": coin, "prediction": pred_30d, "model_confidence": confidence}
                )
                
                insights.append(insight)
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            self.logger.info(f"Generated {len(insights)} AI insights")
            return insights
            
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            return []
    
    async def explain_prediction_confidence(self, coin: str, prediction: float, 
                                          confidence: float, features: Dict[str, float]) -> Optional[str]:
        """Generate AI explanation for prediction confidence"""
        
        if not self.is_available():
            return None
        
        try:
            # Prepare feature context
            feature_text = ""
            for feature, value in features.items():
                feature_text += f"- {feature}: {value:.3f}\n"
            
            prompt = f"""
            Explain why our ML model has {confidence:.1%} confidence in predicting {prediction:.1%} growth for {coin}.
            
            Key model features:
            {feature_text}
            
            Provide a clear, concise explanation suitable for traders focusing on:
            1. What factors support this prediction
            2. What creates the confidence level
            3. Key risks or limitations
            
            Keep it under 200 words and practical.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an ML model explainer helping traders understand algorithmic predictions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"Prediction explanation failed: {e}")
            return None

# Global instance
_openai_intelligence: Optional[OpenAIEnhancedIntelligence] = None

def get_openai_intelligence() -> OpenAIEnhancedIntelligence:
    """Get global OpenAI intelligence instance"""
    global _openai_intelligence
    if _openai_intelligence is None:
        _openai_intelligence = OpenAIEnhancedIntelligence()
    return _openai_intelligence