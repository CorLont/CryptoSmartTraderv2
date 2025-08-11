#!/usr/bin/env python3
"""
OpenAI Enhanced Analysis
Advanced AI-powered market analysis using GPT-4o for intelligent insights
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MarketInsight:
    """AI-generated market insight"""
    insight_type: str
    confidence: float
    content: str
    supporting_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    model_used: str = "gpt-4o"

@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result"""
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    key_themes: List[str]
    market_drivers: List[str]
    risk_factors: List[str]
    opportunity_indicators: List[str]

@dataclass
class NewsImpactAssessment:
    """AI assessment of news impact on markets"""
    impact_magnitude: float  # 0 to 1
    impact_direction: str    # positive, negative, neutral
    affected_assets: List[str]
    time_horizon: str       # immediate, short-term, long-term
    confidence: float
    reasoning: str

class OpenAIEnhancedAnalyzer:
    """Advanced market analysis using OpenAI GPT-4o"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = "gpt-4o"  # Latest OpenAI model as of May 13, 2024
        self.logger = logging.getLogger(__name__)
        
        # Analysis templates
        self.sentiment_prompt_template = """
        Analyze the following cryptocurrency market data and news for sentiment:
        
        Price Data: {price_data}
        Volume Data: {volume_data}
        News Headlines: {news_data}
        Social Media Mentions: {social_data}
        
        Provide a comprehensive sentiment analysis in JSON format:
        {{
            "overall_sentiment": "bullish|bearish|neutral",
            "sentiment_score": -1.0 to 1.0,
            "confidence": 0.0 to 1.0,
            "key_themes": ["theme1", "theme2", ...],
            "market_drivers": ["driver1", "driver2", ...],
            "risk_factors": ["risk1", "risk2", ...],
            "opportunity_indicators": ["opp1", "opp2", ...]
        }}
        
        Focus on actionable insights for cryptocurrency trading.
        """
        
        self.news_impact_template = """
        Assess the potential market impact of this news on cryptocurrency markets:
        
        News: {news_content}
        Current Market Context: {market_context}
        Affected Cryptocurrencies: {crypto_symbols}
        
        Provide impact assessment in JSON format:
        {{
            "impact_magnitude": 0.0 to 1.0,
            "impact_direction": "positive|negative|neutral",
            "affected_assets": ["BTC", "ETH", ...],
            "time_horizon": "immediate|short-term|long-term",
            "confidence": 0.0 to 1.0,
            "reasoning": "detailed explanation"
        }}
        
        Consider regulatory, technological, adoption, and market factors.
        """
        
        self.feature_engineering_template = """
        Generate advanced feature engineering suggestions for cryptocurrency price prediction:
        
        Current Features: {current_features}
        Market Data Available: {available_data}
        Target Prediction: {prediction_target}
        
        Suggest 10 intelligent features in JSON format:
        {{
            "features": [
                {{
                    "name": "feature_name",
                    "description": "detailed description",
                    "calculation": "mathematical formula or logic",
                    "predictive_value": "why this helps prediction",
                    "data_requirements": ["required data sources"],
                    "complexity": "low|medium|high"
                }}
            ]
        }}
        
        Focus on features that capture market psychology, regime changes, and non-linear patterns.
        """
    
    def analyze_market_sentiment(
        self,
        price_data: pd.DataFrame,
        news_data: Optional[List[str]] = None,
        social_data: Optional[List[str]] = None
    ) -> SentimentAnalysis:
        """Comprehensive AI-powered sentiment analysis"""
        
        try:
            # Prepare data summaries
            price_summary = self._summarize_price_data(price_data)
            volume_summary = self._summarize_volume_data(price_data)
            
            # Format prompt
            prompt = self.sentiment_prompt_template.format(
                price_data=price_summary,
                volume_data=volume_summary,
                news_data=news_data or ["No recent news data"],
                social_data=social_data or ["No social media data"]
            )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst with deep understanding of market psychology, technical analysis, and fundamental factors. Provide objective, data-driven sentiment analysis."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return SentimentAnalysis(
                overall_sentiment=result.get("overall_sentiment", "neutral"),
                sentiment_score=float(result.get("sentiment_score", 0.0)),
                confidence=float(result.get("confidence", 0.5)),
                key_themes=result.get("key_themes", []),
                market_drivers=result.get("market_drivers", []),
                risk_factors=result.get("risk_factors", []),
                opportunity_indicators=result.get("opportunity_indicators", [])
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return SentimentAnalysis(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=[],
                market_drivers=[],
                risk_factors=["Analysis failed"],
                opportunity_indicators=[]
            )
    
    def assess_news_impact(
        self,
        news_content: str,
        market_context: Dict[str, Any],
        crypto_symbols: List[str] = None
    ) -> NewsImpactAssessment:
        """AI-powered news impact assessment"""
        
        try:
            if crypto_symbols is None:
                crypto_symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]
            
            # Format prompt
            prompt = self.news_impact_template.format(
                news_content=news_content,
                market_context=json.dumps(market_context, indent=2),
                crypto_symbols=crypto_symbols
            )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency market expert specializing in news impact analysis. Assess how news events affect crypto markets with precise, actionable insights."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return NewsImpactAssessment(
                impact_magnitude=float(result.get("impact_magnitude", 0.0)),
                impact_direction=result.get("impact_direction", "neutral"),
                affected_assets=result.get("affected_assets", []),
                time_horizon=result.get("time_horizon", "short-term"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", "No analysis available")
            )
            
        except Exception as e:
            self.logger.error(f"News impact assessment failed: {e}")
            return NewsImpactAssessment(
                impact_magnitude=0.0,
                impact_direction="neutral",
                affected_assets=[],
                time_horizon="unknown",
                confidence=0.0,
                reasoning=f"Analysis failed: {e}"
            )
    
    def generate_intelligent_features(
        self,
        current_features: List[str],
        available_data: Dict[str, Any],
        prediction_target: str = "price_direction"
    ) -> List[Dict[str, Any]]:
        """AI-powered feature engineering suggestions"""
        
        try:
            # Format prompt
            prompt = self.feature_engineering_template.format(
                current_features=current_features,
                available_data=json.dumps(available_data, indent=2),
                prediction_target=prediction_target
            )
            
            # Call OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a machine learning engineer specializing in cryptocurrency prediction models. Generate innovative, mathematically sound feature engineering ideas that capture market dynamics."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4
            )
            
            # Parse response
            result = json.loads(response.choices[0].message.content)
            
            return result.get("features", [])
            
        except Exception as e:
            self.logger.error(f"Feature engineering generation failed: {e}")
            return []
    
    def analyze_market_anomalies(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        technical_indicators: Dict[str, pd.Series]
    ) -> List[MarketInsight]:
        """Detect and analyze market anomalies using AI"""
        
        insights = []
        
        try:
            # Prepare anomaly detection data
            anomaly_data = self._prepare_anomaly_data(price_data, volume_data, technical_indicators)
            
            anomaly_prompt = f"""
            Analyze this cryptocurrency market data for anomalies and unusual patterns:
            
            {json.dumps(anomaly_data, indent=2)}
            
            Identify significant anomalies and provide insights in JSON format:
            {{
                "anomalies": [
                    {{
                        "type": "price_spike|volume_surge|correlation_break|volatility_cluster",
                        "severity": 0.0 to 1.0,
                        "description": "detailed explanation",
                        "implications": "trading implications",
                        "confidence": 0.0 to 1.0
                    }}
                ]
            }}
            
            Focus on actionable trading insights and risk management implications.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative analyst expert in detecting market anomalies and unusual trading patterns in cryptocurrency markets."
                    },
                    {"role": "user", "content": anomaly_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to MarketInsight objects
            for anomaly in result.get("anomalies", []):
                insight = MarketInsight(
                    insight_type=anomaly.get("type", "unknown"),
                    confidence=float(anomaly.get("confidence", 0.5)),
                    content=f"{anomaly.get('description', '')}\n\nImplications: {anomaly.get('implications', '')}",
                    supporting_data={
                        "severity": anomaly.get("severity", 0.0),
                        "anomaly_type": anomaly.get("type", "unknown")
                    }
                )
                insights.append(insight)
                
        except Exception as e:
            self.logger.error(f"Anomaly analysis failed: {e}")
        
        return insights
    
    def generate_trading_strategy_insights(
        self,
        market_data: pd.DataFrame,
        current_positions: Dict[str, float],
        risk_parameters: Dict[str, float]
    ) -> List[MarketInsight]:
        """AI-powered trading strategy insights"""
        
        insights = []
        
        try:
            # Prepare strategy context
            strategy_context = {
                "market_summary": self._summarize_market_data(market_data),
                "position_summary": current_positions,
                "risk_parameters": risk_parameters,
                "recent_performance": self._calculate_recent_performance(market_data)
            }
            
            strategy_prompt = f"""
            Provide strategic trading insights based on current market conditions:
            
            Market Context: {json.dumps(strategy_context, indent=2)}
            
            Generate actionable insights in JSON format:
            {{
                "insights": [
                    {{
                        "category": "entry_opportunity|exit_signal|risk_management|portfolio_adjustment",
                        "priority": "high|medium|low",
                        "description": "detailed insight",
                        "rationale": "supporting reasoning",
                        "action_items": ["specific actions"],
                        "confidence": 0.0 to 1.0
                    }}
                ]
            }}
            
            Focus on practical, risk-aware trading decisions.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency trader with expertise in risk management, portfolio optimization, and market timing. Provide strategic insights for intelligent trading decisions."
                    },
                    {"role": "user", "content": strategy_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Convert to MarketInsight objects
            for insight_data in result.get("insights", []):
                insight = MarketInsight(
                    insight_type=insight_data.get("category", "general"),
                    confidence=float(insight_data.get("confidence", 0.5)),
                    content=f"{insight_data.get('description', '')}\n\nRationale: {insight_data.get('rationale', '')}",
                    supporting_data={
                        "priority": insight_data.get("priority", "medium"),
                        "action_items": insight_data.get("action_items", [])
                    }
                )
                insights.append(insight)
                
        except Exception as e:
            self.logger.error(f"Strategy insights generation failed: {e}")
        
        return insights
    
    def _summarize_price_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize price data for AI analysis"""
        
        if 'close' not in price_data.columns:
            return {"error": "No price data available"}
        
        prices = price_data['close']
        
        return {
            "current_price": float(prices.iloc[-1]) if len(prices) > 0 else 0,
            "price_change_24h": float(prices.pct_change().iloc[-1]) if len(prices) > 1 else 0,
            "price_change_7d": float((prices.iloc[-1] / prices.iloc[-7] - 1)) if len(prices) > 7 else 0,
            "volatility_7d": float(prices.pct_change().rolling(7).std().iloc[-1]) if len(prices) > 7 else 0,
            "recent_high": float(prices.rolling(30).max().iloc[-1]) if len(prices) > 30 else float(prices.max()),
            "recent_low": float(prices.rolling(30).min().iloc[-1]) if len(prices) > 30 else float(prices.min()),
            "trend_direction": "up" if len(prices) > 10 and prices.iloc[-1] > prices.iloc[-10] else "down"
        }
    
    def _summarize_volume_data(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize volume data for AI analysis"""
        
        if 'volume' not in price_data.columns:
            return {"error": "No volume data available"}
        
        volumes = price_data['volume']
        
        return {
            "current_volume": float(volumes.iloc[-1]) if len(volumes) > 0 else 0,
            "avg_volume_7d": float(volumes.rolling(7).mean().iloc[-1]) if len(volumes) > 7 else 0,
            "volume_trend": "increasing" if len(volumes) > 5 and volumes.iloc[-1] > volumes.rolling(5).mean().iloc[-1] else "decreasing",
            "volume_spike": float(volumes.iloc[-1] / volumes.rolling(20).mean().iloc[-1]) if len(volumes) > 20 else 1.0
        }
    
    def _prepare_anomaly_data(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        technical_indicators: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Prepare data for anomaly detection analysis"""
        
        # Calculate recent statistics
        recent_data = {}
        
        if 'close' in price_data.columns:
            prices = price_data['close']
            recent_data["price_stats"] = {
                "recent_returns": prices.pct_change().tail(10).tolist(),
                "volatility_recent": float(prices.pct_change().rolling(5).std().iloc[-1]) if len(prices) > 5 else 0,
                "price_z_score": float((prices.iloc[-1] - prices.rolling(20).mean().iloc[-1]) / prices.rolling(20).std().iloc[-1]) if len(prices) > 20 else 0
            }
        
        if 'volume' in volume_data.columns:
            volumes = volume_data['volume']
            recent_data["volume_stats"] = {
                "volume_z_score": float((volumes.iloc[-1] - volumes.rolling(20).mean().iloc[-1]) / volumes.rolling(20).std().iloc[-1]) if len(volumes) > 20 else 0,
                "volume_trend": volumes.pct_change().tail(5).tolist()
            }
        
        # Add technical indicator anomalies
        recent_data["indicator_stats"] = {}
        for name, series in technical_indicators.items():
            if len(series) > 10:
                recent_data["indicator_stats"][name] = {
                    "current_value": float(series.iloc[-1]),
                    "percentile_rank": float(series.rank(pct=True).iloc[-1]),
                    "recent_change": float(series.pct_change().iloc[-1]) if len(series) > 1 else 0
                }
        
        return recent_data
    
    def _summarize_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize overall market data"""
        
        summary = {}
        
        if 'close' in market_data.columns:
            prices = market_data['close']
            summary["price_summary"] = self._summarize_price_data(market_data)
        
        if 'volume' in market_data.columns:
            summary["volume_summary"] = self._summarize_volume_data(market_data)
        
        return summary
    
    def _calculate_recent_performance(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        
        if 'close' not in market_data.columns or len(market_data) < 10:
            return {"error": "Insufficient data"}
        
        prices = market_data['close']
        returns = prices.pct_change().dropna()
        
        return {
            "total_return_7d": float((prices.iloc[-1] / prices.iloc[-7] - 1)) if len(prices) > 7 else 0,
            "volatility_7d": float(returns.tail(7).std()),
            "sharpe_7d": float(returns.tail(7).mean() / returns.tail(7).std()) if returns.tail(7).std() > 0 else 0,
            "max_drawdown_7d": float((prices.tail(7).max() - prices.tail(7).min()) / prices.tail(7).max())
        }

def create_openai_analyzer() -> OpenAIEnhancedAnalyzer:
    """Create OpenAI enhanced analyzer"""
    return OpenAIEnhancedAnalyzer()

def analyze_market_with_ai(
    price_data: pd.DataFrame,
    news_data: Optional[List[str]] = None,
    include_sentiment: bool = True,
    include_anomalies: bool = True,
    include_strategy: bool = True
) -> Dict[str, Any]:
    """High-level function for comprehensive AI market analysis"""
    
    analyzer = create_openai_analyzer()
    results = {}
    
    try:
        if include_sentiment:
            sentiment = analyzer.analyze_market_sentiment(price_data, news_data)
            results["sentiment_analysis"] = sentiment
        
        if include_anomalies:
            # Prepare technical indicators for anomaly detection
            technical_indicators = {
                "sma_20": price_data['close'].rolling(20).mean(),
                "rsi": _calculate_rsi(price_data['close']),
                "volatility": price_data['close'].pct_change().rolling(10).std()
            }
            
            anomalies = analyzer.analyze_market_anomalies(
                price_data, price_data[['volume']] if 'volume' in price_data.columns else pd.DataFrame(),
                technical_indicators
            )
            results["anomaly_analysis"] = anomalies
        
        if include_strategy:
            strategy_insights = analyzer.generate_trading_strategy_insights(
                price_data, {}, {"max_position_size": 0.1, "max_risk_per_trade": 0.02}
            )
            results["strategy_insights"] = strategy_insights
        
        return results
        
    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi