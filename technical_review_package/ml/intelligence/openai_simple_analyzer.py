#!/usr/bin/env python3
"""
OpenAI Simple Analyzer
Simplified AI-powered market analysis using GPT-4o without complex dependencies
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

warnings.filterwarnings("ignore")


@dataclass
class AIMarketInsight:
    """AI-generated market insight"""

    insight_type: str
    confidence: float
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass
class AISentimentResult:
    """AI sentiment analysis result"""

    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1 to 1
    confidence: float
    key_themes: List[str]
    market_drivers: List[str]


class OpenAISimpleAnalyzer:
    """Simplified OpenAI market analyzer without complex dependencies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")

        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"  # Using GPT-4o as the latest available model - GPT-5 not yet publicly available

        self.logger.info("OpenAI Simple Analyzer initialized with GPT-4o")

    def analyze_market_sentiment(
        self, price_data: pd.DataFrame, news_headlines: Optional[List[str]] = None
    ) -> AISentimentResult:
        """Analyze market sentiment using AI"""

        try:
            # Prepare market summary
            market_summary = self._create_market_summary(price_data)
            news_text = "\n".join(news_headlines) if news_headlines else "No recent news"

            prompt = f"""
            Analyze cryptocurrency market sentiment based on the following data:
            
            MARKET DATA:
            {json.dumps(market_summary, indent=2)}
            
            RECENT NEWS:
            {news_text}
            
            Provide sentiment analysis in JSON format:
            {{
                "overall_sentiment": "bullish|bearish|neutral",
                "sentiment_score": -1.0 to 1.0,
                "confidence": 0.0 to 1.0,
                "key_themes": ["theme1", "theme2", "theme3"],
                "market_drivers": ["driver1", "driver2", "driver3"]
            }}
            
            Focus on actionable insights for cryptocurrency trading decisions.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cryptocurrency market analyst. Provide objective, data-driven sentiment analysis for trading decisions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)

            return AISentimentResult(
                overall_sentiment=result.get("overall_sentiment", "neutral"),
                sentiment_score=float(result.get("sentiment_score", 0.0)),
                confidence=float(result.get("confidence", 0.5)),
                key_themes=result.get("key_themes", []),
                market_drivers=result.get("market_drivers", []),
            )

        except Exception as e:
            self.logger.error(f"AI sentiment analysis failed: {e}")
            return AISentimentResult(
                overall_sentiment="neutral",
                sentiment_score=0.0,
                confidence=0.0,
                key_themes=[],
                market_drivers=[],
            )

    def assess_news_impact(
        self, news_headline: str, news_content: str, crypto_symbols: List[str] = None
    ) -> Dict[str, Any]:
        """Assess news impact on cryptocurrency markets"""

        try:
            if crypto_symbols is None:
                crypto_symbols = ["BTC", "ETH", "ADA", "DOT", "LINK"]

            prompt = f"""
            Assess the market impact of this cryptocurrency news:
            
            HEADLINE: {news_headline}
            CONTENT: {news_content}
            CRYPTOCURRENCIES: {", ".join(crypto_symbols)}
            
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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cryptocurrency news impact specialist. Assess how news affects crypto markets with precise analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            self.logger.error(f"News impact assessment failed: {e}")
            return {
                "impact_magnitude": 0.0,
                "impact_direction": "neutral",
                "affected_assets": [],
                "time_horizon": "unknown",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {e}",
            }

    def generate_trading_insights(
        self, market_data: pd.DataFrame, current_positions: Dict[str, float] = None
    ) -> List[AIMarketInsight]:
        """Generate AI-powered trading insights"""

        insights = []

        try:
            market_summary = self._create_market_summary(market_data)
            positions_summary = current_positions or {}

            prompt = f"""
            Generate strategic cryptocurrency trading insights based on:
            
            MARKET DATA:
            {json.dumps(market_summary, indent=2)}
            
            CURRENT POSITIONS:
            {json.dumps(positions_summary, indent=2)}
            
            Provide 3-5 actionable insights in JSON format:
            {{
                "insights": [
                    {{
                        "type": "entry_opportunity|exit_signal|risk_management|portfolio_adjustment",
                        "priority": "high|medium|low",
                        "description": "detailed insight",
                        "reasoning": "supporting analysis",
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
                        "content": "You are a professional cryptocurrency trader. Generate practical trading insights based on market analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)

            for insight_data in result.get("insights", []):
                insight = AIMarketInsight(
                    insight_type=insight_data.get("type", "general"),
                    confidence=float(insight_data.get("confidence", 0.5)),
                    content=f"{insight_data.get('description', '')}\n\nReasoning: {insight_data.get('reasoning', '')}",
                )
                insights.append(insight)

        except Exception as e:
            self.logger.error(f"Trading insights generation failed: {e}")

        return insights

    def detect_market_anomalies(
        self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None
    ) -> List[AIMarketInsight]:
        """Detect market anomalies using AI analysis"""

        insights = []

        try:
            # Prepare anomaly data
            anomaly_data = self._prepare_anomaly_analysis(price_data, volume_data)

            prompt = f"""
            Analyze this cryptocurrency market data for significant anomalies:
            
            {json.dumps(anomaly_data, indent=2)}
            
            Identify notable anomalies in JSON format:
            {{
                "anomalies": [
                    {{
                        "type": "price_spike|volume_surge|volatility_cluster|trend_break",
                        "severity": 0.0 to 1.0,
                        "description": "detailed explanation",
                        "implications": "trading implications",
                        "confidence": 0.0 to 1.0
                    }}
                ]
            }}
            
            Focus on actionable trading insights and risk management.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a quantitative analyst expert in detecting cryptocurrency market anomalies and unusual patterns.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )

            result = json.loads(response.choices[0].message.content)

            for anomaly in result.get("anomalies", []):
                insight = AIMarketInsight(
                    insight_type=anomaly.get("type", "anomaly"),
                    confidence=float(anomaly.get("confidence", 0.5)),
                    content=f"{anomaly.get('description', '')}\n\nImplications: {anomaly.get('implications', '')}",
                )
                insights.append(insight)

        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")

        return insights

    def generate_feature_suggestions(
        self, current_features: List[str], prediction_target: str = "price_direction"
    ) -> List[Dict[str, str]]:
        """Generate intelligent feature engineering suggestions"""

        try:
            prompt = f"""
            Suggest advanced features for cryptocurrency price prediction:
            
            CURRENT FEATURES: {", ".join(current_features)}
            PREDICTION TARGET: {prediction_target}
            
            Generate 10 innovative features in JSON format:
            {{
                "features": [
                    {{
                        "name": "feature_name",
                        "description": "clear description",
                        "calculation": "how to calculate",
                        "predictive_value": "why it helps prediction",
                        "complexity": "low|medium|high"
                    }}
                ]
            }}
            
            Focus on features that capture market psychology and non-linear patterns.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a machine learning engineer specializing in cryptocurrency prediction. Generate innovative feature ideas.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )

            result = json.loads(response.choices[0].message.content)
            return result.get("features", [])

        except Exception as e:
            self.logger.error(f"Feature suggestion generation failed: {e}")
            return []

    def _create_market_summary(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Create market data summary for AI analysis"""

        if "close" not in price_data.columns:
            return {"error": "No price data available"}

        prices = price_data["close"]

        # Calculate basic statistics
        current_price = float(prices.iloc[-1]) if len(prices) > 0 else 0
        price_change_24h = float(prices.pct_change().iloc[-1]) if len(prices) > 1 else 0

        # Calculate trend
        if len(prices) > 10:
            recent_trend = "up" if prices.iloc[-1] > prices.iloc[-10] else "down"
            trend_strength = abs((prices.iloc[-1] / prices.iloc[-10] - 1))
        else:
            recent_trend = "neutral"
            trend_strength = 0.0

        # Calculate volatility
        if len(prices) > 7:
            volatility_7d = float(prices.pct_change().rolling(7).std().iloc[-1])
        else:
            volatility_7d = 0.0

        # Volume analysis
        volume_info = {}
        if "volume" in price_data.columns:
            volumes = price_data["volume"]
            volume_info = {
                "current_volume": float(volumes.iloc[-1]) if len(volumes) > 0 else 0,
                "volume_trend": "increasing"
                if len(volumes) > 5 and volumes.iloc[-1] > volumes.rolling(5).mean().iloc[-1]
                else "decreasing",
            }

        summary = {
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "recent_trend": recent_trend,
            "trend_strength": trend_strength,
            "volatility_7d": volatility_7d,
            "data_points": len(prices),
        }

        summary.update(volume_info)
        return summary

    def _prepare_anomaly_analysis(
        self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Prepare data for anomaly detection"""

        anomaly_data = {}

        if "close" in price_data.columns:
            prices = price_data["close"]

            # Price statistics
            if len(prices) > 20:
                price_mean = prices.rolling(20).mean().iloc[-1]
                price_std = prices.rolling(20).std().iloc[-1]
                current_z_score = (prices.iloc[-1] - price_mean) / price_std if price_std > 0 else 0

                anomaly_data["price_analysis"] = {
                    "current_z_score": float(current_z_score),
                    "recent_returns": prices.pct_change().tail(5).tolist(),
                    "volatility_recent": float(prices.pct_change().rolling(5).std().iloc[-1])
                    if len(prices) > 5
                    else 0,
                }

        # Volume statistics
        if volume_data is not None and "volume" in volume_data.columns:
            volumes = volume_data["volume"]

            if len(volumes) > 20:
                volume_mean = volumes.rolling(20).mean().iloc[-1]
                volume_std = volumes.rolling(20).std().iloc[-1]
                volume_z_score = (
                    (volumes.iloc[-1] - volume_mean) / volume_std if volume_std > 0 else 0
                )

                anomaly_data["volume_analysis"] = {
                    "volume_z_score": float(volume_z_score),
                    "volume_spike_ratio": float(volumes.iloc[-1] / volume_mean)
                    if volume_mean > 0
                    else 1.0,
                }

        return anomaly_data


def create_simple_ai_analyzer() -> OpenAISimpleAnalyzer:
    """Create simplified OpenAI analyzer"""
    return OpenAISimpleAnalyzer()


def quick_ai_analysis(
    price_data: pd.DataFrame, news_headlines: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Quick AI-powered market analysis"""

    try:
        analyzer = create_simple_ai_analyzer()

        # Perform sentiment analysis
        sentiment = analyzer.analyze_market_sentiment(price_data, news_headlines)

        # Generate trading insights
        insights = analyzer.generate_trading_insights(price_data)

        # Detect anomalies
        anomalies = analyzer.detect_market_anomalies(price_data)

        return {
            "sentiment": {
                "overall": sentiment.overall_sentiment,
                "score": sentiment.sentiment_score,
                "confidence": sentiment.confidence,
                "themes": sentiment.key_themes,
                "drivers": sentiment.market_drivers,
            },
            "trading_insights": [
                {
                    "type": insight.insight_type,
                    "content": insight.content,
                    "confidence": insight.confidence,
                }
                for insight in insights
            ],
            "anomalies": [
                {
                    "type": anomaly.insight_type,
                    "content": anomaly.content,
                    "confidence": anomaly.confidence,
                }
                for anomaly in anomalies
            ],
        }

    except Exception as e:
        return {"error": f"AI analysis failed: {e}"}
