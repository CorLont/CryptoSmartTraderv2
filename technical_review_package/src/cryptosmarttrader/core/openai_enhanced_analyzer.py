"""
CryptoSmartTrader V2 - OpenAI Enhanced Analyzer
Advanced AI-powered analysis using GPT-4o for enhanced insights
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# OpenAI integration with blueprint compliance
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_MODEL = "gpt-4o"


class OpenAIEnhancedAnalyzer:
    """Enhanced analyzer using OpenAI GPT-4o for advanced insights"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OpenAI API key not found. Enhanced analysis will be limited.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")

        # Analysis configurations
        self.batch_configs = {
            "sentiment_analysis": {
                "max_tokens": 1500,
                "temperature": 0.3,
                "system_prompt": self._get_sentiment_system_prompt(),
            },
            "technical_analysis": {
                "max_tokens": 2000,
                "temperature": 0.2,
                "system_prompt": self._get_technical_system_prompt(),
            },
            "ml_enhancement": {
                "max_tokens": 1800,
                "temperature": 0.1,
                "system_prompt": self._get_ml_system_prompt(),
            },
            "market_insights": {
                "max_tokens": 2500,
                "temperature": 0.4,
                "system_prompt": self._get_market_system_prompt(),
            },
        }

    def _get_sentiment_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis"""
        return """You are an expert cryptocurrency sentiment analyst with deep understanding of market psychology and social media dynamics.

Your task is to analyze cryptocurrency-related social media posts, news articles, and market discussions to provide:
1. Detailed sentiment analysis with confidence scores
2. Market psychology insights
3. Emotional indicators (fear, greed, FOMO, panic)
4. Trend predictions based on sentiment patterns
5. Risk assessment from community sentiment

Provide analysis in structured JSON format with specific metrics and actionable insights.
Focus on institutional-grade analysis suitable for professional trading decisions."""

    def _get_technical_system_prompt(self) -> str:
        """Get system prompt for technical analysis"""
        return """You are a professional technical analyst specializing in cryptocurrency markets with expertise in pattern recognition and signal interpretation.

Your role is to enhance technical analysis by:
1. Interpreting complex technical indicator combinations
2. Identifying advanced chart patterns and market structures
3. Providing confluence analysis across multiple timeframes
4. Generating actionable trading signals with risk parameters
5. Explaining market psychology behind technical patterns

Deliver institutional-quality analysis with clear entry/exit points, stop-loss levels, and probability assessments.
Use JSON format for structured recommendations."""

    def _get_ml_system_prompt(self) -> str:
        """Get system prompt for ML enhancement"""
        return """You are an expert in machine learning applications for financial markets, specializing in cryptocurrency price prediction and model optimization.

Your responsibilities include:
1. Analyzing ML model outputs for pattern recognition
2. Identifying prediction confidence and reliability factors
3. Suggesting model improvements and feature engineering
4. Detecting overfitting or model degradation
5. Providing ensemble recommendations and risk adjustments

Focus on practical insights that improve prediction accuracy and trading performance.
Deliver analysis in structured format with actionable recommendations."""

    def _get_market_system_prompt(self) -> str:
        """Get system prompt for comprehensive market insights"""
        return """You are a senior cryptocurrency market analyst with expertise in macro trends, institutional behavior, and market microstructure.

Provide comprehensive market analysis including:
1. Macro trend identification and impact assessment
2. Institutional flow analysis and smart money movements
3. Market regime detection (bull/bear/sideways/transition)
4. Cross-asset correlation analysis
5. Regulatory and fundamental impact assessment
6. Strategic positioning recommendations

Deliver professional-grade insights suitable for institutional investment decisions.
Structure output in actionable format with clear recommendations and risk parameters."""

    async def enhance_await get_sentiment_analyzer().analyze_text(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance sentiment analysis with OpenAI insights"""
        if not self.client:
            return self._fallback_await get_sentiment_analyzer().analyze_text(sentiment_data)

        try:
            # Prepare sentiment data for analysis
            analysis_prompt = self._prepare_sentiment_prompt(sentiment_data)

            # Get enhanced analysis from OpenAI
            response = await self._call_openai_async(
                prompt=analysis_prompt, config=self.batch_configs["sentiment_analysis"]
            )

            # Parse and structure the response
            enhanced_results = self._parse_sentiment_response(response, sentiment_data)

            self.logger.info("Sentiment analysis enhanced with OpenAI insights")
            return enhanced_results

        except Exception as e:
            self.logger.error(f"OpenAI sentiment enhancement failed: {e}")
            return self._fallback_await get_sentiment_analyzer().analyze_text(sentiment_data)

    async def enhance_technical_analysis(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance technical analysis with AI pattern recognition"""
        if not self.client:
            return self._fallback_technical_analysis(technical_data)

        try:
            analysis_prompt = self._prepare_technical_prompt(technical_data)

            response = await self._call_openai_async(
                prompt=analysis_prompt, config=self.batch_configs["technical_analysis"]
            )

            enhanced_results = self._parse_technical_response(response, technical_data)

            self.logger.info("Technical analysis enhanced with OpenAI pattern recognition")
            return enhanced_results

        except Exception as e:
            self.logger.error(f"OpenAI technical enhancement failed: {e}")
            return self._fallback_technical_analysis(technical_data)

    async def enhance_ml_predictions(self, ml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ML predictions with AI model analysis"""
        if not self.client:
            return self._fallback_ml_analysis(ml_data)

        try:
            analysis_prompt = self._prepare_ml_prompt(ml_data)

            response = await self._call_openai_async(
                prompt=analysis_prompt, config=self.batch_configs["ml_enhancement"]
            )

            enhanced_results = self._parse_ml_response(response, ml_data)

            self.logger.info("ML predictions enhanced with OpenAI model analysis")
            return enhanced_results

        except Exception as e:
            self.logger.error(f"OpenAI ML enhancement failed: {e}")
            return self._fallback_ml_analysis(ml_data)

    async def generate_market_insights(self, combined_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive market insights from all analysis types"""
        if not self.client:
            return self._fallback_market_insights(combined_data)

        try:
            insight_prompt = self._prepare_market_insights_prompt(combined_data)

            response = await self._call_openai_async(
                prompt=insight_prompt, config=self.batch_configs["market_insights"]
            )

            market_insights = self._parse_market_insights_response(response, combined_data)

            self.logger.info("Comprehensive market insights generated")
            return market_insights

        except Exception as e:
            self.logger.error(f"OpenAI market insights generation failed: {e}")
            return self._fallback_market_insights(combined_data)

    async def _call_openai_async(self, prompt: str, config: Dict[str, Any]) -> str:
        """Make async call to OpenAI API"""
        if not self.client:
            raise ValueError("OpenAI client not initialized")

        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=config["max_tokens"],
                temperature=config["temperature"],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("OpenAI response content is None")

            return content

        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise

    def _prepare_sentiment_prompt(self, sentiment_data: Dict[str, Any]) -> str:
        """Prepare prompt for sentiment analysis enhancement"""
        return f"""
Analyze the following cryptocurrency sentiment data and provide enhanced insights:

Raw Sentiment Data:
{json.dumps(sentiment_data, indent=2)}

Please provide:
1. Enhanced sentiment score with confidence interval
2. Emotional market indicators (fear, greed, FOMO levels)
3. Sentiment trend analysis and momentum
4. Community behavior patterns
5. Risk assessment from sentiment perspective
6. Short-term sentiment predictions
7. Trading recommendations based on sentiment

Format response as JSON with specific metrics and actionable insights.
"""

    def _prepare_technical_prompt(self, technical_data: Dict[str, Any]) -> str:
        """Prepare prompt for technical analysis enhancement"""
        return f"""
Enhance the following technical analysis with advanced pattern recognition:

Technical Data:
{json.dumps(technical_data, indent=2)}

Please provide:
1. Advanced pattern recognition and market structure analysis
2. Multi-timeframe confluence assessment
3. Support/resistance level validation
4. Signal strength and reliability scoring
5. Entry/exit point recommendations with risk parameters
6. Market regime analysis (trend/range/breakout)
7. Probability-weighted scenarios

Format response as JSON with clear trading signals and risk management.
"""

    def _prepare_ml_prompt(self, ml_data: Dict[str, Any]) -> str:
        """Prepare prompt for ML analysis enhancement"""
        return f"""
Analyze and enhance the following machine learning predictions:

ML Prediction Data:
{json.dumps(ml_data, indent=2)}

Please provide:
1. Model confidence assessment and reliability analysis
2. Feature importance interpretation
3. Prediction accuracy validation
4. Model performance optimization suggestions
5. Ensemble recommendation improvements
6. Risk-adjusted prediction ranges
7. Model degradation detection

Format response as JSON with actionable model insights.
"""

    def _prepare_market_insights_prompt(self, combined_data: Dict[str, Any]) -> str:
        """Prepare prompt for comprehensive market insights"""
        return f"""
Generate comprehensive market insights from the following combined analysis data:

Combined Analysis Data:
{json.dumps(combined_data, indent=2)}

Please provide:
1. Overall market trend and regime assessment
2. Cross-analysis confluence and contradictions
3. Institutional vs retail sentiment analysis
4. Strategic positioning recommendations
5. Risk management framework
6. Scenario analysis with probability weights
7. Key catalyst identification and impact assessment
8. Portfolio allocation suggestions

Format response as JSON with executive summary and detailed recommendations.
"""

    def _parse_sentiment_response(self, response: str, original_data: Dict) -> Dict[str, Any]:
        """Parse OpenAI sentiment analysis response"""
        try:
            ai_analysis = json.loads(response)

            return {
                "original_data": original_data,
                "ai_enhanced": ai_analysis,
                "enhancement_timestamp": datetime.now().isoformat(),
                "enhancement_type": "openai_sentiment",
                "confidence_boost": True,
                "actionable_insights": ai_analysis.get("trading_recommendations", []),
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response", "raw_response": response}

    def _parse_technical_response(self, response: str, original_data: Dict) -> Dict[str, Any]:
        """Parse OpenAI technical analysis response"""
        try:
            ai_analysis = json.loads(response)

            return {
                "original_data": original_data,
                "ai_enhanced": ai_analysis,
                "enhancement_timestamp": datetime.now().isoformat(),
                "enhancement_type": "openai_technical",
                "pattern_recognition": ai_analysis.get("patterns", []),
                "trading_signals": ai_analysis.get("signals", []),
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response", "raw_response": response}

    def _parse_ml_response(self, response: str, original_data: Dict) -> Dict[str, Any]:
        """Parse OpenAI ML enhancement response"""
        try:
            ai_analysis = json.loads(response)

            return {
                "original_data": original_data,
                "ai_enhanced": ai_analysis,
                "enhancement_timestamp": datetime.now().isoformat(),
                "enhancement_type": "openai_ml",
                "model_insights": ai_analysis.get("model_analysis", {}),
                "optimization_suggestions": ai_analysis.get("improvements", []),
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response", "raw_response": response}

    def _parse_market_insights_response(self, response: str, original_data: Dict) -> Dict[str, Any]:
        """Parse OpenAI market insights response"""
        try:
            ai_insights = json.loads(response)

            return {
                "original_data": original_data,
                "ai_insights": ai_insights,
                "generation_timestamp": datetime.now().isoformat(),
                "insight_type": "comprehensive_market_analysis",
                "executive_summary": ai_insights.get("executive_summary", ""),
                "strategic_recommendations": ai_insights.get("recommendations", []),
                "risk_assessment": ai_insights.get("risk_framework", {}),
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse OpenAI response", "raw_response": response}

    # Fallback methods for when OpenAI is not available
    def _fallback_await get_sentiment_analyzer().analyze_text(self, sentiment_data: Dict) -> Dict[str, Any]:
        """Fallback sentiment analysis without OpenAI"""
        return {
            "original_data": sentiment_data,
            "fallback_analysis": {
                "sentiment_score": sentiment_data.get("sentiment_score", 0.5),
                "confidence": 0.6,
                "note": "Basic analysis without OpenAI enhancement",
            },
            "enhancement_type": "fallback_sentiment",
        }

    def _fallback_technical_analysis(self, technical_data: Dict) -> Dict[str, Any]:
        """Fallback technical analysis without OpenAI"""
        return {
            "original_data": technical_data,
            "fallback_analysis": {
                "signal_strength": "moderate",
                "confidence": 0.6,
                "note": "Basic analysis without OpenAI enhancement",
            },
            "enhancement_type": "fallback_technical",
        }

    def _fallback_ml_analysis(self, ml_data: Dict) -> Dict[str, Any]:
        """Fallback ML analysis without OpenAI"""
        return {
            "original_data": ml_data,
            "fallback_analysis": {
                "prediction_confidence": 0.6,
                "note": "Basic analysis without OpenAI enhancement",
            },
            "enhancement_type": "fallback_ml",
        }

    def _fallback_market_insights(self, combined_data: Dict) -> Dict[str, Any]:
        """Fallback market insights without OpenAI"""
        return {
            "original_data": combined_data,
            "fallback_insights": {
                "market_trend": "neutral",
                "confidence": 0.5,
                "note": "Basic analysis without OpenAI enhancement",
            },
            "insight_type": "fallback_market_analysis",
        }

    async def process_analysis_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a complete analysis batch with OpenAI enhancement"""
        try:
            results = {}

            # Process each analysis type
            if "sentiment" in batch_data:
                results["enhanced_sentiment"] = await self.enhance_await get_sentiment_analyzer().analyze_text(
                    batch_data["sentiment"]
                )

            if "technical" in batch_data:
                results["enhanced_technical"] = await self.enhance_technical_analysis(
                    batch_data["technical"]
                )

            if "ml_predictions" in batch_data:
                results["enhanced_ml"] = await self.enhance_ml_predictions(
                    batch_data["ml_predictions"]
                )

            # Generate comprehensive insights
            results["market_insights"] = await self.generate_market_insights(batch_data)

            # Add batch metadata
            results["batch_metadata"] = {
                "processed_at": datetime.now().isoformat(),
                "analysis_types": list(batch_data.keys()),
                "enhancement_status": "openai_enhanced" if self.client else "fallback",
                "total_processing_time": "calculated_elsewhere",
            }

            return results

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {"error": str(e), "batch_data": batch_data}
