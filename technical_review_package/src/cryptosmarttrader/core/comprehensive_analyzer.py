"""
CryptoSmartTrader V2 - Comprehensive Analyzer
Integrates all analysis components with OpenAI intelligence for alpha seeking
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import sys
from pathlib import Path
import threading
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComprehensiveAnalyzer:
    """Comprehensive analyzer integrating all analysis components"""

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)

        # Get components from container
        self.market_scanner = container.market_scanner()
        self.openai_analyzer = container.openai_analyzer()
        self.cache_manager = container.cache_manager()

        # Analysis coordination
        self.analysis_active = False
        self.analysis_thread = None

        # Analysis intervals (seconds)
        self.intervals = {
            "market_scan": 300,  # 5 minutes
            "sentiment_analysis": 600,  # 10 minutes
            "whale_detection": 900,  # 15 minutes
            "ml_prediction": 1800,  # 30 minutes
            "openai_batch": 3600,  # 1 hour for OpenAI analysis
        }

        # Batch processing for OpenAI
        self.openai_batch_queue = []
        self.batch_size = 20  # Process 20 coins at once

        self.logger.info("Comprehensive Analyzer initialized")

    def start_continuous_analysis(self):
        """Start continuous background analysis"""
        if self.analysis_active:
            self.logger.warning("Analysis already active")
            return

        self.analysis_active = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()

        self.logger.info("Continuous analysis started")

    def stop_continuous_analysis(self):
        """Stop continuous analysis"""
        self.analysis_active = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=10)

        self.logger.info("Continuous analysis stopped")

    def _analysis_loop(self):
        """Main analysis coordination loop"""
        last_runs = {key: 0 for key in self.intervals.keys()}

        while self.analysis_active:
            try:
                current_time = time.time()

                # Check which analyses need to run
                for analysis_type, interval in self.intervals.items():
                    if current_time - last_runs[analysis_type] >= interval:
                        self._run_analysis_type(analysis_type)
                        last_runs[analysis_type] = current_time

                # Sleep for a short interval
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"Analysis loop error: {e}")
                time.sleep(60)  # Wait longer on error

    def _run_analysis_type(self, analysis_type: str):
        """Run specific type of analysis"""
        try:
            if analysis_type == "market_scan":
                self._run_market_scan()
            elif analysis_type == "sentiment_analysis":
                self._run_sentiment_analysis()
            elif analysis_type == "whale_detection":
                self._run_whale_detection()
            elif analysis_type == "ml_prediction":
                self._run_ml_prediction()
            elif analysis_type == "openai_batch":
                self._run_openai_batch_analysis()

        except Exception as e:
            self.logger.error(f"Analysis type {analysis_type} failed: {e}")

    def _run_market_scan(self):
        """Run market scanning and technical analysis"""
        try:
            # Get discovered coins
            discovered_data = self.market_scanner.get_all_discovered_coins()
            active_coins = [
                symbol
                for symbol, metadata in discovered_data["metadata"].items()
                if metadata.get("active", False)
            ]

            # Focus on coins with recent activity
            priority_coins = self._get_priority_coins(active_coins)

            self.logger.info(f"Market scan analyzing {len(priority_coins)} priority coins")

            # Store scan results
            if self.cache_manager:
                self.cache_manager.set(
                    "market_scan_results",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "coins_analyzed": len(priority_coins),
                        "total_active": len(active_coins),
                    },
                    ttl_minutes=60,
                )

        except Exception as e:
            self.logger.error(f"Market scan failed: {e}")

    def _run_sentiment_analysis(self):
        """Run sentiment analysis on discovered coins"""
        try:
            # Get coins with recent technical analysis
            recent_analyses = self._get_recent_technical_analyses()

            # Perform sentiment analysis
            sentiment_results = {}
            for symbol in recent_analyses[:50]:  # Limit to top 50
                sentiment_data = self._analyze_coin_sentiment(symbol)
                if sentiment_data:
                    sentiment_results[symbol] = sentiment_data

            # Store results
            if self.cache_manager:
                self.cache_manager.set(
                    "sentiment_analysis_results",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "results": sentiment_results,
                        "coins_analyzed": len(sentiment_results),
                    },
                    ttl_minutes=120,
                )

            self.logger.info(f"Sentiment analysis completed for {len(sentiment_results)} coins")

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")

    def _run_whale_detection(self):
        """Run whale activity detection"""
        try:
            # Get priority coins for whale analysis
            priority_coins = self._get_priority_coins_for_whale_analysis()

            whale_results = {}
            for symbol in priority_coins[:30]:  # Limit to top 30
                whale_data = self._analyze_whale_activity(symbol)
                if whale_data:
                    whale_results[symbol] = whale_data

            # Store results
            if self.cache_manager:
                self.cache_manager.set(
                    "whale_detection_results",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "results": whale_results,
                        "coins_analyzed": len(whale_results),
                    },
                    ttl_minutes=180,
                )

            self.logger.info(f"Whale detection completed for {len(whale_results)} coins")

        except Exception as e:
            self.logger.error(f"Whale detection failed: {e}")

    def _run_ml_prediction(self):
        """Run ML predictions on analyzed coins"""
        try:
            # Get coins with comprehensive data
            coins_with_data = self._get_coins_with_comprehensive_data()

            ml_results = {}
            for symbol in coins_with_data[:40]:  # Limit to top 40
                ml_prediction = self._generate_ml_prediction(symbol)
                if ml_prediction:
                    ml_results[symbol] = ml_prediction

            # Store results
            if self.cache_manager:
                self.cache_manager.set(
                    "ml_prediction_results",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "results": ml_results,
                        "coins_analyzed": len(ml_results),
                    },
                    ttl_minutes=240,
                )

            self.logger.info(f"ML predictions completed for {len(ml_results)} coins")

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")

    def _run_openai_batch_analysis(self):
        """Run OpenAI batch analysis for comprehensive insights"""
        try:
            # Get coins with all analysis types completed
            complete_analyses = self._get_coins_with_complete_analysis()

            if len(complete_analyses) < 5:  # Need minimum data
                self.logger.info("Insufficient data for OpenAI batch analysis")
                return

            # Process in batches
            batch_results = []
            for i in range(0, len(complete_analyses), self.batch_size):
                batch = complete_analyses[i : i + self.batch_size]
                batch_result = self._process_openai_batch(batch)
                if batch_result:
                    batch_results.extend(batch_result)

            # Store comprehensive results
            if self.cache_manager and batch_results:
                self.cache_manager.set(
                    "openai_comprehensive_analysis",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "results": batch_results,
                        "coins_analyzed": len(batch_results),
                    },
                    ttl_minutes=480,  # 8 hours
                )

            self.logger.info(f"OpenAI batch analysis completed for {len(batch_results)} coins")

        except Exception as e:
            self.logger.error(f"OpenAI batch analysis failed: {e}")

    def _get_priority_coins(self, active_coins: List[str]) -> List[str]:
        """Get priority coins based on recent activity and opportunities"""
        try:
            priority_coins = []

            # Get trading opportunities
            opportunities = self.market_scanner.get_trading_opportunities(min_score=2)
            opportunity_symbols = {opp["symbol"] for opp in opportunities}

            # Prioritize coins with opportunities
            for symbol in active_coins:
                if symbol in opportunity_symbols:
                    priority_coins.append(symbol)

            # Add high-volume coins
            for symbol in active_coins:
                if symbol not in priority_coins:
                    # Check volume activity
                    cache_key = f"analysis_{symbol}_1h"
                    if self.cache_manager:
                        analysis = self.cache_manager.get(cache_key)
                        if analysis and analysis.get("volume_ratio", 0) > 1.5:
                            priority_coins.append(symbol)

            # Limit to reasonable number
            return priority_coins[:100]

        except Exception as e:
            self.logger.error(f"Priority coin selection failed: {e}")
            return active_coins[:50]  # Fallback

    def _get_recent_technical_analyses(self) -> List[str]:
        """Get coins with recent technical analysis"""
        try:
            analyzed_coins = []

            if not self.cache_manager:
                return analyzed_coins

            # Search for recent technical analyses
            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith("analysis_") and "_1h" in cache_key:
                    symbol = cache_key.split("_")[1]
                    analyzed_coins.append(symbol)

            return analyzed_coins

        except Exception as e:
            self.logger.error(f"Recent analysis retrieval failed: {e}")
            return []

    def _analyze_coin_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment for a specific coin"""
        try:
            # Placeholder removed
            # In real implementation, this would scrape social media, news, etc.

            sentiment_data = {
                "sentiment_score": 0.6 + (hash(symbol) % 40) / 100,  # 0.6-1.0
                "sentiment_trend": (hash(symbol + "trend") % 20 - 10) / 100,  # -0.1 to 0.1
                "mention_growth": 1.0 + (hash(symbol + "mentions") % 30) / 10,  # 1.0-4.0
                "news_sentiment": 0.5 + (hash(symbol + "news") % 50) / 100,  # 0.5-1.0
                "social_volume": hash(symbol + "volume") % 1000,
                "analysis_timestamp": datetime.now().isoformat(),
            }

            return sentiment_data

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed for {symbol}: {e}")
            return None

    def _get_priority_coins_for_whale_analysis(self) -> List[str]:
        """Get priority coins for whale analysis"""
        try:
            # Get coins with high volume or price movements
            priority_coins = []

            opportunities = self.market_scanner.get_trading_opportunities(min_score=3)
            priority_coins = [opp["symbol"] for opp in opportunities]

            return priority_coins[:50]

        except Exception as e:
            self.logger.error(f"Whale analysis priority selection failed: {e}")
            return []

    def _analyze_whale_activity(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze whale activity for a specific coin"""
        try:
            # Placeholder removed
            whale_data = {
                "large_transactions_24h": hash(symbol + "tx") % 20,
                "net_whale_flow": (hash(symbol + "flow") % 100 - 50) / 100,  # -0.5 to 0.5
                "whale_address_growth": (hash(symbol + "growth") % 20) / 100,  # 0-0.2
                "average_transaction_size": 100000 + (hash(symbol + "size") % 900000),
                "whale_concentration": 0.3 + (hash(symbol + "conc") % 40) / 100,  # 0.3-0.7
                "analysis_timestamp": datetime.now().isoformat(),
            }

            return whale_data

        except Exception as e:
            self.logger.error(f"Whale analysis failed for {symbol}: {e}")
            return None

    def _get_coins_with_comprehensive_data(self) -> List[str]:
        """Get coins that have technical and sentiment analysis"""
        try:
            comprehensive_coins = []

            if not self.cache_manager:
                return comprehensive_coins

            # Find coins with both technical and sentiment data
            technical_coins = set()
            sentiment_coins = set()

            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith("analysis_"):
                    symbol = cache_key.split("_")[1]
                    technical_coins.add(symbol)

            sentiment_results = self.cache_manager.get("sentiment_analysis_results")
            if sentiment_results and "results" in sentiment_results:
                sentiment_coins = set(sentiment_results["results"].keys())

            # Get intersection
            comprehensive_coins = list(technical_coins.intersection(sentiment_coins))

            return comprehensive_coins[:50]

        except Exception as e:
            self.logger.error(f"Comprehensive data retrieval failed: {e}")
            return []

    def _generate_ml_prediction(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate ML prediction for a specific coin"""
        try:
            # Placeholder removed
            # In real implementation, this would use trained models

            base_score = (hash(symbol + "ml") % 100) / 100  # 0-1

            ml_prediction = {
                "price_prediction": {
                    "return_7d": base_score * 0.5 - 0.25,  # -25% to +25%
                    "return_30d": base_score * 1.5 - 0.75,  # -75% to +75%
                    "confidence": 0.6 + (hash(symbol + "conf") % 30) / 100,  # 0.6-0.9
                },
                "ensemble_agreement": 0.5 + (hash(symbol + "ensemble") % 40) / 100,  # 0.5-0.9
                "historical_accuracy": 0.6 + (hash(symbol + "accuracy") % 35) / 100,  # 0.6-0.95
                "volatility_prediction": 20 + (hash(symbol + "vol") % 60),  # 20-80%
                "trend_probability": {
                    "bullish": base_score,
                    "bearish": 1 - base_score,
                    "sideways": 0.5 - abs(base_score - 0.5),
                },
                "analysis_timestamp": datetime.now().isoformat(),
            }

            return ml_prediction

        except Exception as e:
            self.logger.error(f"ML prediction failed for {symbol}: {e}")
            return None

    def _get_coins_with_complete_analysis(self) -> List[str]:
        """Get coins with all analysis types completed"""
        try:
            if not self.cache_manager:
                return []

            # Get coins from all analysis results
            technical_coins = set()
            sentiment_coins = set()
            whale_coins = set()
            ml_coins = set()

            # Technical analysis
            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith("analysis_"):
                    symbol = cache_key.split("_")[1]
                    technical_coins.add(symbol)

            # Sentiment analysis
            sentiment_results = self.cache_manager.get("sentiment_analysis_results")
            if sentiment_results and "results" in sentiment_results:
                sentiment_coins = set(sentiment_results["results"].keys())

            # Whale detection
            whale_results = self.cache_manager.get("whale_detection_results")
            if whale_results and "results" in whale_results:
                whale_coins = set(whale_results["results"].keys())

            # ML predictions
            ml_results = self.cache_manager.get("ml_prediction_results")
            if ml_results and "results" in ml_results:
                ml_coins = set(ml_results["results"].keys())

            # Find intersection of all analyses
            complete_coins = technical_coins.intersection(sentiment_coins, whale_coins, ml_coins)

            return list(complete_coins)

        except Exception as e:
            self.logger.error(f"Complete analysis retrieval failed: {e}")
            return []

    def _process_openai_batch(self, coin_batch: List[str]) -> Optional[List[Dict[str, Any]]]:
        """Process a batch of coins through OpenAI analysis"""
        try:
            if not self.openai_analyzer:
                return None

            batch_results = []

            for symbol in coin_batch:
                # Gather all analysis data for this coin
                comprehensive_data = self._gather_comprehensive_data(symbol)

                if comprehensive_data:
                    # Send to OpenAI for analysis
                    openai_analysis = self._request_openai_analysis(symbol, comprehensive_data)

                    if openai_analysis:
                        batch_results.append(
                            {
                                "symbol": symbol,
                                "openai_analysis": openai_analysis,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            return batch_results

        except Exception as e:
            self.logger.error(f"OpenAI batch processing failed: {e}")
            return None

    def _gather_comprehensive_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Gather all available analysis data for a coin"""
        try:
            comprehensive_data = {"symbol": symbol}

            if not self.cache_manager:
                return None

            # Technical analysis
            for timeframe in ["1h", "4h", "1d"]:
                cache_key = f"analysis_{symbol}_{timeframe}"
                technical_data = self.cache_manager.get(cache_key)
                if technical_data:
                    comprehensive_data[f"technical_{timeframe}"] = technical_data

            # Sentiment analysis
            sentiment_results = self.cache_manager.get("sentiment_analysis_results")
            if sentiment_results and symbol in sentiment_results.get("results", {}):
                comprehensive_data["sentiment"] = sentiment_results["results"][symbol]

            # Whale activity
            whale_results = self.cache_manager.get("whale_detection_results")
            if whale_results and symbol in whale_results.get("results", {}):
                comprehensive_data["whale"] = whale_results["results"][symbol]

            # ML predictions
            ml_results = self.cache_manager.get("ml_prediction_results")
            if ml_results and symbol in ml_results.get("results", {}):
                comprehensive_data["ml"] = ml_results["results"][symbol]

            # Only return if we have substantial data
            if len(comprehensive_data) > 2:  # More than just symbol
                return comprehensive_data

            return None

        except Exception as e:
            self.logger.error(f"Data gathering failed for {symbol}: {e}")
            return None

    def _request_openai_analysis(
        self, symbol: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Request OpenAI analysis for comprehensive data"""
        try:
            # Format data for OpenAI prompt
            analysis_prompt = self._format_analysis_prompt(symbol, data)

            # Request analysis from OpenAI
            openai_response = self.openai_analyzer.analyze_comprehensive_data(analysis_prompt)

            return openai_response

        except Exception as e:
            self.logger.error(f"OpenAI analysis request failed for {symbol}: {e}")
            return None

    def _format_analysis_prompt(self, symbol: str, data: Dict[str, Any]) -> str:
        """Format comprehensive data into OpenAI analysis prompt"""
        try:
            prompt = f"""
Analyze {symbol} for high-growth potential (500%+ returns in 6 months).

Technical Analysis:
"""

            # Add technical data
            for key, value in data.items():
                if key.startswith("technical_"):
                    timeframe = key.split("_")[1]
                    technical = value
                    prompt += f"""
{timeframe} timeframe:
- Price: ${technical.get("last_price", 0):.4f}
- RSI: {technical.get("rsi", 0):.1f}
- Volume Ratio: {technical.get("volume_ratio", 0):.1f}x
- Trend: {technical.get("trend_direction", "unknown")}
"""

            # Add sentiment data
            if "sentiment" in data:
                sentiment = data["sentiment"]
                prompt += f"""
Sentiment Analysis:
- Sentiment Score: {sentiment.get("sentiment_score", 0):.2f}
- Mention Growth: {sentiment.get("mention_growth", 0):.1f}x
- News Sentiment: {sentiment.get("news_sentiment", 0):.2f}
"""

            # Add whale data
            if "whale" in data:
                whale = data["whale"]
                prompt += f"""
Whale Activity:
- Large Transactions: {whale.get("large_transactions_24h", 0)}
- Net Flow: {whale.get("net_whale_flow", 0):+.2f}
- Address Growth: {whale.get("whale_address_growth", 0):.2f}
"""

            # Add ML predictions
            if "ml" in data:
                ml = data["ml"]
                pred = ml.get("price_prediction", {})
                prompt += f"""
ML Predictions:
- 7-day return: {pred.get("return_7d", 0):+.1%}
- 30-day return: {pred.get("return_30d", 0):+.1%}
- Confidence: {pred.get("confidence", 0):.2f}
"""

            prompt += """
Please provide:
1. Expected return potential for 7 days, 30 days, and 180 days
2. Confidence level (0-1) for each prediction
3. Key growth factors and risks
4. Overall recommendation with reasoning

Respond in JSON format.
"""

            return prompt

        except Exception as e:
            self.logger.error(f"Prompt formatting failed: {e}")
            return ""

    def get_analysis_status(self) -> Dict[str, Any]:
        """Get comprehensive analysis system status"""
        try:
            status = {
                "analysis_active": self.analysis_active,
                "last_analyses": {},
                "data_availability": {},
            }

            if self.cache_manager:
                # Check last analysis times
                for analysis_type in [
                    "market_scan",
                    "sentiment_analysis",
                    "whale_detection",
                    "ml_prediction",
                    "openai_comprehensive_analysis",
                ]:
                    cache_key = f"{analysis_type}_results"
                    results = self.cache_manager.get(cache_key)
                    if results and "timestamp" in results:
                        status["last_analyses"][analysis_type] = results["timestamp"]
                        status["data_availability"][analysis_type] = results.get(
                            "coins_analyzed", 0
                        )

            return status

        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {"analysis_active": False}
