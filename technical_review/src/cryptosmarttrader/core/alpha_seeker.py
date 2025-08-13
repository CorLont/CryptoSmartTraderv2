"""
CryptoSmartTrader V2 - Alpha Seeker
Advanced system for identifying high-growth cryptocurrencies with 500%+ return potential
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_accelerator import gpu_accelerator

class AlphaSeeker:
    """Advanced alpha-seeking system for identifying high-growth cryptocurrencies"""

    def __init__(self, config_manager=None, cache_manager=None, openai_analyzer=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.openai_analyzer = openai_analyzer
        self.logger = logging.getLogger(__name__)

        # Alpha seeking parameters
        self.alpha_config = {
            'min_confidence': 0.80,        # 80% probability threshold
            'target_return': 5.0,          # 500% return target
            'max_timeframe_days': 30,      # 30-day prediction horizon
            'volume_growth_threshold': 3.0, # 3x volume increase
            'momentum_threshold': 0.75,    # Strong momentum indicator
            'whale_activity_weight': 0.3,  # Whale activity importance
            'sentiment_weight': 0.25,      # Sentiment analysis weight
            'technical_weight': 0.25,      # Technical analysis weight
            'ml_weight': 0.20              # ML prediction weight
        }

        # Learning system for prediction accuracy
        self.prediction_history = {}
        self.accuracy_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'average_accuracy': 0.0,
            'best_indicators': [],
            'model_performance': {}
        }

        self.logger.info("Alpha Seeker initialized for 500%+ return identification")

    def analyze_growth_potential(self, symbol: str, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cryptocurrency for high-growth potential with confidence scoring"""
        try:
            alpha_score = 0.0
            confidence_factors = []
            risk_factors = []

            # Extract technical indicators
            technical_data = analysis_data.get('technical', {})
            sentiment_data = analysis_data.get('sentiment', {})
            ml_data = analysis_data.get('ml_prediction', {})
            whale_data = analysis_data.get('whale_activity', {})
            volume_data = analysis_data.get('volume_analysis', {})

            # 1. Volume Analysis (Early Adopter Signal)
            volume_score = self._analyze_volume_growth(volume_data)
            alpha_score += volume_score * 0.25

            if volume_score > 0.7:
                confidence_factors.append(f"Strong volume growth: {volume_score:.2f}")
            elif volume_score < 0.3:
                risk_factors.append(f"Low volume activity: {volume_score:.2f}")

            # 2. Technical Momentum Analysis
            momentum_score = self._analyze_technical_momentum(technical_data)
            alpha_score += momentum_score * self.alpha_config['technical_weight']

            if momentum_score > 0.8:
                confidence_factors.append(f"Exceptional technical momentum: {momentum_score:.2f}")

            # 3. Sentiment Analysis (Market Psychology)
            sentiment_score = self._analyze_sentiment_signals(sentiment_data)
            alpha_score += sentiment_score * self.alpha_config['sentiment_weight']

            if sentiment_score > 0.75:
                confidence_factors.append(f"Bullish sentiment trend: {sentiment_score:.2f}")

            # 4. ML Prediction Analysis
            ml_score = self._analyze_ml_predictions(ml_data)
            alpha_score += ml_score * self.alpha_config['ml_weight']

            if ml_score > 0.8:
                confidence_factors.append(f"Strong ML prediction: {ml_score:.2f}")

            # 5. Whale Activity Analysis
            whale_score = self._analyze_whale_signals(whale_data)
            alpha_score += whale_score * self.alpha_config['whale_activity_weight']

            if whale_score > 0.7:
                confidence_factors.append(f"Significant whale accumulation: {whale_score:.2f}")

            # 6. Early Stage Detection
            early_stage_score = self._detect_early_stage_signals(analysis_data)
            alpha_score += early_stage_score * 0.15

            if early_stage_score > 0.8:
                confidence_factors.append(f"Early stage opportunity: {early_stage_score:.2f}")

            # Calculate confidence level
            confidence = self._calculate_confidence(alpha_score, confidence_factors, risk_factors)

            # Predict expected returns
            return_predictions = self._predict_returns(symbol, alpha_score, analysis_data)

            # Store prediction for learning
            self._store_prediction(symbol, alpha_score, confidence, return_predictions)

            return {
                'symbol': symbol,
                'alpha_score': alpha_score,
                'confidence': confidence,
                'expected_returns': return_predictions,
                'confidence_factors': confidence_factors,
                'risk_factors': risk_factors,
                'meets_criteria': confidence >= self.alpha_config['min_confidence'] and
                                return_predictions.get('30_day', 0) >= 1.0,  # 100%+ return minimum
                'analysis_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Alpha analysis failed for {symbol}: {e}")
            return {}

    def _analyze_volume_growth(self, volume_data: Dict[str, Any]) -> float:
        """Analyze volume growth patterns for early detection"""
        try:
            volume_ratio = volume_data.get('volume_ratio', 1.0)
            volume_trend = volume_data.get('volume_trend', 0.0)
            volume_acceleration = volume_data.get('volume_acceleration', 0.0)

            # Volume explosion detection
            score = 0.0

            # Sudden volume spike (whale/institutional interest)
            if volume_ratio > 5.0:
                score += 0.4  # Very strong signal
            elif volume_ratio > 3.0:
                score += 0.3
            elif volume_ratio > 2.0:
                score += 0.2

            # Sustained volume growth
            if volume_trend > 0.5:
                score += 0.3
            elif volume_trend > 0.2:
                score += 0.2

            # Volume acceleration (building momentum)
            if volume_acceleration > 0.3:
                score += 0.3

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Volume analysis failed: {e}")
            return 0.0

    def _analyze_technical_momentum(self, technical_data: Dict[str, Any]) -> float:
        """Analyze technical indicators for momentum signals"""
        try:
            score = 0.0

            # RSI analysis (oversold recovery or strong momentum)
            rsi = technical_data.get('rsi', 50)
            if 30 <= rsi <= 40:  # Oversold recovery
                score += 0.25
            elif 55 <= rsi <= 70:  # Strong momentum
                score += 0.20

            # MACD signals
            macd = technical_data.get('macd', 0)
            macd_signal = technical_data.get('macd_signal', 0)
            if macd > macd_signal and macd > 0:
                score += 0.25

            # Moving average breakouts
            price = technical_data.get('last_price', 0)
            sma_20 = technical_data.get('sma_20', 0)
            sma_50 = technical_data.get('sma_50', 0)

            if price > sma_20 > sma_50:  # Bullish alignment
                score += 0.25

            # Bollinger Band squeeze (potential breakout)
            bb_position = technical_data.get('bb_position', 0.5)
            if bb_position > 0.8 or bb_position < 0.2:
                score += 0.15

            # Trend strength
            trend_strength = technical_data.get('trend_strength', 0)
            if trend_strength > 0.7:
                score += 0.10

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Technical momentum analysis failed: {e}")
            return 0.0

    def _analyze_sentiment_signals(self, sentiment_data: Dict[str, Any]) -> float:
        """Analyze sentiment for early bullish signals"""
        try:
            score = 0.0

            # Social media sentiment trend
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            sentiment_trend = sentiment_data.get('sentiment_trend', 0.0)
            mention_growth = sentiment_data.get('mention_growth', 0.0)

            # Strong positive sentiment
            if sentiment_score > 0.8:
                score += 0.3
            elif sentiment_score > 0.6:
                score += 0.2

            # Improving sentiment trend
            if sentiment_trend > 0.3:
                score += 0.25
            elif sentiment_trend > 0.1:
                score += 0.15

            # Growing mentions (attention increase)
            if mention_growth > 2.0:
                score += 0.25
            elif mention_growth > 1.5:
                score += 0.15

            # News sentiment analysis
            news_sentiment = sentiment_data.get('news_sentiment', 0.5)
            if news_sentiment > 0.7:
                score += 0.20

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return 0.0

    def _analyze_ml_predictions(self, ml_data: Dict[str, Any]) -> float:
        """Analyze ML model predictions for growth signals"""
        try:
            score = 0.0

            # Price prediction confidence
            price_prediction = ml_data.get('price_prediction', {})
            prediction_confidence = price_prediction.get('confidence', 0.0)
            predicted_return = price_prediction.get('return_7d', 0.0)

            # Strong upward prediction
            if predicted_return > 0.5 and prediction_confidence > 0.8:
                score += 0.4
            elif predicted_return > 0.3 and prediction_confidence > 0.7:
                score += 0.3
            elif predicted_return > 0.1 and prediction_confidence > 0.6:
                score += 0.2

            # Model ensemble agreement
            ensemble_agreement = ml_data.get('ensemble_agreement', 0.0)
            if ensemble_agreement > 0.8:
                score += 0.3
            elif ensemble_agreement > 0.6:
                score += 0.2

            # Historical accuracy of model for this asset
            model_accuracy = ml_data.get('historical_accuracy', 0.0)
            if model_accuracy > 0.75:
                score += 0.3

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"ML prediction analysis failed: {e}")
            return 0.0

    def _analyze_whale_signals(self, whale_data: Dict[str, Any]) -> float:
        """Analyze whale activity for accumulation signals"""
        try:
            score = 0.0

            # Large transaction frequency
            large_tx_count = whale_data.get('large_transactions_24h', 0)
            if large_tx_count > 10:
                score += 0.3
            elif large_tx_count > 5:
                score += 0.2

            # Net whale flow (accumulation vs distribution)
            whale_flow = whale_data.get('net_whale_flow', 0.0)
            if whale_flow > 0.5:  # Strong accumulation
                score += 0.4
            elif whale_flow > 0.2:
                score += 0.2

            # Whale addresses increasing
            whale_address_growth = whale_data.get('whale_address_growth', 0.0)
            if whale_address_growth > 0.1:
                score += 0.3

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Whale analysis failed: {e}")
            return 0.0

    def _detect_early_stage_signals(self, analysis_data: Dict[str, Any]) -> float:
        """Detect early stage opportunities before mainstream adoption"""
        try:
            score = 0.0

            # Market cap analysis
            market_cap = analysis_data.get('market_cap', float('inf'))
            if market_cap < 100_000_000:  # Under $100M
                score += 0.3
            elif market_cap < 500_000_000:  # Under $500M
                score += 0.2
            elif market_cap < 1_000_000_000:  # Under $1B
                score += 0.1

            # Exchange listing analysis
            exchange_count = analysis_data.get('exchange_count', 0)
            if exchange_count < 5:  # Limited exchange availability
                score += 0.2

            # Age analysis (newer projects with potential)
            project_age_days = analysis_data.get('project_age_days', 0)
            if 90 <= project_age_days <= 730:  # 3 months to 2 years
                score += 0.25

            # Development activity
            dev_activity = analysis_data.get('development_activity', 0.0)
            if dev_activity > 0.7:
                score += 0.25

            return min(score, 1.0)

        except Exception as e:
            self.logger.error(f"Early stage detection failed: {e}")
            return 0.0

    def _calculate_confidence(self, alpha_score: float, confidence_factors: List[str],
                            risk_factors: List[str]) -> float:
        """Calculate confidence level for the prediction"""
        try:
            base_confidence = min(alpha_score, 1.0)

            # Boost confidence for multiple positive factors
            factor_boost = len(confidence_factors) * 0.05
            base_confidence += factor_boost

            # Reduce confidence for risk factors
            risk_penalty = len(risk_factors) * 0.1
            base_confidence -= risk_penalty

            # Historical accuracy adjustment
            if self.accuracy_metrics['total_predictions'] > 10:
                accuracy_adjustment = (self.accuracy_metrics['average_accuracy'] - 0.5) * 0.2
                base_confidence += accuracy_adjustment

            return max(0.0, min(1.0, base_confidence))

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0

    def _predict_returns(self, symbol: str, alpha_score: float,
                        analysis_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict expected returns for different timeframes"""
        try:
            base_return = alpha_score * 2.0  # Base multiplier

            # Volatility adjustment
            volatility = analysis_data.get('technical', {}).get('volatility', 50)
            volatility_multiplier = min(volatility / 100, 2.0)

            # Market cap adjustment (smaller caps = higher potential)
            market_cap = analysis_data.get('market_cap', 1_000_000_000)
            cap_multiplier = max(1.0, 1_000_000_000 / max(market_cap, 100_000_000))

            # Volume growth multiplier
            volume_ratio = analysis_data.get('volume_analysis', {}).get('volume_ratio', 1.0)
            volume_multiplier = min(volume_ratio / 2.0, 3.0)

            # Calculate returns for different timeframes
            total_multiplier = base_return * volatility_multiplier * cap_multiplier * volume_multiplier

            return {
                '7_day': total_multiplier * 0.3,    # 30% of total potential in 7 days
                '30_day': total_multiplier * 1.0,   # Full potential in 30 days
                '90_day': total_multiplier * 2.0,   # Extended potential in 90 days
                '180_day': total_multiplier * 3.5   # Maximum potential in 6 months
            }

        except Exception as e:
            self.logger.error(f"Return prediction failed: {e}")
            return {'7_day': 0, '30_day': 0, '90_day': 0, '180_day': 0}

    def _store_prediction(self, symbol: str, alpha_score: float, confidence: float,
                         returns: Dict[str, float]):
        """Store prediction for learning and accuracy tracking"""
        try:
            prediction_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            prediction_record = {
                'symbol': symbol,
                'alpha_score': alpha_score,
                'confidence': confidence,
                'predicted_returns': returns,
                'timestamp': datetime.now().isoformat(),
                'actual_returns': None,  # To be filled later
                'accuracy_verified': False
            }

            self.prediction_history[prediction_id] = prediction_record

            # Store in cache for persistence
            if self.cache_manager:
                self.cache_manager.set(
                    f'alpha_prediction_{prediction_id}',
                    prediction_record,
                    ttl_minutes=10080  # 7 days
                )

        except Exception as e:
            self.logger.error(f"Prediction storage failed: {e}")

    def get_top_alpha_opportunities(self, min_confidence: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get top alpha opportunities meeting confidence criteria"""
        try:
            if min_confidence is None:
                min_confidence = self.alpha_config['min_confidence']

            opportunities = []

            if not self.cache_manager:
                return opportunities

            # Search cache for alpha analyses
            for cache_key in self.cache_manager._cache.keys():
                if cache_key.startswith('alpha_analysis_'):
                    analysis = self.cache_manager.get(cache_key)
                    if (analysis and
                        analysis.get('confidence', 0) >= min_confidence and
                        analysis.get('meets_criteria', False)):
                        opportunities.append(analysis)

            # Sort by expected 30-day return
            opportunities.sort(
                key=lambda x: x.get('expected_returns', {}).get('30_day', 0),
                reverse=True
            )

            return opportunities

        except Exception as e:
            self.logger.error(f"Failed to get alpha opportunities: {e}")
            return []

    def update_prediction_accuracy(self, symbol: str, actual_returns: Dict[str, float]):
        """Update prediction accuracy based on actual results"""
        try:
            # Find relevant predictions for this symbol
            for pred_id, prediction in self.prediction_history.items():
                if (prediction['symbol'] == symbol and
                    not prediction['accuracy_verified']):

                    # Calculate accuracy
                    predicted = prediction['predicted_returns']
                    accuracy_scores = []

                    for timeframe in ['7_day', '30_day']:
                        if timeframe in predicted and timeframe in actual_returns:
                            pred_val = predicted[timeframe]
                            actual_val = actual_returns[timeframe]

                            if pred_val > 0 and actual_val > 0:
                                accuracy = 1 - abs(pred_val - actual_val) / max(pred_val, actual_val)
                                accuracy_scores.append(max(0, accuracy))

                    if accuracy_scores:
                        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

                        # Update metrics
                        self.accuracy_metrics['total_predictions'] += 1
                        if avg_accuracy > 0.7:  # 70% accuracy threshold
                            self.accuracy_metrics['correct_predictions'] += 1

                        # Recalculate average accuracy
                        self.accuracy_metrics['average_accuracy'] = (
                            self.accuracy_metrics['correct_predictions'] /
                            self.accuracy_metrics['total_predictions']
                        )

                        # Mark as verified
                        prediction['actual_returns'] = actual_returns
                        prediction['accuracy_verified'] = True
                        prediction['accuracy_score'] = avg_accuracy

                        self.logger.info(f"Updated prediction accuracy for {symbol}: {avg_accuracy:.2f}")

        except Exception as e:
            self.logger.error(f"Accuracy update failed: {e}")

    def get_system_performance(self) -> Dict[str, Any]:
        """Get alpha seeking system performance metrics"""
        return {
            'accuracy_metrics': self.accuracy_metrics,
            'total_opportunities_found': len(self.get_top_alpha_opportunities(0.5)),
            'high_confidence_opportunities': len(self.get_top_alpha_opportunities()),
            'alpha_config': self.alpha_config,
            'prediction_count': len(self.prediction_history)
        }
