"""
Main Regime Detection Service

Orchestrates the regime detection process:
1. Collects market data and features
2. Classifies current regime
3. Adapts trading strategies
4. Provides regime-aware recommendations
"""

import asyncio
import logging
from typing import Dict, Optional, Any
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

from .regime_features import RegimeFeatures, RegimeFeatureSet
from .regime_models import RegimeClassifier, MarketRegime, RegimeClassification
from .regime_strategies import RegimeStrategies, TradingParameters

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Main service for market regime detection and strategy adaptation
    """

    def __init__(self,
                 data_manager=None,
                 update_frequency_minutes: int = 15,
                 model_path: str = "models/regime_classifier.pkl",
                 history_path: str = "data/regime_history.json"):

        self.data_manager = data_manager
        self.update_frequency = update_frequency_minutes
        self.model_path = model_path
        self.history_path = Path(history_path)

        # Core components
        self.feature_calculator = RegimeFeatures()
        self.classifier = RegimeClassifier(model_path)
        self.strategy_adapter = RegimeStrategies()

        # State tracking
        self.current_regime = None
        self.current_features = None
        self.current_strategy = None
        self.regime_history = []
        self.is_running = False

        # Cache for performance
        self._last_update = None
        self._cached_dominance = None
        self._dominance_cache_time = None

    async def initialize(self) -> bool:
        """Initialize the regime detection system"""
        try:
            logger.info("Initializing Regime Detection System...")

            # Load historical regime data if exists
            self._load_regime_history()

            # Try to load existing model
            if not self.classifier.load_model():
                logger.warning("No pre-trained regime model found. Will need training data.")

            # Test data connection
            if self.data_manager is not None:
                test_data = await self._get_market_data()
                if not test_data:
                    logger.error("Cannot connect to market data")
                    return False

            logger.info("Regime Detection System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize regime detector: {e}")
            return False

    async def start_monitoring(self) -> None:
        """Start continuous regime monitoring"""
        self.is_running = True
        logger.info(f"Starting regime monitoring (update every {self.update_frequency} minutes)")

        while self.is_running:
            try:
                await self.update_regime()
                await asyncio.sleep(self.update_frequency * 60)

            except Exception as e:
                logger.error(f"Error in regime monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    def stop_monitoring(self) -> None:
        """Stop regime monitoring"""
        self.is_running = False
        logger.info("Regime monitoring stopped")

    async def update_regime(self) -> RegimeClassification:
        """
        Update current market regime classification

        Returns:
            Latest regime classification
        """
        try:
            logger.debug("Updating market regime classification...")

            # Get fresh market data
            market_data = await self._get_market_data()
            if not market_data:
                logger.warning("No market data available for regime update")
                return self._get_fallback_classification()

            # Get market dominance data
            dominance_data = await self._get_dominance_data()

            # Calculate features
            self.current_features = self.feature_calculator.calculate_all_features(
                market_data=market_data,
                dominance_data=dominance_data,
                funding_data=await self._get_funding_data(),
                oi_data=await self._get_oi_data()
            )

            # Classify regime
            self.current_regime = self.classifier.predict_regime(self.current_features)

            # Update strategy
            self.current_strategy = self.strategy_adapter.get_strategy_for_regime(
                self.current_regime
            )

            # Update history
            self.regime_history.append({
                'timestamp': datetime.now().isoformat(),
                'regime': self.current_regime.primary_regime.value,
                'confidence': self.current_regime.confidence,
                'features': self._serialize_features(self.current_features),
                'strategy': self.current_strategy.strategy_type
            })

            # Keep only recent history (last 7 days)
            cutoff_time = datetime.now() - timedelta(days=7)
            self.regime_history = [
                h for h in self.regime_history
                if datetime.fromisoformat(h['timestamp']) > cutoff_time
            ]

            # Save history
            self._save_regime_history()

            self._last_update = datetime.now()

            logger.info(
                f"Regime updated: {self.current_regime.primary_regime.value} "
                f"(confidence: {self.current_regime.confidence:.2f}, "
                f"strategy: {self.current_strategy.strategy_type})"
            )

            return self.current_regime

        except Exception as e:
            logger.error(f"Failed to update regime: {e}")
            return self._get_fallback_classification()

    def get_current_regime(self) -> Optional[RegimeClassification]:
        """Get the current regime classification"""
        return self.current_regime

    def get_current_strategy(self) -> Optional[TradingParameters]:
        """Get current trading strategy parameters"""
        return self.current_strategy

    def should_trade(self, signal_strength: float, symbol: str = None) -> Dict[str, Any]:
        """
        Determine if trading is recommended given current regime

        Args:
            signal_strength: ML prediction confidence (0-1)
            symbol: Trading symbol (optional)

        Returns:
            Trading recommendation with rationale
        """
        if self.current_regime is None or self.current_strategy is None:
            return {
                "trade": False,
                "reason": "No regime classification available",
                "risk_level": "high"
            }

        # Get base recommendation from strategy
        recommendation = self.strategy_adapter.should_enter_position(
            signal_strength, self.current_strategy
        )

        # Add regime context
        recommendation.update({
            "current_regime": self.current_regime.primary_regime.value,
            "regime_confidence": self.current_regime.confidence,
            "strategy_type": self.current_strategy.strategy_type,
            "last_regime_update": self._last_update.isoformat() if self._last_update else None
        })

        return recommendation

    def get_regime_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive regime analytics

        Returns:
            Analytics about current regime and recent patterns
        """
        try:
            analytics = {
                "current_state": {
                    "regime": self.current_regime.primary_regime.value if self.current_regime else None,
                    "confidence": self.current_regime.confidence if self.current_regime else 0,
                    "last_update": self._last_update.isoformat() if self._last_update else None,
                    "trading_allowed": not self.current_strategy.no_trade if self.current_strategy else False
                },
                "regime_stability": self.classifier.get_regime_stability(),
                "strategy_summary": self.strategy_adapter.get_strategy_summary(),
                "recent_history": self.regime_history[-10:] if len(self.regime_history) > 0 else []
            }

            # Feature analysis if available
            if self.current_features:
                analytics["current_features"] = {
                    "hurst_exponent": self.current_features.hurst_exponent,
                    "trend_strength": self.current_features.adx,
                    "volatility": self.current_features.realized_vol,
                    "market_structure": {
                        "btc_dominance": self.current_features.btc_dominance,
                        "alt_breadth": self.current_features.alt_breadth
                    },
                    "derivatives": {
                        "funding_impulse": self.current_features.funding_impulse,
                        "oi_impulse": self.current_features.oi_impulse
                    }
                }

            # Regime transition analysis
            if len(self.regime_history) > 5:
                recent_regimes = [h['regime'] for h in self.regime_history[-10:]]
                unique_regimes = list(set(recent_regimes))

                analytics["transition_analysis"] = {
                    "regimes_last_10_periods": recent_regimes,
                    "unique_regimes_count": len(unique_regimes),
                    "most_common_regime": max(set(recent_regimes), key=recent_regimes.count),
                    "regime_changes_last_10": len(unique_regimes) - 1
                }

            return analytics

        except Exception as e:
            logger.error(f"Failed to generate regime analytics: {e}")
            return {"error": str(e)}

    async def train_regime_model(self, historical_days: int = 90) -> Dict[str, Any]:
        """
        Train the regime classification model using historical data

        Args:
            historical_days: Number of days of historical data to use

        Returns:
            Training results and metrics
        """
        try:
            logger.info(f"Starting regime model training with {historical_days} days of data...")

            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=historical_days)

            historical_data = await self._get_historical_market_data(start_date, end_date)
            if not historical_data:
                return {"error": "Insufficient historical data for training"}

            # Calculate features for all historical periods
            feature_sets = []
            returns_data = []

            for date, data in historical_data.items():
                dominance = await self._get_dominance_data_for_date(date)

                features = self.feature_calculator.calculate_all_features(
                    market_data=data,
                    dominance_data=dominance
                )

                feature_sets.append(features)

                # Calculate returns for labeling
                btc_data = data.get('BTC/USD')
                if btc_data is not None and len(btc_data) > 1:
                    daily_return = btc_data['close'].pct_change().iloc[-1]
                    returns_data.append(daily_return)
                else:
                    returns_data.append(0.0)

            if len(feature_sets) < 30:
                return {"error": "Insufficient feature data for training"}

            # Train model
            returns_series = pd.Series(returns_data)
            training_results = self.classifier.train(feature_sets, returns_series)

            if "error" in training_results:
                return training_results

            logger.info("Regime model training completed successfully")
            return training_results

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}

    # Private methods

    async def _get_market_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """Get current market data for regime analysis"""
        if self.data_manager is None:
            return self._get_random.choice()

        try:
            # Get data for key cryptocurrencies
            symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD']
            market_data = {}

            for symbol in symbols:
                data = await self.data_manager.get_ohlcv(symbol, timeframe='1h', limit=200)
                if data is not None and len(data) > 50:
                    market_data[symbol] = data

            return market_data if market_data else None

        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return None

    async def _get_dominance_data(self) -> Dict[str, float]:
        """Get market dominance data"""
        # Cache dominance data (updates less frequently)
        if (self._dominance_cache_time and
            datetime.now() - self._dominance_cache_time < timedelta(hours=1)):
            return self._cached_dominance

        try:
            # This would connect to a dominance API
            # For now, return approximate values
            dominance_data = {
                'BTC': 45.2,
                'ETH': 18.5,
                'others': 36.3
            }

            self._cached_dominance = dominance_data
            self._dominance_cache_time = datetime.now()

            return dominance_data

        except Exception as e:
            logger.error(f"Failed to get dominance data: {e}")
            return {'BTC': 45.0, 'ETH': 18.0, 'others': 37.0}

    async def _get_funding_data(self) -> Optional[pd.Series]:
        """Get funding rates data"""
        # Placeholder - would connect to derivatives data
        return None

    async def _get_oi_data(self) -> Optional[pd.Series]:
        """Get open interest data"""
        # Placeholder - would connect to derivatives data
        return None

    def _get_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate mock market data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')

        # Simple random walk for BTC
        np_random = np.random.RandomState(42)
        price_changes = np_random.choice
        prices = 50000 * (1 + price_changes).cumprod()

        # REMOVED: Mock data pattern not allowed in production{
            'timestamp': dates,
            'open': prices * 0.999,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np_random.choice
        })
        # REMOVED: Mock data pattern not allowed in production'timestamp', inplace=True)

        return {'BTC/USD': mock_data}

    def _serialize_features(self, features: RegimeFeatureSet) -> Dict[str, Any]:
        """Convert features to serializable format"""
        return {
            'hurst_exponent': features.hurst_exponent,
            'adx': features.adx,
            'realized_vol': features.realized_vol,
            'atr_normalized': features.atr_normalized,
            'btc_dominance': features.btc_dominance,
            'alt_breadth': features.alt_breadth,
            'funding_impulse': features.funding_impulse,
            'oi_impulse': features.oi_impulse,
            'volatility_regime': features.volatility_regime,
            'trend_strength': features.trend_strength
        }

    def _save_regime_history(self) -> None:
        """Save regime history to disk"""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_path, 'w') as f:
                json.dump(self.regime_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save regime history: {e}")

    def _load_regime_history(self) -> None:
        """Load regime history from disk"""
        try:
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    self.regime_history = json.load(f)
                logger.info(f"Loaded {len(self.regime_history)} regime history entries")
        except Exception as e:
            logger.error(f"Failed to load regime history: {e}")
            self.regime_history = []

    def _get_fallback_classification(self) -> RegimeClassification:
        """Return fallback classification when detection fails"""
        from .regime_models import RegimeClassification, MarketRegime
        return RegimeClassification(
            primary_regime=MarketRegime.LOW_VOL_DRIFT,
            confidence=0.5,
            probabilities={MarketRegime.LOW_VOL_DRIFT: 0.5},
            feature_importance={},
            timestamp=pd.Timestamp.now(),
            should_trade=False
        )

    async def _get_historical_market_data(self, start_date: datetime,
                                        end_date: datetime) -> Optional[Dict]:
        """Get historical market data for training"""
        # Placeholder for historical data retrieval
        # In real implementation, would fetch from data provider
        return None

    async def _get_dominance_data_for_date(self, date: datetime) -> Dict[str, float]:
        """Get dominance data for specific date"""
        # Placeholder - would fetch historical dominance data
        return {'BTC': 45.0, 'ETH': 18.0, 'others': 37.0}
