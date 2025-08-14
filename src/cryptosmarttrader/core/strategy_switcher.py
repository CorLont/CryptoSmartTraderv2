"""Strategy switching system with regime-aware allocation and risk management."""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
import threading

from .structured_logger import get_logger
from .regime_detector import RegimeDetector, MarketRegime


class StrategyType(Enum):
    """Available trading strategies."""

    MOMENTUM_LONG = "momentum_long"
    MOMENTUM_SHORT = "momentum_short"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT_MOMENTUM = "breakout_momentum"
    RANGE_TRADING = "range_trading"
    VOLATILITY_CAPTURE = "volatility_capture"
    CONTRARIAN = "contrarian"
    TREND_FOLLOWING = "trend_following"


@dataclass
class StrategyAllocation:
    """Strategy allocation configuration."""

    strategy_type: StrategyType
    weight: float
    confidence: float
    max_position_size: float
    risk_multiplier: float
    rebalance_frequency: str
    active: bool = True
    last_rebalance: datetime = field(default_factory=datetime.now)


@dataclass
class PositionTarget:
    """Target position for a symbol in a strategy."""

    symbol: str
    target_weight: float
    current_weight: float
    confidence: float
    strategy_source: StrategyType
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterLimit:
    """Risk limits for asset clusters."""

    cluster_name: str
    max_weight: float
    current_weight: float
    symbols: List[str]
    correlation_threshold: float


class StrategySwitcher:
    """Advanced strategy switching with regime awareness and risk management."""

    def __init__(self, regime_detector: RegimeDetector, initial_capital: float = 100000.0):
        """Initialize strategy switcher."""
        self.logger = get_logger("strategy_switcher")
        self.regime_detector = regime_detector
        self.initial_capital = initial_capital

        # Current state
        self.current_allocations: Dict[StrategyType, StrategyAllocation] = {}
        self.position_targets: Dict[str, PositionTarget] = {}
        self.cluster_limits: Dict[str, ClusterLimit] = {}

        # Portfolio tracking
        self.current_portfolio_value = initial_capital
        self.target_volatility = 0.15  # 15% annualized target
        self.max_leverage = 2.0
        self.cash_buffer = 0.05  # 5% cash buffer

        # Strategy performance tracking
        self.strategy_performance: Dict[StrategyType, Dict[str, float]] = {}
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {}

        # Risk management
        self.max_single_position = 0.1  # 10% max position
        self.max_sector_exposure = 0.3  # 30% max sector
        self.correlation_threshold = 0.7  # High correlation threshold

        # Rebalancing parameters
        self.rebalance_threshold = 0.05  # 5% drift threshold
        self.min_trade_size = 100.0  # Minimum trade size in USD

        # Thread safety
        self._lock = threading.RLock()

        # Initialize cluster limits
        self._initialize_cluster_limits()

        # Initialize strategy allocations based on current regime
        self._initialize_strategy_allocations()

        # Persistence
        self.data_path = Path("data/strategy_switching")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Strategy switcher initialized",
            initial_capital=initial_capital,
            target_volatility=self.target_volatility,
        )

    def _initialize_cluster_limits(self) -> None:
        """Initialize asset cluster limits for risk management."""
        # Define crypto asset clusters
        clusters = {
            "large_cap": {
                "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "max_weight": 0.6,  # 60% max in large caps
                "correlation_threshold": 0.8,
            },
            "defi": {
                "symbols": ["UNI/USDT", "AAVE/USDT", "COMP/USDT", "MKR/USDT"],
                "max_weight": 0.25,  # 25% max in DeFi
                "correlation_threshold": 0.75,
            },
            "layer1": {
                "symbols": ["ADA/USDT", "SOL/USDT", "DOT/USDT", "AVAX/USDT"],
                "max_weight": 0.3,  # 30% max in Layer 1s
                "correlation_threshold": 0.7,
            },
            "gaming_metaverse": {
                "symbols": ["AXS/USDT", "SAND/USDT", "MANA/USDT", "ENJ/USDT"],
                "max_weight": 0.15,  # 15% max in gaming/metaverse
                "correlation_threshold": 0.65,
            },
            "privacy": {
                "symbols": ["XMR/USDT", "ZEC/USDT", "DASH/USDT"],
                "max_weight": 0.1,  # 10% max in privacy coins
                "correlation_threshold": 0.6,
            },
        }

        for cluster_name, config in clusters.items():
            self.cluster_limits[cluster_name] = ClusterLimit(
                cluster_name=cluster_name,
                max_weight=config["max_weight"],
                current_weight=0.0,
                symbols=config["symbols"],
                correlation_threshold=config["correlation_threshold"],
            )

    def _initialize_strategy_allocations(self) -> None:
        """Initialize strategy allocations based on current regime."""
        current_regime = self.regime_detector.current_regime
        regime_config = self.regime_detector.get_current_strategy_config()

        # Base allocation template
        base_allocations = {
            StrategyType.MOMENTUM_LONG: 0.0,
            StrategyType.MOMENTUM_SHORT: 0.0,
            StrategyType.MEAN_REVERSION: 0.0,
            StrategyType.BREAKOUT_MOMENTUM: 0.0,
            StrategyType.RANGE_TRADING: 0.0,
            StrategyType.VOLATILITY_CAPTURE: 0.0,
            StrategyType.CONTRARIAN: 0.0,
            StrategyType.TREND_FOLLOWING: 0.0,
        }

        # Regime-specific allocations
        if current_regime == MarketRegime.BULL_TRENDING:
            base_allocations[StrategyType.MOMENTUM_LONG] = 0.4
            base_allocations[StrategyType.TREND_FOLLOWING] = 0.3
            base_allocations[StrategyType.BREAKOUT_MOMENTUM] = 0.2
            base_allocations[StrategyType.MEAN_REVERSION] = 0.1

        elif current_regime == MarketRegime.BEAR_TRENDING:
            base_allocations[StrategyType.MOMENTUM_SHORT] = 0.4
            base_allocations[StrategyType.CONTRARIAN] = 0.3
            base_allocations[StrategyType.MEAN_REVERSION] = 0.2
            base_allocations[StrategyType.VOLATILITY_CAPTURE] = 0.1

        elif current_regime == MarketRegime.SIDEWAYS_LOW_VOL:
            base_allocations[StrategyType.MEAN_REVERSION] = 0.5
            base_allocations[StrategyType.RANGE_TRADING] = 0.3
            base_allocations[StrategyType.MOMENTUM_LONG] = 0.1
            base_allocations[StrategyType.MOMENTUM_SHORT] = 0.1

        elif current_regime == MarketRegime.SIDEWAYS_HIGH_VOL:
            base_allocations[StrategyType.VOLATILITY_CAPTURE] = 0.4
            base_allocations[StrategyType.MEAN_REVERSION] = 0.3
            base_allocations[StrategyType.RANGE_TRADING] = 0.2
            base_allocations[StrategyType.CONTRARIAN] = 0.1

        elif current_regime == MarketRegime.BREAKOUT:
            base_allocations[StrategyType.BREAKOUT_MOMENTUM] = 0.5
            base_allocations[StrategyType.MOMENTUM_LONG] = 0.3
            base_allocations[StrategyType.TREND_FOLLOWING] = 0.2

        elif current_regime == MarketRegime.REVERSAL:
            base_allocations[StrategyType.CONTRARIAN] = 0.4
            base_allocations[StrategyType.MEAN_REVERSION] = 0.3
            base_allocations[StrategyType.VOLATILITY_CAPTURE] = 0.2
            base_allocations[StrategyType.RANGE_TRADING] = 0.1

        # Create strategy allocation objects
        for strategy_type, weight in base_allocations.items():
            if weight > 0:
                self.current_allocations[strategy_type] = StrategyAllocation(
                    strategy_type=strategy_type,
                    weight=weight,
                    confidence=self.regime_detector.regime_confidence,
                    max_position_size=regime_config.get("position_sizing", 1.0)
                    * self.max_single_position,
                    risk_multiplier=regime_config.get("risk_multiplier", 1.0),
                    rebalance_frequency=regime_config.get("rebalance_frequency", "daily"),
                )

    def update_regime_allocation(self) -> bool:
        """Update strategy allocation based on regime change."""
        previous_regime = getattr(self, "_previous_regime", None)
        current_regime = self.regime_detector.current_regime

        if previous_regime != current_regime:
            self.logger.info(f"Updating allocation for regime change: {current_regime.value}")

            # Save previous allocation for transition analysis
            if hasattr(self, "_previous_allocations"):
                self._analyze_regime_transition_performance()

            self._previous_allocations = self.current_allocations.copy()
            self._previous_regime = current_regime

            # Reinitialize allocations for new regime
            self._initialize_strategy_allocations()

            return True

        return False

    def calculate_volatility_target_sizing(
        self, symbol: str, base_weight: float, historical_returns: pd.Series
    ) -> float:
        """Calculate position size based on volatility targeting."""
        try:
            if len(historical_returns) < 20:
                return base_weight

            # Calculate realized volatility
            symbol_vol = historical_returns.std() * np.sqrt(252)  # Annualized

            if symbol_vol <= 0:
                return 0.0

            # Volatility targeting: scale position inverse to volatility
            vol_scalar = self.target_volatility / symbol_vol
            vol_scalar = np.clip(vol_scalar, 0.1, 3.0)  # Reasonable bounds

            # Apply volatility scaling
            adjusted_weight = base_weight * vol_scalar

            # Ensure within position limits
            max_pos = self.max_single_position
            adjusted_weight = min(adjusted_weight, max_pos)

            self.logger.debug(
                f"Vol targeting for {symbol}",
                base_weight=base_weight,
                symbol_vol=symbol_vol,
                vol_scalar=vol_scalar,
                adjusted_weight=adjusted_weight,
            )

            return adjusted_weight

        except Exception as e:
            self.logger.error(f"Error in volatility targeting for {symbol}: {e}")
            return base_weight

    def check_cluster_limits(self, new_positions: Dict[str, float]) -> Dict[str, float]:
        """Check and adjust positions against cluster limits."""
        adjusted_positions = new_positions.copy()

        for cluster_name, cluster in self.cluster_limits.items():
            # Calculate cluster exposure
            cluster_exposure = sum(
                adjusted_positions.get(symbol, 0.0) for symbol in cluster.symbols
            )

            # Check if cluster limit is exceeded
            if cluster_exposure > cluster.max_weight:
                self.logger.warning(
                    f"Cluster {cluster_name} exposure {cluster_exposure:.2%} "
                    f"exceeds limit {cluster.max_weight:.2%}"
                )

                # Scale down positions proportionally
                scale_factor = cluster.max_weight / cluster_exposure
                for symbol in cluster.symbols:
                    if symbol in adjusted_positions:
                        adjusted_positions[symbol] *= scale_factor

        return adjusted_positions

    def generate_position_targets(
        self, market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, PositionTarget]:
        """Generate position targets based on current strategy allocations."""
        position_targets = {}

        for strategy_type, allocation in self.current_allocations.items():
            if not allocation.active or allocation.weight <= 0:
                continue

            # Get strategy-specific position targets
            strategy_targets = self._get_strategy_positions(strategy_type, allocation, market_data)

            # Merge with overall targets
            for symbol, target_weight in strategy_targets.items():
                if symbol in position_targets:
                    # Combine multiple strategy signals
                    existing_target = position_targets[symbol]
                    combined_weight = (
                        existing_target.target_weight * existing_target.confidence
                        + target_weight * allocation.confidence
                    ) / (existing_target.confidence + allocation.confidence)

                    position_targets[symbol] = PositionTarget(
                        symbol=symbol,
                        target_weight=combined_weight,
                        current_weight=existing_target.current_weight,
                        confidence=(existing_target.confidence + allocation.confidence) / 2,
                        strategy_source=strategy_type,  # Last strategy wins
                    )
                else:
                    position_targets[symbol] = PositionTarget(
                        symbol=symbol,
                        target_weight=target_weight,
                        current_weight=0.0,  # Will be updated with actual positions
                        confidence=allocation.confidence,
                        strategy_source=strategy_type,
                    )

        # Apply volatility targeting
        for symbol, target in position_targets.items():
            if symbol in market_data:
                returns = market_data[symbol]["close"].pct_change().dropna()
                target.target_weight = self.calculate_volatility_target_sizing(
                    symbol, target.target_weight, returns
                )

        # Apply cluster limits
        raw_weights = {symbol: target.target_weight for symbol, target in position_targets.items()}
        adjusted_weights = self.check_cluster_limits(raw_weights)

        # Update targets with adjusted weights
        for symbol, adjusted_weight in adjusted_weights.items():
            if symbol in position_targets:
                position_targets[symbol].target_weight = adjusted_weight

        return position_targets

    def _get_strategy_positions(
        self,
        strategy_type: StrategyType,
        allocation: StrategyAllocation,
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Get position targets for specific strategy."""
        positions = {}

        # Strategy-specific logic (simplified for demo)
        if strategy_type == StrategyType.MOMENTUM_LONG:
            positions = self._momentum_long_positions(market_data, allocation)
        elif strategy_type == StrategyType.MEAN_REVERSION:
            positions = self._mean_reversion_positions(market_data, allocation)
        elif strategy_type == StrategyType.BREAKOUT_MOMENTUM:
            positions = self._breakout_momentum_positions(market_data, allocation)
        elif strategy_type == StrategyType.RANGE_TRADING:
            positions = self._range_trading_positions(market_data, allocation)
        elif strategy_type == StrategyType.VOLATILITY_CAPTURE:
            positions = self._volatility_capture_positions(market_data, allocation)
        # Add other strategies...

        return positions

    def _momentum_long_positions(
        self, market_data: Dict[str, pd.DataFrame], allocation: StrategyAllocation
    ) -> Dict[str, float]:
        """Generate momentum long positions."""
        positions = {}

        # Calculate momentum scores for all symbols
        momentum_scores = {}
        for symbol, df in market_data.items():
            if len(df) < 20:
                continue

            # Simple momentum: 20-day return
            momentum = df["close"].iloc[-1] / df["close"].iloc[-21] - 1
            momentum_scores[symbol] = momentum

        # Select top momentum symbols
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = sorted_symbols[:5]  # Top 5 momentum

        # Equal weight allocation within strategy budget
        weight_per_symbol = allocation.weight / len(top_symbols) if top_symbols else 0

        for symbol, momentum in top_symbols:
            if momentum > 0.05:  # At least 5% momentum
                positions[symbol] = weight_per_symbol

        return positions

    def _mean_reversion_positions(
        self, market_data: Dict[str, pd.DataFrame], allocation: StrategyAllocation
    ) -> Dict[str, float]:
        """Generate mean reversion positions."""
        positions = {}

        for symbol, df in market_data.items():
            if len(df) < 50:
                continue

            # Calculate distance from 50-day MA
            ma_50 = df["close"].rolling(50).mean().iloc[-1]
            current_price = df["close"].iloc[-1]
            distance = (current_price - ma_50) / ma_50

            # RSI for oversold/overbought
            rsi = self._get_technical_analyzer().calculate_indicator("RSI", df["close"]).values.iloc[-1]

            # Mean reversion signal
            if distance < -0.1 and rsi < 30:  # Oversold
                weight = allocation.weight * 0.3  # Partial allocation
                positions[symbol] = weight
            elif distance > 0.1 and rsi > 70:  # Overbought (short signal)
                weight = -allocation.weight * 0.2  # Short position
                positions[symbol] = weight

        return positions

    def _breakout_momentum_positions(
        self, market_data: Dict[str, pd.DataFrame], allocation: StrategyAllocation
    ) -> Dict[str, float]:
        """Generate breakout momentum positions."""
        positions = {}

        for symbol, df in market_data.items():
            if len(df) < 20:
                continue

            # Bollinger Band breakout
            ma_20 = df["close"].rolling(20).mean()
            std_20 = df["close"].rolling(20).std()
            upper_band = ma_20 + (2 * std_20)
            lower_band = ma_20 - (2 * std_20)

            current_price = df["close"].iloc[-1]
            volume_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]

            # Breakout signals
            if current_price > upper_band.iloc[-1] and volume_ratio > 1.5:
                positions[symbol] = allocation.weight * 0.4
            elif current_price < lower_band.iloc[-1] and volume_ratio > 1.5:
                positions[symbol] = -allocation.weight * 0.3  # Short breakout

        return positions

    def _range_trading_positions(
        self, market_data: Dict[str, pd.DataFrame], allocation: StrategyAllocation
    ) -> Dict[str, float]:
        """Generate range trading positions."""
        # Similar implementation to mean reversion but with different parameters
        return self._mean_reversion_positions(market_data, allocation)

    def _volatility_capture_positions(
        self, market_data: Dict[str, pd.DataFrame], allocation: StrategyAllocation
    ) -> Dict[str, float]:
        """Generate volatility capture positions."""
        positions = {}

        for symbol, df in market_data.items():
            if len(df) < 30:
                continue

            # Volatility expansion/contraction signals
            current_vol = df["close"].pct_change().rolling(20).std().iloc[-1]
            avg_vol = df["close"].pct_change().rolling(60).std().mean()

            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

            # Volatility regime positions
            if vol_ratio > 1.5:  # High volatility
                positions[symbol] = allocation.weight * 0.2
            elif vol_ratio < 0.7:  # Low volatility (anticipate expansion)
                positions[symbol] = allocation.weight * 0.3

        return positions

    def _get_technical_analyzer().calculate_indicator("RSI", self, prices: pd.Series, period: int = 14).values -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def should_rebalance(self, frequency: str) -> bool:
        """Check if rebalancing is needed based on frequency."""
        now = datetime.now()

        if frequency == "hourly":
            return True  # Always rebalance for hourly
        elif frequency == "daily":
            # Rebalance once per day
            return getattr(self, "_last_daily_rebalance", datetime.min).date() < now.date()
        elif frequency == "weekly":
            # Rebalance once per week
            last_rebalance = getattr(self, "_last_weekly_rebalance", datetime.min)
            return (now - last_rebalance).days >= 7

        return False

    def calculate_required_trades(
        self, current_positions: Dict[str, float], target_positions: Dict[str, PositionTarget]
    ) -> Dict[str, float]:
        """Calculate required trades to reach target positions."""
        trades = {}

        # Get all symbols (current and target)
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_weight = current_positions.get(symbol, 0.0)
            target_weight = (
                target_positions[symbol].target_weight if symbol in target_positions else 0.0
            )

            trade_size = target_weight - current_weight

            # Only trade if above minimum threshold
            if abs(trade_size) > self.rebalance_threshold:
                trade_value = trade_size * self.current_portfolio_value
                if abs(trade_value) > self.min_trade_size:
                    trades[symbol] = trade_size

        return trades

    def _analyze_regime_transition_performance(self) -> None:
        """Analyze performance during regime transitions."""
        # This would track performance metrics during regime changes
        # Implementation would analyze returns, drawdowns, etc.
        pass

    def get_strategy_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current strategy allocations."""
        return {
            "current_regime": self.regime_detector.current_regime.value,
            "regime_confidence": self.regime_detector.regime_confidence,
            "allocations": {
                strategy.strategy_type.value: {
                    "weight": strategy.weight,
                    "confidence": strategy.confidence,
                    "max_position_size": strategy.max_position_size,
                    "active": strategy.active,
                }
                for strategy in self.current_allocations.values()
            },
            "cluster_limits": {
                cluster.cluster_name: {
                    "max_weight": cluster.max_weight,
                    "current_weight": cluster.current_weight,
                    "utilization": cluster.current_weight / cluster.max_weight,
                }
                for cluster in self.cluster_limits.values()
            },
        }

    def save_strategy_state(self) -> None:
        """Save current strategy state."""
        state = {
            "current_allocations": {
                strategy_type.value: {
                    "weight": allocation.weight,
                    "confidence": allocation.confidence,
                    "max_position_size": allocation.max_position_size,
                    "risk_multiplier": allocation.risk_multiplier,
                    "rebalance_frequency": allocation.rebalance_frequency,
                    "active": allocation.active,
                }
                for strategy_type, allocation in self.current_allocations.items()
            },
            "cluster_limits": {
                name: {
                    "max_weight": cluster.max_weight,
                    "current_weight": cluster.current_weight,
                    "symbols": cluster.symbols,
                }
                for name, cluster in self.cluster_limits.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        try:
            with open(self.data_path / "strategy_state.json", "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save strategy state: {e}")


def create_strategy_switcher(
    regime_detector: RegimeDetector, initial_capital: float = 100000.0
) -> StrategySwitcher:
    """Factory function to create StrategySwitcher instance."""
    return StrategySwitcher(regime_detector, initial_capital)
