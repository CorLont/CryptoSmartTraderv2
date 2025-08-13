"""
Strategy Switcher System

Regime-adaptive trading strategy management with per-regime
stop losses, take profits, and position sizing parameters.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .regime_detection import MarketRegime, RegimeClassification

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Trading strategy types"""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    HOLD = "hold"
    DEFENSIVE = "defensive"


@dataclass
class StrategyParameters:
    """Strategy-specific parameters"""

    strategy_type: StrategyType

    # Position sizing
    base_position_size: float = 1.0  # Base size multiplier
    max_position_size: float = 2.0  # Maximum size multiplier
    volatility_adjustment: bool = True

    # Stop losses
    stop_loss_pct: float = 2.0  # Stop loss percentage
    trailing_stop: bool = False
    stop_loss_atr_multiple: float = 2.0

    # Take profits
    take_profit_pct: float = 4.0  # Take profit percentage
    partial_profit_levels: List[float] = field(default_factory=lambda: [2.0, 4.0])
    profit_scaling: List[float] = field(default_factory=lambda: [0.5, 0.5])

    # Entry conditions
    entry_signal_strength: float = 0.7  # Minimum signal strength
    confirmation_required: bool = True
    max_concurrent_positions: int = 3

    # Risk management
    max_daily_trades: int = 10
    max_drawdown_stop: float = 5.0  # Stop trading if drawdown exceeds
    correlation_limit: float = 0.7  # Max correlation between positions

    # Timing
    hold_time_min_minutes: int = 15  # Minimum hold time
    hold_time_max_minutes: int = 480  # Maximum hold time (8 hours)
    exit_on_regime_change: bool = True


@dataclass
class RegimeStrategy:
    """Complete strategy configuration for a market regime"""

    regime: MarketRegime
    primary_strategy: StrategyParameters
    fallback_strategy: Optional[StrategyParameters] = None

    # Regime-specific adjustments
    confidence_threshold: float = 0.6  # Minimum confidence to use this strategy
    regime_duration_factor: float = 1.0  # Adjust based on regime duration

    # Performance tracking
    trades_executed: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    turnover_rate: float = 0.0

    # Adaptive parameters
    performance_score: float = 0.0
    last_optimization: Optional[datetime] = None


class StrategySwitcher:
    """
    Regime-adaptive strategy switching system
    """

    def __init__(self):
        # Strategy configurations per regime
        self.regime_strategies: Dict[MarketRegime, RegimeStrategy] = {}

        # Current active strategy
        self.active_strategy: Optional[RegimeStrategy] = None
        self.last_regime_change: Optional[datetime] = None

        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.regime_performance: Dict[str, Dict[str, float]] = {}

        # Setup default strategies
        self._setup_default_strategies()

        # Strategy switching rules
        self.min_regime_duration_minutes = 30  # Min time before switching
        self.confidence_decay_factor = 0.95  # Confidence decay over time

    def _setup_default_strategies(self):
        """Setup default strategy configurations for each regime"""

        # Trend Up Strategy - Aggressive trend following
        trend_up_params = StrategyParameters(
            strategy_type=StrategyType.TREND_FOLLOWING,
            base_position_size=1.2,
            max_position_size=2.0,
            volatility_adjustment=True,
            stop_loss_pct=1.5,
            trailing_stop=True,
            stop_loss_atr_multiple=1.5,
            take_profit_pct=6.0,
            partial_profit_levels=[3.0, 6.0],
            profit_scaling=[0.3, 0.7],
            entry_signal_strength=0.6,
            confirmation_required=False,
            max_concurrent_positions=4,
            max_daily_trades=15,
            hold_time_min_minutes=30,
            hold_time_max_minutes=720,
            exit_on_regime_change=True,
        )

        self.regime_strategies[MarketRegime.TREND_UP] = RegimeStrategy(
            regime=MarketRegime.TREND_UP,
            primary_strategy=trend_up_params,
            confidence_threshold=0.65,
        )

        # Trend Down Strategy - Conservative short-side following
        trend_down_params = StrategyParameters(
            strategy_type=StrategyType.TREND_FOLLOWING,
            base_position_size=0.8,  # More conservative
            max_position_size=1.5,
            volatility_adjustment=True,
            stop_loss_pct=2.0,  # Tighter stops
            trailing_stop=True,
            stop_loss_atr_multiple=1.8,
            take_profit_pct=4.0,  # Lower targets
            partial_profit_levels=[2.0, 4.0],
            profit_scaling=[0.5, 0.5],
            entry_signal_strength=0.7,  # Higher threshold
            confirmation_required=True,
            max_concurrent_positions=2,
            max_daily_trades=10,
            hold_time_min_minutes=20,
            hold_time_max_minutes=480,
            exit_on_regime_change=True,
        )

        self.regime_strategies[MarketRegime.TREND_DOWN] = RegimeStrategy(
            regime=MarketRegime.TREND_DOWN,
            primary_strategy=trend_down_params,
            confidence_threshold=0.7,
        )

        # Mean Reversion Strategy - Contrarian approach
        mean_rev_params = StrategyParameters(
            strategy_type=StrategyType.MEAN_REVERSION,
            base_position_size=1.0,
            max_position_size=1.5,
            volatility_adjustment=True,
            stop_loss_pct=2.5,
            trailing_stop=False,  # Fixed stops for mean reversion
            stop_loss_atr_multiple=2.0,
            take_profit_pct=3.0,  # Quick profits
            partial_profit_levels=[1.5, 3.0],
            profit_scaling=[0.6, 0.4],
            entry_signal_strength=0.75,
            confirmation_required=True,
            max_concurrent_positions=3,
            max_daily_trades=20,  # Higher frequency
            hold_time_min_minutes=10,
            hold_time_max_minutes=240,  # Shorter holds
            exit_on_regime_change=True,
        )

        self.regime_strategies[MarketRegime.MEAN_REVERSION] = RegimeStrategy(
            regime=MarketRegime.MEAN_REVERSION,
            primary_strategy=mean_rev_params,
            confidence_threshold=0.6,
        )

        # Chop Strategy - Minimal activity, defensive
        chop_params = StrategyParameters(
            strategy_type=StrategyType.DEFENSIVE,
            base_position_size=0.3,  # Very small positions
            max_position_size=0.6,
            volatility_adjustment=True,
            stop_loss_pct=1.0,  # Very tight stops
            trailing_stop=False,
            stop_loss_atr_multiple=1.0,
            take_profit_pct=1.5,  # Quick scalping profits
            partial_profit_levels=[0.8, 1.5],
            profit_scaling=[0.7, 0.3],
            entry_signal_strength=0.85,  # Very high threshold
            confirmation_required=True,
            max_concurrent_positions=1,
            max_daily_trades=5,  # Very low activity
            hold_time_min_minutes=5,
            hold_time_max_minutes=60,  # Very short holds
            exit_on_regime_change=False,  # Less sensitive to regime changes
        )

        self.regime_strategies[MarketRegime.CHOP] = RegimeStrategy(
            regime=MarketRegime.CHOP, primary_strategy=chop_params, confidence_threshold=0.5
        )

        # Breakout Strategy - Aggressive momentum capture
        breakout_params = StrategyParameters(
            strategy_type=StrategyType.BREAKOUT,
            base_position_size=1.5,  # Larger positions for breakouts
            max_position_size=2.5,
            volatility_adjustment=True,
            stop_loss_pct=3.0,  # Wider stops for volatility
            trailing_stop=True,
            stop_loss_atr_multiple=2.5,
            take_profit_pct=8.0,  # Higher targets
            partial_profit_levels=[4.0, 8.0],
            profit_scaling=[0.4, 0.6],
            entry_signal_strength=0.8,  # High confidence required
            confirmation_required=True,
            max_concurrent_positions=2,
            max_daily_trades=8,
            hold_time_min_minutes=60,
            hold_time_max_minutes=1440,  # Can hold longer
            exit_on_regime_change=False,  # Let breakouts run
        )

        self.regime_strategies[MarketRegime.BREAKOUT] = RegimeStrategy(
            regime=MarketRegime.BREAKOUT,
            primary_strategy=breakout_params,
            confidence_threshold=0.75,
        )

        # Volatility Spike Strategy - Risk-off approach
        vol_spike_params = StrategyParameters(
            strategy_type=StrategyType.HOLD,
            base_position_size=0.1,  # Minimal exposure
            max_position_size=0.2,
            volatility_adjustment=False,
            stop_loss_pct=0.5,  # Very tight stops
            trailing_stop=False,
            stop_loss_atr_multiple=0.5,
            take_profit_pct=1.0,  # Quick profits
            partial_profit_levels=[0.5, 1.0],
            profit_scaling=[0.8, 0.2],
            entry_signal_strength=0.95,  # Almost no trading
            confirmation_required=True,
            max_concurrent_positions=1,
            max_daily_trades=2,  # Minimal activity
            hold_time_min_minutes=5,
            hold_time_max_minutes=30,  # Very short holds
            exit_on_regime_change=True,
        )

        self.regime_strategies[MarketRegime.VOLATILITY_SPIKE] = RegimeStrategy(
            regime=MarketRegime.VOLATILITY_SPIKE,
            primary_strategy=vol_spike_params,
            confidence_threshold=0.8,
        )

    def select_strategy(
        self, regime_classification: RegimeClassification
    ) -> Optional[RegimeStrategy]:
        """Select appropriate strategy based on regime classification"""

        try:
            regime = regime_classification.regime
            confidence = regime_classification.probability

            # Check if regime strategy exists
            if regime not in self.regime_strategies:
                logger.warning(f"No strategy defined for regime: {regime.value}")
                return None

            regime_strategy = self.regime_strategies[regime]

            # Check confidence threshold
            if confidence < regime_strategy.confidence_threshold:
                logger.info(
                    f"Regime confidence ({confidence:.2f}) below threshold ({regime_strategy.confidence_threshold:.2f})"
                )

                # Use defensive strategy for low confidence
                if MarketRegime.CHOP in self.regime_strategies:
                    return self.regime_strategies[MarketRegime.CHOP]
                return None

            # Check regime duration for stability
            if (
                regime_classification.regime_duration_minutes < self.min_regime_duration_minutes
                and self.active_strategy
                and self.active_strategy.regime != regime
            ):
                logger.info(
                    f"Regime duration ({regime_classification.regime_duration_minutes}min) too short for strategy switch"
                )
                return self.active_strategy  # Keep current strategy

            return regime_strategy

        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return None

    def switch_strategy(self, regime_classification: RegimeClassification) -> bool:
        """Switch to regime-appropriate strategy"""

        try:
            new_strategy = self.select_strategy(regime_classification)

            if not new_strategy:
                return False

            # Check if strategy change is needed
            if self.active_strategy and self.active_strategy.regime == new_strategy.regime:
                return True  # Already using correct strategy

            # Log strategy switch
            old_strategy = self.active_strategy.regime.value if self.active_strategy else "None"
            logger.info(f"Strategy switch: {old_strategy} → {new_strategy.regime.value}")

            # Update active strategy
            self.active_strategy = new_strategy
            self.last_regime_change = datetime.now()

            # Record strategy change
            self._record_strategy_change(regime_classification, new_strategy)

            return True

        except Exception as e:
            logger.error(f"Strategy switching failed: {e}")
            return False

    def get_current_parameters(self) -> Optional[StrategyParameters]:
        """Get current strategy parameters"""

        if not self.active_strategy:
            return None

        return self.active_strategy.primary_strategy

    def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        portfolio_value: float,
        risk_per_trade: float = 0.02,
    ) -> float:
        """Calculate regime-adjusted position size"""

        try:
            if not self.active_strategy:
                return 0.0

            params = self.active_strategy.primary_strategy

            # Base size calculation
            base_size = portfolio_value * risk_per_trade

            # Apply strategy multiplier
            strategy_adjusted = base_size * params.base_position_size

            # Signal strength adjustment
            signal_adjusted = strategy_adjusted * min(
                signal_strength / params.entry_signal_strength, 1.0
            )

            # Volatility adjustment
            if params.volatility_adjustment:
                # Reduce size for high volatility
                vol_adjustment = 1.0 / (1.0 + volatility * 2)
                signal_adjusted *= vol_adjustment

            # Apply maximum size limit
            max_size = portfolio_value * risk_per_trade * params.max_position_size
            final_size = min(signal_adjusted, max_size)

            return max(0.0, final_size)

        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0

    def get_stop_loss_level(self, entry_price: float, side: str, atr_value: float) -> float:
        """Calculate regime-appropriate stop loss level"""

        try:
            if not self.active_strategy:
                return entry_price * 0.95 if side == "buy" else entry_price * 1.05

            params = self.active_strategy.primary_strategy

            # Choose between percentage or ATR-based stop
            pct_stop_distance = entry_price * (params.stop_loss_pct / 100)
            atr_stop_distance = atr_value * params.stop_loss_atr_multiple

            # Use the more conservative (smaller) distance
            stop_distance = min(pct_stop_distance, atr_stop_distance)

            if side.lower() == "buy":
                return entry_price - stop_distance
            else:  # sell
                return entry_price + stop_distance

        except Exception as e:
            logger.error(f"Stop loss calculation failed: {e}")
            if side.lower() == "buy":
                return entry_price * 0.98
            else:
                return entry_price * 1.02

    def get_take_profit_levels(self, entry_price: float, side: str) -> List[Tuple[float, float]]:
        """Get regime-appropriate take profit levels with scaling"""

        try:
            if not self.active_strategy:
                # Default take profit
                if side.lower() == "buy":
                    return [(entry_price * 1.02, 1.0)]
                else:
                    return [(entry_price * 0.98, 1.0)]

            params = self.active_strategy.primary_strategy
            tp_levels = []

            for i, (tp_pct, scaling) in enumerate(
                zip(params.partial_profit_levels, params.profit_scaling)
            ):
                if side.lower() == "buy":
                    tp_price = entry_price * (1 + tp_pct / 100)
                else:
                    tp_price = entry_price * (1 - tp_pct / 100)

                tp_levels.append((tp_price, scaling))

            return tp_levels if tp_levels else [(entry_price * 1.02, 1.0)]

        except Exception as e:
            logger.error(f"Take profit calculation failed: {e}")
            return [(entry_price * 1.02, 1.0)]

    def should_enter_trade(
        self, signal_strength: float, current_positions: int, daily_trades_count: int
    ) -> bool:
        """Check if trade entry should be allowed based on regime strategy"""

        try:
            if not self.active_strategy:
                return False

            params = self.active_strategy.primary_strategy

            # Signal strength check
            if signal_strength < params.entry_signal_strength:
                return False

            # Position limit check
            if current_positions >= params.max_concurrent_positions:
                return False

            # Daily trade limit check
            if daily_trades_count >= params.max_daily_trades:
                return False

            return True

        except Exception as e:
            logger.error(f"Trade entry check failed: {e}")
            return False

    def should_exit_on_regime_change(self, new_regime: MarketRegime) -> bool:
        """Check if positions should be closed on regime change"""

        try:
            if not self.active_strategy:
                return False

            # Check if current strategy has exit-on-regime-change enabled
            if not self.active_strategy.primary_strategy.exit_on_regime_change:
                return False

            # Check if regime actually changed
            if self.active_strategy.regime == new_regime:
                return False

            return True

        except Exception as e:
            logger.error(f"Regime change exit check failed: {e}")
            return False

    def update_strategy_performance(
        self, regime: MarketRegime, trade_pnl: float, trade_duration_minutes: int, win: bool
    ):
        """Update performance metrics for regime strategy"""

        try:
            if regime not in self.regime_strategies:
                return

            strategy = self.regime_strategies[regime]
            strategy.trades_executed += 1

            # Update win rate
            if strategy.trades_executed == 1:
                strategy.win_rate = 1.0 if win else 0.0
            else:
                # Exponentially weighted average
                alpha = 0.1
                strategy.win_rate = alpha * (1.0 if win else 0.0) + (1 - alpha) * strategy.win_rate

            # Update turnover rate (simplified)
            strategy.turnover_rate = strategy.trades_executed / max(
                1, (datetime.now() - (datetime.now() - timedelta(days=1))).total_seconds() / 86400
            )

            # Store performance record
            perf_record = {
                "timestamp": datetime.now().isoformat(),
                "regime": regime.value,
                "pnl": trade_pnl,
                "duration_minutes": trade_duration_minutes,
                "win": win,
                "win_rate": strategy.win_rate,
                "turnover": strategy.turnover_rate,
            }

            self.performance_history.append(perf_record)

            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-500:]

        except Exception as e:
            logger.error(f"Performance update failed: {e}")

    def _record_strategy_change(
        self, regime_classification: RegimeClassification, new_strategy: RegimeStrategy
    ):
        """Record strategy change for analysis"""

        change_record = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime_classification.regime.value,
            "confidence": regime_classification.probability,
            "strategy": new_strategy.primary_strategy.strategy_type.value,
            "regime_duration": regime_classification.regime_duration_minutes,
        }

        # Would store in database or log file
        logger.info(f"Strategy change recorded: {change_record}")

    def get_regime_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary by regime"""

        try:
            summary = {}

            for regime, strategy in self.regime_strategies.items():
                regime_trades = [
                    p for p in self.performance_history if p.get("regime") == regime.value
                ]

                if regime_trades:
                    total_pnl = sum(t["pnl"] for t in regime_trades)
                    avg_duration = np.mean([t["duration_minutes"] for t in regime_trades])
                    win_rate = np.mean([t["win"] for t in regime_trades])

                    # Calculate Sharpe ratio (simplified)
                    pnl_values = [t["pnl"] for t in regime_trades]
                    if len(pnl_values) > 1:
                        returns_mean = np.mean(pnl_values)
                        returns_std = np.std(pnl_values)
                        sharpe = returns_mean / returns_std if returns_std > 0 else 0
                    else:
                        sharpe = 0

                    summary[regime.value] = {
                        "trades": len(regime_trades),
                        "total_pnl": total_pnl,
                        "win_rate": win_rate * 100,
                        "avg_duration_minutes": avg_duration,
                        "sharpe_ratio": sharpe,
                        "turnover_rate": strategy.turnover_rate,
                        "current_strategy": strategy.primary_strategy.strategy_type.value,
                    }
                else:
                    summary[regime.value] = {
                        "trades": 0,
                        "total_pnl": 0,
                        "win_rate": 0,
                        "avg_duration_minutes": 0,
                        "sharpe_ratio": 0,
                        "turnover_rate": 0,
                        "current_strategy": strategy.primary_strategy.strategy_type.value,
                    }

            return summary

        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {}

    def optimize_strategy_parameters(self, regime: MarketRegime, performance_window_days: int = 30):
        """Optimize strategy parameters based on recent performance"""

        try:
            if regime not in self.regime_strategies:
                return

            # Get recent performance data
            cutoff_date = datetime.now() - timedelta(days=performance_window_days)
            recent_trades = [
                p
                for p in self.performance_history
                if p.get("regime") == regime.value
                and datetime.fromisoformat(p["timestamp"]) >= cutoff_date
            ]

            if len(recent_trades) < 10:
                logger.info(f"Insufficient data for {regime.value} optimization")
                return

            strategy = self.regime_strategies[regime]
            current_sharpe = np.mean([t.get("pnl", 0) for t in recent_trades]) / (
                np.std([t.get("pnl", 0) for t in recent_trades]) or 1
            )

            # Simple parameter optimization
            # Adjust stop loss based on win rate
            win_rate = np.mean([t.get("win", False) for t in recent_trades])

            if win_rate < 0.4:  # Low win rate, tighten stops
                strategy.primary_strategy.stop_loss_pct *= 0.9
                logger.info(
                    f"Tightened stops for {regime.value}: {strategy.primary_strategy.stop_loss_pct:.2f}%"
                )
            elif win_rate > 0.7:  # High win rate, widen stops
                strategy.primary_strategy.stop_loss_pct *= 1.1
                logger.info(
                    f"Widened stops for {regime.value}: {strategy.primary_strategy.stop_loss_pct:.2f}%"
                )

            # Adjust position size based on Sharpe ratio
            if current_sharpe > 1.5:  # Good performance, increase size
                strategy.primary_strategy.base_position_size *= 1.05
            elif current_sharpe < 0.5:  # Poor performance, decrease size
                strategy.primary_strategy.base_position_size *= 0.95

            # Clamp parameters to reasonable ranges
            strategy.primary_strategy.stop_loss_pct = np.clip(
                strategy.primary_strategy.stop_loss_pct, 0.5, 5.0
            )
            strategy.primary_strategy.base_position_size = np.clip(
                strategy.primary_strategy.base_position_size, 0.1, 2.0
            )

            strategy.last_optimization = datetime.now()

        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")

    def get_strategy_status(self) -> Dict[str, Any]:
        """Get current strategy system status"""

        return {
            "active_strategy": self.active_strategy.regime.value if self.active_strategy else None,
            "strategy_type": self.active_strategy.primary_strategy.strategy_type.value
            if self.active_strategy
            else None,
            "last_regime_change": self.last_regime_change.isoformat()
            if self.last_regime_change
            else None,
            "total_regime_strategies": len(self.regime_strategies),
            "performance_records": len(self.performance_history),
            "current_parameters": {
                "position_size": self.active_strategy.primary_strategy.base_position_size
                if self.active_strategy
                else 0,
                "stop_loss_pct": self.active_strategy.primary_strategy.stop_loss_pct
                if self.active_strategy
                else 0,
                "take_profit_pct": self.active_strategy.primary_strategy.take_profit_pct
                if self.active_strategy
                else 0,
                "max_daily_trades": self.active_strategy.primary_strategy.max_daily_trades
                if self.active_strategy
                else 0,
            },
        }
