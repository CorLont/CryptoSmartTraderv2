"""
Regime-Adaptive Trading Strategies

Different trading approaches for each market regime:
- Trend Following for trending markets
- Mean Reversion for range-bound markets  
- Risk Management for choppy/dangerous markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from .regime_models import MarketRegime, RegimeClassification

logger = logging.getLogger(__name__)

@dataclass
class TradingParameters:
    """Trading parameters adapted for specific regime"""
    entry_threshold: float      # Signal strength required for entry
    exit_threshold: float       # Signal strength for exit
    stop_loss_pct: float       # Stop loss percentage
    take_profit_pct: float     # Take profit percentage
    position_size_pct: float   # Position size as % of capital
    max_hold_time: int         # Maximum holding period (hours)
    rebalance_frequency: int   # Rebalancing frequency (hours)
    no_trade: bool            # Whether to avoid trading entirely
    strategy_type: str        # Strategy name for this regime


class RegimeStrategies:
    """
    Adaptive trading strategies based on market regime
    """
    
    def __init__(self):
        self.strategy_configs = self._initialize_strategy_configs()
        self.current_regime = None
        self.current_parameters = None
        
    def _initialize_strategy_configs(self) -> Dict[MarketRegime, TradingParameters]:
        """Initialize trading parameters for each regime"""
        
        return {
            MarketRegime.TREND_UP: TradingParameters(
                entry_threshold=0.65,       # Lower threshold for trend following
                exit_threshold=0.35,        # Hold until trend weakens
                stop_loss_pct=3.0,         # Wider stops in trending markets
                take_profit_pct=8.0,       # Higher targets
                position_size_pct=15.0,    # Larger positions in strong trends
                max_hold_time=72,          # Longer holding periods
                rebalance_frequency=4,     # Less frequent rebalancing
                no_trade=False,
                strategy_type="momentum_trend"
            ),
            
            MarketRegime.TREND_DOWN: TradingParameters(
                entry_threshold=0.70,       # Slightly higher threshold for shorts
                exit_threshold=0.40,        # Quick exits on reversal signs
                stop_loss_pct=2.5,         # Tighter stops for downtrends
                take_profit_pct=6.0,       # More conservative targets
                position_size_pct=12.0,    # Smaller positions (higher risk)
                max_hold_time=48,          # Shorter holding (downtrends faster)
                rebalance_frequency=3,     # More frequent monitoring
                no_trade=False,
                strategy_type="momentum_short"
            ),
            
            MarketRegime.MEAN_REVERSION: TradingParameters(
                entry_threshold=0.75,       # Higher threshold for mean reversion
                exit_threshold=0.45,        # Quick exits when mean reached
                stop_loss_pct=2.0,         # Tight stops (should revert quickly)
                take_profit_pct=3.0,       # Smaller targets (limited range)
                position_size_pct=10.0,    # Moderate positions
                max_hold_time=24,          # Short holding periods
                rebalance_frequency=2,     # Frequent rebalancing
                no_trade=False,
                strategy_type="mean_reversion"
            ),
            
            MarketRegime.LOW_VOL_DRIFT: TradingParameters(
                entry_threshold=0.80,       # Very high threshold
                exit_threshold=0.30,        # Hold longer in low vol
                stop_loss_pct=1.5,         # Very tight stops
                take_profit_pct=2.5,       # Small targets
                position_size_pct=8.0,     # Small positions
                max_hold_time=36,          # Medium holding
                rebalance_frequency=6,     # Less frequent (low activity)
                no_trade=False,
                strategy_type="low_vol_scalp"
            ),
            
            MarketRegime.HIGH_VOL_CHOP: TradingParameters(
                entry_threshold=0.90,       # Extremely high threshold
                exit_threshold=0.50,        # Quick exits
                stop_loss_pct=1.0,         # Very tight stops
                take_profit_pct=1.5,       # Very small targets
                position_size_pct=3.0,     # Minimal positions
                max_hold_time=12,          # Very short holding
                rebalance_frequency=1,     # Constant monitoring
                no_trade=True,             # Prefer to avoid trading
                strategy_type="survival_mode"
            ),
            
            MarketRegime.RISK_OFF: TradingParameters(
                entry_threshold=0.95,       # Almost no trading
                exit_threshold=0.20,        # Quick exits if in positions
                stop_loss_pct=0.5,         # Extremely tight stops
                take_profit_pct=1.0,       # Tiny targets
                position_size_pct=1.0,     # Minimal exposure
                max_hold_time=6,           # Very short holding
                rebalance_frequency=1,     # Constant monitoring
                no_trade=True,             # Avoid trading
                strategy_type="capital_preservation"
            )
        }
    
    def get_strategy_for_regime(self, regime_classification: RegimeClassification) -> TradingParameters:
        """
        Get trading parameters adapted for current regime
        
        Args:
            regime_classification: Current regime classification
            
        Returns:
            TradingParameters optimized for the regime
        """
        base_params = self.strategy_configs[regime_classification.primary_regime]
        
        # Adjust parameters based on confidence
        confidence_factor = regime_classification.confidence
        
        # Create adjusted parameters
        adjusted_params = TradingParameters(
            entry_threshold=base_params.entry_threshold + (0.1 * (1 - confidence_factor)),
            exit_threshold=base_params.exit_threshold,
            stop_loss_pct=base_params.stop_loss_pct * (2 - confidence_factor),  # Tighter stops when less confident
            take_profit_pct=base_params.take_profit_pct * confidence_factor,    # Smaller targets when less confident
            position_size_pct=base_params.position_size_pct * confidence_factor,  # Smaller positions when less confident
            max_hold_time=base_params.max_hold_time,
            rebalance_frequency=base_params.rebalance_frequency,
            no_trade=base_params.no_trade or confidence_factor < 0.6,  # Avoid trading if low confidence
            strategy_type=f"{base_params.strategy_type}_conf_{confidence_factor:.1f}"
        )
        
        self.current_regime = regime_classification.primary_regime
        self.current_parameters = adjusted_params
        
        return adjusted_params
    
    def should_enter_position(self, signal_strength: float, 
                             regime_params: TradingParameters) -> Dict[str, Any]:
        """
        Determine if conditions are met for position entry
        
        Args:
            signal_strength: ML prediction confidence (0-1)
            regime_params: Current regime trading parameters
            
        Returns:
            Entry decision with rationale
        """
        if regime_params.no_trade:
            return {
                "enter": False,
                "reason": f"No-trade regime: {regime_params.strategy_type}",
                "risk_level": "high"
            }
        
        if signal_strength < regime_params.entry_threshold:
            return {
                "enter": False,
                "reason": f"Signal strength {signal_strength:.3f} below threshold {regime_params.entry_threshold:.3f}",
                "risk_level": "medium"
            }
        
        # Additional regime-specific checks
        regime_check = self._regime_specific_entry_check(signal_strength, regime_params)
        
        if not regime_check["allowed"]:
            return {
                "enter": False,
                "reason": regime_check["reason"],
                "risk_level": regime_check.get("risk_level", "medium")
            }
        
        return {
            "enter": True,
            "reason": f"Signal {signal_strength:.3f} exceeds threshold, regime allows entry",
            "risk_level": "low",
            "position_size": regime_params.position_size_pct,
            "stop_loss": regime_params.stop_loss_pct,
            "take_profit": regime_params.take_profit_pct,
            "max_hold_time": regime_params.max_hold_time
        }
    
    def should_exit_position(self, current_pnl_pct: float, 
                           hold_time_hours: int,
                           signal_strength: float,
                           regime_params: TradingParameters) -> Dict[str, Any]:
        """
        Determine if position should be closed
        
        Args:
            current_pnl_pct: Current P&L as percentage
            hold_time_hours: How long position has been held
            signal_strength: Current ML signal strength
            regime_params: Current regime parameters
            
        Returns:
            Exit decision with rationale
        """
        exit_reasons = []
        
        # Take profit check
        if current_pnl_pct >= regime_params.take_profit_pct:
            return {
                "exit": True,
                "reason": f"Take profit hit: {current_pnl_pct:.2f}% >= {regime_params.take_profit_pct:.2f}%",
                "exit_type": "take_profit"
            }
        
        # Stop loss check
        if current_pnl_pct <= -regime_params.stop_loss_pct:
            return {
                "exit": True,
                "reason": f"Stop loss hit: {current_pnl_pct:.2f}% <= -{regime_params.stop_loss_pct:.2f}%",
                "exit_type": "stop_loss"
            }
        
        # Maximum hold time
        if hold_time_hours >= regime_params.max_hold_time:
            return {
                "exit": True,
                "reason": f"Max hold time reached: {hold_time_hours}h >= {regime_params.max_hold_time}h",
                "exit_type": "time_limit"
            }
        
        # Signal reversal
        if signal_strength <= regime_params.exit_threshold:
            return {
                "exit": True,
                "reason": f"Signal weakened: {signal_strength:.3f} <= {regime_params.exit_threshold:.3f}",
                "exit_type": "signal_reversal"
            }
        
        # Regime change to no-trade
        if regime_params.no_trade:
            return {
                "exit": True,
                "reason": f"Regime changed to no-trade: {regime_params.strategy_type}",
                "exit_type": "regime_change"
            }
        
        return {
            "exit": False,
            "reason": "All exit conditions negative",
            "current_pnl": current_pnl_pct,
            "hold_time": hold_time_hours
        }
    
    def _regime_specific_entry_check(self, signal_strength: float, 
                                   regime_params: TradingParameters) -> Dict[str, Any]:
        """Additional regime-specific entry validation"""
        
        if regime_params.strategy_type.startswith("momentum"):
            # For momentum strategies, need strong persistent signal
            return {
                "allowed": signal_strength > 0.7,
                "reason": "Momentum strategy requires high conviction signal",
                "risk_level": "low" if signal_strength > 0.8 else "medium"
            }
        
        elif regime_params.strategy_type.startswith("mean_reversion"):
            # For mean reversion, need extreme signal (oversold/overbought)
            return {
                "allowed": signal_strength > 0.75,
                "reason": "Mean reversion requires extreme signal",
                "risk_level": "medium"
            }
        
        elif regime_params.strategy_type.startswith("low_vol"):
            # For low volatility, need very high confidence
            return {
                "allowed": signal_strength > 0.8,
                "reason": "Low vol requires very high confidence",
                "risk_level": "low"
            }
        
        else:
            # Default case
            return {
                "allowed": True,
                "reason": "Default entry check passed",
                "risk_level": "medium"
            }
    
    def get_position_sizing_multiplier(self, market_conditions: Dict[str, float]) -> float:
        """
        Adjust position sizing based on market conditions
        
        Args:
            market_conditions: {'volatility': 0.5, 'liquidity': 0.8, 'correlation': 0.3}
            
        Returns:
            Multiplier to apply to base position size (0.1 to 2.0)
        """
        multiplier = 1.0
        
        # Adjust for volatility
        volatility = market_conditions.get('volatility', 0.5)
        if volatility > 0.7:
            multiplier *= 0.5  # Reduce size in high vol
        elif volatility < 0.3:
            multiplier *= 1.3  # Increase size in low vol
        
        # Adjust for liquidity
        liquidity = market_conditions.get('liquidity', 0.8)
        if liquidity < 0.5:
            multiplier *= 0.7  # Reduce size in low liquidity
        
        # Adjust for correlation
        correlation = market_conditions.get('correlation', 0.3)
        if correlation > 0.8:
            multiplier *= 0.8  # Reduce size when everything moves together
        
        # Ensure reasonable bounds
        return max(0.1, min(2.0, multiplier))
    
    def get_rebalancing_schedule(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get rebalancing schedule for current regime
        
        Returns:
            Rebalancing configuration
        """
        params = self.strategy_configs[regime]
        
        return {
            "frequency_hours": params.rebalance_frequency,
            "trigger_threshold": 0.1,  # Rebalance if allocation drifts > 10%
            "max_trades_per_rebalance": 5,
            "min_trade_size": 0.01,  # Minimum 1% position
            "strategy_type": params.strategy_type
        }
    
    def get_risk_limits_for_regime(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get risk limits adapted for current regime
        
        Returns:
            Risk management parameters
        """
        params = self.strategy_configs[regime]
        
        base_limits = {
            "max_portfolio_risk": 0.02,  # 2% portfolio risk per trade
            "max_single_position": 0.20,  # 20% max single position
            "max_sector_exposure": 0.40,  # 40% max sector exposure
            "max_daily_trades": 10,
            "max_correlation_exposure": 0.60
        }
        
        # Adjust based on regime
        if regime in [MarketRegime.HIGH_VOL_CHOP, MarketRegime.RISK_OFF]:
            # Much more conservative in dangerous regimes
            return {
                "max_portfolio_risk": 0.005,  # 0.5%
                "max_single_position": 0.05,   # 5%
                "max_sector_exposure": 0.15,   # 15%
                "max_daily_trades": 3,
                "max_correlation_exposure": 0.30
            }
        
        elif regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]:
            # More aggressive in trending markets
            return {
                "max_portfolio_risk": 0.03,  # 3%
                "max_single_position": 0.30,  # 30%
                "max_sector_exposure": 0.50,  # 50%
                "max_daily_trades": 15,
                "max_correlation_exposure": 0.70
            }
        
        else:
            return base_limits
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of current strategy configuration"""
        if self.current_regime is None or self.current_parameters is None:
            return {"status": "No active regime"}
        
        return {
            "current_regime": self.current_regime.value,
            "strategy_type": self.current_parameters.strategy_type,
            "trading_allowed": not self.current_parameters.no_trade,
            "entry_threshold": self.current_parameters.entry_threshold,
            "position_size_pct": self.current_parameters.position_size_pct,
            "stop_loss_pct": self.current_parameters.stop_loss_pct,
            "take_profit_pct": self.current_parameters.take_profit_pct,
            "max_hold_hours": self.current_parameters.max_hold_time,
            "rebalance_frequency": self.current_parameters.rebalance_frequency
        }