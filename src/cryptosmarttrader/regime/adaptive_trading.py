"""
Adaptive Trading System
Adjusts trading parameters based on detected market regime
"""

from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from .regime_detection import (
    RegimeDetector, MarketRegime, RegimeDetectionResult, RegimeParameters,
    get_regime_detector
)
from ..sizing.kelly_vol_targeting import KellyVolTargetSizer, SizingMethod, get_kelly_sizer
from ..risk.central_risk_guard import CentralRiskGuard, get_risk_guard
from ..execution.execution_discipline import ExecutionPolicy, get_execution_policy

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveTradeSettings:
    """Trade settings adapted for current regime"""
    
    # Position sizing
    position_size_multiplier: float
    max_position_size: float
    min_position_size: float
    
    # Stop loss settings
    stop_loss_distance: float
    trailing_stop_enabled: bool
    breakeven_trigger: float
    
    # Take profit settings
    take_profit_distance: float
    partial_profit_enabled: bool
    profit_scaling_levels: List[float]
    
    # Entry filters
    min_signal_confidence: float
    max_entries_per_hour: int
    min_time_between_entries: int  # seconds
    
    # Risk adjustments
    regime_risk_multiplier: float
    correlation_threshold: float
    
    # Market timing
    avoid_news_events: bool
    session_preference: Optional[str]  # "asian", "london", "ny", None


@dataclass
class TradeDecision:
    """Result of adaptive trade decision"""
    
    should_trade: bool
    rejection_reasons: List[str]
    adapted_settings: AdaptiveTradeSettings
    regime_info: RegimeDetectionResult
    confidence_adjustment: float
    size_adjustment: float
    risk_adjustment: float


class AdaptiveTradingManager:
    """
    Manages adaptive trading based on regime detection
    Integrates with sizing, risk management, and execution systems
    """
    
    def __init__(self):
        self.kelly_sizer = get_kelly_sizer()
        self.risk_guard = get_risk_guard()
        self.execution_policy = get_execution_policy()
        
        # Base trading settings (before regime adaptation)
        self.base_settings = AdaptiveTradeSettings(
            position_size_multiplier=1.0,
            max_position_size=0.20,
            min_position_size=0.01,
            stop_loss_distance=0.02,      # 2% base stop
            trailing_stop_enabled=True,
            breakeven_trigger=2.0,        # Move to breakeven after 2R
            take_profit_distance=0.04,    # 4% base target (2:1 R:R)
            partial_profit_enabled=True,
            profit_scaling_levels=[1.5, 3.0, 5.0],  # R multiples
            min_signal_confidence=0.6,
            max_entries_per_hour=10,
            min_time_between_entries=300,  # 5 minutes
            regime_risk_multiplier=1.0,
            correlation_threshold=0.7,
            avoid_news_events=True,
            session_preference=None
        )
        
        # Track recent trades for throttling
        self.recent_trades: Dict[str, List[datetime]] = {}
    
    def should_take_trade(
        self, 
        symbol: str, 
        signal_strength: float, 
        signal_confidence: float,
        current_price: float,
        volume: float = 0.0
    ) -> TradeDecision:
        """
        Determine if trade should be taken based on current regime
        
        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (-1 to 1)
            signal_confidence: Signal confidence (0 to 1)
            current_price: Current market price
            volume: Current volume
            
        Returns:
            TradeDecision with adapted settings and recommendation
        """
        
        # Update regime detection
        regime_result = detect_current_regime(symbol, current_price, volume)
        
        # Adapt settings based on regime
        adapted_settings = self._adapt_settings_for_regime(
            self.base_settings, 
            regime_result
        )
        
        # Check if trade should be taken
        should_trade = True
        rejection_reasons = []
        
        # 1. Regime confidence check
        if regime_result.confidence_score < 0.3:
            should_trade = False
            rejection_reasons.append("Low regime detection confidence")
        
        # 2. Signal confidence vs regime requirements
        if signal_confidence < adapted_settings.min_signal_confidence:
            should_trade = False
            rejection_reasons.append(
                f"Signal confidence {signal_confidence:.1%} below regime requirement {adapted_settings.min_signal_confidence:.1%}"
            )
        
        # 3. Entry throttling check
        if self._is_entry_throttled(symbol, adapted_settings):
            should_trade = False
            rejection_reasons.append("Entry throttling active")
        
        # 4. Risk guard checks
        trade_size = abs(signal_strength) * adapted_settings.position_size_multiplier
        risk_check = self.risk_guard.check_trade_risk(symbol, trade_size * 10000, "adaptive_trading")
        
        if not risk_check.is_safe:
            should_trade = False
            rejection_reasons.append("Risk guard rejection")
        
        # 5. Regime-specific filters
        regime_filters = self._apply_regime_filters(regime_result, signal_strength)
        if regime_filters:
            should_trade = False
            rejection_reasons.extend(regime_filters)
        
        # Calculate adjustments
        confidence_adjustment = self._calculate_confidence_adjustment(regime_result, signal_confidence)
        size_adjustment = self._calculate_size_adjustment(regime_result, signal_strength)
        risk_adjustment = adapted_settings.regime_risk_multiplier
        
        # Log decision
        if should_trade:
            logger.info(f"Trade approved for {symbol} in {regime_result.primary_regime.value} regime")
        else:
            logger.info(f"Trade rejected for {symbol}: {', '.join(rejection_reasons)}")
        
        return TradeDecision(
            should_trade=should_trade,
            rejection_reasons=rejection_reasons,
            adapted_settings=adapted_settings,
            regime_info=regime_result,
            confidence_adjustment=confidence_adjustment,
            size_adjustment=size_adjustment,
            risk_adjustment=risk_adjustment
        )
    
    def _adapt_settings_for_regime(
        self, 
        base_settings: AdaptiveTradeSettings, 
        regime_result: RegimeDetectionResult
    ) -> AdaptiveTradeSettings:
        """Adapt trading settings based on detected regime"""
        
        regime_params = regime_result.parameters
        
        # Create adapted settings
        adapted = AdaptiveTradeSettings(
            # Position sizing adaptations
            position_size_multiplier=base_settings.position_size_multiplier * regime_params.sizing_multiplier,
            max_position_size=min(base_settings.max_position_size, regime_params.max_position_size),
            min_position_size=max(base_settings.min_position_size, regime_params.min_position_size),
            
            # Stop loss adaptations
            stop_loss_distance=base_settings.stop_loss_distance * regime_params.stop_loss_multiplier,
            trailing_stop_enabled=regime_params.trailing_stop_enabled,
            breakeven_trigger=regime_params.breakeven_move_ratio,
            
            # Take profit adaptations
            take_profit_distance=base_settings.take_profit_distance * regime_params.take_profit_multiplier,
            partial_profit_enabled=regime_params.profit_scaling_enabled,
            profit_scaling_levels=regime_params.partial_profit_levels,
            
            # Entry filter adaptations
            min_signal_confidence=max(base_settings.min_signal_confidence, regime_params.entry_confidence_threshold),
            max_entries_per_hour=min(base_settings.max_entries_per_hour, regime_params.max_entries_per_hour),
            min_time_between_entries=max(base_settings.min_time_between_entries, regime_params.min_time_between_entries),
            
            # Risk adaptations
            regime_risk_multiplier=regime_params.risk_multiplier,
            correlation_threshold=min(base_settings.correlation_threshold, regime_params.correlation_threshold),
            
            # Market timing (regime-specific)
            avoid_news_events=self._should_avoid_news_in_regime(regime_result.primary_regime),
            session_preference=self._get_session_preference_for_regime(regime_result.primary_regime)
        )
        
        return adapted
    
    def _is_entry_throttled(self, symbol: str, settings: AdaptiveTradeSettings) -> bool:
        """Check if entry is throttled based on recent trades"""
        
        current_time = datetime.now()
        
        # Get recent trades for this symbol
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        
        recent_trades = self.recent_trades[symbol]
        
        # Remove old trades (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        recent_trades = [t for t in recent_trades if t > cutoff_time]
        self.recent_trades[symbol] = recent_trades
        
        # Check hourly limit
        if len(recent_trades) >= settings.max_entries_per_hour:
            return True
        
        # Check minimum time between entries
        if recent_trades:
            last_trade = max(recent_trades)
            time_since_last = (current_time - last_trade).total_seconds()
            if time_since_last < settings.min_time_between_entries:
                return True
        
        return False
    
    def _apply_regime_filters(self, regime_result: RegimeDetectionResult, signal_strength: float) -> List[str]:
        """Apply regime-specific trade filters"""
        
        filters = []
        regime = regime_result.primary_regime
        metrics = regime_result.metrics
        
        # High volatility regime filters
        if regime == MarketRegime.HIGH_VOLATILITY:
            if regime_result.confidence_score < 0.7:
                filters.append("High volatility requires high confidence")
        
        # Choppy market filters
        if regime == MarketRegime.CHOPPY:
            if abs(signal_strength) < 0.8:
                filters.append("Choppy market requires strong signals")
            if metrics.choppiness_index > 80:
                filters.append("Extremely choppy conditions")
        
        # Mean reverting filters
        if regime == MarketRegime.MEAN_REVERTING:
            # Favor counter-trend signals
            if (signal_strength > 0 and metrics.trend_strength > 0.3) or \
               (signal_strength < 0 and metrics.trend_strength < -0.3):
                filters.append("Mean reversion regime favors counter-trend trades")
        
        # Trending regime filters
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            # Favor trend-following signals
            if regime == MarketRegime.TRENDING_UP and signal_strength < 0:
                filters.append("Strong uptrend - avoid short signals")
            elif regime == MarketRegime.TRENDING_DOWN and signal_strength > 0:
                filters.append("Strong downtrend - avoid long signals")
        
        # Low volatility filters
        if regime == MarketRegime.LOW_VOLATILITY:
            if abs(signal_strength) < 0.4:
                filters.append("Low volatility requires stronger signals")
        
        return filters
    
    def _calculate_confidence_adjustment(self, regime_result: RegimeDetectionResult, signal_confidence: float) -> float:
        """Calculate confidence adjustment based on regime"""
        
        base_adjustment = 1.0
        
        # Regime confidence impact
        regime_confidence_multiplier = {
            "very_low": 0.5,
            "low": 0.7,
            "medium": 1.0,
            "high": 1.2,
            "very_high": 1.3
        }
        
        confidence_adj = regime_confidence_multiplier.get(regime_result.confidence.value, 1.0)
        
        # Regime-specific adjustments
        if regime_result.primary_regime == MarketRegime.HIGH_VOLATILITY:
            confidence_adj *= 0.8  # Reduce confidence in high vol
        elif regime_result.primary_regime == MarketRegime.TRENDING_UP:
            confidence_adj *= 1.1  # Boost confidence in uptrend
        elif regime_result.primary_regime == MarketRegime.CHOPPY:
            confidence_adj *= 0.6  # Much lower confidence in chop
        
        return base_adjustment * confidence_adj
    
    def _calculate_size_adjustment(self, regime_result: RegimeDetectionResult, signal_strength: float) -> float:
        """Calculate position size adjustment based on regime"""
        
        # Base size from regime parameters
        size_multiplier = regime_result.parameters.sizing_multiplier
        
        # Adjust for regime confidence
        confidence_adjustment = regime_result.confidence_score
        
        # Adjust for signal strength
        signal_adjustment = abs(signal_strength)
        
        # Combine adjustments
        final_adjustment = size_multiplier * confidence_adjustment * signal_adjustment
        
        # Clamp to reasonable bounds
        return max(0.1, min(2.0, final_adjustment))
    
    def _should_avoid_news_in_regime(self, regime: MarketRegime) -> bool:
        """Determine if news events should be avoided in this regime"""
        
        # Avoid news in volatile or unpredictable regimes
        return regime in [
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.CHOPPY,
            MarketRegime.BREAKOUT
        ]
    
    def _get_session_preference_for_regime(self, regime: MarketRegime) -> Optional[str]:
        """Get preferred trading session for regime"""
        
        # Session preferences based on regime
        preferences = {
            MarketRegime.HIGH_VOLATILITY: "ny",     # NY session for high vol
            MarketRegime.LOW_VOLATILITY: "asian",   # Asian session for low vol
            MarketRegime.TRENDING_UP: "london",     # London for trends
            MarketRegime.TRENDING_DOWN: "london",
            MarketRegime.BREAKOUT: "ny"             # NY for breakouts
        }
        
        return preferences.get(regime)
    
    def record_trade_entry(self, symbol: str):
        """Record that a trade was entered for throttling purposes"""
        
        current_time = datetime.now()
        
        if symbol not in self.recent_trades:
            self.recent_trades[symbol] = []
        
        self.recent_trades[symbol].append(current_time)
        
        # Keep only last 24 hours
        cutoff_time = current_time - timedelta(hours=24)
        self.recent_trades[symbol] = [
            t for t in self.recent_trades[symbol] if t > cutoff_time
        ]
    
    def get_regime_summary_for_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get regime summary for multiple symbols"""
        
        summaries = {}
        
        for symbol in symbols:
            detector = get_regime_detector(symbol)
            summaries[symbol] = detector.get_regime_summary()
        
        return summaries
    
    def get_adaptive_settings_summary(self, symbol: str) -> Dict:
        """Get current adaptive settings for symbol"""
        
        detector = get_regime_detector(symbol)
        regime_result = detector.detect_regime()
        
        adapted_settings = self._adapt_settings_for_regime(self.base_settings, regime_result)
        
        return {
            'symbol': symbol,
            'regime': regime_result.primary_regime.value,
            'confidence': regime_result.confidence.value,
            'confidence_score': regime_result.confidence_score,
            'regime_duration_minutes': regime_result.regime_duration,
            'adapted_settings': asdict(adapted_settings),
            'base_vs_adapted': {
                'size_multiplier': adapted_settings.position_size_multiplier / self.base_settings.position_size_multiplier,
                'stop_multiplier': adapted_settings.stop_loss_distance / self.base_settings.stop_loss_distance,
                'tp_multiplier': adapted_settings.take_profit_distance / self.base_settings.take_profit_distance,
                'confidence_requirement': adapted_settings.min_signal_confidence / self.base_settings.min_signal_confidence
            }
        }


# Global adaptive trading manager
_adaptive_manager: Optional[AdaptiveTradingManager] = None


def get_adaptive_trading_manager() -> AdaptiveTradingManager:
    """Get global adaptive trading manager"""
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveTradingManager()
    return _adaptive_manager


def should_take_adaptive_trade(
    symbol: str, 
    signal_strength: float, 
    signal_confidence: float,
    current_price: float
) -> TradeDecision:
    """Convenience function for adaptive trade decisions"""
    
    return get_adaptive_trading_manager().should_take_trade(
        symbol, signal_strength, signal_confidence, current_price
    )


if __name__ == "__main__":
    # Example usage
    manager = AdaptiveTradingManager()
    
    # Test adaptive trade decision
    decision = manager.should_take_trade(
        symbol="BTC/USD",
        signal_strength=0.8,
        signal_confidence=0.7,
        current_price=50000.0
    )
    
    print(f"Should trade: {decision.should_trade}")
    print(f"Regime: {decision.regime_info.primary_regime.value}")
    print(f"Confidence: {decision.regime_info.confidence.value}")
    print(f"Size adjustment: {decision.size_adjustment:.2f}")
    print(f"Adapted stop distance: {decision.adapted_settings.stop_loss_distance:.1%}")
    print(f"Max entries/hour: {decision.adapted_settings.max_entries_per_hour}")
