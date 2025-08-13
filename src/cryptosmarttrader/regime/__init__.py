"""
CryptoSmartTrader V2 Regime Switching System

Detects market regimes (trend/mean-revert/chop/high-vol) and adapts
trading parameters (stops/TP/sizing/throttle) accordingly.
"""

from .regime_detection import (
    RegimeDetector, MarketRegime, RegimeConfidence, RegimeMetrics,
    RegimeParameters, RegimeDetectionResult, RegimeIndicators,
    get_regime_detector, detect_current_regime
)

from .adaptive_trading import (
    AdaptiveTradingManager, AdaptiveTradeSettings, TradeDecision,
    get_adaptive_trading_manager, should_take_adaptive_trade
)

__all__ = [
    # Regime Detection
    'RegimeDetector', 'MarketRegime', 'RegimeConfidence', 'RegimeMetrics',
    'RegimeParameters', 'RegimeDetectionResult', 'RegimeIndicators',
    'get_regime_detector', 'detect_current_regime',
    
    # Adaptive Trading
    'AdaptiveTradingManager', 'AdaptiveTradeSettings', 'TradeDecision',
    'get_adaptive_trading_manager', 'should_take_adaptive_trade'
]

# Version info
__version__ = '2.0.0'
__title__ = 'CryptoSmartTrader Regime Switching'
__description__ = 'Market regime detection and adaptive trading parameters'
