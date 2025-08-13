"""CryptoSmartTrader V2 - Enterprise Cryptocurrency Trading Intelligence Platform."""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"

# Core exports
from .core.config_manager import ConfigManager
from .core.structured_logger import StructuredLogger, get_logger
from .core.risk_guard import RiskGuard, RiskLevel, TradingMode
from .core.execution_policy import ExecutionPolicy, OrderRequest, OrderType, TimeInForce

# Monitoring exports  
from .monitoring.prometheus_metrics import get_metrics, CryptoSmartTraderMetrics
from .monitoring.alert_rules import AlertManager, AlertSeverity

# Testing exports
from .testing.simulation_tester import SimulationTester, FailureScenario

# Fase 3 exports
from .core.regime_detector import RegimeDetector, MarketRegime
from .core.strategy_switcher import StrategySwitcher, StrategyType
from .analysis.backtest_parity import BacktestParityAnalyzer
from .deployment.canary_system import CanaryDeploymentSystem, CanaryStage

__all__ = [
    # Core
    "ConfigManager", 
    "StructuredLogger",
    "get_logger",
    "RiskGuard",
    "RiskLevel", 
    "TradingMode",
    "ExecutionPolicy",
    "OrderRequest",
    "OrderType",
    "TimeInForce",
    
    # Monitoring
    "get_metrics",
    "CryptoSmartTraderMetrics", 
    "AlertManager",
    "AlertSeverity",
    
    # Testing
    "SimulationTester",
    "FailureScenario",
    
    # Fase 3 - Alpha & Parity
    "RegimeDetector",
    "MarketRegime",
    "StrategySwitcher",
    "StrategyType",
    "BacktestParityAnalyzer",
    "CanaryDeploymentSystem",
    "CanaryStage"
]