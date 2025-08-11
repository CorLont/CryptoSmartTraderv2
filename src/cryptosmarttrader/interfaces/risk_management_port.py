"""
Risk Management Port - Interface for risk assessment and management

Defines the contract for risk management implementations enabling
swappable risk engines without affecting trading logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class RiskType(Enum):
    """Types of risks to assess"""
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    CORRELATION_RISK = "correlation_risk"
    VOLATILITY_RISK = "volatility_risk"
    LEVERAGE_RISK = "leverage_risk"
    OPERATIONAL_RISK = "operational_risk"

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    symbol: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    risk_types: Dict[RiskType, float]
    recommendations: List[str]
    timestamp: datetime
    metadata: Optional[Dict] = None

@dataclass
class PositionRisk:
    """Position-level risk metrics"""
    symbol: str
    position_size: float
    market_value: float
    var_1d: float  # Value at Risk 1 day
    var_1w: float  # Value at Risk 1 week
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    beta: Optional[float] = None

class RiskManagementPort(ABC):
    """
    Abstract interface for risk management systems
    
    This port defines the contract for risk assessment and management,
    enabling different risk models and strategies to be plugged in.
    """
    
    @abstractmethod
    def assess_symbol_risk(self, symbol: str, market_data: pd.DataFrame) -> RiskAssessment:
        """
        Assess risk for a specific symbol
        
        Args:
            symbol: Trading symbol to assess
            market_data: Historical market data for assessment
            
        Returns:
            RiskAssessment with risk level and recommendations
        """
        pass
    
    @abstractmethod
    def assess_portfolio_risk(self, positions: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess overall portfolio risk
        
        Args:
            positions: Dictionary of symbol -> position size
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, risk_budget: float,
                              confidence: float, price: float) -> float:
        """
        Calculate appropriate position size based on risk
        
        Args:
            symbol: Trading symbol
            risk_budget: Maximum risk budget for position
            confidence: Prediction confidence level
            price: Current price
            
        Returns:
            Recommended position size
        """
        pass
    
    @abstractmethod
    def validate_trade(self, symbol: str, size: float, price: float,
                      current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        Validate if a trade is within risk limits
        
        Args:
            symbol: Trading symbol
            size: Proposed trade size
            price: Trade price
            current_positions: Current portfolio positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        pass
    
    @abstractmethod
    def get_risk_limits(self) -> Dict[str, Any]:
        """
        Get current risk limits and parameters
        
        Returns:
            Dictionary with risk limits configuration
        """
        pass
    
    @abstractmethod
    def update_risk_limits(self, limits: Dict[str, Any]) -> bool:
        """
        Update risk limits configuration
        
        Args:
            limits: New risk limits to apply
            
        Returns:
            True if limits were updated successfully
        """
        pass

class NotificationPort(ABC):
    """Interface for notification systems"""
    
    @abstractmethod
    def send_alert(self, message: str, severity: str = "info",
                  channels: Optional[List[str]] = None) -> bool:
        """Send alert notification"""
        pass
    
    @abstractmethod
    def send_report(self, report_data: Dict[str, Any],
                   report_type: str = "daily") -> bool:
        """Send periodic report"""
        pass