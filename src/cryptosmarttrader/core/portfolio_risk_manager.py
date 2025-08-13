#!/usr/bin/env python3
"""
Advanced Portfolio Risk Manager with Per-Coin Caps and Automated Position Controls
Implements hard position limits, liquidity constraints, and kill-switch mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

from ..core.logging_manager import get_logger
from ..core.data_quality_manager import get_data_quality_manager

class RiskStatus(str, Enum):
    """Risk management status levels"""
    GREEN = "green"      # Normal operations
    YELLOW = "yellow"    # Warning - increased monitoring
    ORANGE = "orange"    # Caution - restricted operations
    RED = "red"          # Critical - emergency actions required
    BLACK = "black"      # Kill switch - all positions flat

class PositionAction(str, Enum):
    """Available position management actions"""
    ALLOW = "allow"
    RESTRICT = "restrict"
    REDUCE = "reduce"
    FLATTEN = "flatten"
    KILL = "kill"

@dataclass
class CoinLimits:
    """Per-coin position and risk limits"""
    symbol: str
    max_position_value_usd: float
    max_position_percent_adv: float  # Percentage of Average Daily Volume
    max_portfolio_weight: float      # Percentage of total portfolio
    min_liquidity_usd: float        # Minimum required liquidity
    correlation_limit: float        # Max correlation with other positions
    data_quality_threshold: float   # Minimum data quality score
    enabled: bool = True

@dataclass
class PositionLimit:
    """Current position limits with dynamic adjustments"""
    symbol: str
    current_limit_usd: float
    current_limit_shares: float
    utilization_percent: float
    limit_reason: str
    last_updated: datetime

@dataclass
class RiskMetrics:
    """Current portfolio risk metrics"""
    timestamp: datetime
    total_portfolio_value: float
    total_exposure: float
    net_exposure: float
    gross_leverage: float
    net_leverage: float
    var_1d_95: float  # 1-day Value at Risk 95%
    var_1d_99: float  # 1-day Value at Risk 99%
    max_correlation: float
    concentration_risk: float
    liquidity_risk: float
    data_quality_score: float

@dataclass
class RiskAlert:
    """Risk alert record"""
    alert_id: str
    symbol: Optional[str]
    alert_type: str
    severity: RiskStatus
    message: str
    current_value: float
    limit_value: float
    action_required: PositionAction
    timestamp: datetime
    resolved: bool = False

@dataclass
class Position:
    """Enhanced position tracking with risk metrics"""
    symbol: str
    size: float
    market_value: float
    unrealized_pnl: float
    entry_price: float
    current_price: float
    weight: float  # Portfolio weight
    daily_volume: float
    adv_utilization: float  # Percentage of ADV
    correlation_max: float  # Max correlation with other positions
    data_quality_score: float
    liquidity_score: float
    last_updated: datetime

class LiquidityAnalyzer:
    """Analyzes market liquidity and ADV for position sizing"""

    def __init__(self):
        self.logger = get_logger()
        self.volume_history = {}  # Symbol -> volume history
        self.liquidity_cache = {}

    def calculate_adv(self, symbol: str, volume_data: List[float], days: int = 20) -> float:
        """Calculate Average Daily Volume"""
        if not volume_data or len(volume_data) < days:
            return 0.0

        recent_volumes = volume_data[-days:]
        adv = np.mean(recent_volumes)

        # Cache result
        self.volume_history[symbol] = {
            'adv': adv,
            'last_updated': datetime.now(),
            'sample_size': len(recent_volumes)
        }

        return adv

    def estimate_market_impact(
        self,
        symbol: str,
        position_size_usd: float,
        adv_usd: float
    ) -> float:
        """Estimate market impact for position size"""

        if adv_usd <= 0:
            return 1.0  # High impact if no volume data

        # Position size as percentage of ADV
        adv_percent = position_size_usd / adv_usd

        # Market impact model (square root law)
        market_impact = 0.1 * np.sqrt(adv_percent)  # 10% base impact coefficient

        return min(market_impact, 0.5)  # Cap at 50% impact

    def calculate_liquidity_score(
        self,
        symbol: str,
        current_volume: float,
        adv: float,
        bid_ask_spread: float,
        order_book_depth: float
    ) -> float:
        """Calculate comprehensive liquidity score (0-1)"""

        # Volume component (30% weight)
        volume_ratio = current_volume / adv if adv > 0 else 0
        volume_score = min(volume_ratio, 2.0) / 2.0  # Cap at 2x ADV

        # Spread component (40% weight)
        spread_score = max(0, 1 - bid_ask_spread / 0.01)  # Penalty for >1% spread

        # Depth component (30% weight)
        depth_score = min(order_book_depth / 100000, 1.0)  # Normalize to $100k depth

        liquidity_score = (
            volume_score * 0.3 +
            spread_score * 0.4 +
            depth_score * 0.3
        )

        return max(0, min(1, liquidity_score))

class CorrelationMonitor:
    """Monitors portfolio correlations and concentration risk"""

    def __init__(self):
        self.logger = get_logger()
        self.correlation_matrix = pd.DataFrame()
        self.correlation_history = []

    def update_correlations(self, returns_data: pd.DataFrame):
        """Update correlation matrix with latest returns"""

        if returns_data.empty or len(returns_data.columns) < 2:
            return

        # Calculate rolling correlation
        self.correlation_matrix = returns_data.corr()

        # Store correlation snapshot
        self.correlation_history.append({
            'timestamp': datetime.now(),
            'correlations': self.correlation_matrix.to_dict()
        })

        # Keep last 100 snapshots
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]

    def get_max_correlation(self, symbol: str) -> float:
        """Get maximum correlation with any other position"""

        if symbol not in self.correlation_matrix.columns:
            return 0.0

        # Get correlations with other symbols (exclude self)
        correlations = self.correlation_matrix[symbol].drop(symbol, errors='ignore')

        if correlations.empty:
            return 0.0

        return correlations.abs().max()

    def calculate_concentration_risk(self, positions: Dict[str, Position]) -> float:
        """Calculate portfolio concentration risk"""

        if not positions:
            return 0.0

        # Calculate Herfindahl-Hirschman Index
        weights = [pos.weight for pos in positions.values()]
        hhi = sum(w**2 for w in weights)

        # Normalize to 0-1 scale (1 = max concentration)
        max_hhi = 1.0  # Single position portfolio
        min_hhi = 1.0 / len(positions)  # Equally weighted

        if max_hhi == min_hhi:
            return 0.0

        concentration_risk = (hhi - min_hhi) / (max_hhi - min_hhi)

        return concentration_risk

class PortfolioRiskManager:
    """Main portfolio risk management engine"""

    def __init__(self):
        self.logger = get_logger()
        self.data_quality_manager = get_data_quality_manager()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.correlation_monitor = CorrelationMonitor()

        # Risk limits and controls
        self.coin_limits = {}
        self.portfolio_limits = self._initialize_portfolio_limits()
        self.risk_alerts = []
        self.position_limits = {}

        # Current state
        self.current_risk_status = RiskStatus.GREEN
        self.kill_switch_active = False
        self.emergency_mode = False

        # Performance tracking
        self.risk_metrics_history = []
        self.alert_history = []

        self.logger.info("Portfolio Risk Manager initialized")

    def _initialize_portfolio_limits(self) -> Dict[str, float]:
        """Initialize portfolio-level risk limits"""

        return {
            'max_portfolio_value': 10000000,  # $10M max portfolio
            'max_gross_leverage': 3.0,        # 3x gross leverage
            'max_net_leverage': 2.0,          # 2x net leverage
            'max_concentration': 0.25,        # 25% max single position
            'max_correlation': 0.8,           # 80% max correlation
            'min_data_quality': 0.7,          # 70% min data quality
            'max_var_95': 0.05,              # 5% daily VaR limit
            'min_liquidity_score': 0.3,      # 30% min liquidity score
            'max_adv_utilization': 0.2       # 20% max ADV utilization
        }

    def set_coin_limits(self, symbol: str, limits: CoinLimits):
        """Set risk limits for specific coin"""

        self.coin_limits[symbol] = limits

        self.logger.info(
            f"Coin limits set for {symbol}",
            extra={
                'symbol': symbol,
                'max_value_usd': limits.max_position_value_usd,
                'max_adv_percent': limits.max_position_percent_adv,
                'max_portfolio_weight': limits.max_portfolio_weight
            }
        )

    def initialize_default_limits(self):
        """Initialize default limits for major cryptocurrencies"""

        major_coins = {
            'BTC/USD': CoinLimits(
                symbol='BTC/USD',
                max_position_value_usd=2000000,  # $2M max
                max_position_percent_adv=0.15,   # 15% of ADV
                max_portfolio_weight=0.30,       # 30% of portfolio
                min_liquidity_usd=10000000,      # $10M min liquidity
                correlation_limit=0.7,           # 70% max correlation
                data_quality_threshold=0.8       # 80% data quality
            ),
            'ETH/USD': CoinLimits(
                symbol='ETH/USD',
                max_position_value_usd=1500000,  # $1.5M max
                max_position_percent_adv=0.20,   # 20% of ADV
                max_portfolio_weight=0.25,       # 25% of portfolio
                min_liquidity_usd=5000000,       # $5M min liquidity
                correlation_limit=0.7,
                data_quality_threshold=0.8
            ),
            'ADA/USD': CoinLimits(
                symbol='ADA/USD',
                max_position_value_usd=500000,   # $500k max
                max_position_percent_adv=0.25,   # 25% of ADV
                max_portfolio_weight=0.10,       # 10% of portfolio
                min_liquidity_usd=1000000,       # $1M min liquidity
                correlation_limit=0.6,
                data_quality_threshold=0.7
            )
        }

        for symbol, limits in major_coins.items():
            self.set_coin_limits(symbol, limits)

    def validate_position(
        self,
        symbol: str,
        proposed_size: float,
        current_price: float,
        market_data: Dict[str, Any],
        current_positions: Dict[str, Position]
    ) -> Tuple[PositionAction, List[RiskAlert]]:
        """Comprehensive position validation with all risk checks"""

        alerts = []
        proposed_value = abs(proposed_size * current_price)

        # Get coin limits
        if symbol not in self.coin_limits:
            # Create conservative default limits
            self.coin_limits[symbol] = CoinLimits(
                symbol=symbol,
                max_position_value_usd=100000,   # $100k default
                max_position_percent_adv=0.10,   # 10% of ADV
                max_portfolio_weight=0.05,       # 5% of portfolio
                min_liquidity_usd=500000,        # $500k min liquidity
                correlation_limit=0.5,
                data_quality_threshold=0.6
            )

        coin_limits = self.coin_limits[symbol]

        # 1. Position value check
        if proposed_value > coin_limits.max_position_value_usd:
            alerts.append(RiskAlert(
                alert_id=f"value_limit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                alert_type="position_value_limit",
                severity=RiskStatus.RED,
                message=f"Position value ${proposed_value:,.0f} exceeds limit ${coin_limits.max_position_value_usd:,.0f}",
                current_value=proposed_value,
                limit_value=coin_limits.max_position_value_usd,
                action_required=PositionAction.RESTRICT,
                timestamp=datetime.now()
            ))

        # 2. ADV utilization check
        adv_usd = market_data.get('adv_usd', 0)
        if adv_usd > 0:
            adv_utilization = proposed_value / adv_usd
            if adv_utilization > coin_limits.max_position_percent_adv:
                alerts.append(RiskAlert(
                    alert_id=f"adv_limit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    alert_type="adv_utilization_limit",
                    severity=RiskStatus.ORANGE,
                    message=f"ADV utilization {adv_utilization:.1%} exceeds limit {coin_limits.max_position_percent_adv:.1%}",
                    current_value=adv_utilization,
                    limit_value=coin_limits.max_position_percent_adv,
                    action_required=PositionAction.REDUCE,
                    timestamp=datetime.now()
                ))

        # 3. Portfolio weight check
        total_portfolio_value = sum(pos.market_value for pos in current_positions.values()) + proposed_value
        portfolio_weight = proposed_value / total_portfolio_value if total_portfolio_value > 0 else 0

        if portfolio_weight > coin_limits.max_portfolio_weight:
            alerts.append(RiskAlert(
                alert_id=f"weight_limit_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                alert_type="portfolio_weight_limit",
                severity=RiskStatus.ORANGE,
                message=f"Portfolio weight {portfolio_weight:.1%} exceeds limit {coin_limits.max_portfolio_weight:.1%}",
                current_value=portfolio_weight,
                limit_value=coin_limits.max_portfolio_weight,
                action_required=PositionAction.REDUCE,
                timestamp=datetime.now()
            ))

        # 4. Data quality check
        data_quality_summary = self.data_quality_manager.get_quality_summary()
        current_quality = data_quality_summary.get('overall_completeness', 0)

        if current_quality < coin_limits.data_quality_threshold:
            alerts.append(RiskAlert(
                alert_id=f"data_quality_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                alert_type="data_quality_breach",
                severity=RiskStatus.RED,
                message=f"Data quality {current_quality:.1%} below threshold {coin_limits.data_quality_threshold:.1%}",
                current_value=current_quality,
                limit_value=coin_limits.data_quality_threshold,
                action_required=PositionAction.KILL,
                timestamp=datetime.now()
            ))

        # 5. Liquidity check
        liquidity_score = market_data.get('liquidity_score', 0)
        if liquidity_score < self.portfolio_limits['min_liquidity_score']:
            alerts.append(RiskAlert(
                alert_id=f"liquidity_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                alert_type="liquidity_insufficient",
                severity=RiskStatus.YELLOW,
                message=f"Liquidity score {liquidity_score:.2f} below minimum {self.portfolio_limits['min_liquidity_score']:.2f}",
                current_value=liquidity_score,
                limit_value=self.portfolio_limits['min_liquidity_score'],
                action_required=PositionAction.RESTRICT,
                timestamp=datetime.now()
            ))

        # 6. Correlation check
        max_correlation = self._calculate_position_correlation(symbol, current_positions)
        if max_correlation > coin_limits.correlation_limit:
            alerts.append(RiskAlert(
                alert_id=f"correlation_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=symbol,
                alert_type="correlation_limit",
                severity=RiskStatus.YELLOW,
                message=f"Max correlation {max_correlation:.2f} exceeds limit {coin_limits.correlation_limit:.2f}",
                current_value=max_correlation,
                limit_value=coin_limits.correlation_limit,
                action_required=PositionAction.RESTRICT,
                timestamp=datetime.now()
            ))

        # Determine overall action
        action = self._determine_position_action(alerts)

        # Store alerts
        self.risk_alerts.extend(alerts)
        self.alert_history.extend(alerts)

        return action, alerts

    def _calculate_position_correlation(
        self,
        symbol: str,
        current_positions: Dict[str, Position]
    ) -> float:
        """Calculate maximum correlation with existing positions"""

        if not current_positions:
            return 0.0

        return self.correlation_monitor.get_max_correlation(symbol)

    def _determine_position_action(self, alerts: List[RiskAlert]) -> PositionAction:
        """Determine required action based on risk alerts"""

        if not alerts:
            return PositionAction.ALLOW

        # Find highest severity
        max_severity = max(alert.severity for alert in alerts)

        # Get required actions
        actions = [alert.action_required for alert in alerts]

        # Determine most restrictive action
        if PositionAction.KILL in actions:
            return PositionAction.KILL
        elif PositionAction.FLATTEN in actions:
            return PositionAction.FLATTEN
        elif PositionAction.REDUCE in actions:
            return PositionAction.REDUCE
        elif PositionAction.RESTRICT in actions:
            return PositionAction.RESTRICT
        else:
            return PositionAction.ALLOW

    def check_kill_switch_conditions(
        self,
        current_positions: Dict[str, Position],
        market_data: Dict[str, Any]
    ) -> bool:
        """Check if kill switch should be activated"""

        kill_conditions = []

        # 1. Data quality kill switch
        data_quality_summary = self.data_quality_manager.get_quality_summary()
        overall_quality = data_quality_summary.get('overall_completeness', 0)

        if overall_quality < 0.5:  # Less than 50% data quality
            kill_conditions.append(f"Data quality critical: {overall_quality:.1%}")

        # 2. Portfolio VaR kill switch
        current_metrics = self.calculate_risk_metrics(current_positions, market_data)
        if current_metrics.var_1d_95 > 0.15:  # 15% daily VaR
            kill_conditions.append(f"VaR critical: {current_metrics.var_1d_95:.1%}")

        # 3. Liquidity kill switch
        avg_liquidity = np.mean([pos.liquidity_score for pos in current_positions.values()])
        if avg_liquidity < 0.1:  # Less than 10% liquidity
            kill_conditions.append(f"Liquidity critical: {avg_liquidity:.1%}")

        # 4. Data gap kill switch
        problematic_coins = data_quality_summary.get('problematic_coins_count', 0)
        total_coins = data_quality_summary.get('total_coins_tracked', 1)
        if problematic_coins / total_coins > 0.5:  # More than 50% problematic
            kill_conditions.append(f"Data gaps critical: {problematic_coins}/{total_coins}")

        # 5. Correlation kill switch
        if current_metrics.max_correlation > 0.95:  # Near perfect correlation
            kill_conditions.append(f"Correlation critical: {current_metrics.max_correlation:.2f}")

        if kill_conditions:
            self.logger.critical(
                "KILL SWITCH ACTIVATED",
                extra={
                    'kill_conditions': kill_conditions,
                    'action': 'flatten_all_positions'
                }
            )

            # Create emergency alert
            emergency_alert = RiskAlert(
                alert_id=f"kill_switch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                symbol=None,
                alert_type="kill_switch_activation",
                severity=RiskStatus.BLACK,
                message=f"Kill switch activated: {'; '.join(kill_conditions)}",
                current_value=0,
                limit_value=0,
                action_required=PositionAction.KILL,
                timestamp=datetime.now()
            )

            self.risk_alerts.append(emergency_alert)
            self.kill_switch_active = True
            self.current_risk_status = RiskStatus.BLACK

            return True

        return False

    def calculate_risk_metrics(
        self,
        current_positions: Dict[str, Position],
        market_data: Dict[str, Any]
    ) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""

        if not current_positions:
            return RiskMetrics(
                timestamp=datetime.now(),
                total_portfolio_value=0,
                total_exposure=0,
                net_exposure=0,
                gross_leverage=0,
                net_leverage=0,
                var_1d_95=0,
                var_1d_99=0,
                max_correlation=0,
                concentration_risk=0,
                liquidity_risk=0,
                data_quality_score=1.0
            )

        # Portfolio values
        long_exposure = sum(pos.market_value for pos in current_positions.values() if pos.size > 0)
        short_exposure = sum(abs(pos.market_value) for pos in current_positions.values() if pos.size < 0)
        total_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        total_portfolio_value = sum(pos.market_value for pos in current_positions.values())

        # Leverage calculations
        gross_leverage = total_exposure / abs(total_portfolio_value) if total_portfolio_value != 0 else 0
        net_leverage = abs(net_exposure) / abs(total_portfolio_value) if total_portfolio_value != 0 else 0

        # VaR calculation (simplified)
        position_values = [pos.market_value for pos in current_positions.values()]
        portfolio_volatility = np.std(position_values) / np.mean(position_values) if position_values else 0
        var_1d_95 = 1.645 * portfolio_volatility  # 95% confidence
        var_1d_99 = 2.326 * portfolio_volatility  # 99% confidence

        # Correlation risk
        max_correlation = max([pos.correlation_max for pos in current_positions.values()], default=0)

        # Concentration risk
        concentration_risk = self.correlation_monitor.calculate_concentration_risk(current_positions)

        # Liquidity risk
        liquidity_scores = [pos.liquidity_score for pos in current_positions.values()]
        liquidity_risk = 1 - np.mean(liquidity_scores) if liquidity_scores else 1.0

        # Data quality score
        data_quality_scores = [pos.data_quality_score for pos in current_positions.values()]
        data_quality_score = np.mean(data_quality_scores) if data_quality_scores else 0.0

        metrics = RiskMetrics(
            timestamp=datetime.now(),
            total_portfolio_value=total_portfolio_value,
            total_exposure=total_exposure,
            net_exposure=net_exposure,
            gross_leverage=gross_leverage,
            net_leverage=net_leverage,
            var_1d_95=var_1d_95,
            var_1d_99=var_1d_99,
            max_correlation=max_correlation,
            concentration_risk=concentration_risk,
            liquidity_risk=liquidity_risk,
            data_quality_score=data_quality_score
        )

        # Store metrics history
        self.risk_metrics_history.append(metrics)
        if len(self.risk_metrics_history) > 1000:  # Keep last 1000 records
            self.risk_metrics_history = self.risk_metrics_history[-1000:]

        return metrics

    def get_position_recommendations(
        self,
        current_positions: Dict[str, Position]
    ) -> Dict[str, Dict[str, Any]]:
        """Get position recommendations based on current risk analysis"""

        recommendations = {}

        for symbol, position in current_positions.items():
            recommendation = {
                'action': 'hold',
                'reason': 'position within limits',
                'urgency': 'low',
                'max_increase': 0.0,
                'suggested_target': position.market_value
            }

            # Check against limits
            if symbol in self.coin_limits:
                limits = self.coin_limits[symbol]

                # Value limit check
                if position.market_value > limits.max_position_value_usd * 0.9:  # 90% of limit
                    recommendation.update({
                        'action': 'reduce',
                        'reason': 'approaching value limit',
                        'urgency': 'medium',
                        'suggested_target': limits.max_position_value_usd * 0.8
                    })

                # Data quality check
                if position.data_quality_score < limits.data_quality_threshold:
                    recommendation.update({
                        'action': 'flatten',
                        'reason': 'data quality below threshold',
                        'urgency': 'high',
                        'suggested_target': 0.0
                    })

                # Correlation check
                if position.correlation_max > limits.correlation_limit:
                    recommendation.update({
                        'action': 'reduce',
                        'reason': 'high correlation risk',
                        'urgency': 'medium',
                        'suggested_target': position.market_value * 0.7
                    })

            recommendations[symbol] = recommendation

        return recommendations

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""

        return {
            'timestamp': datetime.now().isoformat(),
            'risk_status': self.current_risk_status.value,
            'kill_switch_active': self.kill_switch_active,
            'active_alerts': len([a for a in self.risk_alerts if not a.resolved]),
            'portfolio_limits': self.portfolio_limits,
            'coin_limits_count': len(self.coin_limits),
            'recent_alerts': [
                {
                    'symbol': alert.symbol,
                    'type': alert.alert_type,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in self.risk_alerts[-10:]  # Last 10 alerts
            ],
            'risk_metrics': self.risk_metrics_history[-1].__dict__ if self.risk_metrics_history else {},
            'data_quality_status': self.data_quality_manager.get_quality_summary()
        }

# Global instance
_portfolio_risk_manager = None

def get_portfolio_risk_manager() -> PortfolioRiskManager:
    """Get global portfolio risk manager instance"""
    global _portfolio_risk_manager
    if _portfolio_risk_manager is None:
        _portfolio_risk_manager = PortfolioRiskManager()
    return _portfolio_risk_manager
