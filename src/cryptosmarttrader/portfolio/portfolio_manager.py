"""
Integrated Portfolio Manager
Combineert Kelly Vol Sizing, Regime Detection, en Risk Guards
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import pandas as pd

from .kelly_vol_sizing import KellyVolSizer, AssetMetrics, PositionSize, MarketRegime
from .regime_detector import RegimeDetector
from ..risk.central_risk_guard import CentralRiskGuard, TradingOperation, RiskDecision

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRecommendation:
    """Portfolio rebalancing recommendation"""
    symbol: str
    current_size_usd: float
    target_size_usd: float
    recommended_action: str  # "buy", "sell", "hold", "rebalance"
    size_change_usd: float
    size_change_pct: float
    priority: int  # 1-5, 1 = highest priority
    reasoning: List[str]


@dataclass
class PortfolioState:
    """Complete portfolio state"""
    total_equity: float
    cash_balance: float
    positions: Dict[str, float]  # symbol -> size_usd
    target_allocations: Dict[str, float]  # symbol -> target_pct
    regime: MarketRegime
    regime_confidence: float
    risk_utilization: float  # 0-1
    cluster_exposures: Dict[str, float]


class IntegratedPortfolioManager:
    """
    Complete portfolio management systeem met:
    - Kelly Vol Sizing
    - Regime Detection & Throttling  
    - Risk Guard Integration
    - Rebalancing Logic
    """
    
    def __init__(self):
        self.kelly_sizer = KellyVolSizer()
        self.regime_detector = RegimeDetector()
        self.risk_guard = CentralRiskGuard()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.total_equity = 100000.0
        self.positions: Dict[str, float] = {}
        self.target_allocations: Dict[str, float] = {}
        self.last_rebalance_time = 0.0
        self.rebalance_threshold = 0.05  # 5% drift threshold
        
        # Performance tracking
        self.portfolio_history: List[Dict] = []
        self.trade_recommendations: List[PortfolioRecommendation] = []
    
    def update_market_data(self, market_data: Dict[str, Dict[str, float]]):
        """
        Update market data for all components
        
        Args:
            market_data: {symbol: {"price": price, "volume": volume, ...}}
        """
        
        # Extract prices for regime detection
        prices = {symbol: data["price"] for symbol, data in market_data.items()}
        self.regime_detector.update_market_data(prices)
        
        # Update Kelly sizer with current regime
        current_regime = self.regime_detector.current_regime
        self.kelly_sizer.set_market_regime(current_regime)
        
        self.logger.info(f"📊 Updated market data for {len(market_data)} assets")
    
    def update_asset_metrics(self, asset_metrics: Dict[str, AssetMetrics]):
        """Update asset performance metrics"""
        self.kelly_sizer.update_asset_metrics(asset_metrics)
        self.logger.info(f"📈 Updated metrics for {len(asset_metrics)} assets")
    
    def update_correlation_matrix(self, correlation_matrix: pd.DataFrame):
        """Update asset correlation matrix"""
        self.kelly_sizer.update_correlation_matrix(correlation_matrix)
        self.logger.info(f"🔗 Updated correlation matrix: {correlation_matrix.shape}")
    
    def update_portfolio_state(self, total_equity: float, positions: Dict[str, float]):
        """Update current portfolio state"""
        self.total_equity = total_equity
        self.positions = positions.copy()
        
        # Update all components
        self.kelly_sizer.update_portfolio_state(total_equity, positions)
        
        # Update risk guard
        daily_pnl = 0.0  # Would need to calculate from previous day
        total_exposure = sum(positions.values())
        position_sizes = {k: v for k, v in positions.items()}
        
        self.risk_guard.update_portfolio_state(
            total_equity=total_equity,
            daily_pnl=daily_pnl,
            open_positions=len(positions),
            total_exposure_usd=total_exposure,
            position_sizes=position_sizes
        )
        
        self.logger.info(f"💼 Portfolio updated: ${total_equity:,.0f}, {len(positions)} positions")
    
    def calculate_optimal_portfolio(
        self, 
        signals: Dict[str, float],  # symbol -> signal_strength (0-1)
        rebalance_mode: bool = False
    ) -> Dict[str, PositionSize]:
        """
        Calculate optimal portfolio allocations
        
        Args:
            signals: Trading signals with strength
            rebalance_mode: If True, considers current positions
            
        Returns:
            Dict of symbol -> PositionSize
        """
        
        # Get position sizes from Kelly Vol Sizer
        position_sizes = self.kelly_sizer.calculate_portfolio_sizes(signals)
        
        # Apply risk guard validation if not rebalance mode
        if not rebalance_mode:
            position_sizes = self._apply_risk_guard_validation(position_sizes)
        
        # Store target allocations
        self.target_allocations = {
            symbol: ps.target_size_pct for symbol, ps in position_sizes.items()
        }
        
        return position_sizes
    
    def _apply_risk_guard_validation(
        self, 
        position_sizes: Dict[str, PositionSize]
    ) -> Dict[str, PositionSize]:
        """Apply risk guard validation to position sizes"""
        
        validated_sizes = {}
        
        for symbol, position_size in position_sizes.items():
            # Create trading operation for risk evaluation
            operation = TradingOperation(
                operation_type="entry",
                symbol=symbol,
                side="buy",  # Assume long positions for now
                size_usd=position_size.target_size_usd,
                current_price=50000.0,  # Would need actual price
                strategy_id="portfolio_optimization"
            )
            
            # Evaluate with risk guard
            risk_eval = self.risk_guard.evaluate_operation(operation)
            
            if risk_eval.decision == RiskDecision.APPROVE:
                validated_sizes[symbol] = position_size
            elif risk_eval.decision == RiskDecision.REDUCE_SIZE:
                # Create reduced position size
                reduced_pct = risk_eval.approved_size_usd / self.total_equity
                
                validated_sizes[symbol] = PositionSize(
                    symbol=symbol,
                    target_size_pct=reduced_pct,
                    target_size_usd=risk_eval.approved_size_usd,
                    kelly_size_pct=position_size.kelly_size_pct,
                    vol_adjusted_size_pct=position_size.vol_adjusted_size_pct,
                    regime_adjusted_size_pct=position_size.regime_adjusted_size_pct,
                    cluster_adjusted_size_pct=reduced_pct,
                    final_size_pct=reduced_pct,
                    reasoning=position_size.reasoning + [
                        f"Risk guard reduced to ${risk_eval.approved_size_usd:,.0f}: {'; '.join(risk_eval.reasons)}"
                    ]
                )
            else:
                # Position rejected by risk guard
                self.logger.warning(
                    f"⚠️  {symbol} rejected by risk guard: {'; '.join(risk_eval.reasons)}"
                )
        
        return validated_sizes
    
    def generate_rebalancing_recommendations(
        self, 
        force_rebalance: bool = False
    ) -> List[PortfolioRecommendation]:
        """
        Generate portfolio rebalancing recommendations
        
        Args:
            force_rebalance: Force rebalancing regardless of thresholds
            
        Returns:
            List of rebalancing recommendations
        """
        
        recommendations = []
        
        # Check if rebalancing is needed
        if not force_rebalance and not self._needs_rebalancing():
            self.logger.info("📊 Portfolio within rebalancing thresholds")
            return recommendations
        
        # Generate recommendations for each target allocation
        for symbol, target_pct in self.target_allocations.items():
            current_size = self.positions.get(symbol, 0.0)
            target_size = target_pct * self.total_equity
            size_diff = target_size - current_size
            
            if abs(size_diff) < self.total_equity * 0.01:  # Skip changes < 1%
                continue
            
            # Determine action
            if size_diff > 0:
                action = "buy" if current_size == 0 else "increase"
            else:
                action = "sell" if target_size == 0 else "reduce"
            
            # Calculate priority (larger changes = higher priority)
            change_pct = abs(size_diff) / max(current_size, target_size, self.total_equity * 0.01)
            priority = min(5, max(1, int(change_pct * 10)))
            
            recommendation = PortfolioRecommendation(
                symbol=symbol,
                current_size_usd=current_size,
                target_size_usd=target_size,
                recommended_action=action,
                size_change_usd=size_diff,
                size_change_pct=size_diff / current_size if current_size > 0 else 0,
                priority=priority,
                reasoning=[
                    f"Target allocation: {target_pct:.1%}",
                    f"Current allocation: {current_size/self.total_equity:.1%}",
                    f"Drift: {abs(size_diff)/self.total_equity:.1%}"
                ]
            )
            
            recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        
        self.trade_recommendations = recommendations
        self.last_rebalance_time = time.time()
        
        self.logger.info(f"📝 Generated {len(recommendations)} rebalancing recommendations")
        
        return recommendations
    
    def _needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing"""
        
        total_drift = 0.0
        max_drift = 0.0
        
        for symbol, target_pct in self.target_allocations.items():
            current_pct = self.positions.get(symbol, 0.0) / self.total_equity
            drift = abs(target_pct - current_pct)
            total_drift += drift
            max_drift = max(max_drift, drift)
        
        # Rebalance if max drift > threshold or total drift > 2x threshold
        needs_rebalance = (
            max_drift > self.rebalance_threshold or
            total_drift > 2 * self.rebalance_threshold
        )
        
        if needs_rebalance:
            self.logger.info(
                f"📊 Rebalancing needed: max drift {max_drift:.1%}, "
                f"total drift {total_drift:.1%}"
            )
        
        return needs_rebalance
    
    def get_portfolio_summary(self) -> PortfolioState:
        """Get comprehensive portfolio summary"""
        
        # Calculate allocations
        total_allocated = sum(self.positions.values())
        cash_balance = self.total_equity - total_allocated
        
        # Get cluster exposures
        cluster_exposures = self.kelly_sizer.get_cluster_exposures()
        cluster_utilization = {
            cluster_id: data["utilization"] 
            for cluster_id, data in cluster_exposures.items()
        }
        
        # Get risk utilization
        risk_status = self.risk_guard.get_risk_status()
        risk_utilization = max(
            risk_status["utilization"]["exposure_utilization"] / 100,
            risk_status["utilization"]["position_utilization"] / 100,
            risk_status["utilization"]["drawdown_utilization"] / 100
        )
        
        return PortfolioState(
            total_equity=self.total_equity,
            cash_balance=cash_balance,
            positions=self.positions.copy(),
            target_allocations=self.target_allocations.copy(),
            regime=self.regime_detector.current_regime,
            regime_confidence=self.regime_detector.regime_confidence,
            risk_utilization=risk_utilization,
            cluster_exposures=cluster_utilization
        )
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of all portfolio components"""
        
        return {
            "portfolio_summary": self.get_portfolio_summary().__dict__,
            "kelly_sizer_status": self.kelly_sizer.get_sizing_summary(),
            "regime_status": self.regime_detector.get_regime_status(),
            "risk_status": self.risk_guard.get_risk_status(),
            "rebalancing": {
                "last_rebalance": self.last_rebalance_time,
                "threshold": self.rebalance_threshold,
                "pending_recommendations": len(self.trade_recommendations),
                "needs_rebalancing": self._needs_rebalancing()
            }
        }


# Global portfolio manager instance
_global_portfolio_manager: Optional[IntegratedPortfolioManager] = None


def get_global_portfolio_manager() -> IntegratedPortfolioManager:
    """Get or create global portfolio manager"""
    global _global_portfolio_manager
    if _global_portfolio_manager is None:
        _global_portfolio_manager = IntegratedPortfolioManager()
        logger.info("✅ Global IntegratedPortfolioManager initialized")
    return _global_portfolio_manager


def reset_global_portfolio_manager():
    """Reset global portfolio manager (for testing)"""
    global _global_portfolio_manager
    _global_portfolio_manager = None