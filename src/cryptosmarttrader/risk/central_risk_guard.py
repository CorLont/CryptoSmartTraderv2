"""
Central RiskGuard: Poortwachter voor alle trading operaties
Verplichte risk checks vóór elke entry/resize/hedge broker-call
"""

import time
import logging
import threading
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REDUCE_SIZE = "reduce_size"
    KILL_SWITCH_ACTIVATED = "kill_switch"


class RiskViolationType(Enum):
    DAY_LOSS = "day_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    MAX_POSITIONS = "max_positions"
    DATA_GAP = "data_gap"
    KILL_SWITCH = "kill_switch"
    POSITION_SIZE = "position_size"
    CORRELATION_LIMIT = "correlation_limit"


@dataclass
class RiskLimits:
    """Centrale risk limits configuratie"""
    max_day_loss_pct: float = 2.0  # Max 2% daily loss
    max_drawdown_pct: float = 10.0  # Max 10% drawdown from peak
    max_total_exposure_pct: float = 50.0  # Max 50% total exposure
    max_positions: int = 10  # Max 10 open positions
    max_position_size_pct: float = 10.0  # Max 10% per position
    max_correlation_exposure: float = 20.0  # Max 20% in correlated assets
    max_data_gap_minutes: int = 5  # Max 5 minute data gap
    kill_switch_active: bool = False
    kill_switch_reason: Optional[str] = None
    emergency_only_mode: bool = False


@dataclass
class PortfolioState:
    """Huidige portfolio state voor risk evaluatie"""
    total_equity: float
    daily_pnl: float
    peak_equity: float
    current_drawdown_pct: float
    open_positions: int
    total_exposure_usd: float
    total_exposure_pct: float
    position_sizes: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    last_data_update: float = 0.0


@dataclass
class TradingOperation:
    """Trading operatie voor risk evaluatie"""
    operation_type: str  # "entry", "resize", "hedge", "exit"
    symbol: str
    side: str  # "buy", "sell"
    size_usd: float
    current_price: float
    strategy_id: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RiskEvaluation:
    """Resultaat van risk evaluatie"""
    decision: RiskDecision
    approved_size_usd: float
    violations: List[RiskViolationType]
    reasons: List[str]
    risk_score: float  # 0-100, higher = more risky
    recommendations: List[str]
    gate_results: Dict[str, bool] = field(default_factory=dict)


class CentralRiskGuard:
    """
    Centrale RiskGuard als poortwachter voor alle trading operaties
    VERPLICHT voor elke entry/resize/hedge vóór broker-calls
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.portfolio_state = PortfolioState(
            total_equity=100000.0,  # Default starting equity
            daily_pnl=0.0,
            peak_equity=100000.0,
            current_drawdown_pct=0.0,
            open_positions=0,
            total_exposure_usd=0.0,
            total_exposure_pct=0.0
        )
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        
        # Risk monitoring
        self.violation_count = 0
        self.total_evaluations = 0
        self.kill_switch_history: List[Dict] = []
        
        # Data gap monitoring
        self.last_market_data_update = time.time()
        self.data_gap_violations = 0
    
    def update_portfolio_state(
        self,
        total_equity: float,
        daily_pnl: float,
        open_positions: int,
        total_exposure_usd: float,
        position_sizes: Dict[str, float],
        correlations: Dict[str, float] = None
    ):
        """Update portfolio state voor risk evaluatie"""
        with self._lock:
            self.portfolio_state.total_equity = total_equity
            self.portfolio_state.daily_pnl = daily_pnl
            self.portfolio_state.open_positions = open_positions
            self.portfolio_state.total_exposure_usd = total_exposure_usd
            self.portfolio_state.total_exposure_pct = (total_exposure_usd / total_equity) * 100
            self.portfolio_state.position_sizes = position_sizes.copy()
            self.portfolio_state.correlations = correlations.copy() if correlations else {}
            
            # Update peak equity and drawdown
            if total_equity > self.portfolio_state.peak_equity:
                self.portfolio_state.peak_equity = total_equity
            
            drawdown = ((self.portfolio_state.peak_equity - total_equity) / self.portfolio_state.peak_equity) * 100
            self.portfolio_state.current_drawdown_pct = max(0, drawdown)
            
            self.portfolio_state.last_data_update = time.time()
            self.last_market_data_update = time.time()
    
    def evaluate_operation(self, operation: TradingOperation) -> RiskEvaluation:
        """
        🛡️  CENTRALE RISK EVALUATIE - Verplichte poortwachter
        Evalueert alle trading operaties vóór broker execution
        """
        with self._lock:
            self.total_evaluations += 1
            
            violations = []
            reasons = []
            gate_results = {}
            risk_score = 0.0
            approved_size = operation.size_usd
            
            self.logger.info(f"🛡️  RiskGuard evaluating {operation.operation_type}: {operation.symbol} {operation.size_usd:,.0f} USD")
            
            # Gate 1: Kill Switch Check (CRITICAL)
            kill_switch_ok = not self.risk_limits.kill_switch_active
            gate_results["kill_switch"] = kill_switch_ok
            
            if not kill_switch_ok:
                violations.append(RiskViolationType.KILL_SWITCH)
                reasons.append(f"Kill switch activated: {self.risk_limits.kill_switch_reason}")
                return RiskEvaluation(
                    decision=RiskDecision.KILL_SWITCH_ACTIVATED,
                    approved_size_usd=0.0,
                    violations=violations,
                    reasons=reasons,
                    risk_score=100.0,
                    recommendations=["Wait for kill switch deactivation"],
                    gate_results=gate_results
                )
            
            # Gate 2: Data Gap Check
            data_gap_minutes = (time.time() - self.last_market_data_update) / 60
            data_gap_ok = data_gap_minutes <= self.risk_limits.max_data_gap_minutes
            gate_results["data_gap"] = data_gap_ok
            
            if not data_gap_ok:
                violations.append(RiskViolationType.DATA_GAP)
                reasons.append(f"Data gap too large: {data_gap_minutes:.1f} minutes > {self.risk_limits.max_data_gap_minutes}")
                risk_score += 30
                self.data_gap_violations += 1
            
            # Gate 3: Daily Loss Check
            daily_loss_pct = (self.portfolio_state.daily_pnl / self.portfolio_state.total_equity) * 100
            day_loss_ok = daily_loss_pct >= -self.risk_limits.max_day_loss_pct
            gate_results["day_loss"] = day_loss_ok
            
            if not day_loss_ok:
                violations.append(RiskViolationType.DAY_LOSS)
                reasons.append(f"Daily loss limit exceeded: {daily_loss_pct:.2f}% > {self.risk_limits.max_day_loss_pct}%")
                risk_score += 40
            
            # Gate 4: Max Drawdown Check
            drawdown_ok = self.portfolio_state.current_drawdown_pct <= self.risk_limits.max_drawdown_pct
            gate_results["drawdown"] = drawdown_ok
            
            if not drawdown_ok:
                violations.append(RiskViolationType.MAX_DRAWDOWN)
                reasons.append(f"Max drawdown exceeded: {self.portfolio_state.current_drawdown_pct:.2f}% > {self.risk_limits.max_drawdown_pct}%")
                risk_score += 50
            
            # Gate 5: Position Count Check
            new_position_count = self.portfolio_state.open_positions
            if operation.operation_type == "entry":
                new_position_count += 1
            
            position_count_ok = new_position_count <= self.risk_limits.max_positions
            gate_results["position_count"] = position_count_ok
            
            if not position_count_ok:
                violations.append(RiskViolationType.MAX_POSITIONS)
                reasons.append(f"Max positions exceeded: {new_position_count} > {self.risk_limits.max_positions}")
                risk_score += 25
            
            # Gate 6: Total Exposure Check
            new_exposure_usd = self.portfolio_state.total_exposure_usd + operation.size_usd
            new_exposure_pct = (new_exposure_usd / self.portfolio_state.total_equity) * 100
            exposure_ok = new_exposure_pct <= self.risk_limits.max_total_exposure_pct
            gate_results["total_exposure"] = exposure_ok
            
            if not exposure_ok:
                violations.append(RiskViolationType.MAX_EXPOSURE)
                reasons.append(f"Total exposure limit exceeded: {new_exposure_pct:.1f}% > {self.risk_limits.max_total_exposure_pct}%")
                risk_score += 35
                
                # Calculate reduced size to stay within limits
                max_allowed_exposure = (self.risk_limits.max_total_exposure_pct / 100) * self.portfolio_state.total_equity
                available_exposure = max_allowed_exposure - self.portfolio_state.total_exposure_usd
                approved_size = max(0, available_exposure)
            
            # Gate 7: Position Size Check
            position_size_pct = (operation.size_usd / self.portfolio_state.total_equity) * 100
            position_size_ok = position_size_pct <= self.risk_limits.max_position_size_pct
            gate_results["position_size"] = position_size_ok
            
            if not position_size_ok:
                violations.append(RiskViolationType.POSITION_SIZE)
                reasons.append(f"Position size too large: {position_size_pct:.2f}% > {self.risk_limits.max_position_size_pct}%")
                risk_score += 20
                
                # Calculate max allowed position size
                max_position_size = (self.risk_limits.max_position_size_pct / 100) * self.portfolio_state.total_equity
                approved_size = min(approved_size, max_position_size)
            
            # Gate 8: Correlation Check
            if operation.symbol in self.portfolio_state.correlations:
                correlation_exposure = sum(
                    size for symbol, size in self.portfolio_state.position_sizes.items()
                    if self.portfolio_state.correlations.get(symbol, 0) > 0.7  # High correlation
                )
                correlation_exposure_pct = (correlation_exposure / self.portfolio_state.total_equity) * 100
                correlation_ok = correlation_exposure_pct <= self.risk_limits.max_correlation_exposure
                gate_results["correlation"] = correlation_ok
                
                if not correlation_ok:
                    violations.append(RiskViolationType.CORRELATION_LIMIT)
                    reasons.append(f"Correlation exposure too high: {correlation_exposure_pct:.1f}% > {self.risk_limits.max_correlation_exposure}%")
                    risk_score += 15
            else:
                gate_results["correlation"] = True
            
            # Determine final decision
            critical_violations = [
                RiskViolationType.KILL_SWITCH,
                RiskViolationType.MAX_DRAWDOWN,
                RiskViolationType.DAY_LOSS
            ]
            
            has_critical_violation = any(v in violations for v in critical_violations)
            
            if has_critical_violation or not data_gap_ok:
                decision = RiskDecision.REJECT
                approved_size = 0.0
                self.violation_count += 1
            elif approved_size < operation.size_usd * 0.1:  # Less than 10% of requested size
                decision = RiskDecision.REJECT
                approved_size = 0.0
                reasons.append("Approved size too small to be meaningful")
            elif approved_size < operation.size_usd:
                decision = RiskDecision.REDUCE_SIZE
            else:
                decision = RiskDecision.APPROVE
            
            # Generate recommendations
            recommendations = self._generate_recommendations(violations, risk_score)
            
            result = RiskEvaluation(
                decision=decision,
                approved_size_usd=approved_size,
                violations=violations,
                reasons=reasons,
                risk_score=risk_score,
                recommendations=recommendations,
                gate_results=gate_results
            )
            
            self.logger.info(
                f"🛡️  RiskGuard decision: {decision.value} "
                f"(approved: ${approved_size:,.0f}, risk: {risk_score:.1f})"
            )
            
            return result
    
    def _generate_recommendations(self, violations: List[RiskViolationType], risk_score: float) -> List[str]:
        """Generate actionable recommendations based on violations"""
        recommendations = []
        
        if RiskViolationType.DAY_LOSS in violations:
            recommendations.append("Consider stopping trading for today to limit losses")
        
        if RiskViolationType.MAX_DRAWDOWN in violations:
            recommendations.append("Reduce position sizes and implement stricter stop-losses")
        
        if RiskViolationType.MAX_EXPOSURE in violations:
            recommendations.append("Close some positions to reduce total exposure")
        
        if RiskViolationType.MAX_POSITIONS in violations:
            recommendations.append("Close least profitable positions before opening new ones")
        
        if RiskViolationType.DATA_GAP in violations:
            recommendations.append("Wait for fresh market data before trading")
        
        if RiskViolationType.POSITION_SIZE in violations:
            recommendations.append("Reduce position size to stay within risk limits")
        
        if RiskViolationType.CORRELATION_LIMIT in violations:
            recommendations.append("Diversify into uncorrelated assets")
        
        if risk_score > 70:
            recommendations.append("Overall risk too high - consider defensive positioning")
        elif risk_score > 50:
            recommendations.append("Moderate risk detected - proceed with caution")
        
        return recommendations
    
    def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch"""
        with self._lock:
            self.risk_limits.kill_switch_active = True
            self.risk_limits.kill_switch_reason = reason
            
            self.kill_switch_history.append({
                "timestamp": time.time(),
                "reason": reason,
                "action": "activated",
                "portfolio_equity": self.portfolio_state.total_equity,
                "daily_pnl": self.portfolio_state.daily_pnl,
                "drawdown": self.portfolio_state.current_drawdown_pct
            })
            
            self.logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate_kill_switch(self, reason: str):
        """Deactivate kill switch"""
        with self._lock:
            self.risk_limits.kill_switch_active = False
            self.risk_limits.kill_switch_reason = None
            
            self.kill_switch_history.append({
                "timestamp": time.time(),
                "reason": reason,
                "action": "deactivated",
                "portfolio_equity": self.portfolio_state.total_equity,
                "daily_pnl": self.portfolio_state.daily_pnl,
                "drawdown": self.portfolio_state.current_drawdown_pct
            })
            
            self.logger.info(f"✅ Kill switch deactivated: {reason}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        with self._lock:
            return {
                "risk_limits": {
                    "max_day_loss_pct": self.risk_limits.max_day_loss_pct,
                    "max_drawdown_pct": self.risk_limits.max_drawdown_pct,
                    "max_total_exposure_pct": self.risk_limits.max_total_exposure_pct,
                    "max_positions": self.risk_limits.max_positions,
                    "kill_switch_active": self.risk_limits.kill_switch_active,
                    "kill_switch_reason": self.risk_limits.kill_switch_reason
                },
                "portfolio_state": {
                    "total_equity": self.portfolio_state.total_equity,
                    "daily_pnl": self.portfolio_state.daily_pnl,
                    "daily_pnl_pct": (self.portfolio_state.daily_pnl / self.portfolio_state.total_equity) * 100,
                    "current_drawdown_pct": self.portfolio_state.current_drawdown_pct,
                    "open_positions": self.portfolio_state.open_positions,
                    "total_exposure_pct": self.portfolio_state.total_exposure_pct,
                    "data_age_minutes": (time.time() - self.last_market_data_update) / 60
                },
                "statistics": {
                    "total_evaluations": self.total_evaluations,
                    "violation_count": self.violation_count,
                    "violation_rate": self.violation_count / max(1, self.total_evaluations),
                    "data_gap_violations": self.data_gap_violations,
                    "kill_switch_activations": len(self.kill_switch_history)
                },
                "utilization": {
                    "exposure_utilization": min(100, self.portfolio_state.total_exposure_pct / self.risk_limits.max_total_exposure_pct * 100),
                    "position_utilization": min(100, self.portfolio_state.open_positions / self.risk_limits.max_positions * 100),
                    "drawdown_utilization": min(100, self.portfolio_state.current_drawdown_pct / self.risk_limits.max_drawdown_pct * 100)
                }
            }


# Global RiskGuard instance
_global_risk_guard: Optional[CentralRiskGuard] = None


def get_global_risk_guard() -> CentralRiskGuard:
    """Get or create global RiskGuard instance"""
    global _global_risk_guard
    if _global_risk_guard is None:
        _global_risk_guard = CentralRiskGuard()
        logger.info("✅ Global CentralRiskGuard initialized")
    return _global_risk_guard


def reset_global_risk_guard():
    """Reset global RiskGuard (for testing)"""
    global _global_risk_guard
    _global_risk_guard = None