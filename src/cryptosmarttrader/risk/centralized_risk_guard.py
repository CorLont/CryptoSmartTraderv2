#!/usr/bin/env python3
"""
Centralized RiskGuard - Hard wire-up voor elke entry/resize/hedge operatie
Controleert day-loss, max-DD, exposure/pos limits, data-gap + kill-switch
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
from pathlib import Path

from ..core.structured_logger import get_logger
from ..risk.risk_guard import RiskGuard, RiskLevel
from ..observability.unified_metrics import UnifiedMetrics


class RiskAction(Enum):
    """Risk management actions."""

    ALLOW = "allow"
    WARN = "warn"
    REDUCE = "reduce"
    BLOCK = "block"
    KILL_SWITCH = "kill_switch"


class OperationType(Enum):
    """Trading operation types."""

    ENTRY = "entry"
    RESIZE = "resize"
    HEDGE = "hedge"
    EXIT = "exit"
    REBALANCE = "rebalance"


@dataclass
class RiskCheckRequest:
    """Request for centralized risk check."""

    operation_type: OperationType
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float

    # Portfolio context
    current_portfolio_value: float
    current_positions: Dict[str, float]  # symbol -> position_size
    current_exposure: float  # Total exposure

    # Performance context
    daily_pnl: float
    total_drawdown_pct: float

    # Data quality context
    data_age_seconds: float
    data_quality_score: float
    last_price_update: datetime

    # Metadata
    strategy_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskCheckResult:
    """Result of centralized risk check."""

    action: RiskAction
    approved: bool

    # Risk analysis
    risk_level: RiskLevel
    violations: List[str]
    warnings: List[str]

    # Position limits
    max_allowed_quantity: float
    position_limit_pct: float

    # Risk metrics
    portfolio_risk_pct: float
    position_concentration: float
    data_quality_score: float

    # Kill switch status
    kill_switch_active: bool
    kill_switch_reason: Optional[str]

    # Execution parameters
    execution_allowed: bool
    max_position_size: float
    risk_budget_used_pct: float

    # Timing
    check_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


class CentralizedRiskGuard:
    """
    Centralized RiskGuard system - Hard wire-up voor alle trading operaties.

    Features:
    - Day-loss limits monitoring (5% trigger, 8% emergency)
    - Max drawdown protection (10% warning, 15% kill-switch)
    - Position/exposure limits (2% per position, 95% total)
    - Data quality gates (age, reliability, gaps)
    - Kill-switch system (manual + automatic)
    - Real-time risk monitoring
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,
        max_drawdown_pct: float = 10.0,
        max_position_size_pct: float = 2.0,
        max_total_exposure_pct: float = 95.0,
        min_data_quality_score: float = 0.7,
        max_data_age_minutes: float = 5.0,
    ):
        """Initialize centralized risk guard."""

        self.logger = get_logger("centralized_risk_guard")

        # Risk limits configuration
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.min_data_quality_score = min_data_quality_score
        self.max_data_age_minutes = max_data_age_minutes

        # Core risk guard integration
        self.risk_guard = RiskGuard()

        # Metrics and monitoring
        self.metrics = UnifiedMetrics("centralized_risk_guard")

        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_timestamp = None

        # Risk tracking
        self.daily_operations = []
        self.risk_violations = []
        self.position_tracking: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self.daily_pnl_history = []
        self.drawdown_history = []

        # Threading
        self._lock = threading.RLock()

        # Persistence
        self.data_path = Path("data/centralized_risk_guard")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Load persistent state
        self._load_persistent_state()

        self.logger.info(
            "CentralizedRiskGuard initialized with hard wire-up",
            daily_loss_limit_pct=max_daily_loss_pct,
            max_drawdown_pct=max_drawdown_pct,
            position_limit_pct=max_position_size_pct,
            total_exposure_limit_pct=max_total_exposure_pct,
            kill_switch_active=self.kill_switch_active,
        )

    async def check_operation_risk(self, request: RiskCheckRequest) -> RiskCheckResult:
        """
        Central risk check voor alle trading operaties.

        Hard wire-up flow:
        1. Check kill-switch status
        2. Validate day-loss limits
        3. Check max drawdown
        4. Validate position/exposure limits
        5. Check data quality gates
        6. Determine risk action
        """

        check_start = time.time()

        with self._lock:
            try:
                # Initialize result
                result = RiskCheckResult(
                    action=RiskAction.ALLOW,
                    approved=False,
                    risk_level=RiskLevel.NORMAL,
                    violations=[],
                    warnings=[],
                    max_allowed_quantity=request.quantity,
                    position_limit_pct=0.0,
                    portfolio_risk_pct=0.0,
                    position_concentration=0.0,
                    data_quality_score=request.data_quality_score,
                    kill_switch_active=self.kill_switch_active,
                    kill_switch_reason=self.kill_switch_reason,
                    execution_allowed=False,
                    max_position_size=0.0,
                    risk_budget_used_pct=0.0,
                    check_time_ms=0.0,
                )

                # Step 1: Kill-switch check (HARD BLOCK)
                if self.kill_switch_active:
                    result.action = RiskAction.KILL_SWITCH
                    result.approved = False
                    result.execution_allowed = False
                    result.violations.append("kill_switch_active")

                    self.logger.warning(
                        "Operation blocked by kill-switch",
                        operation_type=request.operation_type.value,
                        symbol=request.symbol,
                        kill_switch_reason=self.kill_switch_reason,
                    )

                    self._finalize_result(result, check_start)
                    return result

                # Step 2: Day-loss limits check
                day_loss_check = self._check_daily_loss_limits(request)
                if day_loss_check["violation"]:
                    result.violations.append(day_loss_check["violation_type"])
                    if day_loss_check["action"] in ["block", "kill_switch"]:
                        result.action = (
                            RiskAction.BLOCK
                            if day_loss_check["action"] == "block"
                            else RiskAction.KILL_SWITCH
                        )
                        result.approved = False

                        if day_loss_check["action"] == "kill_switch":
                            self._trigger_kill_switch(
                                f"Daily loss limit exceeded: {request.daily_pnl:.2f}%"
                            )

                        self._finalize_result(result, check_start)
                        return result

                # Step 3: Max drawdown check
                drawdown_check = self._check_max_drawdown(request)
                if drawdown_check["violation"]:
                    result.violations.append(drawdown_check["violation_type"])
                    if drawdown_check["action"] in ["block", "kill_switch"]:
                        result.action = (
                            RiskAction.BLOCK
                            if drawdown_check["action"] == "block"
                            else RiskAction.KILL_SWITCH
                        )
                        result.approved = False

                        if drawdown_check["action"] == "kill_switch":
                            self._trigger_kill_switch(
                                f"Max drawdown exceeded: {request.total_drawdown_pct:.2f}%"
                            )

                        self._finalize_result(result, check_start)
                        return result

                # Step 4: Position/exposure limits check
                position_check = self._check_position_limits(request)
                result.max_allowed_quantity = position_check["max_allowed_quantity"]
                result.position_limit_pct = position_check["position_limit_pct"]
                result.max_position_size = position_check["max_position_size"]

                if position_check["violation"]:
                    result.violations.append(position_check["violation_type"])
                    if position_check["action"] == "block":
                        result.action = RiskAction.BLOCK
                        result.approved = False
                        self._finalize_result(result, check_start)
                        return result
                    elif position_check["action"] == "reduce":
                        result.action = RiskAction.REDUCE
                        result.max_allowed_quantity = position_check["reduced_quantity"]

                # Step 5: Data quality gates
                data_check = self._check_data_quality(request)
                if data_check["violation"]:
                    result.violations.append(data_check["violation_type"])
                    if data_check["action"] == "block":
                        result.action = RiskAction.BLOCK
                        result.approved = False
                        self._finalize_result(result, check_start)
                        return result

                # Step 6: Calculate final risk metrics
                result.portfolio_risk_pct = self._calculate_portfolio_risk(request)
                result.position_concentration = self._calculate_position_concentration(request)
                result.risk_budget_used_pct = self._calculate_risk_budget_usage(request)

                # Step 7: Determine final risk level and action
                result.risk_level = self._determine_risk_level(result)

                # Final approval decision
                if len(result.violations) == 0:
                    result.approved = True
                    result.execution_allowed = True
                    if result.action == RiskAction.ALLOW:
                        result.action = RiskAction.ALLOW
                elif result.action == RiskAction.REDUCE and result.max_allowed_quantity > 0:
                    result.approved = True
                    result.execution_allowed = True

                # Log risk check result
                self._log_risk_check(request, result)

                # Update tracking
                self._update_risk_tracking(request, result)

                self._finalize_result(result, check_start)
                return result

            except Exception as e:
                # Handle risk check errors
                check_time = (time.time() - check_start) * 1000

                error_result = RiskCheckResult(
                    action=RiskAction.BLOCK,
                    approved=False,
                    risk_level=RiskLevel.CRITICAL,
                    violations=[f"risk_check_error: {str(e)}"],
                    warnings=[],
                    max_allowed_quantity=0.0,
                    position_limit_pct=0.0,
                    portfolio_risk_pct=0.0,
                    position_concentration=0.0,
                    data_quality_score=0.0,
                    kill_switch_active=self.kill_switch_active,
                    kill_switch_reason=self.kill_switch_reason,
                    execution_allowed=False,
                    max_position_size=0.0,
                    risk_budget_used_pct=0.0,
                    check_time_ms=check_time,
                )

                self.logger.error(
                    "Risk check failed - blocking operation",
                    operation_type=request.operation_type.value,
                    symbol=request.symbol,
                    error=str(e),
                    check_time_ms=check_time,
                )

                return error_result

    def _check_daily_loss_limits(self, request: RiskCheckRequest) -> Dict[str, Any]:
        """Check daily loss limits."""

        daily_loss_pct = abs(request.daily_pnl) if request.daily_pnl < 0 else 0

        # Critical threshold (8% - kill switch)
        if daily_loss_pct >= (self.max_daily_loss_pct * 1.6):  # 8%
            return {
                "violation": True,
                "violation_type": "daily_loss_critical",
                "action": "kill_switch",
                "daily_loss_pct": daily_loss_pct,
            }

        # Warning threshold (5% - block new positions)
        if daily_loss_pct >= self.max_daily_loss_pct:  # 5%
            return {
                "violation": True,
                "violation_type": "daily_loss_limit",
                "action": "block",
                "daily_loss_pct": daily_loss_pct,
            }

        # Caution threshold (3% - warning)
        if daily_loss_pct >= (self.max_daily_loss_pct * 0.6):  # 3%
            return {
                "violation": True,
                "violation_type": "daily_loss_warning",
                "action": "warn",
                "daily_loss_pct": daily_loss_pct,
            }

        return {"violation": False, "daily_loss_pct": daily_loss_pct}

    def _check_max_drawdown(self, request: RiskCheckRequest) -> Dict[str, Any]:
        """Check maximum drawdown limits."""

        drawdown_pct = request.total_drawdown_pct

        # Emergency threshold (15% - kill switch)
        if drawdown_pct >= (self.max_drawdown_pct * 1.5):  # 15%
            return {
                "violation": True,
                "violation_type": "max_drawdown_critical",
                "action": "kill_switch",
                "drawdown_pct": drawdown_pct,
            }

        # Critical threshold (10% - block new positions)
        if drawdown_pct >= self.max_drawdown_pct:  # 10%
            return {
                "violation": True,
                "violation_type": "max_drawdown_limit",
                "action": "block",
                "drawdown_pct": drawdown_pct,
            }

        # Warning threshold (7% - warning)
        if drawdown_pct >= (self.max_drawdown_pct * 0.7):  # 7%
            return {
                "violation": True,
                "violation_type": "max_drawdown_warning",
                "action": "warn",
                "drawdown_pct": drawdown_pct,
            }

        return {"violation": False, "drawdown_pct": drawdown_pct}

    def _check_position_limits(self, request: RiskCheckRequest) -> Dict[str, Any]:
        """Check position and exposure limits."""

        # Calculate new position size after operation
        current_position = request.current_positions.get(request.symbol, 0)
        position_change = request.quantity if request.side == "buy" else -request.quantity
        new_position = current_position + position_change

        # Position value
        position_value = abs(new_position * request.price)
        position_pct = (position_value / request.current_portfolio_value) * 100

        # Max position check (2% per position)
        if position_pct > self.max_position_size_pct:
            # Calculate reduced quantity to stay within limits
            max_position_value = request.current_portfolio_value * self.max_position_size_pct / 100
            max_quantity = max_position_value / request.price
            reduced_quantity = max_quantity - abs(current_position)

            if reduced_quantity <= 0:
                return {
                    "violation": True,
                    "violation_type": "position_size_limit",
                    "action": "block",
                    "position_pct": position_pct,
                    "max_allowed_quantity": 0,
                    "position_limit_pct": self.max_position_size_pct,
                    "max_position_size": max_position_value,
                }
            else:
                return {
                    "violation": True,
                    "violation_type": "position_size_reduce",
                    "action": "reduce",
                    "position_pct": position_pct,
                    "max_allowed_quantity": request.quantity,
                    "reduced_quantity": reduced_quantity,
                    "position_limit_pct": self.max_position_size_pct,
                    "max_position_size": max_position_value,
                }

        # Total exposure check (95% total)
        new_total_exposure = request.current_exposure + position_value
        exposure_pct = (new_total_exposure / request.current_portfolio_value) * 100

        if exposure_pct > self.max_total_exposure_pct:
            return {
                "violation": True,
                "violation_type": "total_exposure_limit",
                "action": "block",
                "exposure_pct": exposure_pct,
                "max_allowed_quantity": 0,
                "position_limit_pct": self.max_position_size_pct,
                "max_position_size": position_value,
            }

        return {
            "violation": False,
            "position_pct": position_pct,
            "max_allowed_quantity": request.quantity,
            "position_limit_pct": position_pct,
            "max_position_size": position_value,
        }

    def _check_data_quality(self, request: RiskCheckRequest) -> Dict[str, Any]:
        """Check data quality gates."""

        # Data age check
        data_age_minutes = request.data_age_seconds / 60
        if data_age_minutes > self.max_data_age_minutes:
            return {
                "violation": True,
                "violation_type": "data_age_exceeded",
                "action": "block",
                "data_age_minutes": data_age_minutes,
            }

        # Data quality score check
        if request.data_quality_score < self.min_data_quality_score:
            return {
                "violation": True,
                "violation_type": "data_quality_low",
                "action": "block",
                "data_quality_score": request.data_quality_score,
            }

        # Price update check (max 10 minutes old)
        if request.last_price_update:
            price_age = (datetime.now() - request.last_price_update).total_seconds() / 60
            if price_age > 10:
                return {
                    "violation": True,
                    "violation_type": "price_data_stale",
                    "action": "block",
                    "price_age_minutes": price_age,
                }

        return {"violation": False}

    def _calculate_portfolio_risk(self, request: RiskCheckRequest) -> float:
        """Calculate current portfolio risk percentage."""

        # Simple risk calculation based on exposure and volatility
        total_exposure_pct = (request.current_exposure / request.current_portfolio_value) * 100
        volatility_factor = 1.2  # Crypto volatility factor

        portfolio_risk = total_exposure_pct * volatility_factor / 100
        return min(portfolio_risk, 100.0)

    def _calculate_position_concentration(self, request: RiskCheckRequest) -> float:
        """Calculate position concentration risk."""

        if not request.current_positions:
            return 0.0

        # Calculate Herfindahl-Hirschman Index for concentration
        position_values = []
        total_value = 0

        for symbol, position in request.current_positions.items():
            # Estimate position value (simplified)
            position_value = abs(position * request.price)  # Using current price as proxy
            position_values.append(position_value)
            total_value += position_value

        if total_value == 0:
            return 0.0

        # Calculate HHI
        hhi = sum((value / total_value) ** 2 for value in position_values)
        concentration_pct = hhi * 100

        return concentration_pct

    def _calculate_risk_budget_usage(self, request: RiskCheckRequest) -> float:
        """Calculate risk budget usage percentage."""

        # Risk budget based on daily loss and drawdown
        daily_risk_used = (
            abs(request.daily_pnl) / self.max_daily_loss_pct * 100 if request.daily_pnl < 0 else 0
        )
        drawdown_risk_used = request.total_drawdown_pct / self.max_drawdown_pct * 100

        # Take maximum of daily and drawdown risk
        risk_budget_used = max(daily_risk_used, drawdown_risk_used)

        return min(risk_budget_used, 100.0)

    def _determine_risk_level(self, result: RiskCheckResult) -> RiskLevel:
        """Determine overall risk level."""

        if result.kill_switch_active or len([v for v in result.violations if "critical" in v]) > 0:
            return RiskLevel.EMERGENCY

        if len(result.violations) > 0:
            return RiskLevel.CRITICAL

        if result.risk_budget_used_pct > 80:
            return RiskLevel.WARNING

        if result.risk_budget_used_pct > 60:
            return RiskLevel.CONSERVATIVE

        return RiskLevel.NORMAL

    def _trigger_kill_switch(self, reason: str) -> None:
        """Trigger kill-switch system."""

        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_timestamp = datetime.now()

        # Save persistent state
        self._save_persistent_state()

        # Record metrics
        self.metrics.record_alert("kill_switch_triggered", "centralized_risk_guard", reason)

        self.logger.critical(
            "KILL-SWITCH TRIGGERED", reason=reason, timestamp=self.kill_switch_timestamp.isoformat()
        )

    def deactivate_kill_switch(self, authorized_user: str = "system") -> bool:
        """Deactivate kill-switch (manual override)."""

        with self._lock:
            if self.kill_switch_active:
                self.kill_switch_active = False
                old_reason = self.kill_switch_reason
                self.kill_switch_reason = None
                self.kill_switch_timestamp = None

                # Save state
                self._save_persistent_state()

                self.logger.warning(
                    "Kill-switch deactivated",
                    authorized_user=authorized_user,
                    previous_reason=old_reason,
                )

                return True

        return False

    def _log_risk_check(self, request: RiskCheckRequest, result: RiskCheckResult) -> None:
        """Log risk check results."""

        if result.approved:
            self.logger.info(
                "Risk check approved",
                operation_type=request.operation_type.value,
                symbol=request.symbol,
                quantity=request.quantity,
                risk_level=result.risk_level.value,
                check_time_ms=result.check_time_ms,
            )
        else:
            self.logger.warning(
                "Risk check blocked",
                operation_type=request.operation_type.value,
                symbol=request.symbol,
                quantity=request.quantity,
                violations=result.violations,
                risk_level=result.risk_level.value,
                check_time_ms=result.check_time_ms,
            )

    def _update_risk_tracking(self, request: RiskCheckRequest, result: RiskCheckResult) -> None:
        """Update risk tracking data."""

        # Track daily operations
        self.daily_operations.append(
            {
                "timestamp": request.timestamp,
                "operation_type": request.operation_type.value,
                "symbol": request.symbol,
                "approved": result.approved,
                "risk_level": result.risk_level.value,
                "violations": result.violations,
            }
        )

        # Track violations
        if result.violations:
            self.risk_violations.append(
                {
                    "timestamp": request.timestamp,
                    "violations": result.violations,
                    "operation_type": request.operation_type.value,
                    "symbol": request.symbol,
                }
            )

        # Clean old data (keep last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.daily_operations = [
            op for op in self.daily_operations if op["timestamp"] > cutoff_time
        ]
        self.risk_violations = [v for v in self.risk_violations if v["timestamp"] > cutoff_time]

    def _finalize_result(self, result: RiskCheckResult, start_time: float) -> None:
        """Finalize risk check result."""

        result.check_time_ms = (time.time() - start_time) * 1000

        # Record metrics
        status = "approved" if result.approved else "blocked"
        self.metrics.record_order(status, "risk_check", "centralized_risk_guard")

    def _load_persistent_state(self) -> None:
        """Load persistent kill-switch state."""

        state_file = self.data_path / "kill_switch_state.json"

        try:
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)

                self.kill_switch_active = state.get("kill_switch_active", False)
                self.kill_switch_reason = state.get("kill_switch_reason")

                if state.get("kill_switch_timestamp"):
                    self.kill_switch_timestamp = datetime.fromisoformat(
                        state["kill_switch_timestamp"]
                    )

                self.logger.info(
                    "Loaded persistent kill-switch state",
                    kill_switch_active=self.kill_switch_active,
                    reason=self.kill_switch_reason,
                )
        except Exception as e:
            self.logger.error("Failed to load persistent state", error=str(e))

    def _save_persistent_state(self) -> None:
        """Save persistent kill-switch state."""

        state_file = self.data_path / "kill_switch_state.json"

        try:
            state = {
                "kill_switch_active": self.kill_switch_active,
                "kill_switch_reason": self.kill_switch_reason,
                "kill_switch_timestamp": self.kill_switch_timestamp.isoformat()
                if self.kill_switch_timestamp
                else None,
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            self.logger.error("Failed to save persistent state", error=str(e))

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status and statistics."""

        # Recent operations stats
        recent_ops = len(self.daily_operations)
        approved_ops = len([op for op in self.daily_operations if op["approved"]])
        blocked_ops = recent_ops - approved_ops

        return {
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "kill_switch_timestamp": self.kill_switch_timestamp.isoformat()
            if self.kill_switch_timestamp
            else None,
            "risk_limits": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_position_size_pct": self.max_position_size_pct,
                "max_total_exposure_pct": self.max_total_exposure_pct,
                "min_data_quality_score": self.min_data_quality_score,
                "max_data_age_minutes": self.max_data_age_minutes,
            },
            "daily_statistics": {
                "operations_total": recent_ops,
                "operations_approved": approved_ops,
                "operations_blocked": blocked_ops,
                "approval_rate_pct": (approved_ops / max(recent_ops, 1)) * 100,
                "violations_count": len(self.risk_violations),
            },
            "recent_violations": [
                {
                    "timestamp": v["timestamp"].isoformat(),
                    "violations": v["violations"],
                    "operation_type": v["operation_type"],
                    "symbol": v["symbol"],
                }
                for v in self.risk_violations[-10:]  # Last 10 violations
            ],
        }
