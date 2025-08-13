"""
Drawdown Monitor

Advanced drawdown tracking and adaptive risk scaling system.
Monitors portfolio drawdowns and triggers risk reduction measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DrawdownLevel(Enum):
    """Drawdown severity levels"""
    NORMAL = "normal"           # < 5% drawdown
    CAUTION = "caution"         # 5-10% drawdown
    WARNING = "warning"         # 10-15% drawdown
    DANGER = "danger"           # 15-25% drawdown
    CRITICAL = "critical"       # > 25% drawdown


class RiskReduction(Enum):
    """Risk reduction strategies"""
    NONE = "none"                   # No reduction
    KELLY_HALF = "kelly_half"       # Halve Kelly fraction
    VOLUME_HALF = "volume_half"     # Halve volume target
    BOTH_HALF = "both_half"         # Halve both Kelly and volume
    PAPER_ONLY = "paper_only"       # Switch to paper trading
    FULL_STOP = "full_stop"         # Stop all trading


@dataclass
class DrawdownEvent:
    """Drawdown event record"""
    event_id: str
    start_time: datetime
    end_time: Optional[datetime]

    # Drawdown metrics
    peak_value: float
    trough_value: float
    max_drawdown_pct: float
    current_drawdown_pct: float

    # Duration metrics
    duration_hours: float
    underwater_time: float      # Time below peak
    recovery_time: Optional[float] = None

    # Classification
    level: DrawdownLevel
    triggered_reductions: List[RiskReduction] = None

    # Recovery tracking
    is_recovered: bool = False
    recovery_high: Optional[float] = None

    def __post_init__(self):
        if self.triggered_reductions is None:
            self.triggered_reductions = []

    @property
    def is_active(self) -> bool:
        """Check if drawdown is still active"""
        return self.end_time is None

    @property
    def severity_score(self) -> float:
        """Calculate severity score (0-1)"""
        return min(1.0, self.max_drawdown_pct / 0.5)  # Max at 50% DD


@dataclass
class DrawdownStats:
    """Portfolio drawdown statistics"""
    current_drawdown_pct: float
    max_drawdown_pct: float
    average_drawdown_pct: float
    drawdown_frequency: float           # Drawdowns per month
    average_recovery_time_hours: float
    longest_drawdown_hours: float
    current_underwater_time_hours: float

    # Risk-adjusted metrics
    calmar_ratio: float                 # Return / Max DD
    sterling_ratio: float               # Return / Avg DD
    ulcer_index: float                  # Sqrt(mean(squared drawdowns))

    # Recent performance
    drawdowns_last_30d: int
    worst_drawdown_30d_pct: float
    recovery_efficiency: float         # How fast recoveries happen


class DrawdownMonitor:
    """
    Advanced drawdown monitoring and adaptive risk management
    """

    def __init__(self):
        self.equity_history = []        # List of (timestamp, equity_value)
        self.drawdown_events = []
        self.current_drawdown = None

        # Configuration
        self.daily_loss_limit_pct = 0.05    # 5% daily loss limit
        self.rolling_dd_thresholds = {
            DrawdownLevel.CAUTION: 0.05,    # 5%
            DrawdownLevel.WARNING: 0.10,    # 10%
            DrawdownLevel.DANGER: 0.15,     # 15%
            DrawdownLevel.CRITICAL: 0.25    # 25%
        }

        # Risk reduction mapping
        self.risk_reduction_map = {
            DrawdownLevel.CAUTION: RiskReduction.KELLY_HALF,
            DrawdownLevel.WARNING: RiskReduction.BOTH_HALF,
            DrawdownLevel.DANGER: RiskReduction.PAPER_ONLY,
            DrawdownLevel.CRITICAL: RiskReduction.FULL_STOP
        }

        # State tracking
        self.current_risk_level = DrawdownLevel.NORMAL
        self.active_reductions = set()
        self.last_peak_value = 0.0
        self.last_peak_time = datetime.now()

    def update_equity(self, equity_value: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Update equity and calculate drawdown metrics"""
        try:
            if timestamp is None:
                timestamp = datetime.now()

            # Store equity point
            self.equity_history.append((timestamp, equity_value))

            # Keep only recent history (1 year)
            cutoff_time = timestamp - timedelta(days=365)
            self.equity_history = [
                (ts, val) for ts, val in self.equity_history
                if ts >= cutoff_time
            ]

            # Update peak tracking
            if equity_value > self.last_peak_value:
                self.last_peak_value = equity_value
                self.last_peak_time = timestamp

                # Check if we recovered from current drawdown
                if self.current_drawdown and not self.current_drawdown.is_recovered:
                    self._check_drawdown_recovery(equity_value, timestamp)

            # Calculate current drawdown
            current_dd_pct = (self.last_peak_value - equity_value) / self.last_peak_value if self.last_peak_value > 0 else 0

            # Determine risk level
            new_risk_level = self._classify_drawdown_level(current_dd_pct)

            # Check for new drawdown event
            if current_dd_pct > 0.01 and (not self.current_drawdown or self.current_drawdown.is_recovered):
                self._start_new_drawdown(equity_value, timestamp)

            # Update current drawdown
            if self.current_drawdown and not self.current_drawdown.is_recovered:
                self._update_current_drawdown(equity_value, timestamp)

            # Check for risk level changes
            risk_change_info = None
            if new_risk_level != self.current_risk_level:
                risk_change_info = self._handle_risk_level_change(new_risk_level, current_dd_pct)

            self.current_risk_level = new_risk_level

            # Check daily loss limit
            daily_loss_info = self._check_daily_loss_limit(timestamp)

            result = {
                "timestamp": timestamp,
                "equity_value": equity_value,
                "current_drawdown_pct": current_dd_pct,
                "risk_level": new_risk_level.value,
                "active_reductions": list(self.active_reductions),
                "peak_value": self.last_peak_value,
                "underwater_time_hours": (timestamp - self.last_peak_time).total_seconds() / 3600,
                "risk_change": risk_change_info,
                "daily_loss_breach": daily_loss_info
            }

            return result

        except Exception as e:
            logger.error(f"Equity update failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_risk_adjustment(self) -> Dict[str, Any]:
        """Get current risk adjustment parameters"""
        try:
            adjustment = {
                "risk_level": self.current_risk_level.value,
                "active_reductions": list(self.active_reductions),
                "kelly_multiplier": self._calculate_kelly_multiplier(),
                "volume_multiplier": self._calculate_volume_multiplier(),
                "trading_allowed": self._is_trading_allowed(),
                "paper_trading_only": RiskReduction.PAPER_ONLY in self.active_reductions,
                "position_size_limit": self._get_position_size_limit(),
                "new_position_allowed": self._are_new_positions_allowed()
            }

            return adjustment

        except Exception as e:
            logger.error(f"Risk adjustment calculation failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_drawdown_statistics(self, days_back: int = 30) -> DrawdownStats:
        """Calculate comprehensive drawdown statistics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter recent equity history
            recent_equity = [
                (ts, val) for ts, val in self.equity_history
                if ts >= cutoff_time
            ]

            if len(recent_equity) < 2:
                return self._get_empty_stats()

            # Calculate drawdown series
            drawdown_series = self._calculate_drawdown_series(recent_equity)

            # Current metrics
            current_dd = drawdown_series[-1] if drawdown_series else 0.0
            max_dd = max(drawdown_series) if drawdown_series else 0.0
            avg_dd = np.mean([dd for dd in drawdown_series if dd > 0]) if drawdown_series else 0.0

            # Recent drawdown events
            recent_events = [
                event for event in self.drawdown_events
                if event.start_time >= cutoff_time
            ]

            # Recovery metrics
            recovery_times = [
                event.recovery_time for event in recent_events
                if event.recovery_time is not None
            ]

            avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0

            # Duration metrics
            durations = [event.duration_hours for event in recent_events]
            longest_dd = max(durations) if durations else 0.0

            # Current underwater time
            current_underwater = (datetime.now() - self.last_peak_time).total_seconds() / 3600

            # Risk-adjusted metrics
            if len(recent_equity) > 1:
                returns = [(recent_equity[i][1] / recent_equity[i-1][1] - 1) for i in range(1, len(recent_equity))]
                annual_return = np.mean(returns) * 365 * 24  # Assuming hourly data
                calmar_ratio = annual_return / max_dd if max_dd > 0 else 0
                sterling_ratio = annual_return / avg_dd if avg_dd > 0 else 0
                ulcer_index = np.sqrt(np.mean([dd**2 for dd in drawdown_series]))
            else:
                calmar_ratio = sterling_ratio = ulcer_index = 0.0

            # Recovery efficiency
            recovery_efficiency = len(recovery_times) / len(recent_events) if recent_events else 0.0

            stats = DrawdownStats(
                current_drawdown_pct=current_dd,
                max_drawdown_pct=max_dd,
                average_drawdown_pct=avg_dd,
                drawdown_frequency=len(recent_events) / (days_back / 30),  # Per month
                average_recovery_time_hours=avg_recovery_time,
                longest_drawdown_hours=longest_dd,
                current_underwater_time_hours=current_underwater,
                calmar_ratio=calmar_ratio,
                sterling_ratio=sterling_ratio,
                ulcer_index=ulcer_index,
                drawdowns_last_30d=len(recent_events),
                worst_drawdown_30d_pct=max_dd,
                recovery_efficiency=recovery_efficiency
            )

            return stats

        except Exception as e:
            logger.error(f"Drawdown statistics calculation failed: {e}")
            return self._get_empty_stats()

    def force_risk_reduction(self, reduction: RiskReduction, reason: str) -> bool:
        """Manually force a risk reduction"""
        try:
            self.active_reductions.add(reduction)

            logger.warning(f"Forced risk reduction: {reduction.value} - {reason}")

            # Log the event
            if self.current_drawdown:
                self.current_drawdown.triggered_reductions.append(reduction)

            return True

        except Exception as e:
            logger.error(f"Failed to force risk reduction: {e}")
            return False

    def clear_risk_reductions(self, reason: str = "Manual override") -> bool:
        """Clear all active risk reductions"""
        try:
            cleared_reductions = list(self.active_reductions)
            self.active_reductions.clear()

            logger.info(f"Cleared risk reductions: {cleared_reductions} - {reason}")

            return True

        except Exception as e:
            logger.error(f"Failed to clear risk reductions: {e}")
            return False

    def _classify_drawdown_level(self, drawdown_pct: float) -> DrawdownLevel:
        """Classify drawdown severity level"""
        for level in [DrawdownLevel.CRITICAL, DrawdownLevel.DANGER,
                     DrawdownLevel.WARNING, DrawdownLevel.CAUTION]:
            if drawdown_pct >= self.rolling_dd_thresholds[level]:
                return level

        return DrawdownLevel.NORMAL

    def _start_new_drawdown(self, equity_value: float, timestamp: datetime) -> None:
        """Start tracking a new drawdown event"""
        try:
            event_id = f"dd_{timestamp.timestamp()}"

            self.current_drawdown = DrawdownEvent(
                event_id=event_id,
                start_time=timestamp,
                end_time=None,
                peak_value=self.last_peak_value,
                trough_value=equity_value,
                max_drawdown_pct=(self.last_peak_value - equity_value) / self.last_peak_value,
                current_drawdown_pct=(self.last_peak_value - equity_value) / self.last_peak_value,
                duration_hours=0.0,
                underwater_time=(timestamp - self.last_peak_time).total_seconds() / 3600,
                level=self._classify_drawdown_level((self.last_peak_value - equity_value) / self.last_peak_value)
            )

            logger.info(f"Started new drawdown event: {event_id}")

        except Exception as e:
            logger.error(f"Failed to start new drawdown: {e}")

    def _update_current_drawdown(self, equity_value: float, timestamp: datetime) -> None:
        """Update current drawdown event"""
        try:
            if not self.current_drawdown:
                return

            # Update trough if this is lower
            if equity_value < self.current_drawdown.trough_value:
                self.current_drawdown.trough_value = equity_value

            # Update drawdown metrics
            current_dd = (self.last_peak_value - equity_value) / self.last_peak_value
            self.current_drawdown.current_drawdown_pct = current_dd

            if current_dd > self.current_drawdown.max_drawdown_pct:
                self.current_drawdown.max_drawdown_pct = current_dd

            # Update timing
            self.current_drawdown.duration_hours = (timestamp - self.current_drawdown.start_time).total_seconds() / 3600
            self.current_drawdown.underwater_time = (timestamp - self.last_peak_time).total_seconds() / 3600

            # Update level
            self.current_drawdown.level = self._classify_drawdown_level(current_dd)

        except Exception as e:
            logger.error(f"Failed to update current drawdown: {e}")

    def _check_drawdown_recovery(self, equity_value: float, timestamp: datetime) -> None:
        """Check if current drawdown has recovered"""
        try:
            if not self.current_drawdown or self.current_drawdown.is_recovered:
                return

            # Mark as recovered
            self.current_drawdown.is_recovered = True
            self.current_drawdown.end_time = timestamp
            self.current_drawdown.recovery_high = equity_value
            self.current_drawdown.recovery_time = (
                timestamp - self.current_drawdown.start_time
            ).total_seconds() / 3600

            # Add to completed events
            self.drawdown_events.append(self.current_drawdown)

            # Clear risk reductions on recovery
            self.active_reductions.clear()

            logger.info(f"Drawdown recovered: {self.current_drawdown.event_id}")

        except Exception as e:
            logger.error(f"Failed to check drawdown recovery: {e}")

    def _handle_risk_level_change(self, new_level: DrawdownLevel, current_dd: float) -> Dict[str, Any]:
        """Handle risk level changes and trigger appropriate reductions"""
        try:
            change_info = {
                "from_level": self.current_risk_level.value,
                "to_level": new_level.value,
                "drawdown_pct": current_dd,
                "new_reductions": [],
                "cleared_reductions": []
            }

            # Determine required reduction for new level
            if new_level in self.risk_reduction_map:
                required_reduction = self.risk_reduction_map[new_level]

                # Add new reduction if not already active
                if required_reduction not in self.active_reductions:
                    self.active_reductions.add(required_reduction)
                    change_info["new_reductions"].append(required_reduction.value)

                    # Clear lesser reductions
                    reductions_to_clear = []
                    for active_reduction in list(self.active_reductions):
                        if self._is_lesser_reduction(active_reduction, required_reduction):
                            reductions_to_clear.append(active_reduction)

                    for reduction in reductions_to_clear:
                        self.active_reductions.remove(reduction)
                        change_info["cleared_reductions"].append(reduction.value)

            # If improving, may clear some reductions
            elif new_level == DrawdownLevel.NORMAL:
                cleared = list(self.active_reductions)
                self.active_reductions.clear()
                change_info["cleared_reductions"] = [r.value for r in cleared]

            # Log risk reductions to current drawdown
            if self.current_drawdown:
                for reduction_str in change_info["new_reductions"]:
                    reduction = RiskReduction(reduction_str)
                    if reduction not in self.current_drawdown.triggered_reductions:
                        self.current_drawdown.triggered_reductions.append(reduction)

            return change_info

        except Exception as e:
            logger.error(f"Risk level change handling failed: {e}")
            return {"error": str(e)}

    def _check_daily_loss_limit(self, current_time: datetime) -> Optional[Dict[str, Any]]:
        """Check if daily loss limit has been breached"""
        try:
            # Get equity from 24 hours ago
            day_ago = current_time - timedelta(hours=24)

            # Find closest equity point to 24 hours ago
            day_ago_equity = None
            min_time_diff = float('inf')

            for ts, equity in self.equity_history:
                time_diff = abs((ts - day_ago).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    day_ago_equity = equity

            if day_ago_equity is None or len(self.equity_history) == 0:
                return None

            # Calculate 24h loss
            current_equity = self.equity_history[-1][1]
            daily_loss_pct = (day_ago_equity - current_equity) / day_ago_equity

            if daily_loss_pct > self.daily_loss_limit_pct:
                # Force full stop on daily limit breach
                self.force_risk_reduction(RiskReduction.FULL_STOP, f"Daily loss limit breached: {daily_loss_pct:.1%}")

                return {
                    "breach_detected": True,
                    "daily_loss_pct": daily_loss_pct,
                    "limit_pct": self.daily_loss_limit_pct,
                    "action_taken": "FULL_STOP"
                }

            return {
                "breach_detected": False,
                "daily_loss_pct": daily_loss_pct,
                "limit_pct": self.daily_loss_limit_pct
            }

        except Exception as e:
            logger.error(f"Daily loss limit check failed: {e}")
            return None

    def _calculate_kelly_multiplier(self) -> float:
        """Calculate Kelly fraction multiplier based on active reductions"""
        if RiskReduction.FULL_STOP in self.active_reductions:
            return 0.0
        elif RiskReduction.PAPER_ONLY in self.active_reductions:
            return 0.0
        elif RiskReduction.BOTH_HALF in self.active_reductions or RiskReduction.KELLY_HALF in self.active_reductions:
            return 0.5
        else:
            return 1.0

    def _calculate_volume_multiplier(self) -> float:
        """Calculate volume target multiplier based on active reductions"""
        if RiskReduction.FULL_STOP in self.active_reductions:
            return 0.0
        elif RiskReduction.PAPER_ONLY in self.active_reductions:
            return 0.0
        elif RiskReduction.BOTH_HALF in self.active_reductions or RiskReduction.VOLUME_HALF in self.active_reductions:
            return 0.5
        else:
            return 1.0

    def _is_trading_allowed(self) -> bool:
        """Check if trading is allowed"""
        return RiskReduction.FULL_STOP not in self.active_reductions

    def _are_new_positions_allowed(self) -> bool:
        """Check if new positions are allowed"""
        return (RiskReduction.FULL_STOP not in self.active_reductions and
                RiskReduction.PAPER_ONLY not in self.active_reductions)

    def _get_position_size_limit(self) -> float:
        """Get position size limit multiplier"""
        if RiskReduction.CRITICAL in [r for r in RiskReduction if r in self.active_reductions]:
            return 0.1  # 10% of normal
        elif RiskReduction.DANGER in [r for r in RiskReduction if r in self.active_reductions]:
            return 0.25  # 25% of normal
        elif any(r in self.active_reductions for r in [RiskReduction.BOTH_HALF, RiskReduction.KELLY_HALF]):
            return 0.5  # 50% of normal
        else:
            return 1.0  # Normal size

    def _is_lesser_reduction(self, reduction1: RiskReduction, reduction2: RiskReduction) -> bool:
        """Check if reduction1 is less severe than reduction2"""
        severity_order = [
            RiskReduction.NONE,
            RiskReduction.KELLY_HALF,
            RiskReduction.VOLUME_HALF,
            RiskReduction.BOTH_HALF,
            RiskReduction.PAPER_ONLY,
            RiskReduction.FULL_STOP
        ]

        try:
            return severity_order.index(reduction1) < severity_order.index(reduction2)
        except ValueError:
            return False

    def _calculate_drawdown_series(self, equity_points: List[Tuple[datetime, float]]) -> List[float]:
        """Calculate drawdown series from equity points"""
        try:
            if len(equity_points) < 2:
                return []

            equity_values = [val for _, val in equity_points]
            drawdowns = []
            running_max = equity_values[0]

            for equity in equity_values:
                if equity > running_max:
                    running_max = equity

                drawdown = (running_max - equity) / running_max if running_max > 0 else 0
                drawdowns.append(drawdown)

            return drawdowns

        except Exception as e:
            logger.error(f"Drawdown series calculation failed: {e}")
            return []

    def _get_empty_stats(self) -> DrawdownStats:
        """Get empty drawdown statistics"""
        return DrawdownStats(
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            average_drawdown_pct=0.0,
            drawdown_frequency=0.0,
            average_recovery_time_hours=0.0,
            longest_drawdown_hours=0.0,
            current_underwater_time_hours=0.0,
            calmar_ratio=0.0,
            sterling_ratio=0.0,
            ulcer_index=0.0,
            drawdowns_last_30d=0,
            worst_drawdown_30d_pct=0.0,
            recovery_efficiency=0.0
        )
