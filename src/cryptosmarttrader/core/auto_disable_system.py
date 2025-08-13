#!/usr/bin/env python3
"""
Auto-Disable System
Automatically disables live trading when health < 60, enables paper trading only
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ..core.structured_logger import get_logger


class TradingMode(Enum):
    """Trading mode enumeration"""

    LIVE = "live"
    PAPER = "paper"
    DISABLED = "disabled"


class DisableReason(Enum):
    """Reasons for disabling live trading"""

    LOW_HEALTH_SCORE = "low_health_score"
    SYSTEM_ERROR = "system_error"
    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MANUAL_DISABLE = "manual_disable"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    API_FAILURE = "api_failure"
    RISK_THRESHOLD_EXCEEDED = "risk_threshold_exceeded"


@dataclass
class TradingStatusChange:
    """Trading status change record"""

    timestamp: datetime
    previous_mode: TradingMode
    new_mode: TradingMode
    reason: DisableReason
    trigger_value: Optional[float]
    threshold: Optional[float]
    description: str
    auto_generated: bool


@dataclass
class HealthThreshold:
    """Health score threshold configuration"""

    disable_threshold: float = 60.0  # Below this: disable live trading
    enable_threshold: float = 85.0  # Above this: can enable live trading
    critical_threshold: float = 30.0  # Below this: disable all trading


class AutoDisableSystem:
    """Automatic trading disable system based on health scores"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("AutoDisableSystem")

        # Default configuration
        self.config = {
            "health_thresholds": HealthThreshold(),
            "check_interval_seconds": 60,  # Check every minute
            "grace_period_minutes": 5,  # Wait before auto-enable
            "max_consecutive_failures": 3,
            "notification_enabled": True,
            "auto_enable": True,  # Automatically re-enable when health improves
            "require_manual_enable": False,  # Require manual intervention for critical failures
        }

        if config:
            if "health_thresholds" in config:
                # Update thresholds
                threshold_config = config.pop("health_thresholds")
                for key, value in threshold_config.items():
                    setattr(self.config["health_thresholds"], key, value)
            self.config.update(config)

        # Current state
        self.current_mode = TradingMode.PAPER  # Start in paper mode for safety
        self.last_health_score: Optional[float] = None
        self.consecutive_failures = 0
        self.last_check_time: Optional[datetime] = None
        self.grace_period_start: Optional[datetime] = None

        # Status history
        self.status_history: List[TradingStatusChange] = []

        # Callbacks for external notification
        self.status_change_callbacks: List[Callable] = []

        # Thread safety
        self.lock = threading.Lock()

        # Initialize with current status
        self._record_status_change(
            previous_mode=TradingMode.DISABLED,
            new_mode=TradingMode.PAPER,
            reason=DisableReason.MANUAL_DISABLE,
            description="System initialized in paper trading mode",
            auto_generated=True,
        )

    def add_status_change_callback(self, callback: Callable[[TradingStatusChange], None]) -> None:
        """Add callback for status changes"""
        self.status_change_callbacks.append(callback)
        self.logger.info("Added status change callback")

    def check_health_and_update_status(
        self, health_score: float, additional_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check health score and update trading status accordingly"""

        with self.lock:
            self.last_health_score = health_score
            self.last_check_time = datetime.now()

            thresholds = self.config["health_thresholds"]
            status_changed = False

            try:
                # Determine required action based on health score
                if health_score < thresholds.critical_threshold:
                    # Critical: disable all trading
                    if self.current_mode != TradingMode.DISABLED:
                        self._change_trading_mode(
                            new_mode=TradingMode.DISABLED,
                            reason=DisableReason.LOW_HEALTH_SCORE,
                            trigger_value=health_score,
                            threshold=thresholds.critical_threshold,
                            description=f"Critical health score {health_score:.1f} < {thresholds.critical_threshold}. "
                            f"All trading disabled.",
                            auto_generated=True,
                        )
                        status_changed = True
                        self.consecutive_failures += 1

                elif health_score < thresholds.disable_threshold:
                    # Low health: disable live trading, enable paper trading
                    if self.current_mode == TradingMode.LIVE:
                        self._change_trading_mode(
                            new_mode=TradingMode.PAPER,
                            reason=DisableReason.LOW_HEALTH_SCORE,
                            trigger_value=health_score,
                            threshold=thresholds.disable_threshold,
                            description=f"Health score {health_score:.1f} < {thresholds.disable_threshold}. "
                            f"Switched to paper trading only.",
                            auto_generated=True,
                        )
                        status_changed = True
                        self.consecutive_failures += 1
                    elif self.current_mode == TradingMode.DISABLED:
                        # Allow paper trading if coming from disabled state
                        self._change_trading_mode(
                            new_mode=TradingMode.PAPER,
                            reason=DisableReason.LOW_HEALTH_SCORE,
                            trigger_value=health_score,
                            threshold=thresholds.critical_threshold,
                            description=f"Health score {health_score:.1f} improved above critical. "
                            f"Enabled paper trading.",
                            auto_generated=True,
                        )
                        status_changed = True

                elif health_score >= thresholds.enable_threshold:
                    # Good health: can enable live trading (if auto-enable is on)
                    if (
                        self.config["auto_enable"]
                        and self.current_mode in [TradingMode.PAPER, TradingMode.DISABLED]
                        and not self.config["require_manual_enable"]
                    ):
                        # Check grace period
                        if self._check_grace_period():
                            self._change_trading_mode(
                                new_mode=TradingMode.LIVE,
                                reason=DisableReason.LOW_HEALTH_SCORE,  # Resolution of previous issue
                                trigger_value=health_score,
                                threshold=thresholds.enable_threshold,
                                description=f"Health score {health_score:.1f} >= {thresholds.enable_threshold}. "
                                f"Auto-enabled live trading.",
                                auto_generated=True,
                            )
                            status_changed = True
                            self.consecutive_failures = 0
                        else:
                            # Start grace period if not already started
                            if self.grace_period_start is None:
                                self.grace_period_start = datetime.now()
                                self.logger.info(
                                    f"Health score improved to {health_score:.1f}. "
                                    f"Starting grace period before auto-enable."
                                )

                # Log health check
                self.logger.debug(
                    f"Health check: score={health_score:.1f}, mode={self.current_mode.value}, "
                    f"consecutive_failures={self.consecutive_failures}"
                )

                # Add to context if additional info provided
                if additional_context:
                    self.logger.debug(f"Additional context: {additional_context}")

                return status_changed

            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                return False

    def manual_disable(self, reason: str, disable_all: bool = False) -> bool:
        """Manually disable trading"""

        with self.lock:
            new_mode = TradingMode.DISABLED if disable_all else TradingMode.PAPER

            if self.current_mode != new_mode:
                self._change_trading_mode(
                    new_mode=new_mode,
                    reason=DisableReason.MANUAL_DISABLE,
                    description=f"Manual disable: {reason}",
                    auto_generated=False,
                )

                self.logger.info(f"Trading manually disabled: {reason}")
                return True

            return False

    def manual_enable(self, force: bool = False) -> bool:
        """Manually enable live trading"""

        with self.lock:
            # Check if manual enable is allowed
            if not force:
                # Verify health score is acceptable
                if (
                    self.last_health_score is not None
                    and self.last_health_score < self.config["health_thresholds"].disable_threshold
                ):
                    self.logger.warning(
                        f"Manual enable rejected: health score {self.last_health_score:.1f} "
                        f"below threshold {self.config['health_thresholds'].disable_threshold}"
                    )
                    return False

            if self.current_mode != TradingMode.LIVE:
                self._change_trading_mode(
                    new_mode=TradingMode.LIVE,
                    reason=DisableReason.MANUAL_DISABLE,  # Resolution of manual disable
                    description="Manual enable of live trading" + (" (forced)" if force else ""),
                    auto_generated=False,
                )

                self.consecutive_failures = 0
                self.grace_period_start = None

                self.logger.info("Trading manually enabled" + (" (forced)" if force else ""))
                return True

            return False

    def disable_for_reason(
        self,
        reason: DisableReason,
        description: str,
        disable_all: bool = False,
        trigger_value: Optional[float] = None,
    ) -> bool:
        """Disable trading for specific reason"""

        with self.lock:
            new_mode = TradingMode.DISABLED if disable_all else TradingMode.PAPER

            if self.current_mode != new_mode:
                self._change_trading_mode(
                    new_mode=new_mode,
                    reason=reason,
                    trigger_value=trigger_value,
                    description=description,
                    auto_generated=True,
                )

                self.consecutive_failures += 1
                self.logger.warning(f"Trading disabled for {reason.value}: {description}")
                return True

            return False

    def _change_trading_mode(
        self,
        new_mode: TradingMode,
        reason: DisableReason,
        trigger_value: Optional[float] = None,
        threshold: Optional[float] = None,
        description: str = "",
        auto_generated: bool = True,
    ) -> None:
        """Internal method to change trading mode"""

        previous_mode = self.current_mode
        self.current_mode = new_mode

        # Record status change
        change = self._record_status_change(
            previous_mode=previous_mode,
            new_mode=new_mode,
            reason=reason,
            trigger_value=trigger_value,
            threshold=threshold,
            description=description,
            auto_generated=auto_generated,
        )

        # Reset grace period if transitioning to worse state
        if new_mode.value in ["paper", "disabled"]:
            self.grace_period_start = None

        # Notify callbacks
        for callback in self.status_change_callbacks:
            try:
                callback(change)
            except Exception as e:
                self.logger.error(f"Status change callback failed: {e}")

        # Log the change
        if new_mode != TradingMode.LIVE:
            self.logger.warning(
                f"Trading mode changed: {previous_mode.value} → {new_mode.value} "
                f"(reason: {reason.value})"
            )
        else:
            self.logger.info(
                f"Trading mode changed: {previous_mode.value} → {new_mode.value} "
                f"(reason: {reason.value})"
            )

    def _record_status_change(
        self,
        previous_mode: TradingMode,
        new_mode: TradingMode,
        reason: DisableReason,
        trigger_value: Optional[float] = None,
        threshold: Optional[float] = None,
        description: str = "",
        auto_generated: bool = True,
    ) -> TradingStatusChange:
        """Record a status change"""

        change = TradingStatusChange(
            timestamp=datetime.now(),
            previous_mode=previous_mode,
            new_mode=new_mode,
            reason=reason,
            trigger_value=trigger_value,
            threshold=threshold,
            description=description,
            auto_generated=auto_generated,
        )

        self.status_history.append(change)

        # Keep only recent history (last 1000 changes)
        if len(self.status_history) > 1000:
            self.status_history = self.status_history[-1000:]

        return change

    def _check_grace_period(self) -> bool:
        """Check if grace period has elapsed"""

        if self.grace_period_start is None:
            return False

        elapsed = datetime.now() - self.grace_period_start
        grace_period = timedelta(minutes=self.config["grace_period_minutes"])

        return elapsed >= grace_period

    def get_current_status(self) -> Dict[str, Any]:
        """Get current trading status"""

        with self.lock:
            status = {
                "current_mode": self.current_mode.value,
                "last_health_score": self.last_health_score,
                "last_check_time": self.last_check_time.isoformat()
                if self.last_check_time
                else None,
                "consecutive_failures": self.consecutive_failures,
                "grace_period_active": self.grace_period_start is not None,
                "grace_period_remaining": None,
                "thresholds": {
                    "disable_threshold": self.config["health_thresholds"].disable_threshold,
                    "enable_threshold": self.config["health_thresholds"].enable_threshold,
                    "critical_threshold": self.config["health_thresholds"].critical_threshold,
                },
                "auto_enable": self.config["auto_enable"],
                "require_manual_enable": self.config["require_manual_enable"],
            }

            # Calculate grace period remaining
            if self.grace_period_start:
                elapsed = datetime.now() - self.grace_period_start
                grace_period = timedelta(minutes=self.config["grace_period_minutes"])
                remaining = grace_period - elapsed

                if remaining.total_seconds() > 0:
                    status["grace_period_remaining"] = remaining.total_seconds()
                else:
                    status["grace_period_remaining"] = 0

            return status

    def get_recent_changes(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent status changes"""

        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_changes = [
            {
                "timestamp": change.timestamp.isoformat(),
                "previous_mode": change.previous_mode.value,
                "new_mode": change.new_mode.value,
                "reason": change.reason.value,
                "trigger_value": change.trigger_value,
                "threshold": change.threshold,
                "description": change.description,
                "auto_generated": change.auto_generated,
            }
            for change in self.status_history
            if change.timestamp > cutoff_time
        ]

        return recent_changes

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary"""

        recent_changes = self.get_recent_changes(24)

        # Count changes by reason
        reason_counts = {}
        for change in recent_changes:
            reason = change["reason"]
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Calculate uptime percentage (time in live mode)
        if recent_changes:
            live_time = 0
            total_time = 0

            for i in range(len(recent_changes)):
                change = recent_changes[i]

                # Calculate duration of this mode
                if i < len(recent_changes) - 1:
                    next_change = recent_changes[i + 1]
                    duration = (
                        datetime.fromisoformat(next_change["timestamp"])
                        - datetime.fromisoformat(change["timestamp"]).total_seconds()
                else:
                    # Current mode duration
                    duration = (
                        datetime.now() - datetime.fromisoformat(change["timestamp"]).total_seconds()

                total_time += duration
                if change["new_mode"] == "live":
                    live_time += duration

            uptime_percentage = (live_time / max(total_time, 1)) * 100
        else:
            uptime_percentage = 0

        summary = {
            "current_status": self.get_current_status(),
            "changes_24h": len(recent_changes),
            "uptime_percentage_24h": uptime_percentage,
            "reason_breakdown": reason_counts,
            "total_status_changes": len(self.status_history),
            "system_health": {
                "consecutive_failures": self.consecutive_failures,
                "max_consecutive_failures": self.config["max_consecutive_failures"],
                "grace_period_enabled": self.config["grace_period_minutes"] > 0,
            },
        }

        return summary


if __name__ == "__main__":

    def test_auto_disable_system():
        """Test auto-disable system"""

        print("🔍 TESTING AUTO-DISABLE SYSTEM")
        print("=" * 60)

        # Create auto-disable system
        auto_disable = AutoDisableSystem()

        print("📊 Initial status:")
        status = auto_disable.get_current_status()
        print(f"   Mode: {status['current_mode']}")
        print(
            f"   Thresholds: disable={status['thresholds']['disable_threshold']}, "
            f"enable={status['thresholds']['enable_threshold']}"
        )

        # Test health score scenarios
        print("\n🧪 Testing health score scenarios...")

        # Scenario 1: Good health (should enable live trading)
        print("\n   Scenario 1: Good health score (90)")
        changed = auto_disable.check_health_and_update_status(90.0)
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Scenario 2: Low health (should disable live trading)
        print("\n   Scenario 2: Low health score (50)")
        changed = auto_disable.check_health_and_update_status(50.0)
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Scenario 3: Critical health (should disable all trading)
        print("\n   Scenario 3: Critical health score (20)")
        changed = auto_disable.check_health_and_update_status(20.0)
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Scenario 4: Recovery
        print("\n   Scenario 4: Health recovery (85)")
        changed = auto_disable.check_health_and_update_status(85.0)
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Test manual controls
        print("\n🎮 Testing manual controls...")

        # Manual disable
        print("\n   Manual disable")
        changed = auto_disable.manual_disable("Testing manual disable")
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Manual enable
        print("\n   Manual enable")
        changed = auto_disable.manual_enable()
        print(f"   Status changed: {changed}")
        print(f"   Current mode: {auto_disable.get_current_status()['current_mode']}")

        # Test specific disable reasons
        print("\n🚨 Testing specific disable reasons...")

        auto_disable.disable_for_reason(
            DisableReason.DRIFT_DETECTED,
            "Model drift detected in sentiment analyzer",
            trigger_value=0.8,
        )
        print(f"   After drift detection: {auto_disable.get_current_status()['current_mode']}")

        # Get recent changes
        print("\n📈 Recent status changes:")
        recent_changes = auto_disable.get_recent_changes(1)  # Last hour
        for change in recent_changes[-5:]:  # Show last 5
            print(
                f"   {change['timestamp']}: {change['previous_mode']} → {change['new_mode']} "
                f"({change['reason']})"
            )

        # System summary
        print("\n📊 System summary:")
        summary = auto_disable.get_system_summary()
        print(f"   Current mode: {summary['current_status']['current_mode']}")
        print(f"   Changes (24h): {summary['changes_24h']}")
        print(f"   Uptime (24h): {summary['uptime_percentage_24h']:.1f}%")
        print(f"   Consecutive failures: {summary['system_health']['consecutive_failures']}")

        print("\n📊 Reason breakdown:")
        for reason, count in summary["reason_breakdown"].items():
            print(f"   {reason}: {count}")

        print("\n✅ AUTO-DISABLE SYSTEM TEST COMPLETED")
        return len(recent_changes) > 0

    # Run test
    success = test_auto_disable_system()
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
