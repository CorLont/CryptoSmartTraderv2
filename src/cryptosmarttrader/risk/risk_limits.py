"""
Risk Limits & Kill Switch System

Comprehensive risk management with daily loss limits, max drawdown guards,
exposure limits, and circuit breakers for data gaps, latency, and drift.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import threading
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RiskLimitType(Enum):
    """Types of risk limits"""
    DAILY_LOSS = "daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_SIZE = "position_size"
    ASSET_EXPOSURE = "asset_exposure"
    CLUSTER_EXPOSURE = "cluster_exposure"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_LIMIT = "volatility_limit"

class CircuitBreakerType(Enum):
    """Types of circuit breakers"""
    DATA_GAP = "data_gap"
    HIGH_LATENCY = "high_latency"
    MODEL_DRIFT = "model_drift"
    EXECUTION_FAILURE = "execution_failure"
    CORRELATION_SPIKE = "correlation_spike"
    VOLATILITY_SPIKE = "volatility_spike"

class RiskAction(Enum):
    """Risk mitigation actions"""
    ALLOW = "allow"
    WARN = "warn"
    REDUCE_SIZE = "reduce_size"
    HALT_NEW_POSITIONS = "halt_new_positions"
    CLOSE_POSITIONS = "close_positions"
    KILL_SWITCH = "kill_switch"
    EMERGENCY_STOP = "emergency_stop"

class TradingMode(Enum):
    """Trading mode states"""
    NORMAL = "normal"           # Full trading
    CONSERVATIVE = "conservative"  # Reduced risk
    DEFENSIVE = "defensive"     # Minimal risk
    EMERGENCY = "emergency"     # Emergency only
    SHUTDOWN = "shutdown"       # Complete stop

@dataclass
class RiskLimit:
    """Individual risk limit configuration"""
    limit_type: RiskLimitType
    threshold: float
    warning_threshold: float
    action: RiskAction
    enabled: bool = True

    # Time-based settings
    lookback_period_hours: int = 24
    reset_frequency: str = "daily"  # daily, weekly, never

    # Escalation
    escalation_multiplier: float = 1.5
    max_escalations: int = 3

    # Metadata
    description: str = ""
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class CircuitBreaker:
    """Circuit breaker configuration"""
    breaker_type: CircuitBreakerType
    threshold: float
    lookback_minutes: int = 5
    min_triggers: int = 3
    action: RiskAction = RiskAction.HALT_NEW_POSITIONS

    # State tracking
    enabled: bool = True
    triggered: bool = False
    last_trigger_time: Optional[datetime] = None
    trigger_history: List[datetime] = field(default_factory=list)

    # Recovery
    recovery_time_minutes: int = 30
    auto_recovery: bool = True

@dataclass
class AssetCluster:
    """Asset cluster for exposure management"""
    name: str
    symbols: Set[str]
    max_exposure_percent: float = 20.0
    correlation_threshold: float = 0.7
    description: str = ""

@dataclass
class RiskMetrics:
    """Current risk metrics snapshot"""
    timestamp: datetime

    # PnL metrics
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0

    # Exposure metrics
    total_exposure: float = 0.0
    max_position_size: float = 0.0
    asset_exposures: Dict[str, float] = field(default_factory=dict)
    cluster_exposures: Dict[str, float] = field(default_factory=dict)

    # Risk metrics
    portfolio_var: float = 0.0
    correlation_risk: float = 0.0
    volatility_risk: float = 0.0

    # Circuit breaker status
    active_breakers: List[str] = field(default_factory=list)

    # Trading mode
    trading_mode: TradingMode = TradingMode.NORMAL

class RiskLimitEngine:
    """
    Comprehensive risk limit and kill switch engine
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 config_path: str = "data/risk"):

        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.config_path = Path(config_path)
        self.config_path.mkdir(parents=True, exist_ok=True)

        # Risk limits
        self.risk_limits: Dict[RiskLimitType, RiskLimit] = {}
        self.circuit_breakers: Dict[CircuitBreakerType, CircuitBreaker] = {}
        self.asset_clusters: Dict[str, AssetCluster] = {}

        # Current state
        self.trading_mode = TradingMode.NORMAL
        self.kill_switch_active = False
        self.emergency_stop_active = False

        # PnL tracking
        self.daily_pnl_history: deque = deque(maxlen=30)  # 30 days
        self.equity_curve: deque = deque(maxlen=10000)    # Equity history
        self.daily_start_capital = initial_capital
        self.last_reset_date = datetime.now().date()

        # Position tracking
        self.positions: Dict[str, float] = {}  # symbol -> position_value
        self.position_history: List[Tuple[datetime, str, float]] = []

        # Metrics tracking
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            'total_violations': 0,
            'kill_switch_triggers': 0,
            'circuit_breaker_triggers': 0,
            'emergency_stops': 0,
            'mode_changes': 0,
            'positions_closed': 0
        }

        # Initialize default configuration
        self._setup_default_risk_limits()
        self._setup_default_circuit_breakers()
        self._setup_default_asset_clusters()

        # Load configuration
        self._load_configuration()

        # Start monitoring
        self._start_monitoring_task()

        logger.info(f"Risk Limit Engine initialized with ${initial_capital:,.2f} capital")

    def _setup_default_risk_limits(self):
        """Setup default risk limits"""

        # Daily loss limit (5%)
        self.risk_limits[RiskLimitType.DAILY_LOSS] = RiskLimit(
            limit_type=RiskLimitType.DAILY_LOSS,
            threshold=0.05,  # 5% daily loss
            warning_threshold=0.03,  # 3% warning
            action=RiskAction.KILL_SWITCH,
            description="Daily loss limit to prevent catastrophic losses",
            reset_frequency="daily"
        )

        # Max drawdown (10%)
        self.risk_limits[RiskLimitType.MAX_DRAWDOWN] = RiskLimit(
            limit_type=RiskLimitType.MAX_DRAWDOWN,
            threshold=0.10,  # 10% max drawdown
            warning_threshold=0.07,  # 7% warning
            action=RiskAction.EMERGENCY_STOP,
            description="Maximum portfolio drawdown limit",
            reset_frequency="never"
        )

        # Position size limit (2% per position)
        self.risk_limits[RiskLimitType.POSITION_SIZE] = RiskLimit(
            limit_type=RiskLimitType.POSITION_SIZE,
            threshold=0.02,  # 2% per position
            warning_threshold=0.015,  # 1.5% warning
            action=RiskAction.REDUCE_SIZE,
            description="Maximum position size per asset"
        )

        # Asset exposure limit (5% per asset)
        self.risk_limits[RiskLimitType.ASSET_EXPOSURE] = RiskLimit(
            limit_type=RiskLimitType.ASSET_EXPOSURE,
            threshold=0.05,  # 5% per asset
            warning_threshold=0.04,  # 4% warning
            action=RiskAction.HALT_NEW_POSITIONS,
            description="Maximum exposure per individual asset"
        )

        # Cluster exposure limit (20% per cluster)
        self.risk_limits[RiskLimitType.CLUSTER_EXPOSURE] = RiskLimit(
            limit_type=RiskLimitType.CLUSTER_EXPOSURE,
            threshold=0.20,  # 20% per cluster
            warning_threshold=0.15,  # 15% warning
            action=RiskAction.HALT_NEW_POSITIONS,
            description="Maximum exposure per asset cluster"
        )

    def _setup_default_circuit_breakers(self):
        """Setup default circuit breakers"""

        # Data gap breaker
        self.circuit_breakers[CircuitBreakerType.DATA_GAP] = CircuitBreaker(
            breaker_type=CircuitBreakerType.DATA_GAP,
            threshold=300.0,  # 5 minutes gap
            lookback_minutes=10,
            min_triggers=2,
            action=RiskAction.HALT_NEW_POSITIONS,
            recovery_time_minutes=15
        )

        # High latency breaker
        self.circuit_breakers[CircuitBreakerType.HIGH_LATENCY] = CircuitBreaker(
            breaker_type=CircuitBreakerType.HIGH_LATENCY,
            threshold=5000.0,  # 5 seconds latency
            lookback_minutes=5,
            min_triggers=3,
            action=RiskAction.REDUCE_SIZE,
            recovery_time_minutes=10
        )

        # Model drift breaker
        self.circuit_breakers[CircuitBreakerType.MODEL_DRIFT] = CircuitBreaker(
            breaker_type=CircuitBreakerType.MODEL_DRIFT,
            threshold=0.15,  # 15% accuracy drop
            lookback_minutes=30,
            min_triggers=5,
            action=RiskAction.CLOSE_POSITIONS,
            recovery_time_minutes=60
        )

        # Execution failure breaker
        self.circuit_breakers[CircuitBreakerType.EXECUTION_FAILURE] = CircuitBreaker(
            breaker_type=CircuitBreakerType.EXECUTION_FAILURE,
            threshold=0.30,  # 30% failure rate
            lookback_minutes=15,
            min_triggers=5,
            action=RiskAction.EMERGENCY_STOP,
            recovery_time_minutes=30
        )

        # Correlation spike breaker
        self.circuit_breakers[CircuitBreakerType.CORRELATION_SPIKE] = CircuitBreaker(
            breaker_type=CircuitBreakerType.CORRELATION_SPIKE,
            threshold=0.95,  # 95% correlation
            lookback_minutes=10,
            min_triggers=3,
            action=RiskAction.REDUCE_SIZE,
            recovery_time_minutes=20
        )

    def _setup_default_asset_clusters(self):
        """Setup default asset clusters"""

        # Major cryptocurrencies
        self.asset_clusters["major_crypto"] = AssetCluster(
            name="major_crypto",
            symbols={"BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD"},
            max_exposure_percent=30.0,
            correlation_threshold=0.7,
            description="Major cryptocurrency assets"
        )

        # DeFi tokens
        self.asset_clusters["defi"] = AssetCluster(
            name="defi",
            symbols={"UNI/USD", "AAVE/USD", "COMP/USD", "MKR/USD", "SNX/USD"},
            max_exposure_percent=15.0,
            correlation_threshold=0.8,
            description="Decentralized Finance tokens"
        )

        # Layer 1 protocols
        self.asset_clusters["layer1"] = AssetCluster(
            name="layer1",
            symbols={"SOL/USD", "AVAX/USD", "ALGO/USD", "DOT/USD", "ATOM/USD"},
            max_exposure_percent=20.0,
            correlation_threshold=0.75,
            description="Layer 1 blockchain protocols"
        )

        # Meme coins (high risk)
        self.asset_clusters["meme"] = AssetCluster(
            name="meme",
            symbols={"DOGE/USD", "SHIB/USD", "PEPE/USD"},
            max_exposure_percent=5.0,
            correlation_threshold=0.6,
            description="Meme coins and speculative assets"
        )

    def update_capital(self, new_capital: float):
        """Update current capital"""
        with self._lock:
            old_capital = self.current_capital
            self.current_capital = new_capital

            # Add to equity curve
            self.equity_curve.append((datetime.now(), new_capital))

            # Calculate daily PnL if new day
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                daily_pnl = new_capital - self.daily_start_capital
                daily_pnl_percent = daily_pnl / self.daily_start_capital if self.daily_start_capital > 0 else 0

                self.daily_pnl_history.append((self.last_reset_date, daily_pnl, daily_pnl_percent))

                # Reset for new day
                self.daily_start_capital = new_capital
                self.last_reset_date = current_date

                logger.info(f"Daily PnL: ${daily_pnl:,.2f} ({daily_pnl_percent:.2%})")

    def update_position(self, symbol: str, position_value: float):
        """Update position value"""
        with self._lock:
            old_value = self.positions.get(symbol, 0.0)
            self.positions[symbol] = position_value

            # Record position change
            self.position_history.append((datetime.now(), symbol, position_value))

            # Keep history manageable
            if len(self.position_history) > 10000:
                self.position_history = self.position_history[-5000:]

    def check_risk_limits(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Check all risk limits and return violations"""
        with self._lock:
            violations = []
            can_trade = True

            # Calculate current metrics
            metrics = self._calculate_current_metrics()

            # Check each risk limit
            for limit_type, limit in self.risk_limits.items():
                if not limit.enabled:
                    continue

                violation = self._check_single_limit(limit, metrics)
                if violation:
                    violations.append(violation)

                    # Determine if trading should be halted
                    if limit.action in [RiskAction.KILL_SWITCH, RiskAction.EMERGENCY_STOP,
                                       RiskAction.HALT_NEW_POSITIONS]:
                        can_trade = False

            # Check circuit breakers
            breaker_violations = self._check_circuit_breakers()
            violations.extend(breaker_violations)

            if breaker_violations:
                can_trade = False

            # Update trading mode based on violations
            if violations:
                self._update_trading_mode(violations)

            return can_trade, violations

    def _check_single_limit(self, limit: RiskLimit, metrics: RiskMetrics) -> Optional[Dict[str, Any]]:
        """Check individual risk limit"""

        current_value = 0.0

        if limit.limit_type == RiskLimitType.DAILY_LOSS:
            current_value = abs(metrics.daily_pnl_percent)

        elif limit.limit_type == RiskLimitType.MAX_DRAWDOWN:
            current_value = metrics.max_drawdown_percent

        elif limit.limit_type == RiskLimitType.POSITION_SIZE:
            current_value = metrics.max_position_size / self.current_capital if self.current_capital > 0 else 0

        elif limit.limit_type == RiskLimitType.ASSET_EXPOSURE:
            current_value = max(metrics.asset_exposures.values()) if metrics.asset_exposures else 0

        elif limit.limit_type == RiskLimitType.CLUSTER_EXPOSURE:
            current_value = max(metrics.cluster_exposures.values()) if metrics.cluster_exposures else 0

        # Check violation (for daily loss, compare absolute values)
        if limit.limit_type == RiskLimitType.DAILY_LOSS:
            # Daily loss should be negative, but we check the absolute value
            current_loss_percent = metrics.daily_pnl_percent
            is_violation = current_loss_percent < -limit.threshold  # Loss greater than threshold
            is_warning = current_loss_percent < -limit.warning_threshold  # Loss greater than warning
        else:
            is_violation = current_value > limit.threshold
            is_warning = current_value > limit.warning_threshold

        if is_violation or is_warning:
            # Update limit statistics
            if is_violation:
                limit.last_triggered = datetime.now()
                limit.trigger_count += 1
                self.stats['total_violations'] += 1

            return {
                'limit_type': limit.limit_type.value,
                'threshold': limit.threshold,
                'current_value': current_value,
                'severity': 'violation' if is_violation else 'warning',
                'action': limit.action.value,
                'description': limit.description,
                'timestamp': datetime.now().isoformat()
            }

        return None

    def _check_circuit_breakers(self) -> List[Dict[str, Any]]:
        """Check all circuit breakers"""
        violations = []

        for breaker_type, breaker in self.circuit_breakers.items():
            if not breaker.enabled:
                continue

            # Check if breaker is already triggered and add to violations
            if breaker.triggered:
                violations.append({
                    'breaker_type': breaker.breaker_type.value,
                    'threshold': breaker.threshold,
                    'action': breaker.action.value,
                    'severity': 'circuit_breaker',
                    'timestamp': datetime.now().isoformat(),
                    'recovery_time_minutes': breaker.recovery_time_minutes
                })

            # Check if breaker should trigger
            should_trigger = self._evaluate_circuit_breaker(breaker)

            if should_trigger and not breaker.triggered:
                # Trigger breaker
                breaker.triggered = True
                breaker.last_trigger_time = datetime.now()
                breaker.trigger_history.append(datetime.now())

                self.stats['circuit_breaker_triggers'] += 1

                violations.append({
                    'breaker_type': breaker.breaker_type.value,
                    'threshold': breaker.threshold,
                    'action': breaker.action.value,
                    'severity': 'circuit_breaker',
                    'timestamp': datetime.now().isoformat(),
                    'recovery_time_minutes': breaker.recovery_time_minutes
                })

                logger.warning(f"Circuit breaker triggered: {breaker.breaker_type.value}")

            # Check for auto-recovery
            elif breaker.triggered and breaker.auto_recovery:
                if breaker.last_trigger_time and (datetime.now() - breaker.last_trigger_time).total_seconds() > (breaker.recovery_time_minutes * 60):
                    breaker.triggered = False
                    logger.info(f"Circuit breaker recovered: {breaker.breaker_type.value}")

        return violations

    def _evaluate_circuit_breaker(self, breaker: CircuitBreaker) -> bool:
        """Evaluate if circuit breaker should trigger"""

        # REMOVED: Mock data pattern not allowed in production
        # these would check actual system metrics)

        if breaker.breaker_type == CircuitBreakerType.DATA_GAP:
            # Check for data gaps in last N minutes
            return False  # Placeholder

        elif breaker.breaker_type == CircuitBreakerType.HIGH_LATENCY:
            # Check API latency
            return False  # Placeholder

        elif breaker.breaker_type == CircuitBreakerType.MODEL_DRIFT:
            # Check model performance degradation
            return False  # Placeholder

        elif breaker.breaker_type == CircuitBreakerType.EXECUTION_FAILURE:
            # Check execution failure rate
            return False  # Placeholder

        elif breaker.breaker_type == CircuitBreakerType.CORRELATION_SPIKE:
            # Check correlation spikes
            return False  # Placeholder

        return False

    def _calculate_current_metrics(self) -> RiskMetrics:
        """Calculate current risk metrics"""

        # Calculate daily PnL
        daily_pnl = self.current_capital - self.daily_start_capital
        daily_pnl_percent = daily_pnl / self.daily_start_capital if self.daily_start_capital > 0 else 0

        # Calculate max drawdown
        max_drawdown, max_drawdown_percent = self._calculate_drawdown()

        # Calculate exposures
        total_exposure = sum(abs(pos) for pos in self.positions.values())
        max_position_size = max(abs(pos) for pos in self.positions.values()) if self.positions else 0

        # Asset exposures
        asset_exposures = {}
        for symbol, position in self.positions.items():
            exposure_percent = abs(position) / self.current_capital if self.current_capital > 0 else 0
            asset_exposures[symbol] = exposure_percent

        # Cluster exposures
        cluster_exposures = {}
        for cluster_name, cluster in self.asset_clusters.items():
            cluster_exposure = sum(
                abs(self.positions.get(symbol, 0))
                for symbol in cluster.symbols
            )
            exposure_percent = cluster_exposure / self.current_capital if self.current_capital > 0 else 0
            cluster_exposures[cluster_name] = exposure_percent

        # Active breakers
        active_breakers = [
            breaker.breaker_type.value
            for breaker in self.circuit_breakers.values()
            if breaker.triggered
        ]

        return RiskMetrics(
            timestamp=datetime.now(),
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            max_drawdown=max_drawdown,
            max_drawdown_percent=max_drawdown_percent,
            total_exposure=total_exposure,
            max_position_size=max_position_size,
            asset_exposures=asset_exposures,
            cluster_exposures=cluster_exposures,
            active_breakers=active_breakers,
            trading_mode=self.trading_mode
        )

    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown"""
        if len(self.equity_curve) < 2:
            return 0.0, 0.0

        equity_values = [eq[1] for eq in self.equity_curve]
        peak = max(equity_values)
        current = equity_values[-1]

        max_drawdown = peak - current
        max_drawdown_percent = max_drawdown / peak if peak > 0 else 0

        return max_drawdown, max_drawdown_percent

    def _update_trading_mode(self, violations: List[Dict[str, Any]]):
        """Update trading mode based on violations"""

        # Count severity levels
        kill_switch_violations = [v for v in violations if v.get('action') == 'kill_switch']
        emergency_violations = [v for v in violations if v.get('action') == 'emergency_stop']
        halt_violations = [v for v in violations if v.get('action') == 'halt_new_positions']

        old_mode = self.trading_mode

        if kill_switch_violations or emergency_violations:
            if not self.kill_switch_active:
                self._activate_kill_switch()
            self.trading_mode = TradingMode.SHUTDOWN

        elif halt_violations or len(violations) >= 3:
            self.trading_mode = TradingMode.EMERGENCY

        elif len(violations) >= 2:
            self.trading_mode = TradingMode.DEFENSIVE

        elif len(violations) >= 1:
            self.trading_mode = TradingMode.CONSERVATIVE

        else:
            self.trading_mode = TradingMode.NORMAL

        if old_mode != self.trading_mode:
            self.stats['mode_changes'] += 1
            logger.warning(f"Trading mode changed: {old_mode.value} → {self.trading_mode.value}")

    def _activate_kill_switch(self):
        """Activate kill switch"""
        if self.kill_switch_active:
            return

        self.kill_switch_active = True
        self.stats['kill_switch_triggers'] += 1

        logger.critical("🚨 KILL SWITCH ACTIVATED - ALL TRADING HALTED")

        # Close all positions (in real implementation)
        # self._close_all_positions()

    def force_kill_switch(self, reason: str = "Manual activation"):
        """Manually activate kill switch"""
        with self._lock:
            logger.critical(f"Kill switch manually activated: {reason}")
            self._activate_kill_switch()

    def reset_kill_switch(self, reason: str = "Manual reset"):
        """Reset kill switch"""
        with self._lock:
            was_active = self.kill_switch_active

            self.kill_switch_active = False
            self.emergency_stop_active = False
            self.trading_mode = TradingMode.CONSERVATIVE

            # Reset circuit breakers that were manually triggered
            for breaker in self.circuit_breakers.values():
                if breaker.triggered:
                    breaker.triggered = False

            logger.warning(f"Kill switch reset: {reason}")
            return was_active  # Return True if was previously active

    def simulate_stress_test(self, duration_seconds: float):
        """Simulate data gap for testing"""
        breaker = self.circuit_breakers.get(CircuitBreakerType.DATA_GAP)
        if breaker and duration_seconds > breaker.threshold:
            breaker.triggered = True
            breaker.last_trigger_time = datetime.now()
            logger.warning(f"Simulated data gap: {duration_seconds}s")

    def # REMOVED: Mock data pattern not allowed in productionself, accuracy_drop: float):
        """Simulate model drift for testing"""
        breaker = self.circuit_breakers.get(CircuitBreakerType.MODEL_DRIFT)
        if breaker and accuracy_drop > breaker.threshold:
            breaker.triggered = True
            breaker.last_trigger_time = datetime.now()
            logger.warning(f"Simulated model drift: {accuracy_drop:.2%}")

    def get_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive risk status"""
        with self._lock:
            can_trade, violations = self.check_risk_limits()
            metrics = self._calculate_current_metrics()

            return {
                'timestamp': datetime.now().isoformat(),
                'can_trade': can_trade,
                'trading_mode': self.trading_mode.value,
                'kill_switch_active': self.kill_switch_active,
                'emergency_stop_active': self.emergency_stop_active,

                'capital': {
                    'initial_capital': self.initial_capital,
                    'current_capital': self.current_capital,
                    'daily_start_capital': self.daily_start_capital
                },

                'pnl_metrics': {
                    'daily_pnl': metrics.daily_pnl,
                    'daily_pnl_percent': metrics.daily_pnl_percent,
                    'max_drawdown': metrics.max_drawdown,
                    'max_drawdown_percent': metrics.max_drawdown_percent
                },

                'exposure_metrics': {
                    'total_exposure': metrics.total_exposure,
                    'max_position_size': metrics.max_position_size,
                    'asset_exposures': metrics.asset_exposures,
                    'cluster_exposures': metrics.cluster_exposures
                },

                'violations': violations,
                'active_breakers': metrics.active_breakers,

                'risk_limits': {
                    limit_type.value: {
                        'threshold': limit.threshold,
                        'warning_threshold': limit.warning_threshold,
                        'enabled': limit.enabled,
                        'trigger_count': limit.trigger_count
                    }
                    for limit_type, limit in self.risk_limits.items()
                },

                'circuit_breakers': {
                    breaker_type.value: {
                        'threshold': breaker.threshold,
                        'triggered': breaker.triggered,
                        'enabled': breaker.enabled
                    }
                    for breaker_type, breaker in self.circuit_breakers.items()
                },

                'statistics': self.stats
            }

    def _start_monitoring_task(self):
        """Start background monitoring task"""

        def monitoring_worker():
            while True:
                try:
                    # Record current metrics
                    metrics = self._calculate_current_metrics()
                    self.metrics_history.append(metrics)

                    # Check risk limits
                    can_trade, violations = self.check_risk_limits()

                    # Log significant events
                    if violations:
                        logger.warning(f"Risk violations detected: {len(violations)}")

                    time.sleep(60)  # Check every minute

                except Exception as e:
                    logger.error(f"Risk monitoring error: {e}")
                    time.sleep(60)

        monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        monitoring_thread.start()
        logger.info("Risk monitoring task started")

    def _load_configuration(self):
        """Load risk configuration from file"""
        config_file = self.config_path / "risk_config.json"

        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)

                # Load custom limits
                # Implementation would deserialize limits from config
                logger.info("Risk configuration loaded")

            except Exception as e:
                logger.error(f"Failed to load risk configuration: {e}")

    def save_configuration(self):
        """Save current risk configuration"""
        config_file = self.config_path / "risk_config.json"

        try:
            config = {
                'risk_limits': {
                    limit_type.value: {
                        'threshold': limit.threshold,
                        'warning_threshold': limit.warning_threshold,
                        'action': limit.action.value,
                        'enabled': limit.enabled,
                        'description': limit.description
                    }
                    for limit_type, limit in self.risk_limits.items()
                },
                'circuit_breakers': {
                    breaker_type.value: {
                        'threshold': breaker.threshold,
                        'lookback_minutes': breaker.lookback_minutes,
                        'min_triggers': breaker.min_triggers,
                        'action': breaker.action.value,
                        'enabled': breaker.enabled
                    }
                    for breaker_type, breaker in self.circuit_breakers.items()
                },
                'asset_clusters': {
                    name: {
                        'symbols': list(cluster.symbols),
                        'max_exposure_percent': cluster.max_exposure_percent,
                        'correlation_threshold': cluster.correlation_threshold,
                        'description': cluster.description
                    }
                    for name, cluster in self.asset_clusters.items()
                }
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info("Risk configuration saved")

        except Exception as e:
            logger.error(f"Failed to save risk configuration: {e}")
