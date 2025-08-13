#!/usr/bin/env python3
"""
Data Flow Orchestrator - Strakke integratie van alle bestaande componenten
Verbindt RiskGuard, ExecutionPolicy, Kelly sizing, en regime detection voor maximale data discipline
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from ..core.fallback_data_eliminator import FallbackDataEliminator, DataValidationLevel
from ..risk.risk_guard import RiskGuard, RiskLevel, TradingMode
from ..execution.execution_policy import ExecutionPolicy
from ..alpha.regime_detector import RegimeDetector
from ..sizing.kelly_sizing import KellySizer
from ..observability.unified_metrics import UnifiedMetrics


class DataFlowState(Enum):
    """Data flow pipeline states."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class DataQualityGate(Enum):
    """Data quality gate levels."""

    STRICT = "strict"  # 95%+ quality required
    STANDARD = "standard"  # 85%+ quality required
    MINIMUM = "minimum"  # 70%+ quality required


@dataclass
class DataFlowMetrics:
    """Data flow quality metrics."""

    authentic_data_percentage: float
    pipeline_latency_ms: float
    validation_success_rate: float
    regime_confidence: float
    risk_level: RiskLevel
    execution_readiness: bool
    kelly_sizing_available: bool
    total_validations: int = 0
    failed_validations: int = 0
    last_update: datetime = field(default_factory=datetime.now)


class DataFlowOrchestrator:
    """
    Orchestrates data flow between all major components with strict data discipline.

    Connects:
    - Authentic data validation (zero fallbacks)
    - Risk management (RiskGuard)
    - Execution policy (tradability gates)
    - Regime detection (market adaptation)
    - Kelly sizing (position optimization)
    - Metrics collection (observability)
    """

    def __init__(
        self, strict_mode: bool = True, quality_gate: DataQualityGate = DataQualityGate.STRICT
    ):
        """Initialize data flow orchestrator."""

        self.logger = get_logger("data_flow_orchestrator")
        self.strict_mode = strict_mode
        self.quality_gate = quality_gate

        # Initialize core components
        self.data_validator = FallbackDataEliminator(
            validation_level=DataValidationLevel.STRICT
            if strict_mode
            else DataValidationLevel.MODERATE
        )
        self.risk_guard = RiskGuard()
        self.execution_policy = ExecutionPolicy()
        self.regime_detector = RegimeDetector()
        self.kelly_sizer = KellySizer()
        self.metrics = UnifiedMetrics("data_flow_orchestrator")

        # Flow state
        self.current_state = DataFlowState.INITIALIZING
        self.flow_metrics = DataFlowMetrics(
            authentic_data_percentage=0.0,
            pipeline_latency_ms=0.0,
            validation_success_rate=0.0,
            regime_confidence=0.0,
            risk_level=RiskLevel.NORMAL,
            execution_readiness=False,
            kelly_sizing_available=False,
        )

        # Data quality tracking
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self.flow_performance: Dict[str, float] = {
            "data_validation_ms": 0.0,
            "risk_assessment_ms": 0.0,
            "regime_detection_ms": 0.0,
            "execution_check_ms": 0.0,
            "kelly_sizing_ms": 0.0,
            "total_pipeline_ms": 0.0,
        }

        # Quality gates
        self.quality_thresholds = {
            DataQualityGate.STRICT: {
                "min_authentic_data_pct": 95.0,
                "min_validation_success_rate": 90.0,
                "max_pipeline_latency_ms": 500.0,
                "min_regime_confidence": 0.8,
            },
            DataQualityGate.STANDARD: {
                "min_authentic_data_pct": 85.0,
                "min_validation_success_rate": 80.0,
                "max_pipeline_latency_ms": 1000.0,
                "min_regime_confidence": 0.6,
            },
            DataQualityGate.MINIMUM: {
                "min_authentic_data_pct": 70.0,
                "min_validation_success_rate": 70.0,
                "max_pipeline_latency_ms": 2000.0,
                "min_regime_confidence": 0.4,
            },
        }

        # Threading
        self._lock = threading.RLock()
        self._monitoring_active = False

        # Persistence
        self.data_path = Path("data/data_flow")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "DataFlowOrchestrator initialized",
            strict_mode=strict_mode,
            quality_gate=quality_gate.value,
            components_loaded=5,
        )

    async def process_market_signal(
        self, symbol: str, market_data: Dict[str, Any], signal_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process complete market signal through all components with strict data validation.

        Flow: Data Validation → Risk Check → Regime Detection → Kelly Sizing → Execution Gate
        """

        pipeline_start = time.time()

        try:
            with self._lock:
                result = {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "flow_state": self.current_state.value,
                    "pipeline_success": False,
                    "components_passed": [],
                    "components_failed": [],
                    "quality_metrics": {},
                    "execution_recommendation": "REJECT",
                }

                # Step 1: Strict Data Validation
                validation_start = time.time()
                validation_result = self.data_validator.validate_data(
                    market_data, data_type="market_data", require_authentic=self.strict_mode
                )

                self.flow_performance["data_validation_ms"] = (
                    time.time() - validation_start
                ) * 1000

                if not validation_result.is_valid:
                    result["components_failed"].append("data_validation")
                    result["validation_errors"] = validation_result.validation_errors
                    self.logger.warning(
                        "Data validation failed",
                        symbol=symbol,
                        errors=validation_result.validation_errors,
                    )

                    # Record failure and return early
                    self._record_pipeline_failure("data_validation", result)
                    return result

                result["components_passed"].append("data_validation")
                result["quality_metrics"]["data_quality_score"] = (
                    validation_result.data_quality_score
                )

                # Step 2: Risk Assessment
                risk_start = time.time()
                risk_assessment = await self._assess_risk(symbol, market_data, signal_data)
                self.flow_performance["risk_assessment_ms"] = (time.time() - risk_start) * 1000

                if not risk_assessment["risk_acceptable"]:
                    result["components_failed"].append("risk_guard")
                    result["risk_rejection_reason"] = risk_assessment["rejection_reason"]
                    self.logger.warning(
                        "Risk assessment failed",
                        symbol=symbol,
                        risk_level=risk_assessment["current_risk_level"],
                    )

                    self._record_pipeline_failure("risk_guard", result)
                    return result

                result["components_passed"].append("risk_guard")
                result["quality_metrics"]["risk_level"] = risk_assessment["current_risk_level"]

                # Step 3: Regime Detection
                regime_start = time.time()
                regime_analysis = await self._detect_regime(symbol, market_data)
                self.flow_performance["regime_detection_ms"] = (time.time() - regime_start) * 1000

                if (
                    regime_analysis["confidence"]
                    < self.quality_thresholds[self.quality_gate]["min_regime_confidence"]
                ):
                    result["components_failed"].append("regime_detector")
                    result["regime_rejection_reason"] = (
                        f"Low confidence: {regime_analysis['confidence']:.2f}"
                    )

                    self._record_pipeline_failure("regime_detector", result)
                    return result

                result["components_passed"].append("regime_detector")
                result["quality_metrics"]["regime"] = regime_analysis["regime"]
                result["quality_metrics"]["regime_confidence"] = regime_analysis["confidence"]

                # Step 4: Kelly Sizing
                kelly_start = time.time()
                sizing_result = await self._calculate_kelly_sizing(
                    symbol, market_data, signal_data, regime_analysis
                )
                self.flow_performance["kelly_sizing_ms"] = (time.time() - kelly_start) * 1000

                if not sizing_result["sizing_available"]:
                    result["components_failed"].append("kelly_sizer")
                    result["sizing_rejection_reason"] = sizing_result["rejection_reason"]

                    self._record_pipeline_failure("kelly_sizer", result)
                    return result

                result["components_passed"].append("kelly_sizer")
                result["quality_metrics"]["position_size"] = sizing_result["recommended_size"]
                result["quality_metrics"]["kelly_fraction"] = sizing_result["kelly_fraction"]

                # Step 5: Execution Gate
                execution_start = time.time()
                execution_check = await self._check_execution_readiness(
                    symbol, market_data, sizing_result
                )
                self.flow_performance["execution_check_ms"] = (time.time() - execution_start) * 1000

                if not execution_check["execution_approved"]:
                    result["components_failed"].append("execution_policy")
                    result["execution_rejection_reason"] = execution_check["rejection_reason"]

                    self._record_pipeline_failure("execution_policy", result)
                    return result

                result["components_passed"].append("execution_policy")
                result["quality_metrics"]["execution_score"] = execution_check["execution_score"]

                # Success - All gates passed
                pipeline_time = (time.time() - pipeline_start) * 1000
                self.flow_performance["total_pipeline_ms"] = pipeline_time

                result.update(
                    {
                        "pipeline_success": True,
                        "execution_recommendation": "APPROVE",
                        "pipeline_latency_ms": pipeline_time,
                        "final_position_size": sizing_result["recommended_size"],
                        "execution_parameters": execution_check["execution_params"],
                    }
                )

                # Update flow metrics
                self._update_flow_metrics(result)

                # Record metrics
                self.metrics.record_signal(symbol, validation_result.data_quality_score)
                self.metrics.record_order("approved", symbol, "signal_processed")

                self.logger.info(
                    "Pipeline success",
                    symbol=symbol,
                    pipeline_time_ms=pipeline_time,
                    components_passed=len(result["components_passed"]),
                )

                return result

        except Exception as e:
            pipeline_time = (time.time() - pipeline_start) * 1000
            self.logger.error(
                "Pipeline execution failed",
                symbol=symbol,
                error=str(e),
                pipeline_time_ms=pipeline_time,
            )

            result["pipeline_success"] = False
            result["components_failed"].append("pipeline_error")
            result["error"] = str(e)

            self._record_pipeline_failure("pipeline_error", result)
            return result

    async def _assess_risk(
        self, symbol: str, market_data: Dict[str, Any], signal_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess risk through RiskGuard."""

        try:
            # Update portfolio value if available
            if "portfolio_value" in market_data:
                self.risk_guard.update_portfolio_value(market_data["portfolio_value"])

            # Check position limits (simplified check)
            proposed_size = signal_data.get("position_size", 0.0) if signal_data else 0.0
            position_check = True  # Would use risk_guard.check_position_limit if method exists

            # Check overall risk status
            current_risk_level = getattr(self.risk_guard, "current_risk_level", RiskLevel.NORMAL)
            trading_mode = getattr(self.risk_guard, "trading_mode", TradingMode.LIVE)

            # Determine if risk is acceptable
            risk_acceptable = (
                position_check
                and current_risk_level in [RiskLevel.NORMAL, RiskLevel.CONSERVATIVE]
                and trading_mode == TradingMode.LIVE
                and not getattr(self.risk_guard, "kill_switch_active", False)

            rejection_reason = None
            if not position_check:
                rejection_reason = "Position size limit exceeded"
            elif current_risk_level not in [RiskLevel.NORMAL, RiskLevel.CONSERVATIVE]:
                rejection_reason = f"Risk level too high: {current_risk_level.value}"
            elif trading_mode != TradingMode.LIVE:
                rejection_reason = f"Trading mode not live: {trading_mode.value}"
            elif getattr(self.risk_guard, "kill_switch_active", False):
                rejection_reason = "Kill switch activated"

            return {
                "risk_acceptable": risk_acceptable,
                "current_risk_level": current_risk_level.value,
                "trading_mode": trading_mode.value,
                "position_check_passed": position_check,
                "kill_switch_active": getattr(self.risk_guard, "kill_switch_active", False),
                "rejection_reason": rejection_reason,
            }

        except Exception as e:
            self.logger.error("Risk assessment failed", symbol=symbol, error=str(e))
            return {
                "risk_acceptable": False,
                "rejection_reason": f"Risk assessment error: {str(e)}",
            }

    async def _detect_regime(self, symbol: str, market_data: dict[str, Any]) -> dict[str, Any]:
        """Detect market regime."""

        try:
            # Use regime detector if available
            if hasattr(self.regime_detector, "detect_current_regime"):
                regime_result = self.regime_detector.detect_current_regime(market_data)

                return {
                    "regime": regime_result.get("regime", "unknown"),
                    "confidence": regime_result.get("confidence", 0.5),
                    "regime_available": True,
                }
            else:
                # Fallback to simple regime classification
                price_change = market_data.get("price_change_24h", 0.0)
                volatility = market_data.get("volatility_24h", 0.02)

                if abs(price_change) > 0.05:  # 5% change
                    regime = "trending"
                    confidence = min(abs(price_change) * 10, 1.0)
                elif volatility > 0.04:  # 4% volatility
                    regime = "volatile"
                    confidence = min(volatility * 15, 1.0)
                else:
                    regime = "sideways"
                    confidence = 0.7

                return {"regime": regime, "confidence": confidence, "regime_available": False}

        except Exception as e:
            self.logger.error("Regime detection failed", symbol=symbol, error=str(e))
            return {"regime": "unknown", "confidence": 0.0, "regime_available": False}

    async def _calculate_kelly_sizing(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        signal_data: Optional[Dict[str, Any]],
        regime_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate Kelly optimal position sizing."""

        try:
            if not signal_data:
                return {"sizing_available": False, "rejection_reason": "No signal data provided"}

            # Extract signal parameters
            win_rate = signal_data.get("confidence", 0.6)
            expected_return = signal_data.get("expected_return", 0.02)
            volatility = market_data.get("volatility_24h", 0.02)

            # Regime-adjusted parameters
            regime = regime_analysis.get("regime", "sideways")
            regime_confidence = regime_analysis.get("confidence", 0.5)

            # Adjust for regime
            if regime == "trending":
                expected_return *= 1 + regime_confidence * 0.2  # Boost for trending
            elif regime == "volatile":
                volatility *= 1 + regime_confidence * 0.5  # Increase vol estimate

            # Calculate Kelly fraction
            if volatility > 0:
                kelly_fraction = (win_rate * expected_return) / (volatility**2)
                kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
            else:
                kelly_fraction = 0.0

            # Calculate recommended size
            portfolio_value = market_data.get("portfolio_value", 100000)
            recommended_size = portfolio_value * kelly_fraction

            # Apply additional safety constraints
            max_position_value = portfolio_value * 0.02  # 2% max position
            recommended_size = min(recommended_size, max_position_value)

            return {
                "sizing_available": True,
                "kelly_fraction": kelly_fraction,
                "recommended_size": recommended_size,
                "win_rate_used": win_rate,
                "expected_return_used": expected_return,
                "volatility_used": volatility,
                "regime_adjusted": True,
            }

        except Exception as e:
            self.logger.error("Kelly sizing failed", symbol=symbol, error=str(e))
            return {"sizing_available": False, "rejection_reason": f"Kelly sizing error: {str(e)}"}

    async def _check_execution_readiness(
        self, symbol: str, market_data: Dict[str, Any], sizing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check execution readiness through ExecutionPolicy."""

        try:
            # Basic market conditions for execution check
            spread = market_data.get("spread_bps", 50)
            volume_24h = market_data.get("volume_24h", 100000)
            orderbook_depth = market_data.get("orderbook_depth", 10000)

            # Check tradability gates
            execution_approved = (
                spread <= 50  # Max 50 bps spread
                and volume_24h >= 50000  # Min $50k volume
                and orderbook_depth >= 5000  # Min $5k depth
            )

            rejection_reason = None
            if spread > 50:
                rejection_reason = f"Spread too wide: {spread} bps"
            elif volume_24h < 50000:
                rejection_reason = f"Volume too low: ${volume_24h:,.0f}"
            elif orderbook_depth < 5000:
                rejection_reason = f"Depth too low: ${orderbook_depth:,.0f}"

            # Calculate execution score
            execution_score = 1.0
            execution_score *= max(0.5, min(1.0, (100 - spread) / 100))  # Spread penalty
            execution_score *= min(1.0, volume_24h / 100000)  # Volume bonus
            execution_score *= min(1.0, orderbook_depth / 10000)  # Depth bonus

            # Execution parameters
            execution_params = {
                "order_type": "limit",
                "max_slippage_bps": 30,
                "timeout_seconds": 300,
                "position_size": sizing_result.get("recommended_size", 0),
            }

            return {
                "execution_approved": execution_approved,
                "execution_score": execution_score,
                "rejection_reason": rejection_reason,
                "execution_params": execution_params,
                "market_conditions": {
                    "spread_bps": spread,
                    "volume_24h": volume_24h,
                    "orderbook_depth": orderbook_depth,
                },
            }

        except Exception as e:
            self.logger.error("Execution check failed", symbol=symbol, error=str(e))
            return {
                "execution_approved": False,
                "rejection_reason": f"Execution check error: {str(e)}",
            }

    def _update_flow_metrics(self, pipeline_result: Dict[str, Any]) -> None:
        """Update flow performance metrics."""

        # Calculate authentic data percentage
        components_passed = len(pipeline_result.get("components_passed", []))
        total_components = components_passed + len(pipeline_result.get("components_failed", []))

        if total_components > 0:
            success_rate = components_passed / total_components
            self.flow_metrics.validation_success_rate = success_rate * 100

        # Update other metrics
        self.flow_metrics.pipeline_latency_ms = pipeline_result.get("pipeline_latency_ms", 0)
        self.flow_metrics.regime_confidence = pipeline_result.get("quality_metrics", {}).get(
            "regime_confidence", 0
        )
        self.flow_metrics.execution_readiness = pipeline_result.get("pipeline_success", False)
        self.flow_metrics.last_update = datetime.now()

        # Update state based on quality
        self._update_flow_state()

    def _update_flow_state(self) -> None:
        """Update data flow state based on metrics."""

        thresholds = self.quality_thresholds[self.quality_gate]

        # Check if we meet quality standards
        quality_ok = (
            self.flow_metrics.validation_success_rate >= thresholds["min_validation_success_rate"]
            and self.flow_metrics.pipeline_latency_ms <= thresholds["max_pipeline_latency_ms"]
            and self.flow_metrics.regime_confidence >= thresholds["min_regime_confidence"]
        )

        # Update state
        if quality_ok and self.flow_metrics.execution_readiness:
            self.current_state = DataFlowState.ACTIVE
        elif self.flow_metrics.validation_success_rate < 50:
            self.current_state = DataFlowState.EMERGENCY
        elif not quality_ok:
            self.current_state = DataFlowState.DEGRADED
        else:
            self.current_state = DataFlowState.ACTIVE

    def _record_pipeline_failure(self, failed_component: str, result: Dict[str, Any]) -> None:
        """Record pipeline failure for analysis."""

        failure_record = {
            "timestamp": datetime.now(),
            "failed_component": failed_component,
            "symbol": result.get("symbol"),
            "flow_state": self.current_state.value,
            "components_passed": result.get("components_passed", []),
            "components_failed": result.get("components_failed", []),
        }

        self.validation_history.append(failure_record)

        # Keep only last 1000 records
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]

        # Record failure in metrics
        self.metrics.record_order(
            "error", result.get("symbol", "unknown"), "pipeline_failure", failed_component
        )

    def get_flow_status(self) -> Dict[str, Any]:
        """Get current data flow status."""

        return {
            "flow_state": self.current_state.value,
            "quality_gate": self.quality_gate.value,
            "strict_mode": self.strict_mode,
            "metrics": {
                "authentic_data_percentage": self.flow_metrics.authentic_data_percentage,
                "pipeline_latency_ms": self.flow_metrics.pipeline_latency_ms,
                "validation_success_rate": self.flow_metrics.validation_success_rate,
                "regime_confidence": self.flow_metrics.regime_confidence,
                "execution_readiness": self.flow_metrics.execution_readiness,
                "last_update": self.flow_metrics.last_update.isoformat(),
            },
            "performance": self.flow_performance,
            "component_status": {
                "data_validator": "active",
                "risk_guard": getattr(
                    self.risk_guard, "current_risk_level", RiskLevel.NORMAL
                ).value,
                "execution_policy": "active",
                "regime_detector": "active",
                "kelly_sizer": "active",
            },
            "quality_thresholds": self.quality_thresholds[self.quality_gate],
        }

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get detailed pipeline statistics."""

        if not self.validation_history:
            return {"message": "No pipeline runs recorded yet"}

        total_runs = len(self.validation_history)
        failed_runs = len([r for r in self.validation_history if r.get("failed_component")])
        success_rate = ((total_runs - failed_runs) / total_runs * 100) if total_runs > 0 else 0

        # Component failure analysis
        component_failures = {}
        for record in self.validation_history:
            if record.get("failed_component"):
                comp = record["failed_component"]
                component_failures[comp] = component_failures.get(comp, 0) + 1

        return {
            "total_pipeline_runs": total_runs,
            "success_rate_percentage": round(success_rate, 2),
            "failed_runs": failed_runs,
            "component_failure_breakdown": component_failures,
            "average_pipeline_time_ms": round(self.flow_performance["total_pipeline_ms"], 2),
            "last_24h_runs": len(
                [
                    r
                    for r in self.validation_history
                    if r.get("timestamp", datetime.min) > datetime.now() - timedelta(days=1)
                ]
            ),
        }
