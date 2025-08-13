#!/usr/bin/env python3
"""
Integrated Trading Engine - Verbindt alle componenten voor live trading
Gebruikt DataFlowOrchestrator voor strakke data discipline en component integratie
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from ..core.data_flow_orchestrator import DataFlowOrchestrator, DataQualityGate, DataFlowState
from ..execution.order_pipeline import OrderPipeline, OrderRequest, OrderType, TimeInForce
from ..risk.centralized_risk_guard import CentralizedRiskGuard, RiskCheckRequest, OperationType
from ..portfolio.volatility_targeting_kelly import VolatilityTargetingKelly, SizingResult

# from ..adapters.kraken_data_adapter import KrakenDataAdapter  # Would be imported when available
from ..observability.unified_metrics import UnifiedMetrics


class TradingEngineState(Enum):
    """Trading engine states."""

    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY = "emergency"


@dataclass
class TradingSession:
    """Trading session information."""

    session_id: str
    start_time: datetime
    symbols_tracked: List[str]
    signals_processed: int = 0
    orders_executed: int = 0
    orders_rejected: int = 0
    total_pnl: float = 0.0
    state: TradingEngineState = TradingEngineState.STOPPED


class IntegratedTradingEngine:
    """
    Integrated trading engine combining all components for live trading.

    Features:
    - Strict data validation through DataFlowOrchestrator
    - Real-time signal processing
    - Risk-aware execution
    - Regime-adaptive positioning
    - Kelly-optimal sizing
    - Comprehensive monitoring
    """

    def __init__(
        self,
        symbols: List[str],
        data_quality_gate: DataQualityGate = DataQualityGate.STRICT,
        max_concurrent_signals: int = 10,
    ):
        """Initialize integrated trading engine."""

        self.logger = get_logger("integrated_trading_engine")
        self.symbols = symbols
        self.max_concurrent_signals = max_concurrent_signals

        # Core orchestrator
        self.data_orchestrator = DataFlowOrchestrator(
            strict_mode=True, quality_gate=data_quality_gate
        )

        # HARD WIRED ORDER PIPELINE - All orders go through ExecutionPolicy.decide
        self.order_pipeline = OrderPipeline(
            default_slippage_budget_bps=30.0,
            order_deduplication_window_minutes=60,
            max_concurrent_orders=max_concurrent_signals,
        )

        # HARD WIRED CENTRALIZED RISK GUARD - All operations checked for day-loss, max-DD, limits
        self.centralized_risk_guard = CentralizedRiskGuard(
            max_daily_loss_pct=5.0,
            max_drawdown_pct=10.0,
            max_position_size_pct=2.0,
            max_total_exposure_pct=95.0,
            min_data_quality_score=0.7,
            max_data_age_minutes=5.0,
        )

        # VOLATILITY TARGETING & KELLY SIZING - fractional Kelly × vol-target × correlation caps
        self.volatility_targeting_kelly = VolatilityTargetingKelly(
            target_volatility=0.15,  # 15% target volatility
            kelly_fraction=0.25,  # 25% of Kelly fraction
            max_asset_exposure_pct=2.0,  # 2% per asset cap
            max_cluster_exposure_pct=20.0,  # 20% per cluster cap
            correlation_threshold=0.7,  # High correlation threshold
            max_leverage=3.0,  # Max 3x leverage
        )

        # Data adapter (placeholder for now)
        self.data_adapter = None  # Would initialize KrakenDataAdapter when available

        # Metrics and monitoring
        self.metrics = UnifiedMetrics("integrated_trading_engine")

        # Engine state
        self.current_state = TradingEngineState.STOPPED
        self.current_session: Optional[TradingSession] = None

        # Signal processing
        self.signal_queue = asyncio.Queue(maxsize=100)
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_signals)

        # Performance tracking
        self.performance_stats = {
            "signals_per_minute": 0.0,
            "average_processing_time_ms": 0.0,
            "pipeline_success_rate": 0.0,
            "execution_success_rate": 0.0,
            "data_quality_score": 0.0,
        }

        # Threading
        self._shutdown_event = asyncio.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._signal_processor_task: Optional[asyncio.Task] = None

        # Persistence
        self.data_path = Path("data/trading_engine")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "IntegratedTradingEngine initialized",
            symbols=len(symbols),
            quality_gate=data_quality_gate.value,
            max_concurrent=max_concurrent_signals,
        )

    async def start_trading_session(self, session_id: Optional[str] = None) -> str:
        """Start new trading session."""

        if self.current_state != TradingEngineState.STOPPED:
            raise RuntimeError(f"Cannot start session in state: {self.current_state.value}")

        self.current_state = TradingEngineState.STARTING

        # Create session
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session = TradingSession(
            session_id=session_id, start_time=datetime.now(), symbols_tracked=self.symbols.copy()
        )

        try:
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._signal_processor_task = asyncio.create_task(self._signal_processor_loop())

            # Verify data flow readiness
            flow_status = self.data_orchestrator.get_flow_status()
            if flow_status["flow_state"] not in ["active", "degraded"]:
                self.logger.warning(
                    "Data flow not optimal for trading", flow_state=flow_status["flow_state"]
                )

            self.current_state = TradingEngineState.ACTIVE

            self.logger.info(
                "Trading session started",
                session_id=session_id,
                symbols=len(self.symbols),
                flow_state=flow_status["flow_state"],
            )

            # Record session start
            self.metrics.record_signal("system", 1.0)  # Session start signal

            return session_id

        except Exception as e:
            self.current_state = TradingEngineState.STOPPED
            self.logger.error("Failed to start trading session", error=str(e))
            raise

    async def stop_trading_session(self) -> Dict[str, Any]:
        """Stop current trading session."""

        if self.current_state == TradingEngineState.STOPPED:
            return {"message": "No active session to stop"}

        self.logger.info(
            "Stopping trading session",
            session_id=self.current_session.session_id if self.current_session else None,
        )

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for tasks to complete
        if self._monitoring_task:
            await self._monitoring_task
        if self._signal_processor_task:
            await self._signal_processor_task

        # Finalize session
        session_summary = {}
        if self.current_session:
            session_duration = datetime.now() - self.current_session.start_time
            session_summary = {
                "session_id": self.current_session.session_id,
                "duration_minutes": session_duration.total_seconds() / 60,
                "signals_processed": self.current_session.signals_processed,
                "orders_executed": self.current_session.orders_executed,
                "orders_rejected": self.current_session.orders_rejected,
                "total_pnl": self.current_session.total_pnl,
                "performance_stats": self.performance_stats.copy(),
            }

        # Reset state
        self.current_state = TradingEngineState.STOPPED
        self.current_session = None
        self._shutdown_event.clear()

        self.logger.info("Trading session stopped", session_summary=session_summary)
        return session_summary

    async def process_market_signal(
        self, symbol: str, signal_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process individual market signal through complete pipeline."""

        if self.current_state != TradingEngineState.ACTIVE:
            return {"success": False, "reason": f"Engine not active: {self.current_state.value}"}

        signal_start = time.time()

        try:
            # Get current market data
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return {"success": False, "reason": "Failed to get market data"}

            # Process through data flow orchestrator
            async with self.processing_semaphore:
                pipeline_result = await self.data_orchestrator.process_market_signal(
                    symbol=symbol, market_data=market_data, signal_data=signal_data
                )

            processing_time = (time.time() - signal_start) * 1000

            # Update session stats and EXECUTE ORDER through pipeline if approved
            if self.current_session:
                self.current_session.signals_processed += 1

                if pipeline_result["pipeline_success"]:
                    # HARD WIRE-UP: Check centralized risk guard FIRST
                    risk_check_result = await self._check_centralized_risk_guard(
                        symbol=symbol, pipeline_result=pipeline_result, signal_data=signal_data
                    )

                    if risk_check_result.approved:
                        # Calculate optimal position size using vol-targeting & Kelly
                        sizing_result = self._calculate_volatility_kelly_size(
                            symbol=symbol,
                            pipeline_result=pipeline_result,
                            signal_data=signal_data,
                            risk_check_result=risk_check_result,
                        )

                        # Execute order through centralized OrderPipeline with optimal sizing
                        order_result = await self._execute_order_through_pipeline(
                            symbol=symbol,
                            pipeline_result=pipeline_result,
                            signal_data=signal_data,
                            risk_check_result=risk_check_result,
                            sizing_result=sizing_result,
                        )
                    else:
                        # Order blocked by risk guard
                        self.current_session.orders_rejected += 1
                        self.logger.warning(
                            "Order blocked by centralized risk guard",
                            symbol=symbol,
                            risk_violations=risk_check_result.violations,
                            risk_level=risk_check_result.risk_level.value,
                            kill_switch_active=risk_check_result.kill_switch_active,
                        )
                        # Skip to next signal - order blocked

                    if order_result.status.value in ["filled", "partially_filled"]:
                        self.current_session.orders_executed += 1
                        self.logger.info(
                            "Order executed through pipeline",
                            symbol=symbol,
                            client_order_id=order_result.client_order_id,
                            filled_quantity=order_result.filled_quantity,
                            slippage_bps=order_result.slippage_bps,
                            processing_time_ms=processing_time,
                        )
                    else:
                        self.current_session.orders_rejected += 1
                        self.logger.info(
                            "Order rejected by execution pipeline",
                            symbol=symbol,
                            client_order_id=order_result.client_order_id,
                            rejection_reason=order_result.rejection_reason,
                            processing_time_ms=processing_time,
                        )
                else:
                    self.current_session.orders_rejected += 1
                    self.logger.info(
                        "Signal rejected",
                        symbol=symbol,
                        rejection_reasons=pipeline_result.get("components_failed", []),
                        processing_time_ms=processing_time,
                    )

            # Update performance stats
            self._update_performance_stats(processing_time, pipeline_result)

            # Record metrics
            if pipeline_result["pipeline_success"]:
                self.metrics.record_order("approved", symbol, "signal_processed")
            else:
                self.metrics.record_order("rejected", symbol, "signal_processed")

            return {
                "success": pipeline_result["pipeline_success"],
                "processing_time_ms": processing_time,
                "pipeline_result": pipeline_result,
                "execution_recommendation": pipeline_result.get(
                    "execution_recommendation", "REJECT"
                ),
            }

        except Exception as e:
            processing_time = (time.time() - signal_start) * 1000
            self.logger.error(
                "Signal processing failed",
                symbol=symbol,
                error=str(e),
                processing_time_ms=processing_time,
            )

            return {
                "success": False,
                "reason": f"Processing error: {str(e)}",
                "processing_time_ms": processing_time,
            }

    async def _check_centralized_risk_guard(
        self, symbol: str, pipeline_result: Dict[str, Any], signal_data: Dict[str, Any]
    ) -> Any:
        """Check centralized risk guard before order execution."""

        try:
            # Extract position parameters
            position_size = pipeline_result.get("final_position_size", 0)

            # Create risk check request
            risk_request = RiskCheckRequest(
                operation_type=OperationType.ENTRY,  # Assume entry for now
                symbol=symbol,
                side="buy" if position_size > 0 else "sell",
                quantity=abs(position_size),
                price=45000.0,  # Mock price - would get from market data
                # Portfolio context (mock values - would get from portfolio manager)
                current_portfolio_value=100000.0,
                current_positions={symbol: 0.0},  # Mock - would get actual positions
                current_exposure=50000.0,
                # Performance context (mock values)
                daily_pnl=0.0,  # Would get from performance tracker
                total_drawdown_pct=2.0,  # Would get from performance tracker
                # Data quality context
                data_age_seconds=30.0,  # Mock - would get from data feed
                data_quality_score=0.8,  # Mock - would get from data validator
                last_price_update=datetime.now(),
                # Metadata
                strategy_id=signal_data.get("strategy_id", "integrated_engine"),
            )

            # Check centralized risk guard
            risk_result = await self.centralized_risk_guard.check_operation_risk(risk_request)

            return risk_result

        except Exception as e:
            self.logger.error("Centralized risk guard check failed", symbol=symbol, error=str(e))

            # Return blocking result on error
            from ..risk.centralized_risk_guard import RiskCheckResult, RiskAction, RiskLevel

            return RiskCheckResult(
                action=RiskAction.BLOCK,
                approved=False,
                risk_level=RiskLevel.CRITICAL,
                violations=[f"risk_guard_error: {str(e)}"],
                warnings=[],
                max_allowed_quantity=0.0,
                position_limit_pct=0.0,
                portfolio_risk_pct=0.0,
                position_concentration=0.0,
                data_quality_score=0.0,
                kill_switch_active=False,
                kill_switch_reason=None,
                execution_allowed=False,
                max_position_size=0.0,
                risk_budget_used_pct=0.0,
                check_time_ms=0.0,
            )

    def _calculate_volatility_kelly_size(
        self,
        symbol: str,
        pipeline_result: Dict[str, Any],
        signal_data: Dict[str, Any],
        risk_check_result: Any,
    ) -> SizingResult:
        """Calculate optimal position size using volatility targeting & Kelly sizing."""

        try:
            # Extract signal parameters
            signal_confidence = signal_data.get("confidence", 0.8)
            expected_return = signal_data.get("expected_return", 0.02)  # 2% default
            current_price = 45000.0  # Mock price - would get from market data

            # Get portfolio value (mock - would get from portfolio manager)
            portfolio_value = 100000.0

            # Calculate optimal position size
            sizing_result = self.volatility_targeting_kelly.calculate_position_size(
                symbol=symbol,
                signal_confidence=signal_confidence,
                expected_return=expected_return,
                current_price=current_price,
                portfolio_value=portfolio_value,
            )

            self.logger.info(
                "Vol-targeting Kelly sizing calculated",
                symbol=symbol,
                signal_confidence=signal_confidence,
                expected_return=expected_return,
                kelly_fraction=sizing_result.kelly_fraction,
                vol_scaling=sizing_result.vol_scaling_factor,
                final_size=sizing_result.final_position_size,
                caps_applied=sizing_result.asset_cap_applied or sizing_result.cluster_cap_applied,
            )

            return sizing_result

        except Exception as e:
            self.logger.error("Vol-targeting Kelly sizing failed", symbol=symbol, error=str(e))

            # Return minimal sizing result
            return SizingResult(
                symbol=symbol,
                signal_confidence=signal_data.get("confidence", 0.0),
                expected_return=signal_data.get("expected_return", 0.0),
                kelly_fraction=0.0,
                fractional_kelly=0.0,
                target_volatility=0.15,
                realized_volatility=0.0,
                vol_scaling_factor=0.0,
                base_size=0.0,
                vol_adjusted_size=0.0,
                correlation_adjusted_size=0.0,
                final_position_size=0.01,  # Minimal position
                warnings=[f"Sizing calculation error: {str(e)}"],
            )

    async def _execute_order_through_pipeline(
        self,
        symbol: str,
        pipeline_result: Dict[str, Any],
        signal_data: Dict[str, Any],
        risk_check_result: Any = None,
        sizing_result: SizingResult = None,
    ) -> Any:
        """Execute order through centralized OrderPipeline with ExecutionPolicy gates."""

        try:
            # Extract execution parameters from pipeline result
            base_position_size = pipeline_result.get("final_position_size", 0)
            execution_params = pipeline_result.get("execution_parameters", {})

            # Use vol-targeting Kelly size if available
            if sizing_result:
                position_size = sizing_result.final_position_size
                self.logger.info(
                    "Using vol-targeting Kelly position size",
                    symbol=symbol,
                    base_size=base_position_size,
                    kelly_size=sizing_result.final_position_size,
                    kelly_fraction=sizing_result.kelly_fraction,
                    vol_scaling=sizing_result.vol_scaling_factor,
                )
            else:
                position_size = base_position_size

            # Apply risk guard limits if provided
            if risk_check_result and hasattr(risk_check_result, "max_allowed_quantity"):
                position_size = min(abs(position_size), risk_check_result.max_allowed_quantity)
                if base_position_size < 0:
                    position_size = -position_size

            # Determine order side based on signal
            side = "buy" if position_size > 0 else "sell"
            quantity = abs(position_size)

            # Create order request with hard wire-up parameters
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.LIMIT,  # Default to limit orders
                time_in_force=TimeInForce.POST_ONLY,  # Default to post-only (maker)
                # Hard wire-up execution parameters
                max_slippage_bps=execution_params.get("max_slippage_bps", 30.0),
                min_fill_ratio=0.1,
                timeout_seconds=300,
                # Metadata from signal
                strategy_id=signal_data.get("strategy_id", "integrated_engine"),
                signal_id=signal_data.get("signal_id", f"signal_{int(time.time())}"),
            )

            # Submit order through centralized pipeline (HARD GATE)
            order_result = await self.order_pipeline.submit_order(order_request)

            return order_result

        except Exception as e:
            self.logger.error(
                "Order execution through pipeline failed", symbol=symbol, error=str(e)
            )

            # Return mock failure result
            from ..execution.order_pipeline import OrderResult, OrderStatus

            return OrderResult(
                client_order_id=f"failed_{symbol}_{int(time.time())}",
                status=OrderStatus.FAILED,
                rejection_reason=f"Pipeline execution error: {str(e)}",
            )

    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol."""

        try:
            # Get market data from adapter (mock for now)
            if self.data_adapter:
                ticker_data = await self.data_adapter.get_ticker_data(symbol)
                orderbook_data = await self.data_adapter.get_orderbook_data(symbol)
            else:
                # Mock data for demonstration
                ticker_data = {
                    "last": 45000.0,
                    "volume": 1000000,
                    "change": 0.02,
                    "spread_bps": 25,
                    "volatility": 0.025,
                }
                orderbook_data = {"depth": 15000, "spread": 0.0025}

            if not ticker_data:
                return None

            # Combine data
            market_data = {
                "symbol": symbol,
                "price": ticker_data.get("last", 0),
                "volume_24h": ticker_data.get("volume", 0),
                "price_change_24h": ticker_data.get("change", 0),
                "spread_bps": ticker_data.get("spread_bps", 50),
                "volatility_24h": ticker_data.get("volatility", 0.02),
                "timestamp": datetime.now(),
                "portfolio_value": 100000,  # Would come from portfolio manager
            }

            # Add orderbook data if available
            if orderbook_data:
                market_data["orderbook_depth"] = orderbook_data.get("depth", 10000)
                market_data["bid_ask_spread"] = orderbook_data.get("spread", 0.001)

            return market_data

        except Exception as e:
            self.logger.error("Failed to get market data", symbol=symbol, error=str(e))
            return None

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""

        self.logger.info("Starting monitoring loop")

        try:
            while not self._shutdown_event.is_set():
                # Monitor data flow health
                flow_status = self.data_orchestrator.get_flow_status()

                # Check for degraded state
                if flow_status["flow_state"] == "emergency":
                    self.logger.warning("Data flow in emergency state - pausing trading")
                    self.current_state = TradingEngineState.EMERGENCY
                elif flow_status["flow_state"] == "degraded":
                    self.logger.warning("Data flow degraded - monitoring closely")

                # Update performance metrics
                self._calculate_performance_metrics()

                # Record monitoring metrics
                self.metrics.update_drawdown(0.0)  # Would come from portfolio

                # Sleep before next check
                await asyncio.sleep(30)  # Monitor every 30 seconds

        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error("Monitoring loop error", error=str(e))

    async def _signal_processor_loop(self) -> None:
        """Background signal processor loop."""

        self.logger.info("Starting signal processor loop")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for signals (with timeout)
                    signal_data = await asyncio.wait_for(self.signal_queue.get(), timeout=5.0)

                    # Process signal
                    result = await self.process_market_signal(
                        signal_data["symbol"], signal_data["signal"]
                    )

                    # Mark task done
                    self.signal_queue.task_done()

                except asyncio.TimeoutError:
                    # No signals to process - continue monitoring
                    continue

        except asyncio.CancelledError:
            self.logger.info("Signal processor loop cancelled")
        except Exception as e:
            self.logger.error("Signal processor loop error", error=str(e))

    def _update_performance_stats(
        self, processing_time_ms: float, pipeline_result: Dict[str, Any]
    ) -> None:
        """Update performance statistics."""

        # Update processing time (exponential moving average)
        alpha = 0.1
        self.performance_stats["average_processing_time_ms"] = (
            alpha * processing_time_ms
            + (1 - alpha) * self.performance_stats["average_processing_time_ms"]
        )

        # Update success rate
        if pipeline_result["pipeline_success"]:
            success = 1.0
        else:
            success = 0.0

        self.performance_stats["pipeline_success_rate"] = (
            alpha * success + (1 - alpha) * self.performance_stats["pipeline_success_rate"]
        )

        # Update data quality score
        quality_score = pipeline_result.get("quality_metrics", {}).get("data_quality_score", 0.5)
        self.performance_stats["data_quality_score"] = (
            alpha * quality_score + (1 - alpha) * self.performance_stats["data_quality_score"]
        )

    def _calculate_performance_metrics(self) -> None:
        """Calculate overall performance metrics."""

        if not self.current_session:
            return

        # Calculate signals per minute
        session_duration = datetime.now() - self.current_session.start_time
        minutes = session_duration.total_seconds() / 60

        if minutes > 0:
            self.performance_stats["signals_per_minute"] = (
                self.current_session.signals_processed / minutes
            )

        # Calculate execution success rate
        total_orders = self.current_session.orders_executed + self.current_session.orders_rejected
        if total_orders > 0:
            self.performance_stats["execution_success_rate"] = (
                self.current_session.orders_executed / total_orders
            )

    async def queue_signal(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Queue signal for processing."""

        try:
            await self.signal_queue.put(
                {"symbol": symbol, "signal": signal_data, "timestamp": datetime.now()}
            )
            return True

        except asyncio.QueueFull:
            self.logger.warning("Signal queue full - dropping signal", symbol=symbol)
            return False

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status."""

        flow_status = self.data_orchestrator.get_flow_status()

        # Get order pipeline status
        pipeline_status = self.order_pipeline.get_pipeline_status()

        # Get centralized risk guard status
        risk_guard_status = self.centralized_risk_guard.get_risk_status()

        # Get vol-targeting Kelly sizing status
        sizing_stats = self.volatility_targeting_kelly.get_sizing_statistics()

        status = {
            "engine_state": self.current_state.value,
            "data_flow_state": flow_status["flow_state"],
            "performance_stats": self.performance_stats.copy(),
            "queue_size": self.signal_queue.qsize(),
            "symbols_tracked": len(self.symbols),
            "order_pipeline_status": {
                "active_orders": pipeline_status["active_orders"],
                "total_orders": pipeline_status["total_orders_submitted"],
                "policy_rejection_rate": pipeline_status["pipeline_stats"]["policy_rejection_rate"],
                "average_slippage_bps": pipeline_status["average_slippage_bps"],
                "execution_policy_status": pipeline_status["execution_policy_status"],
            },
            "centralized_risk_guard_status": {
                "kill_switch_active": risk_guard_status["kill_switch_active"],
                "kill_switch_reason": risk_guard_status["kill_switch_reason"],
                "operations_approval_rate": risk_guard_status["daily_statistics"][
                    "approval_rate_pct"
                ],
                "operations_blocked_today": risk_guard_status["daily_statistics"][
                    "operations_blocked"
                ],
                "recent_violations_count": len(risk_guard_status["recent_violations"]),
                "risk_limits_active": True,
            },
            "volatility_targeting_kelly_status": {
                "target_volatility": sizing_stats["target_volatility"],
                "kelly_fraction": sizing_stats["kelly_fraction"],
                "total_calculations": sizing_stats["total_calculations"],
                "avg_kelly_fraction": sizing_stats["avg_kelly_fraction"],
                "avg_vol_scaling": sizing_stats["avg_vol_scaling"],
                "caps_applied_pct": sizing_stats["caps_applied_pct"],
                "assets_tracked": sizing_stats["assets_tracked"],
                "clusters_tracked": sizing_stats["clusters_tracked"],
                "max_asset_exposure_pct": sizing_stats["max_asset_exposure_pct"],
                "max_cluster_exposure_pct": sizing_stats["max_cluster_exposure_pct"],
            },
        }

        if self.current_session:
            session_duration = datetime.now() - self.current_session.start_time
            status["current_session"] = {
                "session_id": self.current_session.session_id,
                "duration_minutes": round(session_duration.total_seconds() / 60, 2),
                "signals_processed": self.current_session.signals_processed,
                "orders_executed": self.current_session.orders_executed,
                "orders_rejected": self.current_session.orders_rejected,
            }

        return status

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed engine and pipeline statistics."""

        flow_stats = self.data_orchestrator.get_pipeline_statistics()
        flow_status = self.data_orchestrator.get_flow_status()

        return {
            "engine_performance": self.performance_stats,
            "data_flow_statistics": flow_stats,
            "data_flow_status": flow_status,
            "component_performance": flow_status.get("performance", {}),
            "session_summary": {
                "current_session": self.current_session.__dict__ if self.current_session else None,
                "engine_state": self.current_state.value,
            },
        }
