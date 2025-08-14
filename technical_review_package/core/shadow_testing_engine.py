#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Shadow Testing & Live Monitoring Engine
Model runs in "paper trading" mode before going live with comprehensive validation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import asyncio
import json
from pathlib import Path
import json  # SECURITY: Replaced pickle with JSON for external data
import warnings

warnings.filterwarnings("ignore")


class ShadowTestStatus(Enum):
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    PAPER_TRADING = "paper_trading"
    VALIDATION = "validation"
    APPROVED = "approved"
    REJECTED = "rejected"
    LIVE_MONITORING = "live_monitoring"


class ValidationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"
    MANUAL_REVIEW = "manual_review"


@dataclass
class ShadowTradeRecord:
    """Record of a shadow trade execution"""

    trade_id: str
    timestamp: datetime
    symbol: str
    action: str  # 'buy' or 'sell'
    quantity: float
    price: float
    confidence: float
    model_version: str
    shadow_pnl: float = 0.0
    actual_price: Optional[float] = None
    slippage: Optional[float] = None
    execution_time_ms: Optional[float] = None


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance metrics"""

    model_id: str
    shadow_period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy: float
    precision: float
    recall: float
    average_confidence: float
    risk_adjusted_return: float
    validation_score: float
    timestamp: datetime


@dataclass
class ShadowTestConfig:
    """Configuration for shadow testing"""

    shadow_period_days: int = 30
    min_trades_for_validation: int = 100
    min_accuracy_threshold: float = 0.6
    min_sharpe_ratio: float = 1.0
    max_drawdown_threshold: float = 0.15
    confidence_threshold: float = 0.8
    paper_trading_balance: float = 100000.0
    risk_per_trade: float = 0.02
    validation_criteria: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.6,
            "sharpe_ratio": 1.0,
            "max_drawdown": 0.15,
            "total_trades": 100,
            "risk_adjusted_return": 0.1,
        }
    )


class ShadowTestingEngine:
    """Advanced shadow testing engine with comprehensive model validation"""

    def __init__(self, config: Optional[ShadowTestConfig] = None):
        self.config = config or ShadowTestConfig()
        self.logger = logging.getLogger(f"{__name__}.ShadowTestingEngine")

        # Shadow testing state
        self.active_shadow_tests: Dict[str, Dict] = {}
        self.shadow_trade_history: List[ShadowTradeRecord] = []
        self.model_performance_history: List[ModelPerformanceMetrics] = []

        # Paper trading portfolio
        self.paper_portfolios: Dict[str, Dict] = {}

        # Live monitoring state
        self.live_models: Dict[str, Dict] = {}
        self.live_performance_tracking: Dict[str, List] = {}

        # Validation callbacks
        self.validation_callbacks: List[Callable] = []

        # Data storage
        self.data_dir = Path("data/shadow_testing")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Thread safety
        self._lock = threading.RLock()

        self.logger.info("Shadow Testing Engine initialized for model validation")

    def start_shadow_test(
        self, model_id: str, model_instance: Any, model_metadata: Dict[str, Any]
    ) -> bool:
        """
        Start shadow testing for a new model

        Args:
            model_id: Unique identifier for the model
            model_instance: The trained model instance
            model_metadata: Metadata about the model

        Returns:
            True if shadow test started successfully
        """
        with self._lock:
            try:
                if model_id in self.active_shadow_tests:
                    self.logger.warning(f"Shadow test already active for model {model_id}")
                    return False

                # Initialize shadow test
                shadow_test = {
                    "model_id": model_id,
                    "model_instance": model_instance,
                    "metadata": model_metadata,
                    "status": ShadowTestStatus.INITIALIZING,
                    "start_time": datetime.now(),
                    "end_time": None,
                    "trades": [],
                    "performance_metrics": {},
                    "validation_result": ValidationResult.PENDING,
                }

                # Initialize paper trading portfolio
                self.paper_portfolios[model_id] = {
                    "balance": self.config.paper_trading_balance,
                    "positions": {},
                    "total_pnl": 0.0,
                    "trade_history": [],
                    "peak_balance": self.config.paper_trading_balance,
                    "drawdown": 0.0,
                }

                self.active_shadow_tests[model_id] = shadow_test

                # Update status
                self._update_shadow_test_status(model_id, ShadowTestStatus.PAPER_TRADING)

                self.logger.info(f"Started shadow test for model {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to start shadow test for {model_id}: {e}")
                return False

    def execute_shadow_trade(
        self, model_id: str, trade_signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[ShadowTradeRecord]:
        """
        Execute a shadow trade based on model signal

        Args:
            model_id: Model identifier
            trade_signal: Trading signal from the model
            market_data: Current market data

        Returns:
            Shadow trade record if executed
        """
        with self._lock:
            try:
                if model_id not in self.active_shadow_tests:
                    return None

                shadow_test = self.active_shadow_tests[model_id]
                if shadow_test["status"] != ShadowTestStatus.PAPER_TRADING:
                    return None

                # Validate trade signal
                if not self._validate_trade_signal(trade_signal):
                    return None

                # Execute paper trade
                trade_record = self._execute_paper_trade(model_id, trade_signal, market_data)

                if trade_record:
                    # Store trade record
                    shadow_test["trades"].append(trade_record)
                    self.shadow_trade_history.append(trade_record)

                    # Update portfolio
                    self._update_paper_portfolio(model_id, trade_record, market_data)

                    # Check if enough trades for validation
                    if len(shadow_test["trades"]) >= self.config.min_trades_for_validation:
                        self._trigger_validation_check(model_id)

                return trade_record

            except Exception as e:
                self.logger.error(f"Shadow trade execution failed for {model_id}: {e}")
                return None

    def _validate_trade_signal(self, trade_signal: Dict[str, Any]) -> bool:
        """Validate trade signal format and content"""
        required_fields = ["symbol", "action", "quantity", "confidence"]

        if not all(field in trade_signal for field in required_fields):
            return False

        if trade_signal["action"] not in ["buy", "sell"]:
            return False

        if trade_signal["confidence"] < self.config.confidence_threshold:
            return False

        if trade_signal["quantity"] <= 0:
            return False

        return True

    def _execute_paper_trade(
        self, model_id: str, trade_signal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> Optional[ShadowTradeRecord]:
        """Execute paper trade and return record"""
        try:
            symbol = trade_signal["symbol"]
            action = trade_signal["action"]
            confidence = trade_signal["confidence"]

            # Get current price from market data
            current_price = self._get_current_price(symbol, market_data)
            if current_price is None:
                return None

            # Calculate position size based on risk management
            portfolio = self.paper_portfolios[model_id]
            position_size = self._calculate_position_size(portfolio, current_price, trade_signal)

            if position_size <= 0:
                return None

            # Generate trade ID
            trade_id = f"{model_id}_{symbol}_{action}_{int(datetime.now().timestamp())}"

            # Create trade record
            trade_record = ShadowTradeRecord(
                trade_id=trade_id,
                timestamp=datetime.now(),
                symbol=symbol,
                action=action,
                quantity=position_size,
                price=current_price,
                confidence=confidence,
                model_version=self.active_shadow_tests[model_id]["metadata"].get(
                    "version", "unknown"
                ),
            )

            return trade_record

        except Exception as e:
            self.logger.error(f"Paper trade execution failed: {e}")
            return None

    def _get_current_price(self, symbol: str, market_data: Dict[str, Any]) -> Optional[float]:
        """Extract current price from market data"""
        try:
            # Try different possible price fields
            price_fields = ["price", "close", "last", "current_price"]

            for field in price_fields:
                if field in market_data:
                    return float(market_data[field])

            # Try nested symbol data
            if symbol in market_data:
                symbol_data = market_data[symbol]
                for field in price_fields:
                    if field in symbol_data:
                        return float(symbol_data[field])

            return None

        except Exception:
            return None

    def _calculate_position_size(
        self, portfolio: Dict[str, Any], price: float, trade_signal: Dict[str, Any]
    ) -> float:
        """Calculate position size based on risk management"""
        try:
            balance = portfolio["balance"]
            risk_amount = balance * self.config.risk_per_trade

            # Basic position sizing: risk amount / price
            base_position_size = risk_amount / price

            # Adjust based on confidence
            confidence = trade_signal["confidence"]
            confidence_multiplier = min(confidence / self.config.confidence_threshold, 2.0)

            position_size = base_position_size * confidence_multiplier

            # Ensure we don't exceed available balance
            max_position_value = balance * 0.1  # Max 10% per trade
            max_position_size = max_position_value / price

            return min(position_size, max_position_size)

        except Exception:
            return 0.0

    def _update_paper_portfolio(
        self, model_id: str, trade_record: ShadowTradeRecord, market_data: Dict[str, Any]
    ):
        """Update paper portfolio with trade execution"""
        try:
            portfolio = self.paper_portfolios[model_id]
            symbol = trade_record.symbol
            action = trade_record.action
            quantity = trade_record.quantity
            price = trade_record.price

            # Update positions
            if symbol not in portfolio["positions"]:
                portfolio["positions"][symbol] = {"quantity": 0.0, "avg_price": 0.0}

            position = portfolio["positions"][symbol]

            if action == "buy":
                # Calculate new average price
                total_value = position["quantity"] * position["avg_price"] + quantity * price
                total_quantity = position["quantity"] + quantity

                if total_quantity > 0:
                    position["avg_price"] = total_value / total_quantity

                position["quantity"] = total_quantity
                portfolio["balance"] -= quantity * price

            elif action == "sell":
                if position["quantity"] >= quantity:
                    # Calculate PnL
                    pnl = quantity * (price - position["avg_price"])
                    trade_record.shadow_pnl = pnl
                    portfolio["total_pnl"] += pnl

                    position["quantity"] -= quantity
                    portfolio["balance"] += quantity * price

                    # Remove position if fully closed
                    if position["quantity"] <= 0:
                        del portfolio["positions"][symbol]

            # Update portfolio metrics
            self._update_portfolio_metrics(portfolio)

            # Store trade in portfolio history
            portfolio["trade_history"].append(
                {
                    "trade_record": trade_record,
                    "portfolio_balance": portfolio["balance"],
                    "total_pnl": portfolio["total_pnl"],
                }
            )

        except Exception as e:
            self.logger.error(f"Portfolio update failed: {e}")

    def _update_portfolio_metrics(self, portfolio: Dict[str, Any]):
        """Update portfolio performance metrics"""
        try:
            current_balance = portfolio["balance"] + portfolio["total_pnl"]

            # Update peak balance and drawdown
            if current_balance > portfolio["peak_balance"]:
                portfolio["peak_balance"] = current_balance
                portfolio["drawdown"] = 0.0
            else:
                portfolio["drawdown"] = (portfolio["peak_balance"] - current_balance) / portfolio[
                    "peak_balance"
                ]

        except Exception as e:
            self.logger.error(f"Portfolio metrics update failed: {e}")

    def _trigger_validation_check(self, model_id: str):
        """Trigger validation check for model performance"""
        try:
            shadow_test = self.active_shadow_tests[model_id]

            # Check if shadow period is complete
            elapsed_days = (datetime.now() - shadow_test["start_time"]).days

            if elapsed_days >= self.config.shadow_period_days:
                self._update_shadow_test_status(model_id, ShadowTestStatus.VALIDATION)
                self._perform_validation(model_id)

        except Exception as e:
            self.logger.error(f"Validation trigger failed for {model_id}: {e}")

    def _perform_validation(self, model_id: str) -> ValidationResult:
        """Perform comprehensive model validation"""
        with self._lock:
            try:
                shadow_test = self.active_shadow_tests[model_id]
                trades = shadow_test["trades"]
                portfolio = self.paper_portfolios[model_id]

                # Calculate performance metrics
                metrics = self._calculate_performance_metrics(model_id, trades, portfolio)

                # Store metrics
                shadow_test["performance_metrics"] = metrics
                self.model_performance_history.append(metrics)

                # Apply validation criteria
                validation_result = self._apply_validation_criteria(metrics)

                # Update shadow test status
                shadow_test["validation_result"] = validation_result
                shadow_test["end_time"] = datetime.now()

                if validation_result == ValidationResult.PASS:
                    self._update_shadow_test_status(model_id, ShadowTestStatus.APPROVED)
                    self._promote_to_live_trading(model_id)
                else:
                    self._update_shadow_test_status(model_id, ShadowTestStatus.REJECTED)

                # Notify callbacks
                for callback in self.validation_callbacks:
                    try:
                        callback(model_id, validation_result, metrics)
                    except Exception as e:
                        self.logger.error(f"Validation callback failed: {e}")

                self.logger.info(f"Validation completed for {model_id}: {validation_result.value}")
                return validation_result

            except Exception as e:
                self.logger.error(f"Validation failed for {model_id}: {e}")
                return ValidationResult.FAIL

    def _calculate_performance_metrics(
        self, model_id: str, trades: List[ShadowTradeRecord], portfolio: Dict[str, Any]
    ) -> ModelPerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            shadow_test = self.active_shadow_tests[model_id]
            shadow_period = (datetime.now() - shadow_test["start_time"]).days

            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.shadow_pnl > 0)
            losing_trades = sum(1 for trade in trades if trade.shadow_pnl < 0)

            total_pnl = portfolio["total_pnl"]
            accuracy = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate returns for Sharpe ratio
            returns = []
            for i, trade_info in enumerate(portfolio["trade_history"]):
                if i > 0:
                    prev_balance = portfolio["trade_history"][i - 1]["portfolio_balance"]
                    curr_balance = trade_info["portfolio_balance"]
                    return_rate = (
                        (curr_balance - prev_balance) / prev_balance if prev_balance > 0 else 0
                    )
                    returns.append(return_rate)

            # Sharpe ratio calculation
            if returns:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            else:
                sharpe_ratio = 0

            # Max drawdown
            max_drawdown = portfolio["drawdown"]

            # Risk-adjusted return
            initial_balance = self.config.paper_trading_balance
            total_return = total_pnl / initial_balance
            risk_adjusted_return = total_return / max(max_drawdown, 0.01)  # Avoid division by zero

            # Average confidence
            average_confidence = np.mean([trade.confidence for trade in trades]) if trades else 0

            # Precision and recall (simplified)
            precision = winning_trades / max(winning_trades + losing_trades, 1)
            recall = winning_trades / max(total_trades, 1)

            # Overall validation score
            validation_score = self._calculate_validation_score(
                accuracy, sharpe_ratio, max_drawdown, risk_adjusted_return, total_trades
            )

            return ModelPerformanceMetrics(
                model_id=model_id,
                shadow_period_days=shadow_period,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                average_confidence=average_confidence,
                risk_adjusted_return=risk_adjusted_return,
                validation_score=validation_score,
                timestamp=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
            return ModelPerformanceMetrics(
                model_id=model_id,
                shadow_period_days=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                max_drawdown=1.0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                average_confidence=0.0,
                risk_adjusted_return=0.0,
                validation_score=0.0,
                timestamp=datetime.now(),
            )

    def _calculate_validation_score(
        self,
        accuracy: float,
        sharpe_ratio: float,
        max_drawdown: float,
        risk_adjusted_return: float,
        total_trades: int,
    ) -> float:
        """Calculate overall validation score"""
        try:
            # Weighted scoring
            weights = {
                "accuracy": 0.25,
                "sharpe_ratio": 0.25,
                "drawdown": 0.20,
                "risk_adjusted_return": 0.20,
                "trade_count": 0.10,
            }

            # Normalize metrics to 0-1 scale
            accuracy_score = accuracy
            sharpe_score = min(sharpe_ratio / 2.0, 1.0)  # Normalize to 2.0 Sharpe
            drawdown_score = max(0, 1.0 - max_drawdown / 0.3)  # Penalty for >30% drawdown
            return_score = min(
                risk_adjusted_return / 0.5, 1.0
            )  # Normalize to 50% risk-adjusted return
            trade_score = min(total_trades / 200, 1.0)  # Normalize to 200 trades

            validation_score = (
                weights["accuracy"] * accuracy_score
                + weights["sharpe_ratio"] * sharpe_score
                + weights["drawdown"] * drawdown_score
                + weights["risk_adjusted_return"] * return_score
                + weights["trade_count"] * trade_score
            )

            return max(0.0, min(1.0, validation_score))

        except Exception:
            return 0.0

    def _apply_validation_criteria(self, metrics: ModelPerformanceMetrics) -> ValidationResult:
        """Apply validation criteria to determine if model passes"""
        try:
            criteria = self.config.validation_criteria

            # Check each criterion
            checks = {
                "accuracy": metrics.accuracy >= criteria["accuracy"],
                "sharpe_ratio": metrics.sharpe_ratio >= criteria["sharpe_ratio"],
                "max_drawdown": metrics.max_drawdown <= criteria["max_drawdown"],
                "total_trades": metrics.total_trades >= criteria["total_trades"],
                "risk_adjusted_return": metrics.risk_adjusted_return
                >= criteria["risk_adjusted_return"],
            }

            # All criteria must pass
            if all(checks.values()):
                return ValidationResult.PASS

            # If close to passing, flag for manual review
            passed_checks = sum(checks.values())
            if passed_checks >= len(checks) - 1:  # Allow one failure
                return ValidationResult.MANUAL_REVIEW

            return ValidationResult.FAIL

        except Exception as e:
            self.logger.error(f"Validation criteria application failed: {e}")
            return ValidationResult.FAIL

    def _promote_to_live_trading(self, model_id: str):
        """Promote model to live trading status"""
        try:
            shadow_test = self.active_shadow_tests[model_id]

            # Move to live models
            self.live_models[model_id] = {
                "model_instance": shadow_test["model_instance"],
                "metadata": shadow_test["metadata"],
                "promotion_time": datetime.now(),
                "shadow_performance": shadow_test["performance_metrics"],
                "status": "active",
            }

            # Initialize live monitoring
            self.live_performance_tracking[model_id] = []

            self.logger.info(f"Model {model_id} promoted to live trading")

        except Exception as e:
            self.logger.error(f"Live promotion failed for {model_id}: {e}")

    def _update_shadow_test_status(self, model_id: str, status: ShadowTestStatus):
        """Update shadow test status"""
        if model_id in self.active_shadow_tests:
            self.active_shadow_tests[model_id]["status"] = status
            self.logger.info(f"Shadow test {model_id} status updated to {status.value}")

    def monitor_live_model_performance(self, model_id: str, actual_trade_result: Dict[str, Any]):
        """Monitor live model performance"""
        with self._lock:
            try:
                if model_id not in self.live_models:
                    return

                # Record live performance
                performance_record = {
                    "timestamp": datetime.now(),
                    "trade_result": actual_trade_result,
                    "model_prediction": actual_trade_result.get("predicted_outcome"),
                    "actual_outcome": actual_trade_result.get("actual_outcome"),
                    "pnl": actual_trade_result.get("pnl", 0),
                }

                self.live_performance_tracking[model_id].append(performance_record)

                # Check for performance degradation
                if len(self.live_performance_tracking[model_id]) >= 20:
                    self._check_live_performance_degradation(model_id)

            except Exception as e:
                self.logger.error(f"Live monitoring failed for {model_id}: {e}")

    def _check_live_performance_degradation(self, model_id: str):
        """Check if live model performance is degrading"""
        try:
            recent_records = self.live_performance_tracking[model_id][-20:]

            # Calculate recent performance metrics
            recent_pnl = sum(record["pnl"] for record in recent_records)
            recent_accuracy = sum(
                1
                for record in recent_records
                if record.get("predicted_outcome") == record.get("actual_outcome")
            ) / len(recent_records)

            # Compare with shadow testing performance
            shadow_metrics = self.live_models[model_id]["shadow_performance"]

            # Check for significant degradation
            accuracy_degradation = shadow_metrics.accuracy - recent_accuracy

            if accuracy_degradation > 0.1 or recent_pnl < -5000:  # 10% accuracy drop or $5K loss
                self._flag_model_for_review(model_id, "Performance degradation detected")

        except Exception as e:
            self.logger.error(f"Performance degradation check failed for {model_id}: {e}")

    def _flag_model_for_review(self, model_id: str, reason: str):
        """Flag model for manual review"""
        self.live_models[model_id]["status"] = "flagged"
        self.live_models[model_id]["flag_reason"] = reason
        self.live_models[model_id]["flag_time"] = datetime.now()

        self.logger.warning(f"Model {model_id} flagged for review: {reason}")

    def get_shadow_test_summary(self) -> Dict[str, Any]:
        """Get comprehensive shadow testing summary"""
        with self._lock:
            summary = {
                "active_shadow_tests": len(self.active_shadow_tests),
                "completed_validations": len(self.model_performance_history),
                "live_models": len(self.live_models),
                "total_shadow_trades": len(self.shadow_trade_history),
                "validation_statistics": self._get_validation_statistics(),
                "active_tests_details": [],
                "live_models_status": {},
            }

            # Active tests details
            for model_id, shadow_test in self.active_shadow_tests.items():
                summary["active_tests_details"].append(
                    {
                        "model_id": model_id,
                        "status": shadow_test["status"].value,
                        "trades_count": len(shadow_test["trades"]),
                        "days_running": (datetime.now() - shadow_test["start_time"]).days,
                        "current_pnl": self.paper_portfolios.get(model_id, {}).get("total_pnl", 0),
                    }
                )

            # Live models status
            for model_id, model_info in self.live_models.items():
                recent_trades = len(self.live_performance_tracking.get(model_id, []))
                summary["live_models_status"][model_id] = {
                    "status": model_info["status"],
                    "promotion_date": model_info["promotion_time"].isoformat(),
                    "recent_trades": recent_trades,
                    "flagged": model_info.get("status") == "flagged",
                }

            return summary

    def _get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics across all models"""
        if not self.model_performance_history:
            return {}

        metrics = self.model_performance_history

        return {
            "total_models_validated": len(metrics),
            "average_accuracy": np.mean([m.accuracy for m in metrics]),
            "average_sharpe_ratio": np.mean([m.sharpe_ratio for m in metrics]),
            "average_max_drawdown": np.mean([m.max_drawdown for m in metrics]),
            "models_passed": len([m for m in metrics if m.validation_score > 0.7]),
            "models_failed": len([m for m in metrics if m.validation_score <= 0.7]),
        }

    def add_validation_callback(self, callback: Callable):
        """Add callback for validation events"""
        self.validation_callbacks.append(callback)

    def save_shadow_test_data(self):
        """Save shadow testing data to disk"""
        try:
            # Save shadow test history
            with open(self.data_dir / "shadow_tests.json", "w") as f:
                json.dump(
                    {
                        "active_tests": {
                            k: {**v, "model_instance": None}
                            for k, v in self.active_shadow_tests.items()
                        },
                        "performance_history": [m.__dict__ for m in self.model_performance_history],
                        "trade_history": [t.__dict__ for t in self.shadow_trade_history],
                    },
                    f,
                    default=str,
                    indent=2,
                )

            # Save portfolio data
            with open(self.data_dir / "paper_portfolios.json", "w") as f:
                json.dump(self.paper_portfolios, f, default=str, indent=2)

            self.logger.info("Shadow test data saved successfully")

        except Exception as e:
            self.logger.error(f"Failed to save shadow test data: {e}")


# Singleton shadow testing engine
_shadow_engine = None
_shadow_lock = threading.Lock()


def get_shadow_testing_engine(config: Optional[ShadowTestConfig] = None) -> ShadowTestingEngine:
    """Get the singleton shadow testing engine"""
    global _shadow_engine

    with _shadow_lock:
        if _shadow_engine is None:
            _shadow_engine = ShadowTestingEngine(config)
        return _shadow_engine


def start_model_shadow_test(model_id: str, model_instance: Any, metadata: Dict[str, Any]) -> bool:
    """Convenient function to start model shadow testing"""
    engine = get_shadow_testing_engine()
    return engine.start_shadow_test(model_id, model_instance, metadata)

"""
SECURITY POLICY: NO PICKLE ALLOWED
This file handles external data.
Pickle usage is FORBIDDEN for security reasons.
Use JSON or msgpack for all serialization.
"""

