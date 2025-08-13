#!/usr/bin/env python3
"""
Enterprise Integrator
Integrates all enterprise-grade fixes into the main application pipeline
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# Core imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Enterprise fixes imports
try:
    from ..core.data_completeness_gate import DataCompletenessGate
    from ..core.secure_logging import get_secure_logger
    from ml.enhanced_calibration import (
        EnhancedCalibratorV2,
        create_confidence_gate_with_calibration,
    )
    from utils.timestamp_validator import normalize_timestamp, validate_timestamp_sequence
    from trading.realistic_execution_engine import RealisticExecutionEngine
    from ml.regime_adaptive_modeling import MarketRegimeDetector
    from ml.uncertainty_quantification import UncertaintyAwarePredictionSystem

    ENTERPRISE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some enterprise modules not available: {e}")
    ENTERPRISE_MODULES_AVAILABLE = False


class EnterpriseIntegratedPipeline:
    """Enterprise-grade trading pipeline with all fixes integrated"""

    def __init__(self):
        self.logger = (
            get_secure_logger("EnterprisePipeline") if ENTERPRISE_MODULES_AVAILABLE else None
        )
        self.data_gate = DataCompletenessGate() if ENTERPRISE_MODULES_AVAILABLE else None
        self.calibrator = EnhancedCalibratorV2() if ENTERPRISE_MODULES_AVAILABLE else None
        self.execution_engine = RealisticExecutionEngine() if ENTERPRISE_MODULES_AVAILABLE else None
        self.regime_detector = MarketRegimeDetector() if ENTERPRISE_MODULES_AVAILABLE else None
        self.uncertainty_system = (
            UncertaintyAwarePredictionSystem() if ENTERPRISE_MODULES_AVAILABLE else None
        )

        # Performance tracking
        self.pipeline_stats = {
            "data_filtered": 0,
            "predictions_calibrated": 0,
            "execution_simulated": 0,
            "regime_adaptive": 0,
        }

    def process_market_data(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process market data through enterprise pipeline"""

        if not ENTERPRISE_MODULES_AVAILABLE:
            return raw_data, {"status": "enterprise_disabled"}

        pipeline_start = datetime.now()
        processing_report = {"steps": []}

        # Step 1: Data Completeness Gate
        try:
            if self.data_gate:
                filtered_data, completeness_report = self.data_gate.validate_completeness(raw_data)
                processing_report["steps"].append(
                    {
                        "step": "data_completeness",
                        "status": "completed",
                        "original_count": completeness_report.get("original_count", 0),
                        "passed_count": completeness_report.get("passed_count", 0),
                        "rejection_rate": completeness_report.get("rejection_rate", 0),
                    }
                )
                self.pipeline_stats["data_filtered"] += completeness_report.get(
                    "rejection_count", 0
                )
            else:
                filtered_data = raw_data
                processing_report["steps"].append(
                    {
                        "step": "data_completeness",
                        "status": "skipped",
                        "reason": "gate_not_available",
                    }
                )

        except Exception as e:
            filtered_data = raw_data
            processing_report["steps"].append(
                {"step": "data_completeness", "status": "failed", "error": str(e)}
            )

        # Step 2: Timestamp Validation
        try:
            if not filtered_data.empty and "timestamp" in filtered_data.columns:
                # Normalize timestamps
                filtered_data["timestamp"] = filtered_data["timestamp"].apply(
                    lambda x: normalize_timestamp(x) if pd.notna(x) else x
                )

                # Validate sequence
                validation_result = validate_timestamp_sequence(filtered_data, "timestamp")
                processing_report["steps"].append(
                    {
                        "step": "timestamp_validation",
                        "status": "completed"
                        if validation_result.get("valid", False)
                        else "warning",
                        "issues": validation_result.get("issues", []),
                    }
                )
            else:
                processing_report["steps"].append(
                    {
                        "step": "timestamp_validation",
                        "status": "skipped",
                        "reason": "no_timestamp_column",
                    }
                )

        except Exception as e:
            processing_report["steps"].append(
                {"step": "timestamp_validation", "status": "failed", "error": str(e)}
            )

        # Step 3: Regime Detection
        try:
            if self.regime_detector and not filtered_data.empty:
                regime_result = self.regime_detector.fit(filtered_data)
                if regime_result.get("success", False):
                    regimes = self.regime_detector.predict_regime(filtered_data)
                    filtered_data["market_regime"] = regimes
                    processing_report["steps"].append(
                        {
                            "step": "regime_detection",
                            "status": "completed",
                            "regimes_detected": regime_result.get("regimes_detected", 0),
                        }
                    )
                    self.pipeline_stats["regime_adaptive"] += 1

        except Exception as e:
            processing_report["steps"].append(
                {"step": "regime_detection", "status": "failed", "error": str(e)}
            )

        # Calculate processing time
        processing_duration = (datetime.now() - pipeline_start).total_seconds()

        processing_report.update(
            {
                "processing_duration_seconds": processing_duration,
                "total_steps": len(processing_report["steps"]),
                "successful_steps": sum(
                    1 for step in processing_report["steps"] if step["status"] == "completed"
                ),
                "pipeline_stats": self.pipeline_stats,
            }
        )

        return filtered_data, processing_report

    def process_predictions(
        self, predictions_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process predictions through enterprise calibration and uncertainty"""

        if not ENTERPRISE_MODULES_AVAILABLE or predictions_df.empty:
            return predictions_df, {"status": "enterprise_disabled_or_empty"}

        processing_report = {"prediction_steps": []}

        # Step 1: Uncertainty Quantification
        try:
            if self.uncertainty_system:
                uncertainty_results, uncertainty_report = (
                    self.uncertainty_system.predict_with_confidence_gate(predictions_df)

                # Add uncertainty metrics to predictions
                predictions_df["uncertainty"] = uncertainty_results.get("uncertainty", 0)
                predictions_df["confidence_score"] = uncertainty_results.get("confidence", 0.5)

                processing_report["prediction_steps"].append(
                    {
                        "step": "uncertainty_quantification",
                        "status": "completed",
                        "high_confidence_rate": uncertainty_report.get("high_confidence_rate", 0),
                    }
                )

        except Exception as e:
            processing_report["prediction_steps"].append(
                {"step": "uncertainty_quantification", "status": "failed", "error": str(e)}
            )

        # Step 2: Confidence Calibration
        try:
            if self.calibrator and "conf_7d" in predictions_df.columns:
                # Apply calibration to confidence scores
                conf_cols = [col for col in predictions_df.columns if col.startswith("conf_")]

                for conf_col in conf_cols:
                    if self.calibrator.is_fitted:
                        predictions_df[conf_col] = self.calibrator.calibrate_probabilities(
                            predictions_df[conf_col].values
                        )
                        self.pipeline_stats["predictions_calibrated"] += len(predictions_df)

                processing_report["prediction_steps"].append(
                    {
                        "step": "confidence_calibration",
                        "status": "completed" if self.calibrator.is_fitted else "skipped",
                        "calibrated_columns": conf_cols,
                    }
                )

        except Exception as e:
            processing_report["prediction_steps"].append(
                {"step": "confidence_calibration", "status": "failed", "error": str(e)}
            )

        return predictions_df, processing_report

    def process_trading_opportunities(self, trading_opportunities: List[Dict]) -> Dict[str, Any]:
        """Simulate realistic execution for trading opportunities"""

        if not ENTERPRISE_MODULES_AVAILABLE or not trading_opportunities:
            return {"status": "enterprise_disabled_or_empty"}

        execution_results = []

        try:
            if self.execution_engine:
                for opportunity in trading_opportunities:
                    # Extract execution parameters
                    order_size = opportunity.get("position_size", 1000)
                    market_price = opportunity.get("current_price", 100)
                    volatility = opportunity.get("volatility", 0.02)
                    volume_24h = opportunity.get("volume_24h", 1000000)

                    # REMOVED: Mock data pattern not allowed in production
                    execution_result = self.execution_engine.execute_order(
                        order_size=order_size,
                        market_price=market_price,
                        volatility=volatility,
                        volume_24h=volume_24h,
                    )

                    execution_results.append(
                        {
                            "symbol": opportunity.get("symbol", "UNKNOWN"),
                            "executed_size": execution_result.executed_size,
                            "executed_price": execution_result.executed_price,
                            "slippage_bps": execution_result.slippage_bps,
                            "success": execution_result.success,
                            "partial_fill": execution_result.partial_fill,
                        }
                    )

                self.pipeline_stats["execution_simulated"] += len(execution_results)

                # Calculate execution statistics
                successful_executions = [r for r in execution_results if r["success"]]

                if successful_executions:
                    avg_slippage = np.mean([r["slippage_bps"] for r in successful_executions])
                    success_rate = len(successful_executions) / len(execution_results)

                    return {
                        "status": "completed",
                        "total_opportunities": len(trading_opportunities),
                        "successful_executions": len(successful_executions),
                        "success_rate": success_rate,
                        "average_slippage_bps": avg_slippage,
                        "execution_details": execution_results[:10],  # First 10 for display
                    }

        except Exception as e:
            return {"status": "failed", "error": str(e), "execution_results": execution_results}

        return {"status": "no_execution_engine"}

    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get enterprise pipeline health status"""

        health_status = {
            "enterprise_modules_available": ENTERPRISE_MODULES_AVAILABLE,
            "components_status": {
                "data_completeness_gate": self.data_gate is not None,
                "secure_logging": self.logger is not None,
                "calibration_system": self.calibrator is not None,
                "execution_engine": self.execution_engine is not None,
                "regime_detector": self.regime_detector is not None,
                "uncertainty_system": self.uncertainty_system is not None,
            },
            "pipeline_stats": self.pipeline_stats,
            "health_score": 0.0,
        }

        # Calculate health score
        available_components = sum(
            1 for available in health_status["components_status"].values() if available
        )
        total_components = len(health_status["components_status"])
        health_status["health_score"] = available_components / total_components

        return health_status


# Global enterprise pipeline instance
_enterprise_pipeline = None


def get_enterprise_pipeline() -> EnterpriseIntegratedPipeline:
    """Get or create enterprise pipeline instance"""
    global _enterprise_pipeline

    if _enterprise_pipeline is None:
        _enterprise_pipeline = EnterpriseIntegratedPipeline()

    return _enterprise_pipeline


def initialize_enterprise_features() -> Dict[str, Any]:
    """Initialize all enterprise features and return status"""

    pipeline = get_enterprise_pipeline()
    health = pipeline.get_pipeline_health()

    initialization_report = {
        "timestamp": datetime.now().isoformat(),
        "enterprise_available": ENTERPRISE_MODULES_AVAILABLE,
        "health_score": health["health_score"],
        "components_initialized": health["components_status"],
        "ready_for_production": health["health_score"] >= 0.8,
    }

    return initialization_report
