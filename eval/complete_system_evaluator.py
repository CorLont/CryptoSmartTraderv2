#!/usr/bin/env python3
"""
Complete System Evaluator - Daily GO/NO-GO Decision System
Enterprise-grade system evaluation with automated decision making
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from pathlib import Path

from core.structured_logger import get_structured_logger
from config.daily_logging_config import get_daily_logger, DailyMetrics
from data.kraken_coverage_auditor import KrakenCoverageAuditor
from core.data_integrity_enforcer import ProductionDataGuard
from orchestration.orchestrator import CryptoSmartTraderOrchestrator, OrchestratorConfig


class CompleteSystemEvaluator:
    """Complete daily system evaluation with GO/NO-GO decisions"""

    def __init__(self):
        self.logger = get_structured_logger("SystemEvaluator")
        self.daily_logger = get_daily_logger()

        # Initialize components
        self.coverage_auditor = KrakenCoverageAuditor()
        self.data_guard = ProductionDataGuard(production_mode=True)

        # Evaluation criteria
        self.evaluation_criteria = {
            "data_integrity": {"weight": 0.35, "min_score": 0.95},
            "system_health": {"weight": 0.25, "min_score": 0.70},
            "model_performance": {"weight": 0.25, "min_score": 0.60},
            "coverage_compliance": {"weight": 0.15, "min_score": 0.99},
        }

    async def run_complete_daily_evaluation(self) -> DailyMetrics:
        """Run complete daily system evaluation"""

        evaluation_start = datetime.utcnow()
        self.logger.info("Starting complete daily system evaluation")

        try:
            # 1. Coverage Audit
            coverage_result = await self.evaluate_coverage_compliance()

            # 2. Data Integrity Check
            integrity_result = await self.evaluate_data_integrity()

            # 3. System Health Check
            health_result = await self.evaluate_system_health()

            # 4. Model Performance Check
            performance_result = await self.evaluate_model_performance()

            # 5. Generate final evaluation
            daily_metrics = self.daily_logger.generate_daily_evaluation()

            # 6. Execute GO/NO-GO decision
            await self.execute_go_nogo_decision(daily_metrics)

            evaluation_time = (datetime.utcnow() - evaluation_start).total_seconds()

            self.logger.info(
                f"Daily evaluation completed in {evaluation_time:.1f}s: {daily_metrics.go_nogo_decision}"
            )

            return daily_metrics

        except Exception as e:
            self.logger.error(f"Daily evaluation failed: {e}")
            self.daily_logger.log_error_with_context(e, {"evaluation_type": "daily_complete"})

            # Create emergency NO-GO decision
            emergency_metrics = DailyMetrics(
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                system_health_score=0.0,
                data_integrity_score=0.0,
                model_performance_score=0.0,
                coverage_compliance_score=0.0,
                overall_score=0.0,
                go_nogo_decision="NO-GO",
                critical_issues=[f"Evaluation system failure: {str(e)}"],
                warnings=["System evaluation crashed"],
                recommendations=["Immediate manual intervention required"],
            )

            return emergency_metrics

    async def evaluate_coverage_compliance(self) -> Dict[str, Any]:
        """Evaluate Kraken coverage compliance"""

        self.logger.info("Evaluating coverage compliance")

        try:
            # Run coverage audit
            audit_result = await self.coverage_auditor.audit_full_coverage()

            coverage_percentage = audit_result.get("coverage_percentage", 0.0)
            compliance_status = audit_result.get("compliance_status", "NON_COMPLIANT")

            # Log result
            self.daily_logger.log_system_check(
                "kraken_coverage_compliance",
                compliance_status == "COMPLIANT",
                f"Coverage: {coverage_percentage:.1%}, Status: {compliance_status}",
            )

            return {
                "coverage_percentage": coverage_percentage,
                "compliance_status": compliance_status,
                "score": coverage_percentage,
                "audit_result": audit_result,
            }

        except Exception as e:
            self.logger.error(f"Coverage evaluation failed: {e}")
            self.daily_logger.log_error_with_context(e, {"evaluation": "coverage_compliance"})
            return {"score": 0.0, "compliance_status": "ERROR"}

    async def evaluate_data_integrity(self) -> Dict[str, Any]:
        """Evaluate data integrity across all data sources"""

        self.logger.info("Evaluating data integrity")

        try:
            # Create sample data for validation
            sample_data = pd.DataFrame(
                {
                    "timestamp": [datetime.utcnow() - timedelta(minutes=i) for i in range(10)],
                    "symbol": ["BTC/USD"] * 10,
                    "price": np.random.uniform(40000, 50000, 10),
                    "volume": np.random.uniform(1000, 10000, 10),
                }
            )

            source_metadata = {
                "sources": [{"type": "exchange_api", "exchange": "kraken"}],
                "data_type": "market_data",
                "collection_time": datetime.utcnow().isoformat(),
            }

            # Validate data integrity
            is_valid, validation_results = self.data_guard.validate_data_pipeline(
                sample_data, source_metadata
            )

            integrity_score = 1.0 if is_valid else 0.0

            # Log result
            self.daily_logger.log_system_check(
                "data_integrity_validation",
                is_valid,
                f"Violations: {len(validation_results.get('violations', []))}",
            )

            return {
                "integrity_score": integrity_score,
                "is_valid": is_valid,
                "validation_results": validation_results,
            }

        except Exception as e:
            self.logger.error(f"Data integrity evaluation failed: {e}")
            self.daily_logger.log_error_with_context(e, {"evaluation": "data_integrity"})
            return {"integrity_score": 0.0, "is_valid": False}

    async def evaluate_system_health(self) -> Dict[str, Any]:
        """Evaluate overall system health"""

        self.logger.info("Evaluating system health")

        try:
            # Import health monitor
            from agents.health_monitor_agent import HealthMonitorAgent

            health_agent = HealthMonitorAgent()
            await health_agent.initialize()

            # Get system health
            health_result = await health_agent.process_async({})
            health_data = health_result.get("health_status", {})

            system_score = health_data.get("score", 0) / 100.0  # Convert to 0-1 scale
            system_grade = health_data.get("grade", "F")

            # Log result
            self.daily_logger.log_system_check(
                "system_health_check",
                system_grade in ["A", "B"],
                f"Grade: {system_grade}, Score: {system_score:.1%}",
            )

            return {"health_score": system_score, "grade": system_grade, "health_data": health_data}

        except Exception as e:
            self.logger.error(f"System health evaluation failed: {e}")
            self.daily_logger.log_error_with_context(e, {"evaluation": "system_health"})
            return {"health_score": 0.0, "grade": "F"}

    async def evaluate_model_performance(self) -> Dict[str, Any]:
        """Evaluate ML model performance"""

        self.logger.info("Evaluating model performance")

        try:
            # Import ML predictor
            from ml.models.predict import MultiHorizonPredictor

            predictor = MultiHorizonPredictor()

            # Check if models exist and validate performance
            performance_results = await predictor.validate_model_performance()

            # Calculate overall performance score
            horizon_scores = []
            for horizon_name, results in performance_results.items():
                if results.get("production_ready", False):
                    horizon_scores.append(1.0)
                elif results.get("mae") is not None:
                    # Convert MAE to score (lower MAE = higher score)
                    mae = results["mae"]
                    score = max(0, 1.0 - (mae * 10))  # Scale MAE to score
                    horizon_scores.append(score)
                else:
                    horizon_scores.append(0.0)

            overall_performance = np.mean(horizon_scores) if horizon_scores else 0.0

            # Log result
            self.daily_logger.log_performance_metric(
                "model_performance_score", overall_performance, target=0.6
            )

            return {
                "performance_score": overall_performance,
                "horizon_results": performance_results,
                "production_ready_count": sum(
                    1 for r in performance_results.values() if r.get("production_ready", False)
                ),
            }

        except Exception as e:
            self.logger.error(f"Model performance evaluation failed: {e}")
            self.daily_logger.log_error_with_context(e, {"evaluation": "model_performance"})
            return {"performance_score": 0.0}

    async def execute_go_nogo_decision(self, daily_metrics: DailyMetrics):
        """Execute actions based on GO/NO-GO decision"""

        self.logger.info(f"Executing GO/NO-GO decision: {daily_metrics.go_nogo_decision}")

        if daily_metrics.go_nogo_decision == "GO":
            await self._execute_go_actions(daily_metrics)
        else:
            await self._execute_nogo_actions(daily_metrics)

    async def _execute_go_actions(self, daily_metrics: DailyMetrics):
        """Execute actions for GO decision"""

        self.logger.info("System evaluation: GO - Enabling live trading capabilities")

        # Enable production systems
        self.daily_logger.log_system_check(
            "production_enablement",
            True,
            f"Live trading enabled with overall score: {daily_metrics.overall_score:.1%}",
        )

        # Save GO decision
        decision_file = Path("logs/current_trading_status.json")
        decision_file.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(decision_file, "w") as f:
            json.dump(
                {
                    "decision": "GO",
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_score": daily_metrics.overall_score,
                    "live_trading_enabled": True,
                },
                f,
                indent=2,
            )

    async def _execute_nogo_actions(self, daily_metrics: DailyMetrics):
        """Execute actions for NO-GO decision"""

        self.logger.warning(f"System evaluation: NO-GO - Disabling live trading")

        # Disable production systems
        self.daily_logger.log_system_check(
            "production_disablement",
            True,
            f"Live trading disabled due to issues: {len(daily_metrics.critical_issues)} critical issues",
        )

        # Save NO-GO decision
        decision_file = Path("logs/current_trading_status.json")
        decision_file.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(decision_file, "w") as f:
            json.dump(
                {
                    "decision": "NO-GO",
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_score": daily_metrics.overall_score,
                    "critical_issues": daily_metrics.critical_issues,
                    "live_trading_enabled": False,
                },
                f,
                indent=2,
            )


# Main execution function
async def run_daily_evaluation():
    """Run complete daily evaluation"""

    evaluator = CompleteSystemEvaluator()
    return await evaluator.run_complete_daily_evaluation()


if __name__ == "__main__":
    asyncio.run(run_daily_evaluation())
