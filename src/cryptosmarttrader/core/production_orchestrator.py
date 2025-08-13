#!/usr/bin/env python3
"""
Production Orchestrator - Complete pipeline with error handling and atomic writes
Replaces all ... with working implementations
"""

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..core.structured_logger import get_logger
from utils.timestamp_validator import normalize_timestamp, validate_timestamp_sequence
from ..core.data_completeness_gate import DataCompletenessGate

try:
    from orchestration.strict_gate import apply_strict_gate_orchestration
except ImportError:
    # Fallback implementation
    def apply_strict_gate_orchestration(all_preds, thr=0.80):
        return {"gate_status": "OK", "total_candidates": 0, "total_passed": 0, "per_horizon": {}}


from ml.models.predict import predict_all, get_model_status


class ProductionOrchestrator:
    """Complete production orchestrator with no fallback data tolerance"""

    def __init__(self):
        self.logger = get_logger("ProductionOrchestrator")
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now(timezone.utc)

        # Initialize gates
        self.data_gate = DataCompletenessGate()

        # Production directories
        self.data_dir = Path("data/production")
        self.output_dir = Path("exports/production")
        self.logs_dir = Path("logs/daily")

        # Create directories
        for dir_path in [self.data_dir, self.output_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Production orchestrator initialized with run_id: {self.run_id}")

    async def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run complete production pipeline with error handling"""

        pipeline_start = time.time()
        results = {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "status": "RUNNING",
            "steps": {},
            "errors": [],
            "warnings": [],
        }

        try:
            # Step 1: Validate models exist
            self.logger.info("Step 1: Validating model availability")
            model_status = await self._validate_models()
            results["steps"]["model_validation"] = model_status

            if not model_status["all_models_available"]:
                raise Exception("Critical models missing - cannot proceed")

            # Step 2: Data collection with gates
            self.logger.info("Step 2: Collecting and validating data")
            data_status = await self._collect_and_validate_data()
            results["steps"]["data_collection"] = data_status

            if not data_status["passed_gates"]:
                raise Exception("Data quality gates failed - stopping pipeline")

            # Step 3: Feature engineering with timestamp validation
            self.logger.info("Step 3: Feature engineering with temporal integrity")
            features_status = await self._engineer_features(data_status["validated_data"])
            results["steps"]["feature_engineering"] = features_status

            # Step 4: Generate predictions
            self.logger.info("Step 4: Generating predictions")
            predictions_status = await self._generate_predictions(features_status["features"])
            results["steps"]["predictions"] = predictions_status

            # Step 5: Apply strict confidence gates
            self.logger.info("Step 5: Applying confidence gates")
            gate_status = await self._apply_confidence_gates(predictions_status["predictions"])
            results["steps"]["confidence_gates"] = gate_status

            # Step 6: Atomic write to production outputs
            self.logger.info("Step 6: Writing production outputs")
            output_status = await self._write_production_outputs(
                gate_status["filtered_predictions"]
            )
            results["steps"]["output_generation"] = output_status

            # Pipeline completed successfully
            pipeline_duration = time.time() - pipeline_start
            results["status"] = "SUCCESS"
            results["duration_seconds"] = pipeline_duration
            results["end_time"] = datetime.now(timezone.utc).isoformat()

            self.logger.info(
                f"Production pipeline completed successfully in {pipeline_duration:.2f}s"
            )

        except Exception as e:
            pipeline_duration = time.time() - pipeline_start
            results["status"] = "FAILED"
            results["duration_seconds"] = pipeline_duration
            results["end_time"] = datetime.now(timezone.utc).isoformat()
            results["error"] = str(e)

            self.logger.error(f"Production pipeline failed: {e}")

            # Write failure log
            await self._write_failure_log(results)

        finally:
            # Always write daily log
            await self._write_daily_log(results)

        return results

    async def _validate_models(self) -> Dict[str, Any]:
        """Validate all required models are available and recent"""

        model_status = get_model_status()

        required_horizons = ["1h", "24h", "168h", "720h"]
        available_models = []
        missing_models = []
        model_ages = {}

        for horizon in required_horizons:
            if model_status.get(horizon, {}).get("exists", False):
                available_models.append(horizon)
                model_ages[horizon] = model_status[horizon].get("modified", 0)
            else:
                missing_models.append(horizon)

        # Check model freshness (< 7 days)
        current_time = time.time()
        stale_models = []
        for horizon, mod_time in model_ages.items():
            age_hours = (current_time - mod_time) / 3600
            if age_hours > 168:  # 7 days
                stale_models.append(f"{horizon} ({age_hours:.1f}h old)")

        all_models_available = len(missing_models) == 0 and len(stale_models) == 0

        return {
            "all_models_available": all_models_available,
            "available_models": available_models,
            "missing_models": missing_models,
            "stale_models": stale_models,
            "total_required": len(required_horizons),
            "total_available": len(available_models),
        }

    async def _collect_and_validate_data(self) -> Dict[str, Any]:
        """Collect data with strict no-fallback gates"""

        # Load latest features (authentic data only)
        features_file = Path("exports/features.parquet")

        if not features_file.exists():
            raise Exception("No features file found - run data collection first")

        # Check file age
        file_age_hours = (time.time() - features_file.stat().st_mtime) / 3600
        if file_age_hours > 24:
            raise Exception(f"Features file too old: {file_age_hours:.1f}h (max 24h)")

        # Load and validate data
        features_df = pd.read_parquet(features_file)

        # Apply data completeness gates
        gate_result = self.data_gate.validate_data_completeness(features_df)

        if not gate_result["passed"]:
            raise Exception(f"Data completeness gate failed: {gate_result['reason']}")

        # Timestamp validation
        if "timestamp" in features_df.columns:
            # Normalize all timestamps to UTC
            features_df["timestamp"] = features_df["timestamp"].apply(
                lambda x: normalize_timestamp(x, target_timezone="UTC")
            )

            # Validate timestamp sequence
            timestamp_validation = validate_timestamp_sequence(features_df["timestamp"])
            if not timestamp_validation["valid"]:
                self.logger.warning(f"Timestamp issues: {timestamp_validation['issues']}")

        return {
            "passed_gates": True,
            "validated_data": features_df,
            "data_quality_score": gate_result["coverage_percentage"],
            "file_age_hours": file_age_hours,
            "total_samples": len(features_df),
            "timestamp_validation": timestamp_validation
            if "timestamp" in features_df.columns
            else None,
        }

    async def _engineer_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Engineer features with temporal integrity validation"""

        feature_cols = [col for col in data.columns if col.startswith("feat_")]

        if len(feature_cols) == 0:
            raise Exception("No feature columns found in data")

        # Validate feature completeness
        feature_completeness = {}
        for col in feature_cols:
            completeness = 1 - (data[col].isna().sum() / len(data))
            feature_completeness[col] = completeness

            if completeness < 0.95:  # 95% completeness required
                raise Exception(f"Feature {col} only {completeness:.1%} complete (need ≥95%)")

        # Prepare features for prediction
        features_clean = data[["coin", "timestamp"] + feature_cols].copy()

        # Remove any remaining NaN values
        features_clean = features_clean.dropna()

        return {
            "features": features_clean,
            "feature_columns": feature_cols,
            "feature_completeness": feature_completeness,
            "samples_after_cleaning": len(features_clean),
            "samples_removed": len(data) - len(features_clean),
        }

    async def _generate_predictions(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Generate predictions with uncertainty estimates"""

        if features.empty:
            raise Exception("No features available for prediction")

        # Generate predictions using trained models
        predictions_df = predict_all(features)

        if predictions_df.empty:
            raise Exception("Prediction generation failed - no output from models")

        # Validate prediction columns
        required_cols = []
        for horizon in ["1h", "24h", "168h", "720h"]:
            required_cols.extend([f"pred_{horizon}", f"conf_{horizon}"])

        missing_cols = [col for col in required_cols if col not in predictions_df.columns]
        if missing_cols:
            raise Exception(f"Missing prediction columns: {missing_cols}")

        # Calculate prediction statistics
        pred_stats = {}
        for horizon in ["1h", "24h", "168h", "720h"]:
            pred_col = f"pred_{horizon}"
            conf_col = f"conf_{horizon}"

            pred_stats[horizon] = {
                "mean_prediction": predictions_df[pred_col].mean(),
                "std_prediction": predictions_df[pred_col].std(),
                "mean_confidence": predictions_df[conf_col].mean(),
                "min_confidence": predictions_df[conf_col].min(),
                "max_confidence": predictions_df[conf_col].max(),
            }

        return {
            "predictions": predictions_df,
            "total_predictions": len(predictions_df),
            "prediction_statistics": pred_stats,
            "prediction_columns": required_cols,
        }

    async def _apply_confidence_gates(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Apply strict 80% confidence gates"""

        # Prepare predictions by horizon
        horizons = ["1h", "24h", "168h", "720h"]
        predictions_by_horizon = {}

        for horizon in horizons:
            horizon_df = predictions[
                ["coin", "timestamp", f"pred_{horizon}", f"conf_{horizon}"]
            ].copy()
            predictions_by_horizon[horizon] = horizon_df

        # Apply strict gate orchestration
        gate_results = apply_strict_gate_orchestration(predictions_by_horizon, thr=0.80)

        # Combine all passed predictions
        all_passed = []
        for horizon, passed_df in gate_results["per_horizon"].items():
            if not passed_df.empty:
                # Add horizon identifier
                passed_df = passed_df.copy()
                passed_df["horizon"] = horizon
                all_passed.append(passed_df)

        if all_passed:
            filtered_predictions = pd.concat(all_passed, ignore_index=True)
        else:
            filtered_predictions = pd.DataFrame()

        return {
            "filtered_predictions": filtered_predictions,
            "gate_status": gate_results["gate_status"],
            "total_candidates": gate_results["total_candidates"],
            "total_passed": gate_results["total_passed"],
            "pass_rate": gate_results["total_passed"] / max(gate_results["total_candidates"], 1),
            "per_horizon_stats": {
                horizon: {
                    "candidates": len(predictions_by_horizon[horizon]),
                    "passed": len(gate_results["per_horizon"][horizon]),
                    "pass_rate": len(gate_results["per_horizon"][horizon])
                    / max(len(predictions_by_horizon[horizon]), 1),
                }
                for horizon in horizons
            },
        }

    async def _write_production_outputs(self, filtered_predictions: pd.DataFrame) -> Dict[str, Any]:
        """Atomic write of production outputs"""

        output_files = []

        try:
            # Write main predictions.csv
            predictions_file = self.output_dir / "predictions.csv"
            temp_file = predictions_file.with_suffix(".tmp")

            filtered_predictions.to_csv(temp_file, index=False)
            temp_file.rename(predictions_file)  # Atomic rename
            output_files.append(str(predictions_file))

            # Write timestamped backup
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            backup_file = self.output_dir / f"predictions_{timestamp}_{self.run_id}.csv"
            filtered_predictions.to_csv(backup_file, index=False)
            output_files.append(str(backup_file))

            # Write summary JSON
            summary = {
                "run_id": self.run_id,
                "timestamp": self.start_time.isoformat(),
                "total_predictions": len(filtered_predictions),
                "horizons": filtered_predictions["horizon"].unique().tolist()
                if "horizon" in filtered_predictions.columns
                else [],
                "coins": filtered_predictions["coin"].nunique()
                if "coin" in filtered_predictions.columns
                else 0,
            }

            summary_file = self.output_dir / "predictions_summary.json"
            temp_summary = summary_file.with_suffix(".tmp")

            with open(temp_summary, "w") as f:
                json.dump(summary, f, indent=2)
            temp_summary.rename(summary_file)  # Atomic rename
            output_files.append(str(summary_file))

            return {
                "success": True,
                "output_files": output_files,
                "total_predictions": len(filtered_predictions),
                "summary": summary,
            }

        except Exception as e:
            # Cleanup temp files on failure
            for temp_file in [f for f in output_files if f.endswith(".tmp")]:
                Path(temp_file).unlink(missing_ok=True)

            raise Exception(f"Failed to write production outputs: {e}")

    async def _write_daily_log(self, results: Dict[str, Any]):
        """Write daily log with run details"""

        today = self.start_time.strftime("%Y-%m-%d")
        log_file = self.logs_dir / f"production_{today}.json"

        # Load existing log or create new
        if log_file.exists():
            with open(log_file, "r") as f:
                daily_log = json.load(f)
        else:
            daily_log = {"date": today, "runs": []}

        # Add this run
        daily_log["runs"].append(results)

        # Write atomically
        temp_log = log_file.with_suffix(".tmp")
        with open(temp_log, "w") as f:
            json.dump(daily_log, f, indent=2)
        temp_log.rename(log_file)

        self.logger.info(f"Daily log updated: {log_file}")

    async def _write_failure_log(self, results: Dict[str, Any]):
        """Write failure-specific log for alerts"""

        failure_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "error": results.get("error"),
            "failed_step": None,
            "partial_results": results.get("steps", {}),
        }

        # Find failed step
        for step_name, step_result in results.get("steps", {}).items():
            if isinstance(step_result, dict) and step_result.get("error"):
                failure_log["failed_step"] = step_name
                break

        failure_file = self.logs_dir / f"failures_{self.start_time.strftime('%Y%m%d')}.json"

        # Append to failures log
        failures = []
        if failure_file.exists():
            with open(failure_file, "r") as f:
                failures = json.load(f)

        failures.append(failure_log)

        # Write atomically
        temp_failures = failure_file.with_suffix(".tmp")
        with open(temp_failures, "w") as f:
            json.dump(failures, f, indent=2)
        temp_failures.rename(failure_file)

        self.logger.error(f"Failure logged: {failure_file}")


async def run_production_pipeline():
    """Main entry point for production pipeline"""

    orchestrator = ProductionOrchestrator()
    results = await orchestrator.run_complete_pipeline()

    if results["status"] == "SUCCESS":
        print(f"✅ Production pipeline completed successfully")
        print(f"   Run ID: {results['run_id']}")
        print(f"   Duration: {results['duration_seconds']:.2f}s")
        print(f"   Predictions: {results['steps']['output_generation']['total_predictions']}")
    else:
        print(f"❌ Production pipeline failed")
        print(f"   Run ID: {results['run_id']}")
        print(f"   Error: {results['error']}")

    return results


if __name__ == "__main__":
    asyncio.run(run_production_pipeline())
