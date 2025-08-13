#!/usr/bin/env python3
"""
Backend enforcement of 80% confidence gate and readiness checks
"""

import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class BackendEnforcement:
    """Enforce production-grade confidence gating and readiness"""

    def __init__(self, confidence_threshold: float = 80.0):
        self.confidence_threshold = confidence_threshold
        self.enforcement_log = []

    def enforce_confidence_gate(self, predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Enforce 80% confidence gate with backend validation"""

        enforcement_start = datetime.now()
        original_count = len(predictions_df)

        # Find confidence columns
        confidence_cols = [col for col in predictions_df.columns if col.startswith("confidence_")]

        if not confidence_cols:
            logger.warning("No confidence columns found - cannot enforce gate")
            return predictions_df, {
                "enforced": False,
                "reason": "no_confidence_columns",
                "original_count": original_count,
                "filtered_count": original_count,
            }

        # Apply confidence filtering
        filtered_predictions = []

        for _, row in predictions_df.iterrows():
            # Get maximum confidence across all horizons
            max_confidence = max([row.get(col, 0) for col in confidence_cols])

            # Enforce strict threshold
            if max_confidence >= self.confidence_threshold:
                # Add enforcement metadata
                row_dict = row.to_dict()
                row_dict["_enforcement_passed"] = True
                row_dict["_max_confidence"] = max_confidence
                row_dict["_enforcement_timestamp"] = enforcement_start.isoformat()

                filtered_predictions.append(row_dict)

        filtered_df = pd.DataFrame(filtered_predictions)
        filtered_count = len(filtered_df)

        # Log enforcement action
        enforcement_result = {
            "enforced": True,
            "threshold": self.confidence_threshold,
            "original_count": original_count,
            "filtered_count": filtered_count,
            "rejection_rate": (original_count - filtered_count) / original_count
            if original_count > 0
            else 0,
            "enforcement_timestamp": enforcement_start.isoformat(),
            "confidence_columns_used": confidence_cols,
        }

        self.enforcement_log.append(enforcement_result)

        logger.info(
            f"Confidence gate enforced: {filtered_count}/{original_count} predictions passed (≥{self.confidence_threshold}%)"
        )

        return filtered_df, enforcement_result

    def check_readiness_gate(self) -> Dict:
        """Check system readiness for production"""

        readiness_checks = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_ready": False,
            "go_no_go": "NO-GO",  # Default to NO-GO
        }

        # Check 1: Models available
        models_dir = Path("models/baseline")
        metadata_file = models_dir / "metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                models_count = len(metadata.get("models_trained", []))
                readiness_checks["checks"]["models"] = {
                    "status": "ready" if models_count > 0 else "not_ready",
                    "models_count": models_count,
                    "last_training": metadata.get("timestamp"),
                }
            except Exception as e:
                readiness_checks["checks"]["models"] = {"status": "error", "error": str(e)}
        else:
            readiness_checks["checks"]["models"] = {
                "status": "not_ready",
                "reason": "no_trained_models",
            }

        # Check 2: Recent data availability
        features_file = Path("data/processed/features.csv")

        if features_file.exists():
            try:
                # Check file age
                file_age_hours = (datetime.now().timestamp() - features_file.stat().st_mtime) / 3600

                readiness_checks["checks"]["data"] = {
                    "status": "ready" if file_age_hours < 24 else "stale",
                    "file_age_hours": file_age_hours,
                    "file_path": str(features_file),
                }
            except Exception as e:
                readiness_checks["checks"]["data"] = {"status": "error", "error": str(e)}
        else:
            readiness_checks["checks"]["data"] = {
                "status": "not_ready",
                "reason": "no_processed_data",
            }

        # Check 3: Predictions quality
        pred_file = Path("exports/production/predictions.parquet")
        if not pred_file.exists():
            pred_file = Path("exports/production/predictions.csv")

        if pred_file.exists():
            try:
                if pred_file.suffix == ".parquet":
                    pred_df = pd.read_parquet(pred_file)
                else:
                    pred_df = pd.read_csv(pred_file)

                # Check prediction quality
                confidence_cols = [col for col in pred_df.columns if col.startswith("confidence_")]

                if confidence_cols:
                    all_confidences = []
                    for col in confidence_cols:
                        all_confidences.extend(pred_df[col].dropna().tolist())

                    if all_confidences:
                        high_conf_rate = (
                            pd.Series(all_confidences) >= self.confidence_threshold
                        ).mean()

                        readiness_checks["checks"]["predictions"] = {
                            "status": "ready" if high_conf_rate > 0.1 else "low_quality",
                            "total_predictions": len(pred_df),
                            "high_confidence_rate": high_conf_rate,
                            "mean_confidence": pd.Series(all_confidences).mean(),
                        }
                    else:
                        readiness_checks["checks"]["predictions"] = {
                            "status": "not_ready",
                            "reason": "no_confidence_data",
                        }
                else:
                    readiness_checks["checks"]["predictions"] = {
                        "status": "not_ready",
                        "reason": "no_confidence_columns",
                    }
            except Exception as e:
                readiness_checks["checks"]["predictions"] = {"status": "error", "error": str(e)}
        else:
            readiness_checks["checks"]["predictions"] = {
                "status": "not_ready",
                "reason": "no_predictions_file",
            }

        # Determine overall readiness
        all_ready = all(
            check.get("status") == "ready" for check in readiness_checks["checks"].values()
        )

        readiness_checks["overall_ready"] = all_ready
        readiness_checks["go_no_go"] = "GO" if all_ready else "NO-GO"

        # Log readiness check
        logger.info(f"Readiness check: {readiness_checks['go_no_go']} - Ready: {all_ready}")

        return readiness_checks

    def save_enforcement_log(self):
        """Save enforcement log"""
        log_dir = Path("logs/enforcement")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"enforcement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        enforcement_data = {
            "timestamp": datetime.now().isoformat(),
            "confidence_threshold": self.confidence_threshold,
            "enforcement_log": self.enforcement_log,
        }

        with open(log_file, "w") as f:
            json.dump(enforcement_data, f, indent=2)

        logger.info(f"Enforcement log saved to {log_file}")


def apply_backend_enforcement(
    predictions_df: pd.DataFrame, confidence_threshold: float = 80.0
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Apply complete backend enforcement"""

    enforcer = BackendEnforcement(confidence_threshold)

    # 1. Enforce confidence gate
    filtered_df, gate_result = enforcer.enforce_confidence_gate(predictions_df)

    # 2. Check readiness
    readiness_result = enforcer.check_readiness_gate()

    # 3. Save enforcement log
    enforcer.save_enforcement_log()

    return filtered_df, gate_result, readiness_result


if __name__ == "__main__":
    # Test enforcement
    import pandas as pd

    # Create test predictions
    test_data = [
        {"coin": "BTC", "confidence_1h": 85, "confidence_24h": 90},
        {"coin": "ETH", "confidence_1h": 75, "confidence_24h": 82},
        {"coin": "ADA", "confidence_1h": 60, "confidence_24h": 70},
    ]

    test_df = pd.DataFrame(test_data)
    filtered_df, gate_result, readiness_result = apply_backend_enforcement(test_df)

    print("=== Backend Enforcement Test ===")
    print(f"Original: {len(test_df)}, Filtered: {len(filtered_df)}")
    print(f"Gate Result: {gate_result}")
    print(f"Readiness: {readiness_result['go_no_go']}")
