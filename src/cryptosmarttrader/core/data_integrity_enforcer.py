#!/usr/bin/env python3
"""
Data Integrity Enforcer - Zero Tolerance for Synthetic/Fallback Data
Production data pipeline blocker for non-authentic data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from config.daily_logging_config import get_daily_logger

class DataIntegrityViolation(Exception):
    """Exception raised when data integrity is violated"""
    pass

class ProductionDataGuard:
    """Production-grade data integrity enforcement"""

    def __init__(self, production_mode: bool = True):
        self.logger = get_logger("DataIntegrityEnforcer")
        self.daily_logger = get_daily_logger()
        self.production_mode = production_mode

        # Zero tolerance settings
        self.allow_synthetic_data = not production_mode
        self.allow_fallback_data = not production_mode
        self.allow_interpolated_data = not production_mode
        self.allow_nan_values = False  # Never allowed

        # Validation rules
        self.validation_rules = {
            "required_fields": ["timestamp", "symbol", "price", "volume"],
            "max_missing_percentage": 0.0,  # 0% missing data allowed
            "min_data_freshness_hours": 1,
            "required_sources": ["exchange_api"],
            "forbidden_sources": ["synthetic", "fallback", "interpolated", "mock", "placeholder"]
        }

        self.logger.info(f"Data integrity enforcer initialized",
                        production_mode=production_mode,
                        zero_tolerance=True)

    def validate_data_pipeline(self, data: pd.DataFrame, source_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validate entire data pipeline for integrity violations"""

        validation_start = datetime.utcnow()
        self.logger.info("Starting zero-tolerance data validation")

        validation_results = {
            "validation_timestamp": validation_start.isoformat(),
            "production_mode": self.production_mode,
            "data_shape": data.shape,
            "source_metadata": source_metadata,
            "violations": [],
            "passed_checks": [],
            "overall_status": "UNKNOWN"
        }

        try:
            # 1. Source Authenticity Check
            source_check = self._validate_data_sources(source_metadata)
            if not source_check["passed"]:
                validation_results["violations"].extend(source_check["violations"])
            else:
                validation_results["passed_checks"].append("source_authenticity")

            # 2. Data Completeness Check
            completeness_check = self._validate_data_completeness(data)
            if not completeness_check["passed"]:
                validation_results["violations"].extend(completeness_check["violations"])
            else:
                validation_results["passed_checks"].append("data_completeness")

            # 3. Data Freshness Check
            freshness_check = self._validate_data_freshness(data)
            if not freshness_check["passed"]:
                validation_results["violations"].extend(freshness_check["violations"])
            else:
                validation_results["passed_checks"].append("data_freshness")

            # 4. Synthetic Data Detection
            synthetic_check = self._detect_synthetic_data(data)
            if not synthetic_check["passed"]:
                validation_results["violations"].extend(synthetic_check["violations"])
            else:
                validation_results["passed_checks"].append("synthetic_detection")

            # 5. NaN/Missing Value Check
            nan_check = self._validate_no_nan_values(data)
            if not nan_check["passed"]:
                validation_results["violations"].extend(nan_check["violations"])
            else:
                validation_results["passed_checks"].append("no_nan_values")

            # Overall validation result
            has_violations = len(validation_results["violations"]) > 0
            validation_results["overall_status"] = "FAILED" if has_violations else "PASSED"

            # Production blocking logic
            if self.production_mode and has_violations:
                self._block_production_pipeline(validation_results)

            # Log validation results
            self.daily_logger.log_system_check(
                "data_integrity_validation",
                not has_violations,
                f"Violations: {len(validation_results['violations'])}, Passed: {len(validation_results['passed_checks'])}"
            )

            validation_time = (datetime.utcnow() - validation_start).total_seconds()
            validation_results["validation_duration_seconds"] = validation_time

            self.logger.info(f"Data validation completed: {validation_results['overall_status']}")

            return not has_violations, validation_results

        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            validation_results["violations"].append({
                "type": "VALIDATION_ERROR",
                "severity": "CRITICAL",
                "message": f"Validation process failed: {str(e)}"
            })
            validation_results["overall_status"] = "ERROR"

            return False, validation_results

    def _validate_data_sources(self, source_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that all data comes from authentic sources"""

        violations = []

        data_sources = source_metadata.get("sources", [])

        for source in data_sources:
            source_type = source.get("type", "unknown")

            # Check for forbidden sources
            if source_type.lower() in self.validation_rules["forbidden_sources"]:
                violations.append({
                    "type": "FORBIDDEN_SOURCE",
                    "severity": "CRITICAL",
                    "message": f"Forbidden data source detected: {source_type}",
                    "source": source
                })

            # Check for required authentic sources
            if source_type.lower() not in self.validation_rules["required_sources"]:
                violations.append({
                    "type": "NON_AUTHENTIC_SOURCE",
                    "severity": "CRITICAL",
                    "message": f"Non-authentic data source: {source_type}",
                    "source": source
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "sources_checked": len(data_sources)
        }

    def _validate_data_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data completeness against zero-tolerance standards"""

        violations = []

        # Check required fields
        missing_fields = set(self.validation_rules["required_fields"]) - set(data.columns)
        if missing_fields:
            violations.append({
                "type": "MISSING_REQUIRED_FIELDS",
                "severity": "CRITICAL",
                "message": f"Required fields missing: {list(missing_fields)}",
                "missing_fields": list(missing_fields)
            })

        # Check for missing data percentage
        for field in self.validation_rules["required_fields"]:
            if field in data.columns:
                missing_percentage = data[field].isnull().sum() / len(data)
                if missing_percentage > self.validation_rules["max_missing_percentage"]:
                    violations.append({
                        "type": "EXCESSIVE_MISSING_DATA",
                        "severity": "CRITICAL",
                        "message": f"Field '{field}' has {missing_percentage:.1%} missing data (max allowed: {self.validation_rules['max_missing_percentage']:.1%})",
                        "field": field,
                        "missing_percentage": missing_percentage
                    })

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "total_records": len(data)
        }

    def _validate_data_freshness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data freshness requirements"""

        violations = []

        if "timestamp" in data.columns:
            try:
                # Convert to datetime if needed
                timestamps = pd.to_datetime(data["timestamp"])
                latest_timestamp = timestamps.max()
                current_time = datetime.utcnow()

                # Check data freshness
                hours_old = (current_time - latest_timestamp).total_seconds() / 3600
                max_hours = self.validation_rules["min_data_freshness_hours"]

                if hours_old > max_hours:
                    violations.append({
                        "type": "STALE_DATA",
                        "severity": "CRITICAL",
                        "message": f"Data is {hours_old:.1f} hours old (max allowed: {max_hours} hours)",
                        "hours_old": hours_old,
                        "latest_timestamp": latest_timestamp.isoformat()
                    })

            except Exception as e:
                violations.append({
                    "type": "TIMESTAMP_VALIDATION_ERROR",
                    "severity": "CRITICAL",
                    "message": f"Cannot validate timestamp freshness: {str(e)}"
                })

        return {
            "passed": len(violations) == 0,
            "violations": violations
        }

    def _detect_synthetic_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect synthetic/generated data patterns"""

        violations = []

        # Check for perfectly regular patterns (synthetic indicator)
        if "price" in data.columns:
            prices = data["price"].dropna()
            if len(prices) > 10:
                # Check for unrealistic price patterns
                price_changes = prices.diff().dropna()

                # Detect constant changes (synthetic pattern)
                if len(price_changes.unique()) == 1 and len(price_changes) > 5:
                    violations.append({
                        "type": "SYNTHETIC_PRICE_PATTERN",
                        "severity": "CRITICAL",
                        "message": "Detected synthetic price pattern (constant changes)",
                        "pattern": "constant_changes"
                    })

                # Detect unrealistic volatility patterns
                price_volatility = price_changes.std()
                if price_volatility == 0 and len(price_changes) > 3:
                    violations.append({
                        "type": "ZERO_VOLATILITY",
                        "severity": "CRITICAL",
                        "message": "Detected zero price volatility (likely synthetic)",
                        "volatility": price_volatility
                    })

        # Check for placeholder values
        for column in data.columns:
            if data[column].dtype in ['object', 'string']:
                placeholder_values = ['placeholder', 'dummy', 'test', 'fake', 'synthetic', 'mock']
                for placeholder in placeholder_values:
                    if data[column].astype(str).str.contains(placeholder, case=False, na=False).any():
                        violations.append({
                            "type": "PLACEHOLDER_VALUES",
                            "severity": "CRITICAL",
                            "message": f"Placeholder values detected in column '{column}': {placeholder}",
                            "column": column,
                            "placeholder": placeholder
                        })

        return {
            "passed": len(violations) == 0,
            "violations": violations
        }

    def _validate_no_nan_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate zero NaN/missing values in critical fields"""

        violations = []

        critical_fields = ["price", "volume", "timestamp", "symbol"]

        for field in critical_fields:
            if field in data.columns:
                nan_count = data[field].isnull().sum()
                if nan_count > 0:
                    violations.append({
                        "type": "NAN_VALUES_DETECTED",
                        "severity": "CRITICAL",
                        "message": f"NaN values detected in critical field '{field}': {nan_count} occurrences",
                        "field": field,
                        "nan_count": nan_count
                    })

        return {
            "passed": len(violations) == 0,
            "violations": violations
        }

    def _block_production_pipeline(self, validation_results: Dict[str, Any]) -> None:
        """Block production pipeline when violations are detected"""

        self.logger.error("PRODUCTION PIPELINE BLOCKED - Data integrity violations detected")

        # Log critical alert
        self.daily_logger.log_security_event(
            "PRODUCTION_PIPELINE_BLOCKED",
            "CRITICAL",
            f"Data integrity violations: {len(validation_results['violations'])} issues detected"
        )

        # Save violation report
        self._save_violation_report(validation_results)

        # Raise exception to halt pipeline
        raise DataIntegrityViolation(
            f"Production pipeline blocked due to {len(validation_results['violations'])} data integrity violations"
        )

    def _save_violation_report(self, validation_results: Dict[str, Any]) -> None:
        """Save detailed violation report"""

        try:
            reports_dir = Path("logs/data_integrity_violations")
            reports_dir.mkdir(parents=True, exist_ok=True)

            report_file = reports_dir / f"violation_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

            with open(report_file, 'w') as f:
                json.dump(validation_results, f, indent=2)

            self.logger.info(f"Violation report saved: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to save violation report: {e}")

# Global instance for production use
production_guard = ProductionDataGuard(production_mode=True)

def validate_production_data(data: pd.DataFrame, source_metadata: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Global function for production data validation"""
    return production_guard.validate_data_pipeline(data, source_metadata)
