#!/usr/bin/env python3
"""
System Readiness Checker - Enterprise production readiness validation

Comprehensive system readiness assessment based on models, data, calibration,
and health status with consistent naming and robust validation logic.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    from core.consolidated_logging_manager import get_consolidated_logger
except ImportError:

    def get_consolidated_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class ReadinessStatus(Enum):
    """System readiness status levels"""

    READY = "ready"  # All systems go
    WARNING = "warning"  # Some issues but operational
    NOT_READY = "not_ready"  # Critical issues prevent operation
    UNKNOWN = "unknown"  # Cannot determine status


@dataclass
class ComponentReadiness:
    """Readiness status for individual system component"""

    component: str
    status: ReadinessStatus
    score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None


@dataclass
class SystemReadinessReport:
    """Complete system readiness assessment"""

    overall_status: ReadinessStatus
    overall_score: float
    go_no_go_decision: bool
    components: List[ComponentReadiness]
    timestamp: datetime
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SystemReadinessChecker:
    """
    Enterprise system readiness checker with consistent validation

    Provides comprehensive readiness assessment with consistent model naming,
    clean issue filtering, and robust health status validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system readiness checker

        Args:
            config: Optional readiness configuration
        """
        self.logger = get_consolidated_logger("SystemReadinessChecker")

        # Load configuration
        self.config = self._load_config(config)

        # Readiness state
        self.last_check: Optional[SystemReadinessReport] = None
        self.check_history: List[SystemReadinessReport] = []

        # Performance tracking
        self.check_count = 0
        self.total_check_time = 0.0

        self.logger.info("System Readiness Checker initialized with enterprise validation")

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load readiness checker configuration with enterprise defaults"""

        default_config = {
            # Model validation - CONSISTENT NAMING PATTERN
            "models": {
                "required_horizons": ["1h", "24h", "7d", "30d"],
                "model_directories": ["models", "ml/models", "data/models", "cache/models"],
                "naming_patterns": {
                    "xgboost": "{horizon}_xgb.pkl",  # Consistent: 1h_xgb.pkl, 24h_xgb.pkl
                    "tree": "{horizon}_tree.pkl",  # Consistent: 1h_tree.pkl, 24h_tree.pkl
                    "neural": "{horizon}_nn.pkl",  # Consistent: 1h_nn.pkl, 24h_nn.pkl
                    "ensemble": "{horizon}_ensemble.pkl",  # Consistent: 1h_ensemble.pkl
                },
                "minimum_model_age_hours": 1,  # Models must be at least 1 hour old
                "maximum_model_age_days": 7,  # Models must be younger than 7 days
            },
            # Data validation
            "data": {
                "required_files": ["data/market_data", "data/cache", "logs"],
                "minimum_data_age_hours": 1,  # Data must be recent
                "coverage_threshold": 0.8,  # 80% coverage required
                "integrity_violation_threshold": 5,  # Max 5 recent violations
            },
            # Calibration validation
            "calibration": {
                "confidence_threshold": 0.7,  # 70% minimum confidence
                "calibration_files": ["data/calibration.json", "models/calibration_metrics.json"],
                "drift_threshold": 0.1,  # 10% maximum drift
            },
            # Health status validation - ROBUST FILE HANDLING
            "health": {
                "required_files": ["health_status.json", "logs/current_trading_status.json"],
                "health_score_threshold": 70.0,  # Minimum health score
                "trading_status_required": True,  # Trading status must be available
                "fallback_behavior": "strict",  # strict/lenient fallback
            },
            # Overall readiness criteria
            "readiness": {
                "minimum_overall_score": 75.0,  # 75% minimum for GO
                "critical_components": [  # Must be ready
                    "models",
                    "data",
                    "health",
                ],
                "warning_threshold": 60.0,  # Warning below 60%
            },
        }

        if config:
            self._deep_merge_dict(default_config, config)

        return default_config

    def check_system_readiness(self) -> SystemReadinessReport:
        """
        Perform comprehensive system readiness check

        Returns:
            SystemReadinessReport with complete assessment
        """

        start_time = datetime.now(timezone.utc)
        self.logger.info("Starting system readiness assessment")

        try:
            components = []

            # Check model readiness
            model_readiness = self._check_model_readiness()
            components.append(model_readiness)

            # Check data readiness
            data_readiness = self._check_data_readiness()
            components.append(data_readiness)

            # Check calibration readiness
            calibration_readiness = self._check_calibration_readiness()
            components.append(calibration_readiness)

            # Check health status
            health_readiness = self._check_health_readiness()
            components.append(health_readiness)

            # Calculate overall readiness
            overall_score = self._calculate_overall_score(components)
            overall_status = self._determine_overall_status(overall_score, components)
            go_no_go = self._make_go_no_go_decision(overall_score, components)

            # Generate recommendations
            recommendations = self._generate_recommendations(components, overall_status)

            # Create report
            report = SystemReadinessReport(
                overall_status=overall_status,
                overall_score=overall_score,
                go_no_go_decision=go_no_go,
                components=components,
                timestamp=start_time,
                summary=self._generate_summary(overall_status, overall_score, go_no_go),
                recommendations=recommendations,
                metadata={
                    "check_duration_seconds": (
                        datetime.now(timezone.utc) - start_time
                    ).total_seconds(),
                    "components_checked": len(components),
                    "config_version": "2.0.0",
                },
            )

            # Update state
            self.last_check = report
            self.check_history.append(report)
            self._cleanup_history()

            # Update performance metrics
            self.check_count += 1
            self.total_check_time += report.metadata["check_duration_seconds"]

            decision = "GO" if go_no_go else "NO-GO"
            self.logger.info(
                f"Readiness check completed: {overall_status.value} ({overall_score:.1f}%) - {decision}"
            )

            return report

        except Exception as e:
            self.logger.error(f"Readiness check failed: {e}")
            return self._create_emergency_report(str(e))

    def _check_model_readiness(self) -> ComponentReadiness:
        """Check model availability and consistency with CONSISTENT NAMING"""

        try:
            model_config = self.config["models"]
            required_horizons = model_config["required_horizons"]
            naming_patterns = model_config["naming_patterns"]

            total_models_found = 0
            horizon_coverage = {}
            model_details = {}
            issues = []

            # Check each model directory
            for model_dir_str in model_config["model_directories"]:
                model_dir = Path(model_dir_str)
                if not model_dir.exists():
                    continue

                model_details[model_dir_str] = {}

                # Check each required horizon with CONSISTENT NAMING
                for horizon in required_horizons:
                    horizon_models = {}

                    # Check each model type with consistent naming pattern
                    for model_type, pattern in naming_patterns.items():
                        model_filename = pattern.format(horizon=horizon)
                        model_path = model_dir / model_filename

                        if model_path.exists():
                            # Validate model age
                            model_age = self._get_file_age_hours(model_path)
                            min_age = model_config["minimum_model_age_hours"]
                            max_age = model_config["maximum_model_age_days"] * 24

                            if min_age <= model_age <= max_age:
                                horizon_models[model_type] = {
                                    "path": str(model_path),
                                    "age_hours": model_age,
                                    "size_mb": model_path.stat().st_size / (1024 * 1024),
                                }
                                total_models_found += 1
                            else:
                                # Add issue for age problems
                                if model_age < min_age:
                                    issues.append(
                                        f"{model_filename} too recent ({model_age:.1f}h < {min_age}h)"
                                    )
                                else:
                                    issues.append(
                                        f"{model_filename} too old ({model_age:.1f}h > {max_age}h)"
                                    )

                    if horizon_models:
                        horizon_coverage[horizon] = horizon_models
                        model_details[model_dir_str][horizon] = horizon_models

            # Calculate coverage statistics
            horizons_covered = len(horizon_coverage)
            total_horizons = len(required_horizons)
            coverage_percentage = horizons_covered / total_horizons if total_horizons > 0 else 0.0

            # Generate clean issues list - NO EMPTY STRINGS
            model_issues = []

            if total_models_found == 0:
                model_issues.append("No valid models found in any directory")

            if coverage_percentage < 1.0:
                missing_horizons = [h for h in required_horizons if h not in horizon_coverage]
                model_issues.append(f"Missing models for horizons: {missing_horizons}")

            # Add age-related issues
            model_issues.extend(issues)

            # Calculate score
            score = min(100.0, (coverage_percentage * 70) + (min(total_models_found, 10) * 3))

            # Determine status
            if score >= 80 and coverage_percentage >= 0.8:
                status = ReadinessStatus.READY
            elif score >= 60:
                status = ReadinessStatus.WARNING
            else:
                status = ReadinessStatus.NOT_READY

            return ComponentReadiness(
                component="models",
                status=status,
                score=score,
                issues=model_issues,
                details={
                    "total_models": total_models_found,
                    "horizons_covered": horizons_covered,
                    "total_horizons": total_horizons,
                    "coverage_percentage": coverage_percentage,
                    "horizon_coverage": horizon_coverage,
                    "model_details": model_details,
                    "naming_patterns": naming_patterns,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            self.logger.error(f"Model readiness check failed: {e}")
            return ComponentReadiness(
                component="models",
                status=ReadinessStatus.NOT_READY,
                score=0.0,
                issues=[f"Model check failed: {str(e)}"],
                last_check=datetime.now(timezone.utc),
            )

    def _check_data_readiness(self) -> ComponentReadiness:
        """Check data availability and completeness with CLEAN ISSUE FILTERING"""

        try:
            data_config = self.config["data"]
            required_files = data_config["required_files"]
            min_age_hours = data_config["minimum_data_age_hours"]
            coverage_threshold = data_config["coverage_threshold"]
            violation_threshold = data_config["integrity_violation_threshold"]

            # Check required data files
            existing_files = 0
            recent_data_files = 0
            file_details = {}

            for required_file in required_files:
                file_path = Path(required_file)

                if file_path.exists():
                    existing_files += 1
                    age_hours = self._get_file_age_hours(file_path)

                    if age_hours <= min_age_hours:
                        recent_data_files += 1

                    file_details[required_file] = {
                        "exists": True,
                        "age_hours": age_hours,
                        "size_mb": self._get_directory_size_mb(file_path)
                        if file_path.is_dir()
                        else file_path.stat().st_size / (1024 * 1024),
                    }
                else:
                    file_details[required_file] = {"exists": False}

            # Check data coverage (mock calculation for now - would be replaced with real logic)
            coverage_percentage = (
                min(1.0, existing_files / len(required_files)) if required_files else 0.0
            )
            coverage_compliance = coverage_percentage >= coverage_threshold

            # Check integrity violations (would integrate with real integrity system)
            recent_violations = 0  # Would be populated from actual integrity logs

            # Generate clean issues list - FILTER OUT NONE VALUES
            data_issues = [
                issue
                for issue in [
                    "No recent data files found" if recent_data_files == 0 else None,
                    f"Data coverage non-compliant: {coverage_percentage:.1%} < {coverage_threshold:.1%}"
                    if not coverage_compliance
                    else None,
                    f"Recent integrity violations detected: {recent_violations}"
                    if recent_violations > violation_threshold
                    else None,
                    f"Missing required files: {len(required_files) - existing_files}/{len(required_files)}"
                    if existing_files < len(required_files)
                    else None,
                ]
                if issue is not None
            ]

            # Calculate score
            file_score = (existing_files / len(required_files)) * 40 if required_files else 0
            recency_score = (recent_data_files / len(required_files)) * 30 if required_files else 0
            coverage_score = coverage_percentage * 20
            integrity_score = max(0, 10 - (recent_violations * 2))

            score = file_score + recency_score + coverage_score + integrity_score

            # Determine status
            if score >= 80 and coverage_compliance and recent_violations <= violation_threshold:
                status = ReadinessStatus.READY
            elif score >= 60:
                status = ReadinessStatus.WARNING
            else:
                status = ReadinessStatus.NOT_READY

            return ComponentReadiness(
                component="data",
                status=status,
                score=score,
                issues=data_issues,
                details={
                    "existing_files": existing_files,
                    "total_required": len(required_files),
                    "recent_data_files": recent_data_files,
                    "coverage_percentage": coverage_percentage,
                    "coverage_compliance": coverage_compliance,
                    "recent_violations": recent_violations,
                    "file_details": file_details,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            self.logger.error(f"Data readiness check failed: {e}")
            return ComponentReadiness(
                component="data",
                status=ReadinessStatus.NOT_READY,
                score=0.0,
                issues=[f"Data check failed: {str(e)}"],
                last_check=datetime.now(timezone.utc),
            )

    def _check_calibration_readiness(self) -> ComponentReadiness:
        """Check calibration status and confidence levels"""

        try:
            calibration_config = self.config["calibration"]
            confidence_threshold = calibration_config["confidence_threshold"]
            calibration_files = calibration_config["calibration_files"]
            drift_threshold = calibration_config["drift_threshold"]

            calibration_data = {}
            calibration_found = False
            calibration_issues = []

            # Check calibration files
            for calibration_file in calibration_files:
                file_path = Path(calibration_file)

                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)

                        calibration_data[calibration_file] = data
                        calibration_found = True

                        # Check confidence levels
                        confidence = data.get("confidence", 0.0)
                        if confidence < confidence_threshold:
                            calibration_issues.append(
                                f"Low confidence in {calibration_file}: {confidence:.1%} < {confidence_threshold:.1%}"
                            )

                        # Check drift levels
                        drift = data.get("drift", 0.0)
                        if drift > drift_threshold:
                            calibration_issues.append(
                                f"High drift in {calibration_file}: {drift:.1%} > {drift_threshold:.1%}"
                            )

                    except (json.JSONDecodeError, IOError) as e:
                        calibration_issues.append(
                            f"Cannot read calibration file {calibration_file}: {e}"
                        )

            if not calibration_found:
                calibration_issues.append("No calibration files found")

            # Calculate score based on calibration quality
            if calibration_found:
                avg_confidence = sum(
                    data.get("confidence", 0.0) for data in calibration_data.values()
                ) / len(calibration_data)
                avg_drift = sum(data.get("drift", 1.0) for data in calibration_data.values()) / len(
                    calibration_data
                )

                confidence_score = min(100, (avg_confidence / confidence_threshold) * 70)
                drift_score = min(30, (1 - (avg_drift / drift_threshold)) * 30)
                score = confidence_score + drift_score
            else:
                score = 0.0

            # Determine status
            if score >= 80 and not calibration_issues:
                status = ReadinessStatus.READY
            elif score >= 60:
                status = ReadinessStatus.WARNING
            else:
                status = ReadinessStatus.NOT_READY

            return ComponentReadiness(
                component="calibration",
                status=status,
                score=score,
                issues=calibration_issues,
                details={
                    "calibration_files_found": len(calibration_data),
                    "calibration_data": calibration_data,
                    "confidence_threshold": confidence_threshold,
                    "drift_threshold": drift_threshold,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            self.logger.error(f"Calibration readiness check failed: {e}")
            return ComponentReadiness(
                component="calibration",
                status=ReadinessStatus.NOT_READY,
                score=0.0,
                issues=[f"Calibration check failed: {str(e)}"],
                last_check=datetime.now(timezone.utc),
            )

    def _check_health_readiness(self) -> ComponentReadiness:
        """Check health status with ROBUST FILE HANDLING"""

        try:
            health_config = self.config["health"]
            required_files = health_config["required_files"]
            health_threshold = health_config["health_score_threshold"]
            trading_required = health_config["trading_status_required"]
            fallback_behavior = health_config["fallback_behavior"]

            health_data = {}
            health_issues = []
            files_found = 0

            # Check each required health file with robust handling
            for health_file in required_files:
                file_path = Path(health_file)

                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)

                        health_data[health_file] = data
                        files_found += 1

                        # Validate health score
                        health_score = data.get("health_score", 0.0)
                        if health_score < health_threshold:
                            health_issues.append(
                                f"Low health score in {health_file}: {health_score:.1f} < {health_threshold}"
                            )

                        # Check trading status if required
                        if trading_required and "trading" in health_file:
                            trading_status = data.get("trading_enabled", False)
                            if not trading_status:
                                health_issues.append(f"Trading disabled in {health_file}")

                    except (json.JSONDecodeError, IOError) as e:
                        if fallback_behavior == "strict":
                            health_issues.append(f"Cannot read health file {health_file}: {e}")
                        else:
                            self.logger.warning(
                                f"Health file {health_file} unreadable, using lenient fallback: {e}"
                            )
                else:
                    if fallback_behavior == "strict":
                        health_issues.append(f"Required health file missing: {health_file}")
                    else:
                        self.logger.warning(
                            f"Health file {health_file} missing, using lenient fallback"
                        )

            # Calculate score based on health status
            if files_found > 0:
                # Average health scores from found files
                health_scores = []
                for data in health_data.values():
                    if "health_score" in data:
                        health_scores.append(data["health_score"])

                if health_scores:
                    avg_health_score = sum(health_scores) / len(health_scores)
                    file_coverage = (files_found / len(required_files)) * 30
                    health_quality = min(70, (avg_health_score / health_threshold) * 70)
                    score = file_coverage + health_quality
                else:
                    score = (
                        files_found / len(required_files)
                    ) * 50  # Partial credit for file existence
            else:
                if fallback_behavior == "strict":
                    score = 0.0
                else:
                    score = 30.0  # Lenient fallback score

            # Determine status based on fallback behavior
            if fallback_behavior == "strict":
                if score >= 80 and files_found == len(required_files) and not health_issues:
                    status = ReadinessStatus.READY
                elif score >= 60:
                    status = ReadinessStatus.WARNING
                else:
                    status = ReadinessStatus.NOT_READY
            else:
                # Lenient behavior - more forgiving
                if score >= 60 and not health_issues:
                    status = ReadinessStatus.READY
                elif score >= 40:
                    status = ReadinessStatus.WARNING
                else:
                    status = ReadinessStatus.NOT_READY

            return ComponentReadiness(
                component="health",
                status=status,
                score=score,
                issues=health_issues,
                details={
                    "files_found": files_found,
                    "total_required": len(required_files),
                    "health_data": health_data,
                    "fallback_behavior": fallback_behavior,
                    "health_threshold": health_threshold,
                },
                last_check=datetime.now(timezone.utc),
            )

        except Exception as e:
            self.logger.error(f"Health readiness check failed: {e}")
            return ComponentReadiness(
                component="health",
                status=ReadinessStatus.NOT_READY,
                score=0.0,
                issues=[f"Health check failed: {str(e)}"],
                last_check=datetime.now(timezone.utc),
            )

    def _calculate_overall_score(self, components: List[ComponentReadiness]) -> float:
        """Calculate overall readiness score"""

        if not components:
            return 0.0

        # Weighted scoring based on component importance
        weights = {"models": 0.3, "data": 0.3, "calibration": 0.2, "health": 0.2}

        total_weighted_score = 0.0
        total_weight = 0.0

        for component in components:
            weight = weights.get(component.component, 0.1)
            total_weighted_score += component.score * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_overall_status(
        self, overall_score: float, components: List[ComponentReadiness]
    ) -> ReadinessStatus:
        """Determine overall readiness status"""

        readiness_config = self.config["readiness"]
        critical_components = readiness_config["critical_components"]
        warning_threshold = readiness_config["warning_threshold"]

        # Check if any critical component is not ready
        for component in components:
            if (
                component.component in critical_components
                and component.status == ReadinessStatus.NOT_READY
            ):
                return ReadinessStatus.NOT_READY

        # Overall score-based determination
        if overall_score >= readiness_config["minimum_overall_score"]:
            return ReadinessStatus.READY
        elif overall_score >= warning_threshold:
            return ReadinessStatus.WARNING
        else:
            return ReadinessStatus.NOT_READY

    def _make_go_no_go_decision(
        self, overall_score: float, components: List[ComponentReadiness]
    ) -> bool:
        """Make GO/NO-GO decision for system operation"""

        readiness_config = self.config["readiness"]
        minimum_score = readiness_config["minimum_overall_score"]
        critical_components = readiness_config["critical_components"]

        # Must meet minimum score
        if overall_score < minimum_score:
            return False

        # All critical components must be ready or warning (not not_ready)
        for component in components:
            if (
                component.component in critical_components
                and component.status == ReadinessStatus.NOT_READY
            ):
                return False

        return True

    def _generate_recommendations(
        self, components: List[ComponentReadiness], overall_status: ReadinessStatus
    ) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # Overall recommendations
        if overall_status == ReadinessStatus.NOT_READY:
            recommendations.append(
                "System not ready for operation - address critical issues before deployment"
            )
        elif overall_status == ReadinessStatus.WARNING:
            recommendations.append(
                "System operational with warnings - monitor closely and address issues"
            )

        # Component-specific recommendations
        for component in components:
            if component.status == ReadinessStatus.NOT_READY:
                if component.issues:
                    recommendations.append(f"Fix {component.component}: {component.issues[0]}")
                else:
                    recommendations.append(f"Address {component.component} readiness issues")
            elif component.status == ReadinessStatus.WARNING and component.issues:
                recommendations.append(f"Monitor {component.component}: {component.issues[0]}")

        return recommendations[:5]  # Limit to 5 most important

    def _generate_summary(self, status: ReadinessStatus, score: float, go_no_go: bool) -> str:
        """Generate human-readable summary"""

        decision = "GO" if go_no_go else "NO-GO"
        return f"System readiness: {status.value.upper()} ({score:.1f}%) - Decision: {decision}"

    def _create_emergency_report(self, error_message: str) -> SystemReadinessReport:
        """Create emergency report when check fails"""

        return SystemReadinessReport(
            overall_status=ReadinessStatus.UNKNOWN,
            overall_score=0.0,
            go_no_go_decision=False,
            components=[],
            timestamp=datetime.now(timezone.utc),
            summary=f"Readiness check failed: {error_message}",
            recommendations=["Fix readiness checker system before proceeding"],
            metadata={"emergency_report": True, "error": error_message},
        )

    def _get_file_age_hours(self, file_path: Path) -> float:
        """Get file age in hours"""

        try:
            if file_path.is_dir():
                # For directories, use the newest file modification time
                newest_time = 0
                for item in file_path.rglob("*"):
                    if item.is_file():
                        newest_time = max(newest_time, item.stat().st_mtime)

                if newest_time > 0:
                    age_seconds = datetime.now().timestamp() - newest_time
                    return age_seconds / 3600
                else:
                    return float("inf")  # Empty directory
            else:
                age_seconds = datetime.now().timestamp() - file_path.stat().st_mtime
                return age_seconds / 3600
        except OSError:
            return float("inf")

    def _get_directory_size_mb(self, dir_path: Path) -> float:
        """Get directory size in MB"""

        try:
            total_size = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
            return total_size / (1024 * 1024)
        except OSError:
            return 0.0

    def _cleanup_history(self):
        """Clean up old readiness check history"""

        max_history = 20
        if len(self.check_history) > max_history:
            self.check_history = self.check_history[-max_history:]

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

        return base

    def get_readiness_summary(self) -> Dict[str, Any]:
        """Get summary of readiness checker status"""

        return {
            "checker_status": "active",
            "last_check": self.last_check.timestamp.isoformat() if self.last_check else None,
            "total_checks": self.check_count,
            "average_check_time": self.total_check_time / max(1, self.check_count),
            "current_status": self.last_check.overall_status.value
            if self.last_check
            else "unknown",
            "current_score": self.last_check.overall_score if self.last_check else 0.0,
            "go_no_go_decision": self.last_check.go_no_go_decision if self.last_check else False,
        }


# Utility functions


def quick_readiness_check() -> Dict[str, Any]:
    """Perform quick system readiness check"""

    checker = SystemReadinessChecker()
    report = checker.check_system_readiness()

    return {
        "status": report.overall_status.value,
        "score": report.overall_score,
        "go_no_go": report.go_no_go_decision,
        "summary": report.summary,
        "issues": sum(len(c.issues) for c in report.components),
        "timestamp": report.timestamp.isoformat(),
    }


if __name__ == "__main__":
    # Test system readiness checking
    print("Testing System Readiness Checker")

    checker = SystemReadinessChecker()
    report = checker.check_system_readiness()

    print(f"\nReadiness Report:")
    print(f"Status: {report.overall_status.value}")
    print(f"Score: {report.overall_score:.1f}%")
    print(f"GO/NO-GO: {'GO' if report.go_no_go_decision else 'NO-GO'}")
    print(f"Summary: {report.summary}")

    print(f"\nComponents ({len(report.components)}):")
    for component in report.components:
        print(f"  {component.component}: {component.status.value} ({component.score:.1f}%)")
        if component.issues:
            for issue in component.issues[:2]:  # Show first 2 issues
                print(f"    - {issue}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")

    print("\n✅ SYSTEM READINESS CHECKER TEST COMPLETE")
