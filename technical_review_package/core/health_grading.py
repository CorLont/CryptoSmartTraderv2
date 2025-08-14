#!/usr/bin/env python3
"""
Health Grading System - Precise A/B/C/D/F scoring with documented formulas
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class HealthGrade(Enum):
    """Health grade enumeration"""

    A = "A"  # Excellent (90-100)
    B = "B"  # Good (80-89)
    C = "C"  # Fair (70-79)
    D = "D"  # Poor (60-69)
    F = "F"  # Fail (0-59)


@dataclass
class ComponentHealth:
    """Component health data"""

    name: str
    score: float  # 0-100
    weight: float  # Weight in overall score
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Comprehensive health report"""

    overall_grade: HealthGrade
    overall_score: float
    component_scores: Dict[str, ComponentHealth]
    trading_enabled: bool
    recommendations: List[str]
    timestamp: datetime
    grade_history: List[Tuple[datetime, str, float]] = field(default_factory=list)


class HealthGradingSystem:
    """
    Enterprise health grading system with precise formulas

    Grading Formula:
    ================

    Overall Score = Σ(ComponentScore × Weight) / Σ(Weight)

    Component Weights:
    - Data Quality: 25%
    - System Performance: 20%
    - Model Performance: 20%
    - API Health: 15%
    - Trading System: 15%
    - Security: 5%

    Grade Boundaries:
    - A: 90-100 (Excellent) - Full trading enabled
    - B: 80-89  (Good)      - Full trading enabled
    - C: 70-79  (Fair)      - Reduced position sizing
    - D: 60-69  (Poor)      - Paper trading only
    - F: 0-59   (Fail)      - Trading disabled

    Component Scoring:
    ==================

    1. Data Quality (25% weight):
       - API Uptime: 40%
       - Data Freshness: 30%
       - Data Completeness: 20%
       - Drift Score: 10%

    2. System Performance (20% weight):
       - CPU Usage: 25%
       - Memory Usage: 25%
       - Disk Usage: 20%
       - Response Time: 30%

    3. Model Performance (20% weight):
       - Prediction Accuracy: 50%
       - Confidence Levels: 30%
       - Model Drift: 20%

    4. API Health (15% weight):
       - Response Time: 40%
       - Error Rate: 40%
       - Rate Limit Usage: 20%

    5. Trading System (15% weight):
       - Execution Success: 50%
       - Slippage Control: 30%
       - Risk Management: 20%

    6. Security (5% weight):
       - Auth Status: 60%
       - Rate Limiting: 40%
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Component weights (must sum to 1.0)
        self.component_weights = {
            "data_quality": 0.25,
            "system_performance": 0.20,
            "model_performance": 0.20,
            "api_health": 0.15,
            "trading_system": 0.15,
            "security": 0.05,
        }

        # Grade boundaries
        self.grade_boundaries = {
            90: HealthGrade.A,
            80: HealthGrade.B,
            70: HealthGrade.C,
            60: HealthGrade.D,
            0: HealthGrade.F,
        }

        # Trading policies by grade
        self.trading_policies = {
            HealthGrade.A: {"trading_enabled": True, "position_size_multiplier": 1.0},
            HealthGrade.B: {"trading_enabled": True, "position_size_multiplier": 1.0},
            HealthGrade.C: {"trading_enabled": True, "position_size_multiplier": 0.5},
            HealthGrade.D: {"trading_enabled": False, "paper_trading": True},
            HealthGrade.F: {"trading_enabled": False, "paper_trading": False},
        }

        # Current component health
        self.component_health: Dict[str, ComponentHealth] = {}

        # Health history for trending
        self.health_history: List[Tuple[datetime, HealthGrade, float]] = []

    def calculate_data_quality_score(
        self,
        api_uptime_pct: float,
        data_freshness_minutes: float,
        data_completeness_pct: float,
        drift_score: float,
    ) -> float:
        """
        Calculate data quality score

        Args:
            api_uptime_pct: API uptime percentage (0-100)
            data_freshness_minutes: Average data age in minutes
            data_completeness_pct: Data completeness percentage (0-100)
            drift_score: Model drift score (0-1, lower is better)

        Returns:
            Data quality score (0-100)
        """

        # API Uptime (40% weight)
        # Score = uptime percentage
        uptime_score = min(api_uptime_pct, 100)

        # Data Freshness (30% weight)
        # Fresh data (< 5 min) = 100, stale data (> 60 min) = 0
        if data_freshness_minutes <= 5:
            freshness_score = 100
        elif data_freshness_minutes >= 60:
            freshness_score = 0
        else:
            freshness_score = 100 * (1 - (data_freshness_minutes - 5) / 55)

        # Data Completeness (20% weight)
        # Score = completeness percentage
        completeness_score = min(data_completeness_pct, 100)

        # Drift Score (10% weight)
        # Low drift (< 0.1) = 100, high drift (> 0.5) = 0
        if drift_score <= 0.1:
            drift_score_normalized = 100
        elif drift_score >= 0.5:
            drift_score_normalized = 0
        else:
            drift_score_normalized = 100 * (1 - (drift_score - 0.1) / 0.4)

        # Weighted combination
        score = (
            uptime_score * 0.40
            + freshness_score * 0.30
            + completeness_score * 0.20
            + drift_score_normalized * 0.10
        )

        return max(0, min(100, score))

    def calculate_system_performance_score(
        self,
        cpu_usage_pct: float,
        memory_usage_pct: float,
        disk_usage_pct: float,
        avg_response_time_ms: float,
    ) -> float:
        """
        Calculate system performance score

        Args:
            cpu_usage_pct: CPU usage percentage (0-100)
            memory_usage_pct: Memory usage percentage (0-100)
            disk_usage_pct: Disk usage percentage (0-100)
            avg_response_time_ms: Average response time in milliseconds

        Returns:
            System performance score (0-100)
        """

        # CPU Usage (25% weight)
        # Good (< 50%) = 100, Warning (50-80%) = linear, Critical (> 80%) = 0
        if cpu_usage_pct <= 50:
            cpu_score = 100
        elif cpu_usage_pct >= 80:
            cpu_score = 0
        else:
            cpu_score = 100 * (1 - (cpu_usage_pct - 50) / 30)

        # Memory Usage (25% weight)
        # Good (< 60%) = 100, Warning (60-85%) = linear, Critical (> 85%) = 0
        if memory_usage_pct <= 60:
            memory_score = 100
        elif memory_usage_pct >= 85:
            memory_score = 0
        else:
            memory_score = 100 * (1 - (memory_usage_pct - 60) / 25)

        # Disk Usage (20% weight)
        # Good (< 70%) = 100, Warning (70-90%) = linear, Critical (> 90%) = 0
        if disk_usage_pct <= 70:
            disk_score = 100
        elif disk_usage_pct >= 90:
            disk_score = 0
        else:
            disk_score = 100 * (1 - (disk_usage_pct - 70) / 20)

        # Response Time (30% weight)
        # Fast (< 100ms) = 100, Slow (100-1000ms) = linear, Very slow (> 1000ms) = 0
        if avg_response_time_ms <= 100:
            response_score = 100
        elif avg_response_time_ms >= 1000:
            response_score = 0
        else:
            response_score = 100 * (1 - (avg_response_time_ms - 100) / 900)

        # Weighted combination
        score = cpu_score * 0.25 + memory_score * 0.25 + disk_score * 0.20 + response_score * 0.30

        return max(0, min(100, score))

    def calculate_model_performance_score(
        self, prediction_accuracy: float, avg_confidence: float, model_drift_score: float
    ) -> float:
        """
        Calculate model performance score

        Args:
            prediction_accuracy: Model accuracy (0-1)
            avg_confidence: Average prediction confidence (0-1)
            model_drift_score: Model drift score (0-1, lower is better)

        Returns:
            Model performance score (0-100)
        """

        # Prediction Accuracy (50% weight)
        # Convert 0-1 to 0-100 scale
        accuracy_score = prediction_accuracy * 100

        # Confidence Levels (30% weight)
        # High confidence (> 0.8) = 100, Low confidence (< 0.5) = 0
        if avg_confidence >= 0.8:
            confidence_score = 100
        elif avg_confidence <= 0.5:
            confidence_score = 0
        else:
            confidence_score = 100 * (avg_confidence - 0.5) / 0.3

        # Model Drift (20% weight)
        # Low drift (< 0.1) = 100, High drift (> 0.3) = 0
        if model_drift_score <= 0.1:
            drift_score = 100
        elif model_drift_score >= 0.3:
            drift_score = 0
        else:
            drift_score = 100 * (1 - (model_drift_score - 0.1) / 0.2)

        # Weighted combination
        score = accuracy_score * 0.50 + confidence_score * 0.30 + drift_score * 0.20

        return max(0, min(100, score))

    def calculate_api_health_score(
        self, avg_response_time_ms: float, error_rate_pct: float, rate_limit_usage_pct: float
    ) -> float:
        """
        Calculate API health score

        Args:
            avg_response_time_ms: Average API response time
            error_rate_pct: API error rate percentage (0-100)
            rate_limit_usage_pct: Rate limit usage percentage (0-100)

        Returns:
            API health score (0-100)
        """

        # Response Time (40% weight)
        # Fast (< 200ms) = 100, Slow (> 2000ms) = 0
        if avg_response_time_ms <= 200:
            response_score = 100
        elif avg_response_time_ms >= 2000:
            response_score = 0
        else:
            response_score = 100 * (1 - (avg_response_time_ms - 200) / 1800)

        # Error Rate (40% weight)
        # No errors (0%) = 100, High errors (> 5%) = 0
        if error_rate_pct <= 0:
            error_score = 100
        elif error_rate_pct >= 5:
            error_score = 0
        else:
            error_score = 100 * (1 - error_rate_pct / 5)

        # Rate Limit Usage (20% weight)
        # Low usage (< 70%) = 100, High usage (> 90%) = 0
        if rate_limit_usage_pct <= 70:
            rate_limit_score = 100
        elif rate_limit_usage_pct >= 90:
            rate_limit_score = 0
        else:
            rate_limit_score = 100 * (1 - (rate_limit_usage_pct - 70) / 20)

        # Weighted combination
        score = response_score * 0.40 + error_score * 0.40 + rate_limit_score * 0.20

        return max(0, min(100, score))

    def calculate_trading_system_score(
        self, execution_success_rate: float, avg_slippage_bps: float, risk_violations: int
    ) -> float:
        """
        Calculate trading system score

        Args:
            execution_success_rate: Order execution success rate (0-1)
            avg_slippage_bps: Average slippage in basis points
            risk_violations: Number of risk violations in last 24h

        Returns:
            Trading system score (0-100)
        """

        # Execution Success (50% weight)
        execution_score = execution_success_rate * 100

        # Slippage Control (30% weight)
        # Low slippage (< 10 bps) = 100, High slippage (> 50 bps) = 0
        if avg_slippage_bps <= 10:
            slippage_score = 100
        elif avg_slippage_bps >= 50:
            slippage_score = 0
        else:
            slippage_score = 100 * (1 - (avg_slippage_bps - 10) / 40)

        # Risk Management (20% weight)
        # No violations = 100, 5+ violations = 0
        if risk_violations <= 0:
            risk_score = 100
        elif risk_violations >= 5:
            risk_score = 0
        else:
            risk_score = 100 * (1 - risk_violations / 5)

        # Weighted combination
        score = execution_score * 0.50 + slippage_score * 0.30 + risk_score * 0.20

        return max(0, min(100, score))

    def calculate_security_score(
        self, auth_success_rate: float, rate_limit_violations: int
    ) -> float:
        """
        Calculate security score

        Args:
            auth_success_rate: Authentication success rate (0-1)
            rate_limit_violations: Rate limit violations in last 24h

        Returns:
            Security score (0-100)
        """

        # Auth Status (60% weight)
        auth_score = auth_success_rate * 100

        # Rate Limiting (40% weight)
        # No violations = 100, 10+ violations = 0
        if rate_limit_violations <= 0:
            rate_limit_score = 100
        elif rate_limit_violations >= 10:
            rate_limit_score = 0
        else:
            rate_limit_score = 100 * (1 - rate_limit_violations / 10)

        # Weighted combination
        score = auth_score * 0.60 + rate_limit_score * 0.40

        return max(0, min(100, score))

    def update_component_health(
        self,
        component_name: str,
        score: float,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Update component health data"""

        if component_name not in self.component_weights:
            self.logger.warning(f"Unknown component: {component_name}")
            return

        self.component_health[component_name] = ComponentHealth(
            name=component_name,
            score=max(0, min(100, score)),
            weight=self.component_weights[component_name],
            status=status,
            last_check=datetime.utcnow(),
            details=details or {},
        )

    def calculate_overall_health(self) -> HealthReport:
        """
        Calculate overall health grade and generate report

        Returns:
            Comprehensive health report
        """

        if not self.component_health:
            return HealthReport(
                overall_grade=HealthGrade.F,
                overall_score=0.0,
                component_scores={},
                trading_enabled=False,
                recommendations=["No health data available"],
                timestamp=datetime.utcnow(),
            )

        # Calculate weighted overall score
        total_weighted_score = 0.0
        total_weight = 0.0

        for component_name, weight in self.component_weights.items():
            if component_name in self.component_health:
                component = self.component_health[component_name]
                total_weighted_score += component.score * weight
                total_weight += weight

        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0

        # Determine grade
        grade = HealthGrade.F
        for threshold, grade_value in sorted(self.grade_boundaries.items(), reverse=True):
            if overall_score >= threshold:
                grade = grade_value
                break

        # Determine trading policy
        policy = self.trading_policies[grade]
        trading_enabled = policy.get("trading_enabled", False)

        # Generate recommendations
        recommendations = self._generate_recommendations(grade, self.component_health)

        # Update history
        self.health_history.append((datetime.utcnow(), grade, overall_score))

        # Keep only last 24 hours of history
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.health_history = [(ts, g, s) for ts, g, s in self.health_history if ts > cutoff_time]

        return HealthReport(
            overall_grade=grade,
            overall_score=overall_score,
            component_scores=dict(self.component_health),
            trading_enabled=trading_enabled,
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
            grade_history=list(self.health_history),
        )

    def _generate_recommendations(
        self, grade: HealthGrade, components: Dict[str, ComponentHealth]
    ) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # Grade-specific recommendations
        if grade == HealthGrade.F:
            recommendations.append("CRITICAL: System health is failing - trading disabled")
            recommendations.append("Immediate investigation required")
        elif grade == HealthGrade.D:
            recommendations.append("WARNING: Poor system health - paper trading only")
            recommendations.append("Address critical issues before enabling live trading")
        elif grade == HealthGrade.C:
            recommendations.append("CAUTION: Fair system health - reduced position sizing")
            recommendations.append("Monitor system closely and fix degraded components")

        # Component-specific recommendations
        for component in components.values():
            if component.score < 70:
                if component.name == "data_quality":
                    recommendations.append(
                        f"Improve data quality: check API connections and data freshness"
                    )
                elif component.name == "system_performance":
                    recommendations.append(
                        f"Optimize system performance: monitor CPU/memory/disk usage"
                    )
                elif component.name == "model_performance":
                    recommendations.append(
                        f"Model performance degraded: consider retraining or validation"
                    )
                elif component.name == "api_health":
                    recommendations.append(
                        f"API health issues: check response times and error rates"
                    )
                elif component.name == "trading_system":
                    recommendations.append(
                        f"Trading system problems: review execution and slippage"
                    )
                elif component.name == "security":
                    recommendations.append(
                        f"Security concerns: review authentication and rate limiting"
                    )

        return recommendations[:5]  # Limit to top 5 recommendations


# Global health grading system
health_grader = HealthGradingSystem()
