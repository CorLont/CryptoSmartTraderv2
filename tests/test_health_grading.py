#!/usr/bin/env python3
"""
Health Grading System Tests - Verify precise scoring formulas
"""

import pytest
from datetime import datetime

from core.health_grading import HealthGradingSystem, HealthGrade


class TestHealthGradingSystem:
    """Test health grading calculations"""

    @pytest.fixture
    def health_grader(self):
        """Create health grading system"""
        return HealthGradingSystem()

    def test_data_quality_score_perfect(self, health_grader):
        """Test perfect data quality score"""
        score = health_grader.calculate_data_quality_score(
            api_uptime_pct=100.0,
            data_freshness_minutes=2.0,
            data_completeness_pct=100.0,
            drift_score=0.05,
        )

        assert score == 100.0

    def test_data_quality_score_poor(self, health_grader):
        """Test poor data quality score"""
        score = health_grader.calculate_data_quality_score(
            api_uptime_pct=50.0,  # Poor uptime
            data_freshness_minutes=120.0,  # Very stale data
            data_completeness_pct=60.0,  # Incomplete data
            drift_score=0.8,  # High drift
        )

        # Should be a low score
        assert score < 50.0

    def test_system_performance_score_excellent(self, health_grader):
        """Test excellent system performance"""
        score = health_grader.calculate_system_performance_score(
            cpu_usage_pct=30.0,
            memory_usage_pct=40.0,
            disk_usage_pct=50.0,
            avg_response_time_ms=50.0,
        )

        assert score == 100.0

    def test_system_performance_score_critical(self, health_grader):
        """Test critical system performance"""
        score = health_grader.calculate_system_performance_score(
            cpu_usage_pct=95.0,  # Critical CPU
            memory_usage_pct=90.0,  # Critical memory
            disk_usage_pct=95.0,  # Critical disk
            avg_response_time_ms=2000.0,  # Very slow
        )

        # Should be very low score
        assert score < 20.0

    def test_model_performance_score_high_accuracy(self, health_grader):
        """Test high model performance"""
        score = health_grader.calculate_model_performance_score(
            prediction_accuracy=0.95, avg_confidence=0.85, model_drift_score=0.05
        )

        assert score > 90.0

    def test_model_performance_score_poor(self, health_grader):
        """Test poor model performance"""
        score = health_grader.calculate_model_performance_score(
            prediction_accuracy=0.60,  # Low accuracy
            avg_confidence=0.45,  # Low confidence
            model_drift_score=0.4,  # High drift
        )

        assert score < 60.0

    def test_api_health_score_optimal(self, health_grader):
        """Test optimal API health"""
        score = health_grader.calculate_api_health_score(
            avg_response_time_ms=100.0, error_rate_pct=0.0, rate_limit_usage_pct=50.0
        )

        assert score == 100.0

    def test_api_health_score_degraded(self, health_grader):
        """Test degraded API health"""
        score = health_grader.calculate_api_health_score(
            avg_response_time_ms=1500.0,  # Slow
            error_rate_pct=3.0,  # Some errors
            rate_limit_usage_pct=85.0,  # High rate limit usage
        )

        assert score < 70.0

    def test_trading_system_score_excellent(self, health_grader):
        """Test excellent trading system"""
        score = health_grader.calculate_trading_system_score(
            execution_success_rate=0.99, avg_slippage_bps=5.0, risk_violations=0
        )

        assert score > 95.0

    def test_trading_system_score_problematic(self, health_grader):
        """Test problematic trading system"""
        score = health_grader.calculate_trading_system_score(
            execution_success_rate=0.85,  # Some failures
            avg_slippage_bps=40.0,  # High slippage
            risk_violations=3,  # Risk violations
        )

        assert score < 80.0

    def test_security_score_perfect(self, health_grader):
        """Test perfect security score"""
        score = health_grader.calculate_security_score(
            auth_success_rate=1.0, rate_limit_violations=0
        )

        assert score == 100.0

    def test_security_score_issues(self, health_grader):
        """Test security with issues"""
        score = health_grader.calculate_security_score(
            auth_success_rate=0.95,  # Some auth failures
            rate_limit_violations=5,  # Rate limit violations
        )

        assert score < 90.0

    def test_overall_health_calculation_grade_a(self, health_grader):
        """Test Grade A calculation"""
        # Set all components to high scores
        health_grader.update_component_health("data_quality", 95.0, "healthy")
        health_grader.update_component_health("system_performance", 92.0, "healthy")
        health_grader.update_component_health("model_performance", 94.0, "healthy")
        health_grader.update_component_health("api_health", 96.0, "healthy")
        health_grader.update_component_health("trading_system", 93.0, "healthy")
        health_grader.update_component_health("security", 98.0, "healthy")

        report = health_grader.calculate_overall_health()

        assert report.overall_grade == HealthGrade.A
        assert report.overall_score >= 90.0
        assert report.trading_enabled is True

    def test_overall_health_calculation_grade_c(self, health_grader):
        """Test Grade C calculation"""
        # Set components to fair scores
        health_grader.update_component_health("data_quality", 75.0, "degraded")
        health_grader.update_component_health("system_performance", 72.0, "degraded")
        health_grader.update_component_health("model_performance", 78.0, "healthy")
        health_grader.update_component_health("api_health", 74.0, "degraded")
        health_grader.update_component_health("trading_system", 76.0, "healthy")
        health_grader.update_component_health("security", 80.0, "healthy")

        report = health_grader.calculate_overall_health()

        assert report.overall_grade == HealthGrade.C
        assert 70.0 <= report.overall_score < 80.0
        assert report.trading_enabled is True  # Still enabled but with restrictions

    def test_overall_health_calculation_grade_f(self, health_grader):
        """Test Grade F calculation"""
        # Set components to failing scores
        health_grader.update_component_health("data_quality", 30.0, "unhealthy")
        health_grader.update_component_health("system_performance", 45.0, "unhealthy")
        health_grader.update_component_health("model_performance", 35.0, "unhealthy")
        health_grader.update_component_health("api_health", 40.0, "unhealthy")
        health_grader.update_component_health("trading_system", 25.0, "unhealthy")
        health_grader.update_component_health("security", 60.0, "degraded")

        report = health_grader.calculate_overall_health()

        assert report.overall_grade == HealthGrade.F
        assert report.overall_score < 60.0
        assert report.trading_enabled is False

    def test_component_weights_sum_to_one(self, health_grader):
        """Test that component weights sum to 1.0"""
        total_weight = sum(health_grader.component_weights.values())
        assert abs(total_weight - 1.0) < 0.001  # Allow for floating point precision

    def test_recommendations_generation(self, health_grader):
        """Test recommendation generation"""
        # Set mixed health scores
        health_grader.update_component_health("data_quality", 65.0, "degraded")  # Below 70
        health_grader.update_component_health("system_performance", 85.0, "healthy")
        health_grader.update_component_health("model_performance", 55.0, "unhealthy")  # Below 70
        health_grader.update_component_health("api_health", 90.0, "healthy")
        health_grader.update_component_health("trading_system", 75.0, "healthy")
        health_grader.update_component_health("security", 95.0, "healthy")

        report = health_grader.calculate_overall_health()

        # Should have recommendations for degraded components
        assert len(report.recommendations) > 0

        # Check for specific recommendations
        recommendation_text = " ".join(report.recommendations).lower()
        assert "data quality" in recommendation_text or "model performance" in recommendation_text

    def test_grade_boundaries(self, health_grader):
        """Test grade boundary conditions"""
        test_cases = [
            (95.0, HealthGrade.A),
            (90.0, HealthGrade.A),
            (89.9, HealthGrade.B),
            (85.0, HealthGrade.B),
            (80.0, HealthGrade.B),
            (79.9, HealthGrade.C),
            (75.0, HealthGrade.C),
            (70.0, HealthGrade.C),
            (69.9, HealthGrade.D),
            (65.0, HealthGrade.D),
            (60.0, HealthGrade.D),
            (59.9, HealthGrade.F),
            (30.0, HealthGrade.F),
            (0.0, HealthGrade.F),
        ]

        for score, expected_grade in test_cases:
            # Set all components to the test score
            for component in health_grader.component_weights.keys():
                health_grader.update_component_health(component, score, "test")

            report = health_grader.calculate_overall_health()
            assert report.overall_grade == expected_grade, (
                f"Score {score} should be grade {expected_grade}, got {report.overall_grade}"
            )

    def test_trading_policies(self, health_grader):
        """Test trading policies by grade"""
        # Test Grade A - Full trading
        for component in health_grader.component_weights.keys():
            health_grader.update_component_health(component, 95.0, "healthy")

        report = health_grader.calculate_overall_health()
        assert report.trading_enabled is True

        # Test Grade D - Paper trading only
        for component in health_grader.component_weights.keys():
            health_grader.update_component_health(component, 65.0, "degraded")

        report = health_grader.calculate_overall_health()
        assert report.trading_enabled is False

        # Test Grade F - No trading
        for component in health_grader.component_weights.keys():
            health_grader.update_component_health(component, 30.0, "unhealthy")

        report = health_grader.calculate_overall_health()
        assert report.trading_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
