"""Unit tests for position sizing components."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.cryptosmarttrader.portfolio.kelly_sizing import KellySizing, PositionSizeResult


class TestKellySizing:
    """Test Kelly criterion position sizing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.kelly_sizing = KellySizing(
            max_position_pct=0.25, min_position_pct=0.01, confidence_threshold=0.8
        )

    def test_kelly_calculation_basic(self):
        """Test basic Kelly calculation."""
        # High confidence, good win rate scenario
        result = self.kelly_sizing.calculate_position_size(
            confidence=0.9, expected_return=0.15, volatility=0.3, portfolio_value=100000
        )

        assert isinstance(result, PositionSizeResult)
        assert 0.01 <= result.position_pct <= 0.25
        assert result.position_value == result.position_pct * 100000
        assert result.confidence >= 0.8
        assert result.kelly_fraction > 0

    def test_kelly_calculation_low_confidence(self):
        """Test Kelly calculation with low confidence."""
        result = self.kelly_sizing.calculate_position_size(
            confidence=0.6,  # Below threshold
            expected_return=0.1,
            volatility=0.4,
            portfolio_value=100000,
        )

        # Should return minimum position or zero
        assert result.position_pct <= self.kelly_sizing.min_position_pct
        assert result.rejection_reason == "confidence_too_low"

    def test_kelly_calculation_negative_expected_return(self):
        """Test Kelly calculation with negative expected return."""
        result = self.kelly_sizing.calculate_position_size(
            confidence=0.9,
            expected_return=-0.05,  # Negative return
            volatility=0.2,
            portfolio_value=100000,
        )

        # Should reject negative expected return
        assert result.position_pct == 0
        assert result.rejection_reason == "negative_expected_return"

    def test_volatility_adjustment(self):
        """Test position size decreases with higher volatility."""
        high_vol_result = self.kelly_sizing.calculate_position_size(
            confidence=0.9,
            expected_return=0.1,
            volatility=0.5,  # High volatility
            portfolio_value=100000,
        )

        low_vol_result = self.kelly_sizing.calculate_position_size(
            confidence=0.9,
            expected_return=0.1,
            volatility=0.2,  # Low volatility
            portfolio_value=100000,
        )

        assert high_vol_result.position_pct < low_vol_result.position_pct

    def test_max_position_limit(self):
        """Test maximum position limit enforcement."""
        result = self.kelly_sizing.calculate_position_size(
            confidence=0.99,
            expected_return=1.0,  # Very high return
            volatility=0.1,  # Very low volatility
            portfolio_value=100000,
        )

        # Should be capped at max_position_pct
        assert result.position_pct <= self.kelly_sizing.max_position_pct

    def test_regime_adjustment(self):
        """Test position sizing adjustment based on market regime."""
        # Bull market - should allow larger positions
        bull_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.12,
            volatility=0.25,
            portfolio_value=100000,
            regime_adjustment=1.2,  # Bull market multiplier
        )

        # Bear market - should reduce positions
        bear_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.12,
            volatility=0.25,
            portfolio_value=100000,
            regime_adjustment=0.8,  # Bear market multiplier
        )

        assert bull_result.position_pct > bear_result.position_pct

    def test_drawdown_scaling(self):
        """Test position scaling during drawdown periods."""
        # Normal conditions
        normal_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.1,
            volatility=0.3,
            portfolio_value=100000,
            current_drawdown=0.02,  # 2% drawdown
        )

        # High drawdown - should reduce positions
        drawdown_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.1,
            volatility=0.3,
            portfolio_value=100000,
            current_drawdown=0.15,  # 15% drawdown
        )

        assert drawdown_result.position_pct < normal_result.position_pct

    def test_correlation_adjustment(self):
        """Test position adjustment for portfolio correlation."""
        # Low correlation portfolio - can take larger positions
        low_corr_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.1,
            volatility=0.3,
            portfolio_value=100000,
            portfolio_correlation=0.3,
        )

        # High correlation portfolio - should reduce positions
        high_corr_result = self.kelly_sizing.calculate_position_size(
            confidence=0.85,
            expected_return=0.1,
            volatility=0.3,
            portfolio_value=100000,
            portfolio_correlation=0.8,
        )

        assert low_corr_result.position_pct > high_corr_result.position_pct

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero volatility
        with pytest.raises(ValueError):
            self.kelly_sizing.calculate_position_size(
                confidence=0.85, expected_return=0.1, volatility=0.0, portfolio_value=100000
            )

        # Zero portfolio value
        with pytest.raises(ValueError):
            self.kelly_sizing.calculate_position_size(
                confidence=0.85, expected_return=0.1, volatility=0.3, portfolio_value=0
            )

        # Invalid confidence (outside 0-1)
        with pytest.raises(ValueError):
            self.kelly_sizing.calculate_position_size(
                confidence=1.5, expected_return=0.1, volatility=0.3, portfolio_value=100000
            )


@pytest.mark.unit
class TestPositionSizeResult:
    """Test PositionSizeResult data class."""

    def test_result_creation(self):
        """Test creation of position size result."""
        result = PositionSizeResult(
            position_pct=0.15,
            position_value=15000,
            confidence=0.85,
            kelly_fraction=0.18,
            adjusted_fraction=0.15,
            regime_adjustment=1.0,
            volatility_adjustment=0.9,
            correlation_adjustment=0.95,
            rejection_reason=None,
        )

        assert result.position_pct == 0.15
        assert result.position_value == 15000
        assert result.is_valid()
        assert result.rejection_reason is None

    def test_rejected_result(self):
        """Test rejected position size result."""
        result = PositionSizeResult(
            position_pct=0.0,
            position_value=0.0,
            confidence=0.6,
            kelly_fraction=0.0,
            rejection_reason="confidence_too_low",
        )

        assert not result.is_valid()
        assert result.rejection_reason == "confidence_too_low"

    def test_result_serialization(self):
        """Test result can be serialized to dict."""
        result = PositionSizeResult(
            position_pct=0.1, position_value=10000, confidence=0.85, kelly_fraction=0.12
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["position_pct"] == 0.1
        assert result_dict["position_value"] == 10000
