"""Integration tests for backtest-live parity validation."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from src.cryptosmarttrader.analysis.backtest_parity import BacktestParityAnalyzer


@pytest.mark.integration
class TestBacktestParityAnalyzer:
    """Test backtest-live parity tracking and validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parity_analyzer = BacktestParityAnalyzer(
            target_tracking_error_bps=20,  # 20 basis points
            alert_threshold_bps=15,
            max_deviation_bps=50
        )
    
    def test_parity_analysis_basic(self):
        """Test basic parity analysis between backtest and live results."""
        # Mock backtest results
        backtest_results = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
            'symbol': 'BTC/USDT',
            'predicted_return': np.random.normal(0.001, 0.02, 100),
            'actual_return': np.random.normal(0.001, 0.025, 100),
            'position_size': np.random.uniform(0.01, 0.05, 100),
            'execution_price': np.random.uniform(45000, 55000, 100),
            'fees': np.random.uniform(10, 50, 100)
        })
        
        # Mock live results with slight deviation
        live_results = backtest_results.copy()
        live_results['actual_return'] += np.random.normal(0, 0.001, 100)  # Small tracking error
        live_results['execution_price'] += np.random.normal(0, 100, 100)  # Price impact
        live_results['fees'] *= 1.1  # Slightly higher live fees
        
        parity_analysis = self.parity_analyzer.analyze_parity(
            backtest_results=backtest_results,
            live_results=live_results
        )
        
        assert parity_analysis.tracking_error_bps is not None
        assert parity_analysis.tracking_error_bps >= 0
        assert parity_analysis.component_attribution is not None
        assert 'fees' in parity_analysis.component_attribution
        assert 'slippage' in parity_analysis.component_attribution
        assert 'timing' in parity_analysis.component_attribution
    
    def test_tracking_error_calculation(self):
        """Test tracking error calculation accuracy."""
        # Create perfectly aligned results (zero tracking error)
        aligned_results = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'backtest_return': [0.01] * 50,
            'live_return': [0.01] * 50
        })
        
        tracking_error = self.parity_analyzer.calculate_tracking_error(
            aligned_results['backtest_return'],
            aligned_results['live_return']
        )
        
        assert abs(tracking_error.daily_te_bps) < 1.0  # Should be near zero
        
        # Create misaligned results (high tracking error)
        misaligned_results = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'backtest_return': [0.01] * 50,
            'live_return': [0.02] * 50  # 100 bps difference
        })
        
        tracking_error_high = self.parity_analyzer.calculate_tracking_error(
            misaligned_results['backtest_return'],
            misaligned_results['live_return']
        )
        
        assert tracking_error_high.daily_te_bps > 50  # Should be high
    
    def test_component_attribution_analysis(self):
        """Test component-wise attribution of tracking error."""
        # Create test data with known attribution sources
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='1H'),
            'symbol': 'ETH/USDT',
            'backtest_return': np.random.normal(0.001, 0.01, 24),
            'live_return': np.random.normal(0.001, 0.01, 24),
            'backtest_fees': [5.0] * 24,
            'live_fees': [7.5] * 24,  # 50% higher fees
            'backtest_slippage': [2.0] * 24,
            'live_slippage': [8.0] * 24,  # 4x higher slippage
            'timing_delay_ms': [50] * 24
        })
        
        attribution = self.parity_analyzer.analyze_component_attribution(test_data)
        
        assert attribution.fee_impact_bps is not None
        assert attribution.slippage_impact_bps is not None
        assert attribution.timing_impact_bps is not None
        assert attribution.alpha_decay_bps is not None
        
        # Slippage should be the largest contributor given the test data
        assert attribution.slippage_impact_bps > attribution.fee_impact_bps
    
    def test_statistical_significance_testing(self):
        """Test statistical significance of tracking error."""
        # Generate statistically significant tracking error
        significant_data = pd.DataFrame({
            'backtest_return': np.random.normal(0.002, 0.01, 1000),
            'live_return': np.random.normal(0.0015, 0.01, 1000)  # 5 bps difference
        })
        
        significance_test = self.parity_analyzer.test_statistical_significance(
            significant_data['backtest_return'],
            significant_data['live_return']
        )
        
        assert significance_test.p_value is not None
        assert 0 <= significance_test.p_value <= 1
        assert significance_test.is_significant is not None
        assert significance_test.confidence_interval is not None
    
    def test_rolling_parity_monitoring(self):
        """Test rolling parity monitoring over time."""
        # Create time series data with evolving parity
        rolling_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=168, freq='1H'),  # 1 week
            'backtest_return': np.random.normal(0.001, 0.015, 168),
            'live_return': np.random.normal(0.001, 0.015, 168)
        })
        
        # Add drift in second half
        rolling_data.loc[84:, 'live_return'] += 0.0005  # 5 bps drift
        
        rolling_analysis = self.parity_analyzer.analyze_rolling_parity(
            rolling_data,
            window_hours=24
        )
        
        assert len(rolling_analysis.rolling_te_bps) > 0
        assert rolling_analysis.drift_detected is not None
        assert rolling_analysis.drift_start_time is not None
        
        # Should detect drift in second half
        if rolling_analysis.drift_detected:
            assert rolling_analysis.drift_start_time >= rolling_data['timestamp'].iloc[84]
    
    def test_execution_quality_comparison(self):
        """Test execution quality comparison between backtest and live."""
        execution_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='30min'),
            'symbol': 'BTC/USDT',
            'backtest_execution_time_ms': [50] * 100,  # Instant in backtest
            'live_execution_time_ms': np.random.uniform(100, 500, 100),  # Variable live
            'backtest_fill_rate': [1.0] * 100,  # Perfect fills in backtest
            'live_fill_rate': np.random.uniform(0.95, 1.0, 100),  # Partial fills possible
            'backtest_slippage_bps': [1.0] * 100,  # Optimistic backtest slippage
            'live_slippage_bps': np.random.uniform(2, 15, 100)  # Realistic live slippage
        })
        
        execution_comparison = self.parity_analyzer.compare_execution_quality(execution_data)
        
        assert execution_comparison.execution_time_impact_bps >= 0
        assert execution_comparison.fill_rate_impact_bps >= 0
        assert execution_comparison.slippage_difference_bps >= 0
        
        # Live execution should generally be worse than backtest
        assert execution_comparison.execution_time_impact_bps > 0
        assert execution_comparison.slippage_difference_bps > 0
    
    def test_market_impact_attribution(self):
        """Test market impact attribution in parity analysis."""
        market_impact_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'symbol': 'ETH/USDT',
            'order_size_usd': np.random.uniform(10000, 100000, 50),
            'market_liquidity_usd': np.random.uniform(500000, 2000000, 50),
            'backtest_market_impact_bps': [0] * 50,  # No market impact in backtest
            'live_market_impact_bps': None  # To be calculated
        })
        
        # Calculate realistic market impact
        market_impact_data['live_market_impact_bps'] = (
            (market_impact_data['order_size_usd'] / market_impact_data['market_liquidity_usd']) * 
            10000 * 0.5  # Square root market impact model
        )
        
        impact_attribution = self.parity_analyzer.analyze_market_impact(market_impact_data)
        
        assert impact_attribution.total_impact_bps >= 0
        assert impact_attribution.linear_impact_bps >= 0
        assert impact_attribution.sqrt_impact_bps >= 0
        assert impact_attribution.correlation_with_size is not None
    
    def test_regime_dependent_parity(self):
        """Test parity analysis across different market regimes."""
        regime_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1H'),
            'regime': ['bull'] * 100 + ['bear'] * 100,
            'backtest_return': np.concatenate([
                np.random.normal(0.002, 0.01, 100),  # Bull market
                np.random.normal(-0.001, 0.02, 100)  # Bear market
            ]),
            'live_return': np.concatenate([
                np.random.normal(0.0018, 0.012, 100),  # Slight underperformance in bull
                np.random.normal(-0.0008, 0.022, 100)  # Slight outperformance in bear
            ])
        })
        
        regime_analysis = self.parity_analyzer.analyze_regime_dependent_parity(regime_data)
        
        assert 'bull' in regime_analysis.regime_tracking_errors
        assert 'bear' in regime_analysis.regime_tracking_errors
        
        bull_te = regime_analysis.regime_tracking_errors['bull']
        bear_te = regime_analysis.regime_tracking_errors['bear']
        
        assert bull_te.tracking_error_bps >= 0
        assert bear_te.tracking_error_bps >= 0
        
        # May have different tracking errors in different regimes
        assert regime_analysis.regime_consistency_score is not None
    
    def test_parity_alert_generation(self):
        """Test alert generation when parity thresholds are breached."""
        # Create data that violates tracking error threshold
        violation_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1H'),
            'backtest_return': [0.001] * 50,
            'live_return': [0.004] * 50  # 30 bps difference (above 20 bps threshold)
        })
        
        alerts = self.parity_analyzer.generate_parity_alerts(violation_data)
        
        assert len(alerts) > 0
        assert any(alert.alert_type == 'tracking_error_breach' for alert in alerts)
        assert any(alert.severity in ['high', 'critical'] for alert in alerts)
        
        breach_alert = next(alert for alert in alerts if alert.alert_type == 'tracking_error_breach')
        assert breach_alert.current_value_bps > self.parity_analyzer.target_tracking_error_bps
    
    def test_continuous_monitoring_integration(self):
        """Test integration with continuous monitoring system."""
        # Simulate real-time data stream
        streaming_data = []
        
        for i in range(100):
            new_data_point = {
                'timestamp': datetime.now() - timedelta(hours=i),
                'symbol': 'BTC/USDT',
                'backtest_return': np.random.normal(0.001, 0.01),
                'live_return': np.random.normal(0.001, 0.01),
                'backtest_sharpe': 1.5,
                'live_sharpe': 1.4
            }
            streaming_data.append(new_data_point)
        
        monitoring_result = self.parity_analyzer.process_streaming_data(
            pd.DataFrame(streaming_data)
        )
        
        assert monitoring_result.current_tracking_error_bps is not None
        assert monitoring_result.trend_direction in ['improving', 'degrading', 'stable']
        assert monitoring_result.requires_attention is not None
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation for tracking error."""
        sample_data = pd.DataFrame({
            'backtest_return': np.random.normal(0.001, 0.02, 252),  # 1 year daily
            'live_return': np.random.normal(0.0012, 0.021, 252)  # Slight difference
        })
        
        confidence_intervals = self.parity_analyzer.calculate_confidence_intervals(
            sample_data['backtest_return'],
            sample_data['live_return'],
            confidence_levels=[0.95, 0.99]
        )
        
        assert 0.95 in confidence_intervals
        assert 0.99 in confidence_intervals
        
        ci_95 = confidence_intervals[0.95]
        ci_99 = confidence_intervals[0.99]
        
        assert ci_95.lower_bound_bps <= ci_95.upper_bound_bps
        assert ci_99.lower_bound_bps <= ci_99.upper_bound_bps
        
        # 99% CI should be wider than 95% CI
        assert (ci_99.upper_bound_bps - ci_99.lower_bound_bps) >= (ci_95.upper_bound_bps - ci_95.lower_bound_bps)
    
    def test_maximum_tracking_error_assertion(self):
        """Test assertion for maximum allowable tracking error."""
        # Test data within acceptable range
        acceptable_data = pd.DataFrame({
            'backtest_return': np.random.normal(0.001, 0.01, 100),
            'live_return': np.random.normal(0.0012, 0.01, 100)  # Small difference
        })
        
        # Should not raise assertion error
        self.parity_analyzer.assert_tracking_error_within_bounds(
            acceptable_data['backtest_return'],
            acceptable_data['live_return']
        )
        
        # Test data exceeding acceptable range
        unacceptable_data = pd.DataFrame({
            'backtest_return': np.random.normal(0.001, 0.01, 100),
            'live_return': np.random.normal(0.01, 0.01, 100)  # 90 bps difference
        })
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            self.parity_analyzer.assert_tracking_error_within_bounds(
                unacceptable_data['backtest_return'],
                unacceptable_data['live_return'],
                max_tracking_error_bps=20
            )


@pytest.mark.integration
@pytest.mark.slow
class TestParityValidationWorkflow:
    """Test complete parity validation workflow."""
    
    def test_end_to_end_parity_validation(self):
        """Test end-to-end parity validation workflow."""
        # This would integrate with actual trading system
        # For now, test with comprehensive mock data
        
        validation_workflow = BacktestParityAnalyzer(
            target_tracking_error_bps=20,
            validation_period_days=7,
            min_samples=100
        )
        
        # Generate realistic test scenario
        test_scenario = self._generate_realistic_test_scenario()
        
        # Run complete validation
        validation_result = validation_workflow.run_complete_validation(test_scenario)
        
        assert validation_result.overall_parity_score is not None
        assert 0 <= validation_result.overall_parity_score <= 100
        assert validation_result.validation_passed is not None
        assert validation_result.recommendations is not None
    
    def _generate_realistic_test_scenario(self):
        """Generate realistic test scenario data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=168, freq='1H'),
            'symbol': 'BTC/USDT',
            'backtest_return': np.random.normal(0.001, 0.02, 168),
            'live_return': np.random.normal(0.0008, 0.022, 168),
            'backtest_sharpe': np.random.uniform(1.3, 1.7, 168),
            'live_sharpe': np.random.uniform(1.2, 1.6, 168),
            'volume_usd': np.random.uniform(50000, 500000, 168),
            'market_regime': np.random.choice(['bull', 'bear', 'sideways'], 168)
        })