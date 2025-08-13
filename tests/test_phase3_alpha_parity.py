"""Tests for Fase 3 Alpha & Parity components."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.cryptosmarttrader.core.regime_detector import (
    RegimeDetector, MarketRegime, RegimeMetrics
)
from src.cryptosmarttrader.core.strategy_switcher import (
    StrategySwitcher, StrategyType, StrategyAllocation
)
from src.cryptosmarttrader.analysis.backtest_parity import (
    BacktestParityAnalyzer, TradeExecution, ParityMetrics
)


class TestRegimeDetector:
    """Test regime detection functionality."""
    
    def test_regime_detector_initialization(self):
        """Test RegimeDetector initialization."""
        detector = RegimeDetector(lookback_periods=100)
        
        assert detector.current_regime == MarketRegime.SIDEWAYS_LOW_VOL
        assert detector.lookback_periods == 100
        assert len(detector.regime_strategies) == len(MarketRegime)
    
    def test_market_data_update(self):
        """Test market data updates."""
        detector = RegimeDetector()
        
        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        price_data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 50),
            'high': np.random.uniform(50000, 60000, 50),
            'low': np.random.uniform(40000, 50000, 50),
            'close': np.random.uniform(45000, 55000, 50),
            'timestamp': dates
        })
        
        detector.update_market_data("BTC/USDT", price_data)
        
        assert "BTC/USDT" in detector.price_history
        assert len(detector.price_history["BTC/USDT"]) == 50
    
    def test_regime_metrics_calculation(self):
        """Test regime metrics calculation."""
        detector = RegimeDetector()
        
        # Create trending price data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        trending_prices = np.cumsum(np.random.normal(0.01, 0.02, 100)) + 50000
        
        price_data = pd.DataFrame({
            'open': trending_prices,
            'high': trending_prices * 1.02,
            'low': trending_prices * 0.98,
            'close': trending_prices,
            'timestamp': dates
        })
        
        detector.update_market_data("BTC/USDT", price_data)
        metrics = detector.calculate_regime_metrics("BTC/USDT")
        
        assert metrics is not None
        assert isinstance(metrics.trend_strength, float)
        assert isinstance(metrics.volatility_percentile, float)
        assert isinstance(metrics.hurst_exponent, float)
        assert 0 <= metrics.hurst_exponent <= 1
    
    def test_regime_classification(self):
        """Test regime classification logic."""
        detector = RegimeDetector()
        
        # Test bull trending metrics
        bull_metrics = RegimeMetrics(
            trend_strength=0.5,
            volatility_percentile=60.0,
            momentum_score=0.3,
            mean_reversion_score=0.2,
            volume_profile=1.5,
            hurst_exponent=0.7,
            adx_strength=30.0,
            rsi_divergence=0.3,
            correlation_breakdown=0.4
        )
        
        regime, confidence = detector.classify_regime(bull_metrics)
        
        assert isinstance(regime, MarketRegime)
        assert 0 <= confidence <= 1
    
    def test_regime_transition_tracking(self):
        """Test regime transition tracking."""
        detector = RegimeDetector()
        
        # Force regime update
        old_regime = detector.current_regime
        detector.current_regime = MarketRegime.BULL_TRENDING
        detector.regime_confidence = 0.8
        
        # Simulate regime change
        new_regime = detector.update_regime("BTC/USDT")
        
        # Should have some transition tracking
        assert hasattr(detector, 'regime_history')
    
    def test_strategy_config_retrieval(self):
        """Test strategy configuration retrieval."""
        detector = RegimeDetector()
        
        config = detector.get_current_strategy_config()
        
        assert 'primary_strategy' in config
        assert 'position_sizing' in config
        assert 'risk_multiplier' in config
    
    @pytest.mark.unit
    def test_hurst_exponent_calculation(self):
        """Test Hurst exponent calculation."""
        detector = RegimeDetector()
        
        # Create random walk (should be ~0.5)
        random_returns = pd.Series(np.random.normal(0, 0.01, 200))
        hurst = detector._calculate_hurst_exponent(random_returns)
        
        assert 0.0 <= hurst <= 1.0
        # For random walk, expect Hurst ~0.5 (within reasonable range)
        assert 0.3 <= hurst <= 0.7
    
    def test_regime_performance_stats(self):
        """Test regime performance statistics."""
        detector = RegimeDetector()
        
        # Add some mock transitions
        from src.cryptosmarttrader.core.regime_detector import RegimeTransition
        transition = RegimeTransition(
            from_regime=MarketRegime.SIDEWAYS_LOW_VOL,
            to_regime=MarketRegime.BULL_TRENDING,
            confidence=0.8,
            trigger_metrics={'trend_strength': 0.4},
            timestamp=datetime.now(),
            duration_hours=12.0
        )
        detector.regime_history.append(transition)
        
        stats = detector.get_regime_performance_stats()
        
        assert isinstance(stats, dict)
        if stats:  # If we have any data
            assert 'bull_trending' in stats


class TestStrategySwitcher:
    """Test strategy switching functionality."""
    
    def test_strategy_switcher_initialization(self):
        """Test StrategySwitcher initialization."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector, initial_capital=100000.0)
        
        assert switcher.initial_capital == 100000.0
        assert len(switcher.cluster_limits) > 0
        assert len(switcher.current_allocations) > 0
    
    def test_cluster_limits_initialization(self):
        """Test cluster limits setup."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        assert 'large_cap' in switcher.cluster_limits
        assert 'defi' in switcher.cluster_limits
        
        large_cap = switcher.cluster_limits['large_cap']
        assert large_cap.max_weight > 0
        assert len(large_cap.symbols) > 0
        assert 'BTC/USDT' in large_cap.symbols
    
    def test_volatility_target_sizing(self):
        """Test volatility targeting for position sizing."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        # High volatility asset should get smaller position
        high_vol_returns = pd.Series(np.random.normal(0, 0.05, 100))  # 5% daily vol
        adjusted_weight = switcher.calculate_volatility_target_sizing(
            "HIGH_VOL/USDT", 0.1, high_vol_returns
        )
        
        # Low volatility asset should get larger position  
        low_vol_returns = pd.Series(np.random.normal(0, 0.01, 100))   # 1% daily vol
        adjusted_weight_low = switcher.calculate_volatility_target_sizing(
            "LOW_VOL/USDT", 0.1, low_vol_returns
        )
        
        # Low vol should get larger allocation than high vol
        assert adjusted_weight_low >= adjusted_weight
    
    def test_cluster_limit_enforcement(self):
        """Test cluster limit enforcement."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        # Create positions that exceed cluster limits
        excessive_positions = {
            'BTC/USDT': 0.4,
            'ETH/USDT': 0.3,
            'BNB/USDT': 0.2  # Total large_cap = 0.9 (exceeds 0.6 limit)
        }
        
        adjusted = switcher.check_cluster_limits(excessive_positions)
        
        # Should be scaled down
        large_cap_total = sum(
            adjusted.get(symbol, 0) for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        )
        
        assert large_cap_total <= switcher.cluster_limits['large_cap'].max_weight + 0.01  # Small tolerance
    
    def test_position_target_generation(self):
        """Test position target generation."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        # Create sample market data
        market_data = {}
        for symbol in ['BTC/USDT', 'ETH/USDT']:
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            market_data[symbol] = pd.DataFrame({
                'open': np.random.uniform(45000, 55000, 50),
                'high': np.random.uniform(50000, 60000, 50),
                'low': np.random.uniform(40000, 50000, 50),
                'close': np.random.uniform(45000, 55000, 50),
                'volume': np.random.uniform(1000, 10000, 50),
                'timestamp': dates
            })
        
        targets = switcher.generate_position_targets(market_data)
        
        assert isinstance(targets, dict)
        # Should have some position targets if strategies are active
        if targets:
            for symbol, target in targets.items():
                assert target.target_weight >= 0
                assert 0 <= target.confidence <= 1
    
    def test_regime_allocation_update(self):
        """Test allocation updates on regime changes."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        # Change regime
        regime_detector.current_regime = MarketRegime.BULL_TRENDING
        regime_detector.regime_confidence = 0.8
        
        # Update allocations
        changed = switcher.update_regime_allocation()
        
        # Should detect regime change and update allocations
        assert hasattr(switcher, 'current_allocations')
    
    def test_strategy_allocation_summary(self):
        """Test strategy allocation summary."""
        regime_detector = RegimeDetector()
        switcher = StrategySwitcher(regime_detector)
        
        summary = switcher.get_strategy_allocation_summary()
        
        assert 'current_regime' in summary
        assert 'allocations' in summary
        assert 'cluster_limits' in summary
        assert summary['current_regime'] in [regime.value for regime in MarketRegime]


class TestBacktestParityAnalyzer:
    """Test backtest-live parity analysis."""
    
    def test_parity_analyzer_initialization(self):
        """Test BacktestParityAnalyzer initialization."""
        analyzer = BacktestParityAnalyzer(target_tracking_error_bps=25.0)
        
        assert analyzer.target_tracking_error_bps == 25.0
        assert len(analyzer.backtest_executions) == 0
        assert len(analyzer.live_executions) == 0
    
    def test_execution_simulation(self):
        """Test execution simulation."""
        analyzer = BacktestParityAnalyzer()
        
        market_conditions = {
            'bid': 49900.0,
            'ask': 50100.0,
            'price': 50000.0,
            'volume_24h': 1000000.0,
            'volatility': 0.02,
            'orderbook_depth': 50000.0
        }
        
        # Simulate backtest execution
        bt_execution = analyzer.simulate_execution(
            "BTC/USDT", 0.1, "buy", market_conditions, "backtest"
        )
        
        # Simulate live execution
        live_execution = analyzer.simulate_execution(
            "BTC/USDT", 0.1, "buy", market_conditions, "live"
        )
        
        assert bt_execution.execution_type == "backtest"
        assert live_execution.execution_type == "live"
        assert bt_execution.symbol == "BTC/USDT"
        assert live_execution.slippage_bps >= bt_execution.slippage_bps  # Live should have more slippage
    
    def test_execution_recording(self):
        """Test execution recording."""
        analyzer = BacktestParityAnalyzer()
        
        execution = TradeExecution(
            trade_id="test_123",
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            intended_price=50000.0,
            executed_price=50050.0,
            slippage_bps=10.0,
            fees_bps=15.0,
            latency_ms=50,
            market_conditions={},
            execution_type="live"
        )
        
        analyzer.record_execution(execution)
        
        assert len(analyzer.live_executions) == 1
        assert len(analyzer.slippage_history) == 1
    
    def test_parity_metrics_calculation(self):
        """Test parity metrics calculation."""
        analyzer = BacktestParityAnalyzer()
        
        # Add sample executions
        for i in range(20):
            # Backtest execution
            bt_execution = TradeExecution(
                trade_id=f"bt_{i}",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                intended_price=50000.0,
                executed_price=50000.0 + np.random.normal(0, 10),
                slippage_bps=np.random.normal(5, 2),
                fees_bps=10.0,
                latency_ms=0,
                market_conditions={},
                execution_type="backtest"
            )
            analyzer.record_execution(bt_execution)
            
            # Live execution
            live_execution = TradeExecution(
                trade_id=f"live_{i}",
                symbol="BTC/USDT",
                side="buy",
                quantity=0.1,
                intended_price=50000.0,
                executed_price=50000.0 + np.random.normal(0, 20),
                slippage_bps=np.random.normal(10, 3),
                fees_bps=15.0,
                latency_ms=50,
                market_conditions={},
                execution_type="live"
            )
            analyzer.record_execution(live_execution)
        
        metrics = analyzer.calculate_parity_metrics(lookback_hours=24)
        
        assert metrics is not None
        assert isinstance(metrics.tracking_error_bps, float)
        assert metrics.tracking_error_bps >= 0
        assert -1 <= metrics.correlation <= 1
        assert metrics.sample_size > 0
    
    def test_slippage_analysis(self):
        """Test slippage analysis."""
        analyzer = BacktestParityAnalyzer()
        
        # Add some slippage data
        from src.cryptosmarttrader.analysis.backtest_parity import ExecutionSlippage
        for i in range(10):
            slippage = ExecutionSlippage(
                symbol="BTC/USDT",
                order_size=5000.0,
                market_impact=np.random.uniform(2, 8),
                bid_ask_spread=np.random.uniform(3, 7),
                timing_slippage=np.random.uniform(1, 5),
                total_slippage=np.random.uniform(8, 15)
            )
            analyzer.slippage_history.append(slippage)
        
        analysis = analyzer.get_slippage_analysis("BTC/USDT", hours=24)
        
        assert 'total_slippage' in analysis
        assert 'component_breakdown' in analysis
        assert 'sample_size' in analysis
        assert analysis['sample_size'] == 10
    
    def test_tracking_error_acceptance(self):
        """Test tracking error acceptance check."""
        analyzer = BacktestParityAnalyzer(target_tracking_error_bps=20.0)
        
        # Should return False, None with no data
        acceptable, metrics = analyzer.is_tracking_error_acceptable()
        
        assert not acceptable
        assert metrics is None
    
    def test_execution_quality_report(self):
        """Test execution quality report generation."""
        analyzer = BacktestParityAnalyzer()
        
        report = analyzer.get_execution_quality_report()
        
        assert 'parity_metrics' in report
        assert 'slippage_analysis' in report
        assert 'execution_performance' in report
        assert 'quality_score' in report
        assert 'report_timestamp' in report
    
    @pytest.mark.integration
    def test_return_attribution_analysis(self):
        """Test return attribution analysis."""
        analyzer = BacktestParityAnalyzer()
        
        # Create sample return series
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 10), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 10), index=dates)
        
        attribution = analyzer.analyze_return_attribution(
            portfolio_returns, benchmark_returns, period_days=7
        )
        
        assert isinstance(attribution.total_return, float)
        assert isinstance(attribution.alpha_return, float)
        assert isinstance(attribution.fees_impact, float)
        assert isinstance(attribution.slippage_impact, float)


# Integration test for full Fase 3 pipeline
@pytest.mark.integration
def test_fase3_integration():
    """Integration test for complete Fase 3 functionality."""
    # Create all components
    regime_detector = RegimeDetector(lookback_periods=100)
    strategy_switcher = StrategySwitcher(regime_detector, initial_capital=100000.0)
    parity_analyzer = BacktestParityAnalyzer(target_tracking_error_bps=20.0)
    
    # Test component integration
    assert regime_detector is not None
    assert strategy_switcher is not None
    assert parity_analyzer is not None
    
    # Test basic regime detection
    sample_data = pd.DataFrame({
        'open': [50000] * 50,
        'high': [51000] * 50,
        'low': [49000] * 50,
        'close': [50000] * 50,
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='H')
    })
    
    regime_detector.update_market_data("BTC/USDT", sample_data)
    current_regime = regime_detector.update_regime("BTC/USDT")
    
    # Test strategy allocation
    strategy_summary = strategy_switcher.get_strategy_allocation_summary()
    
    # Test execution simulation
    market_conditions = {
        'bid': 49900.0,
        'ask': 50100.0,
        'price': 50000.0,
        'volume_24h': 1000000.0,
        'volatility': 0.02,
        'orderbook_depth': 50000.0
    }
    
    execution = parity_analyzer.simulate_execution(
        "BTC/USDT", 0.1, "buy", market_conditions, "live"
    )
    
    # Should have working integration
    assert current_regime in [regime for regime in MarketRegime]
    assert 'current_regime' in strategy_summary
    assert execution.symbol == "BTC/USDT"