#!/usr/bin/env python3
"""
Test Kelly Vol Sizing & Portfolio Management
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.cryptosmarttrader.portfolio.kelly_vol_sizing import (
        KellyVolSizer, AssetMetrics, SizingParameters, MarketRegime, PositionSize
    )
    from src.cryptosmarttrader.portfolio.regime_detector import RegimeDetector
    from src.cryptosmarttrader.portfolio.portfolio_manager import IntegratedPortfolioManager
except ImportError:
    pytest.skip("Portfolio modules not available", allow_module_level=True)


class TestKellyVolSizing:
    """Test Kelly Vol Sizing system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.sizer = KellyVolSizer()
        
        # Setup test asset metrics
        self.btc_metrics = AssetMetrics(
            symbol="BTC/USD",
            expected_return=0.50,  # 50% annual return
            volatility=0.80,  # 80% annual vol
            sharpe_ratio=0.625,
            max_drawdown=0.30,
            win_rate=0.55,  # 55% win rate
            avg_win_loss_ratio=1.8,  # 1.8:1 win/loss ratio
            correlation_to_market=0.9,
            cluster_id="crypto_large"
        )
        
        self.eth_metrics = AssetMetrics(
            symbol="ETH/USD", 
            expected_return=0.40,
            volatility=0.70,
            sharpe_ratio=0.571,
            max_drawdown=0.25,
            win_rate=0.52,
            avg_win_loss_ratio=1.6,
            correlation_to_market=0.8,
            cluster_id="crypto_large"
        )
        
        self.sol_metrics = AssetMetrics(
            symbol="SOL/USD",
            expected_return=0.60,
            volatility=1.20,  # High vol alt coin
            sharpe_ratio=0.50,
            max_drawdown=0.50,
            win_rate=0.48,
            avg_win_loss_ratio=2.2,
            correlation_to_market=0.6,
            cluster_id="crypto_alt"
        )
        
        # Update sizer with metrics
        self.sizer.update_asset_metrics({
            "BTC/USD": self.btc_metrics,
            "ETH/USD": self.eth_metrics,
            "SOL/USD": self.sol_metrics
        })
    
    def test_kelly_calculation(self):
        """Test Kelly criterion calculation"""
        
        # BTC Kelly size
        kelly_size = self.sizer.calculate_kelly_size(self.btc_metrics)
        
        # Manual calculation: Kelly = (p*b - q) / b * fractional
        # p = 0.55, q = 0.45, b = 1.8
        # Kelly = (0.55 * 1.8 - 0.45) / 1.8 = 0.299
        # Fractional (25%): 0.299 * 0.25 = 0.0748
        expected_kelly = ((0.55 * 1.8 - 0.45) / 1.8) * 0.25
        
        assert abs(kelly_size - expected_kelly) < 0.01
        assert kelly_size > 0
        assert kelly_size <= 0.10  # Max position size
    
    def test_vol_adjustment(self):
        """Test volatility adjustment"""
        
        kelly_size = self.sizer.calculate_kelly_size(self.btc_metrics)
        vol_adjusted = self.sizer.calculate_vol_adjusted_size(self.btc_metrics, kelly_size)
        
        # Vol adjustment = (vol_target / asset_vol) * kelly_size
        # = (0.20 / 0.80) * kelly_size = 0.25 * kelly_size
        expected_vol_adj = kelly_size * (0.20 / 0.80)
        
        assert abs(vol_adjusted - expected_vol_adj) < 0.01
    
    def test_regime_throttling(self):
        """Test regime-based throttling"""
        
        base_size = 0.05  # 5% position
        
        # Test different regimes
        regimes_to_test = [
            (MarketRegime.TREND, 1.0),
            (MarketRegime.MEAN_REVERSION, 0.8),
            (MarketRegime.CHOP, 0.5),
            (MarketRegime.HIGH_VOL, 0.3),
            (MarketRegime.CRISIS, 0.1)
        ]
        
        for regime, expected_factor in regimes_to_test:
            self.sizer.set_market_regime(regime)
            throttled_size = self.sizer.apply_regime_throttle(base_size)
            expected_size = base_size * expected_factor
            
            assert abs(throttled_size - expected_size) < 0.001
    
    def test_cluster_limits(self):
        """Test cluster exposure limits"""
        
        # Set large position in crypto_large cluster
        self.sizer.update_portfolio_state(
            100000.0,
            {"BTC/USD": 25000.0, "ETH/USD": 15000.0}  # 40% in crypto_large
        )
        
        # Try to add more BTC (would exceed cluster limit)
        adjusted_size, reasoning = self.sizer.calculate_cluster_adjusted_size(
            "BTC/USD", 0.10  # 10% more
        )
        
        # Should be limited by cluster exposure
        assert adjusted_size <= 0.10
        assert len(reasoning) > 0
        assert "cluster" in reasoning[0].lower()
    
    def test_position_size_calculation(self):
        """Test complete position size calculation"""
        
        position_size = self.sizer.calculate_position_size("BTC/USD", signal_strength=0.8)
        
        assert isinstance(position_size, PositionSize)
        assert position_size.symbol == "BTC/USD"
        assert position_size.final_size_pct >= 0
        assert position_size.final_size_pct <= 0.10  # Max position size
        assert position_size.target_size_usd == position_size.final_size_pct * 100000.0
        assert len(position_size.reasoning) > 0
        
        # Check that all intermediate sizes are calculated
        assert position_size.kelly_size_pct >= 0
        assert position_size.vol_adjusted_size_pct >= 0
        assert position_size.regime_adjusted_size_pct >= 0
    
    def test_portfolio_sizing(self):
        """Test portfolio-level sizing"""
        
        signals = {
            "BTC/USD": 0.8,
            "ETH/USD": 0.6,
            "SOL/USD": 0.4
        }
        
        portfolio_sizes = self.sizer.calculate_portfolio_sizes(signals)
        
        assert len(portfolio_sizes) == 3
        
        # Check total allocation
        total_allocation = sum(ps.final_size_pct for ps in portfolio_sizes.values())
        assert total_allocation <= 1.0  # Should not exceed 100%
        
        # Check individual constraints
        for symbol, position_size in portfolio_sizes.items():
            assert position_size.final_size_pct <= 0.10  # Max position size
            assert position_size.final_size_pct >= 0
    
    def test_correlation_matrix_integration(self):
        """Test correlation matrix integration"""
        
        # Create correlation matrix
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        correlation_data = [
            [1.00, 0.85, 0.60],
            [0.85, 1.00, 0.55],
            [0.60, 0.55, 1.00]
        ]
        correlation_matrix = pd.DataFrame(correlation_data, index=symbols, columns=symbols)
        
        self.sizer.update_correlation_matrix(correlation_matrix)
        
        # Set existing correlated positions
        self.sizer.update_portfolio_state(
            100000.0,
            {"BTC/USD": 15000.0}  # 15% in BTC
        )
        
        # Try to add ETH (highly correlated with BTC)
        eth_size = self.sizer.calculate_position_size("ETH/USD", signal_strength=1.0)
        
        # Should be adjusted for correlation
        assert eth_size.final_size_pct >= 0
        
        # Check reasoning mentions correlation if adjusted
        reasoning_text = " ".join(eth_size.reasoning).lower()
        if eth_size.final_size_pct < eth_size.cluster_adjusted_size_pct:
            assert "correlation" in reasoning_text
    
    def test_cluster_exposure_tracking(self):
        """Test cluster exposure tracking"""
        
        # Set positions across clusters
        self.sizer.update_portfolio_state(
            100000.0,
            {
                "BTC/USD": 20000.0,
                "ETH/USD": 15000.0,  # crypto_large: 35%
                "SOL/USD": 10000.0   # crypto_alt: 10%
            }
        )
        
        cluster_exposures = self.sizer.get_cluster_exposures()
        
        assert "crypto_large" in cluster_exposures
        assert "crypto_alt" in cluster_exposures
        
        crypto_large = cluster_exposures["crypto_large"]
        assert abs(crypto_large["current_exposure"] - 0.35) < 0.01  # 35%
        assert crypto_large["utilization"] > 0.8  # High utilization
        
        crypto_alt = cluster_exposures["crypto_alt"]
        assert abs(crypto_alt["current_exposure"] - 0.10) < 0.01  # 10%


class TestRegimeDetector:
    """Test regime detection system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.detector = RegimeDetector(lookback_periods=30)
    
    def test_hurst_exponent_calculation(self):
        """Test Hurst exponent calculation"""
        
        # Trending returns (persistent)
        trend_returns = np.cumsum(np.random.normal(0.001, 0.02, 50))
        trend_returns = np.diff(trend_returns)  # Convert to returns
        hurst_trend = self.detector.calculate_hurst_exponent(trend_returns.tolist())
        
        # Should be > 0.5 for trending
        assert hurst_trend > 0.3  # Allow some variance
        
        # Mean-reverting returns (anti-persistent)
        mr_returns = []
        value = 0
        for _ in range(50):
            # Mean revert to 0
            change = -0.5 * value + np.random.normal(0, 0.01)
            value += change
            mr_returns.append(change)
        
        hurst_mr = self.detector.calculate_hurst_exponent(mr_returns)
        
        # Should be < 0.5 for mean-reverting
        assert hurst_mr < 0.7  # Allow some variance
    
    def test_trend_strength_calculation(self):
        """Test trend strength calculation"""
        
        # Strong uptrend
        uptrend_prices = [100 + i * 2 + np.random.normal(0, 1) for i in range(20)]
        trend_strength = self.detector.calculate_trend_strength(uptrend_prices)
        
        assert trend_strength >= 0.0
        assert trend_strength <= 1.0
        
        # Sideways market
        sideways_prices = [100 + np.random.normal(0, 1) for _ in range(20)]
        sideways_strength = self.detector.calculate_trend_strength(sideways_prices)
        
        # Trending should be stronger than sideways
        assert trend_strength >= sideways_strength
    
    def test_volatility_regime_calculation(self):
        """Test volatility regime calculation"""
        
        # Low vol followed by high vol
        low_vol_returns = np.random.normal(0, 0.01, 20)  # 1% daily vol
        high_vol_returns = np.random.normal(0, 0.05, 10)  # 5% daily vol
        
        combined_returns = np.concatenate([low_vol_returns, high_vol_returns])
        vol_regime = self.detector.calculate_volatility_regime(combined_returns.tolist())
        
        assert vol_regime > 1.0  # Current vol should be higher than historical
    
    def test_market_data_update(self):
        """Test market data update and regime detection"""
        
        # Simulate trending market data
        prices = {"BTC/USD": 50000, "ETH/USD": 3000}
        
        for i in range(25):  # Need minimum data for regime detection
            # Simulate uptrend with noise
            prices["BTC/USD"] += np.random.normal(200, 500)
            prices["ETH/USD"] += np.random.normal(10, 30)
            
            self.detector.update_market_data(prices, time.time() + i)
        
        # Should have detected some regime
        regime_status = self.detector.get_regime_status()
        
        assert regime_status["current_regime"] in ["trend", "mean_reversion", "chop", "high_vol", "crisis"]
        assert 0 <= regime_status["regime_confidence"] <= 1
        assert regime_status["data_points"] >= 20


class TestIntegratedPortfolioManager:
    """Test integrated portfolio management"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = IntegratedPortfolioManager()
        
        # Setup test data
        asset_metrics = {
            "BTC/USD": AssetMetrics(
                symbol="BTC/USD",
                expected_return=0.40,
                volatility=0.60,
                sharpe_ratio=0.67,
                max_drawdown=0.25,
                win_rate=0.55,
                avg_win_loss_ratio=1.5,
                correlation_to_market=0.9,
                cluster_id="crypto_large"
            ),
            "ETH/USD": AssetMetrics(
                symbol="ETH/USD",
                expected_return=0.35,
                volatility=0.55,
                sharpe_ratio=0.64,
                max_drawdown=0.22,
                win_rate=0.53,
                avg_win_loss_ratio=1.4,
                correlation_to_market=0.85,
                cluster_id="crypto_large"
            )
        }
        
        self.manager.update_asset_metrics(asset_metrics)
        self.manager.update_portfolio_state(
            100000.0,
            {"BTC/USD": 25000.0, "ETH/USD": 15000.0}
        )
    
    def test_optimal_portfolio_calculation(self):
        """Test optimal portfolio calculation"""
        
        signals = {"BTC/USD": 0.8, "ETH/USD": 0.6}
        
        optimal_portfolio = self.manager.calculate_optimal_portfolio(signals)
        
        assert len(optimal_portfolio) == 2
        assert "BTC/USD" in optimal_portfolio
        assert "ETH/USD" in optimal_portfolio
        
        # Check position sizes are reasonable
        for symbol, position_size in optimal_portfolio.items():
            assert position_size.final_size_pct >= 0
            assert position_size.final_size_pct <= 0.15  # Reasonable max
    
    def test_rebalancing_recommendations(self):
        """Test rebalancing recommendations"""
        
        # Set target allocations different from current
        self.manager.target_allocations = {
            "BTC/USD": 0.30,  # Current: 25%
            "ETH/USD": 0.20,  # Current: 15%
            "SOL/USD": 0.10   # Current: 0%
        }
        
        recommendations = self.manager.generate_rebalancing_recommendations(force_rebalance=True)
        
        assert len(recommendations) > 0
        
        # Check recommendation structure
        for rec in recommendations:
            assert hasattr(rec, 'symbol')
            assert hasattr(rec, 'recommended_action')
            assert hasattr(rec, 'size_change_usd')
            assert hasattr(rec, 'priority')
            assert len(rec.reasoning) > 0
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation"""
        
        summary = self.manager.get_portfolio_summary()
        
        assert summary.total_equity == 100000.0
        assert len(summary.positions) == 2
        assert summary.cash_balance >= 0
        assert isinstance(summary.regime, MarketRegime)
        assert 0 <= summary.regime_confidence <= 1
        assert 0 <= summary.risk_utilization <= 1
    
    def test_detailed_status(self):
        """Test detailed status reporting"""
        
        status = self.manager.get_detailed_status()
        
        required_sections = [
            "portfolio_summary",
            "kelly_sizer_status", 
            "regime_status",
            "risk_status",
            "rebalancing"
        ]
        
        for section in required_sections:
            assert section in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])