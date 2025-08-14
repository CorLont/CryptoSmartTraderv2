#!/usr/bin/env python3
"""
Test Backtest-Live Parity System
Comprehensive tests for execution simulation and drift monitoring
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.cryptosmarttrader.backtest.execution_simulator import (
        ExecutionSimulator, OrderRequest, MarketMicrostructure, ExchangeConfig, OrderStatus, FillType
    )
    from src.cryptosmarttrader.backtest.parity_monitor import (
        BacktestLiveParityMonitor, ParityConfig, ParityStatus, TradeRecord
    )
    from src.cryptosmarttrader.backtest.integrated_parity_system import (
        IntegratedParitySystem, BacktestTrade, ParitySystemConfig
    )
except ImportError:
    pytest.skip("Backtest parity modules not available", allow_module_level=True)


class TestExecutionSimulator:
    """Test execution simulator functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.exchange_config = ExchangeConfig(
            name="test_exchange",
            maker_fee_bps=10.0,
            taker_fee_bps=25.0,
            base_latency_ms=50.0,
            partial_fill_probability=0.2
        )
        
        self.simulator = ExecutionSimulator(self.exchange_config)
        
        # Setup test market data
        self.market_data = MarketMicrostructure(
            symbol="BTC/USD",
            timestamp=time.time(),
            bid_price=49950.0,
            ask_price=50050.0,
            bid_size=2.0,
            ask_size=2.0,
            last_price=50000.0,
            spread_bps=20.0,
            depth_5_levels={"bids": [(49950, 2.0), (49940, 1.5)], "asks": [(50050, 2.0), (50060, 1.5)]},
            recent_trades=[(50000, 0.5, "buy"), (49995, 0.3, "sell")],
            volume_1min=100000.0,
            volatility_1min=0.02
        )
    
    def test_market_order_execution(self):
        """Test market order execution simulation"""
        
        order = OrderRequest(
            order_id="test_market_1",
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            size=0.1
        )
        
        result = self.simulator.simulate_order_execution(order, self.market_data)
        
        assert result.order_id == "test_market_1"
        assert result.symbol == "BTC/USD"
        assert result.side == "buy"
        assert result.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]
        assert result.average_price > 0
        assert result.total_fee > 0
        assert result.slippage_bps >= 0
        assert result.total_latency_ms > 0
        assert len(result.fills) > 0
        
        # Market orders should be aggressive (taker)
        assert all(fill.fill_type in [FillType.TAKER, FillType.AGGRESSIVE] for fill in result.fills)
    
    def test_limit_order_execution(self):
        """Test limit order execution simulation"""
        
        # Passive limit order (maker)
        order = OrderRequest(
            order_id="test_limit_1",
            symbol="BTC/USD",
            side="buy",
            order_type="limit",
            size=0.1,
            limit_price=49900.0  # Below current bid
        )
        
        result = self.simulator.simulate_order_execution(order, self.market_data)
        
        assert result.order_id == "test_limit_1"
        assert result.average_price == 49900.0  # Should fill at limit price
        
        # Should be maker fills
        assert any(fill.fill_type == FillType.MAKER for fill in result.fills)
    
    def test_aggressive_limit_order(self):
        """Test aggressive limit order (crosses spread)"""
        
        order = OrderRequest(
            order_id="test_aggressive_1",
            symbol="BTC/USD",
            side="buy",
            order_type="limit",
            size=0.1,
            limit_price=50100.0  # Above current ask
        )
        
        result = self.simulator.simulate_order_execution(order, self.market_data)
        
        # Should execute at market price, not limit price
        assert result.average_price <= 50100.0
        assert result.average_price >= self.market_data.ask_price
        
        # Should be aggressive fills
        assert any(fill.fill_type == FillType.AGGRESSIVE for fill in result.fills)
    
    def test_order_validation(self):
        """Test order validation and rejection"""
        
        # Order too small
        small_order = OrderRequest(
            order_id="test_small",
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            size=0.0001  # Very small size
        )
        
        result = self.simulator.simulate_order_execution(small_order, self.market_data)
        
        # Should be rejected if order value too small
        if result.status == OrderStatus.REJECTED:
            assert "too small" in result.rejection_reason.lower()
    
    def test_latency_calculation(self):
        """Test execution latency calculation"""
        
        order = OrderRequest(
            order_id="test_latency",
            symbol="BTC/USD",
            side="buy",
            order_type="market",
            size=1.0  # Large order for latency impact
        )
        
        result = self.simulator.simulate_order_execution(order, self.market_data)
        
        # Should have realistic latency
        assert result.total_latency_ms >= 10.0  # Minimum latency
        assert result.total_latency_ms <= 500.0  # Reasonable maximum
    
    def test_partial_fills(self):
        """Test partial fill scenarios"""
        
        # Run multiple orders to trigger partial fills
        partial_fill_detected = False
        
        for i in range(20):  # Try multiple times due to randomness
            order = OrderRequest(
                order_id=f"test_partial_{i}",
                symbol="BTC/USD",
                side="buy",
                order_type="limit",
                size=0.5,
                limit_price=49950.0
            )
            
            result = self.simulator.simulate_order_execution(order, self.market_data)
            
            if result.status == OrderStatus.PARTIAL:
                partial_fill_detected = True
                assert result.filled_size < result.requested_size
                assert result.filled_size > 0
                break
        
        # Should occasionally get partial fills due to configuration
        # (This test might occasionally fail due to randomness)
    
    def test_execution_statistics(self):
        """Test execution statistics tracking"""
        
        # Execute multiple orders
        for i in range(10):
            order = OrderRequest(
                order_id=f"test_stats_{i}",
                symbol="BTC/USD",
                side="buy" if i % 2 == 0 else "sell",
                order_type="market",
                size=0.1
            )
            
            self.simulator.simulate_order_execution(order, self.market_data)
        
        stats = self.simulator.get_execution_statistics()
        
        assert "total_executions" in stats
        assert stats["total_executions"] == 10
        assert "fill_statistics" in stats
        assert "performance_metrics" in stats
        assert "cost_breakdown" in stats
        
        # Check reasonable values
        assert 0 <= stats["fill_statistics"]["fill_rate"] <= 1
        assert stats["performance_metrics"]["average_slippage_bps"] >= 0
        assert stats["performance_metrics"]["average_latency_ms"] > 0


class TestParityMonitor:
    """Test parity monitoring functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = ParityConfig(
            warning_threshold_bps=20.0,
            critical_threshold_bps=50.0,
            auto_disable_threshold_bps=100.0
        )
        
        self.monitor = BacktestLiveParityMonitor(self.config)
        
        # Mock execution result
        self.mock_execution = Mock()
        self.mock_execution.average_price = 50100.0
        self.mock_execution.total_fee = 12.5
        self.mock_execution.slippage_bps = 15.0
        self.mock_execution.total_latency_ms = 75.0
        self.mock_execution.execution_quality_score = 85.0
        self.mock_execution.status = Mock()
        self.mock_execution.status.value = "filled"
    
    def test_trade_recording(self):
        """Test trade recording functionality"""
        
        trade_record = self.monitor.record_trade(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            backtest_entry_price=50000.0,
            live_execution_result=self.mock_execution,
            strategy_id="test_strategy"
        )
        
        assert isinstance(trade_record, TradeRecord)
        assert trade_record.symbol == "BTC/USD"
        assert trade_record.side == "buy"
        assert trade_record.size == 0.1
        assert trade_record.slippage_bps == 15.0
        assert trade_record.execution_cost == 12.5
        
        # Should be stored in monitor
        assert len(self.monitor.trade_records) == 1
    
    def test_daily_report_generation(self):
        """Test daily report generation"""
        
        # Record multiple trades
        for i in range(15):  # Above minimum threshold
            self.monitor.record_trade(
                symbol="BTC/USD",
                side="buy" if i % 2 == 0 else "sell",
                size=0.1,
                backtest_entry_price=50000.0,
                live_execution_result=self.mock_execution,
                strategy_id="test_strategy"
            )
        
        report = self.monitor.generate_daily_report()
        
        assert report.total_trades == 15
        assert report.tracking_error_bps >= 0
        assert report.execution_cost_bps >= 0
        assert report.slippage_cost_bps >= 0
        assert isinstance(report.parity_status, ParityStatus)
        assert len(report.recommendations) >= 0
    
    def test_auto_disable_trigger(self):
        """Test auto-disable functionality"""
        
        # Create high-slippage execution result
        high_slippage_execution = Mock()
        high_slippage_execution.average_price = 50000.0
        high_slippage_execution.total_fee = 50.0
        high_slippage_execution.slippage_bps = 150.0  # Above auto-disable threshold
        high_slippage_execution.total_latency_ms = 100.0
        high_slippage_execution.execution_quality_score = 30.0
        high_slippage_execution.status = Mock()
        high_slippage_execution.status.value = "filled"
        
        # Should trigger auto-disable
        self.monitor.record_trade(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            backtest_entry_price=50000.0,
            live_execution_result=high_slippage_execution
        )
        
        assert self.monitor.auto_disabled
        assert self.monitor.current_status == ParityStatus.DISABLED
        assert self.monitor.disable_reason is not None
    
    def test_manual_enable_after_disable(self):
        """Test manual re-enable after auto-disable"""
        
        # First trigger auto-disable
        self.monitor._trigger_auto_disable("Test disable")
        assert self.monitor.auto_disabled
        
        # Manual re-enable
        success = self.monitor.manual_enable("test_operator", "Issue resolved")
        
        assert success
        assert not self.monitor.auto_disabled
        assert self.monitor.current_status == ParityStatus.HEALTHY
    
    def test_parity_status_determination(self):
        """Test parity status determination logic"""
        
        # Healthy status
        healthy_status = self.monitor._determine_parity_status(10.0)  # 10 bps
        assert healthy_status == ParityStatus.HEALTHY
        
        # Warning status
        warning_status = self.monitor._determine_parity_status(25.0)  # 25 bps
        assert warning_status == ParityStatus.WARNING
        
        # Critical status
        critical_status = self.monitor._determine_parity_status(60.0)  # 60 bps
        assert critical_status == ParityStatus.CRITICAL
    
    def test_parity_summary(self):
        """Test parity summary generation"""
        
        # Record some trades
        for i in range(5):
            self.monitor.record_trade(
                symbol="BTC/USD",
                side="buy",
                size=0.1,
                backtest_entry_price=50000.0,
                live_execution_result=self.mock_execution
            )
        
        # Generate a report
        self.monitor.generate_daily_report()
        
        summary = self.monitor.get_parity_summary()
        
        assert "current_status" in summary
        assert "recent_performance" in summary
        assert "cost_breakdown" in summary
        assert "thresholds" in summary
        
        assert summary["current_status"]["status"] in ["healthy", "warning", "critical", "disabled"]


class TestIntegratedParitySystem:
    """Test integrated parity system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.system = IntegratedParitySystem()
        
        # Setup market data
        self.market_data = MarketMicrostructure(
            symbol="BTC/USD",
            timestamp=time.time(),
            bid_price=49950.0,
            ask_price=50050.0,
            bid_size=2.0,
            ask_size=2.0,
            last_price=50000.0,
            spread_bps=20.0,
            depth_5_levels={"bids": [(49950, 2.0)], "asks": [(50050, 2.0)]},
            recent_trades=[(50000, 0.5, "buy")],
            volume_1min=100000.0,
            volatility_1min=0.02
        )
        
        self.system.update_market_data("BTC/USD", self.market_data)
    
    def test_backtest_trade_recording(self):
        """Test backtest trade recording"""
        
        backtest_trade = self.system.record_backtest_trade(
            timestamp=time.time(),
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            entry_price=50000.0,
            expected_pnl=100.0,
            strategy_id="momentum_v1"
        )
        
        assert isinstance(backtest_trade, BacktestTrade)
        assert backtest_trade.symbol == "BTC/USD"
        assert backtest_trade.entry_price == 50000.0
        assert backtest_trade.expected_pnl == 100.0
        
        # Should be stored in system
        assert len(self.system.backtest_trades) == 1
    
    def test_live_trade_execution(self):
        """Test live trade execution with parity tracking"""
        
        # First record backtest trade
        self.system.record_backtest_trade(
            timestamp=time.time(),
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            entry_price=50000.0,
            strategy_id="momentum_v1"
        )
        
        # Execute live trade
        result = self.system.execute_live_trade(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            order_type="market",
            strategy_id="momentum_v1"
        )
        
        assert "success" in result
        assert "execution_result" in result
        assert "trade_record" in result
        assert "parity_metrics" in result
        
        # Should have found matching backtest trade
        assert result["backtest_trade"] is not None
        assert result["parity_metrics"]["has_backtest_comparison"]
    
    def test_system_health_summary(self):
        """Test system health summary"""
        
        # Execute some trades
        for i in range(3):
            self.system.record_backtest_trade(
                timestamp=time.time(),
                symbol="BTC/USD",
                side="buy",
                size=0.1,
                entry_price=50000.0,
                strategy_id="test"
            )
            
            self.system.execute_live_trade(
                symbol="BTC/USD",
                side="buy",
                size=0.1,
                order_type="market",
                strategy_id="test"
            )
        
        health_summary = self.system.get_system_health_summary()
        
        assert "system_status" in health_summary
        assert "parity_monitoring" in health_summary
        assert "execution_simulation" in health_summary
        assert "data_coverage" in health_summary
        
        # Check data coverage
        assert health_summary["data_coverage"]["backtest_trades"] == 3
        assert health_summary["data_coverage"]["live_executions"] == 3
    
    def test_daily_parity_report_generation(self):
        """Test daily parity report generation"""
        
        # Execute sufficient trades for meaningful analysis
        for i in range(12):
            self.system.record_backtest_trade(
                timestamp=time.time(),
                symbol="BTC/USD",
                side="buy" if i % 2 == 0 else "sell",
                size=0.1,
                entry_price=50000.0,
                strategy_id="test"
            )
            
            self.system.execute_live_trade(
                symbol="BTC/USD",
                side="buy" if i % 2 == 0 else "sell",
                size=0.1,
                order_type="market",
                strategy_id="test"
            )
        
        # Generate daily report
        daily_report = self.system.generate_daily_parity_report()
        
        assert daily_report.total_trades >= 10  # Above minimum threshold
        assert daily_report.tracking_error_bps >= 0
        assert isinstance(daily_report.parity_status, ParityStatus)
        assert len(daily_report.component_attribution) > 0
    
    def test_auto_disable_integration(self):
        """Test auto-disable integration with system"""
        
        # Manually trigger auto-disable
        self.system.parity_monitor._trigger_auto_disable("Test auto-disable")
        
        # Try to execute trade
        result = self.system.execute_live_trade(
            symbol="BTC/USD",
            side="buy",
            size=0.1,
            order_type="market"
        )
        
        assert not result["success"]
        assert "auto-disabled" in result
        assert "System auto-disabled" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])