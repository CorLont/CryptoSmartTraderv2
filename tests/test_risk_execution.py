#!/usr/bin/env python3
"""
Unit Tests for Risk Guard and Execution Policy
Tests for kill-switch, data gaps, duplicate orders, and gate enforcement
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.cryptosmarttrader.risk.centralized_risk_guard import (
    CentralizedRiskGuard, RiskLimits, RiskMetrics, KillSwitchState, RiskLevel
)
from src.cryptosmarttrader.execution.execution_policy import (
    ExecutionPolicy, ExecutionGates, OrderRequest, OrderType, TimeInForce, 
    MarketData, ExecutionResult, OrderStatus
)


class TestCentralizedRiskGuard:
    """Test suite for CentralizedRiskGuard"""
    
    def setup_method(self):
        """Setup for each test"""
        self.limits = RiskLimits(
            daily_loss_limit_usd=1000.0,
            max_drawdown_pct=0.10,
            max_total_exposure_usd=10000.0,
            max_open_positions=5,
            max_data_gap_minutes=2
        )
        self.risk_guard = CentralizedRiskGuard(self.limits)
        self.risk_guard.start_monitoring()
    
    def test_daily_loss_limit_trigger(self):
        """Test daily loss limit triggers emergency stop"""
        # Setup metrics with loss exceeding limit
        metrics = RiskMetrics(daily_pnl_usd=-1500.0)  # Exceeds $1000 limit
        
        # Mock alert callback
        alert_callback = Mock()
        self.risk_guard.add_alert_callback(alert_callback)
        
        # Update metrics
        self.risk_guard.update_metrics(metrics)
        
        # Verify kill switch triggered
        assert self.risk_guard.kill_switch_state == KillSwitchState.EMERGENCY
        assert self.risk_guard.risk_level == RiskLevel.EMERGENCY
        
        # Verify alert sent (should be called multiple times)
        assert alert_callback.call_count >= 1
        
        # Check for kill_switch alert in call history
        kill_switch_alerts = [
            call[0][0] for call in alert_callback.call_args_list 
            if call[0][0].get('type') == 'kill_switch'
        ]
        assert len(kill_switch_alerts) >= 1
        assert kill_switch_alerts[0]['state'] == 'emergency'
    
    def test_drawdown_limit_trigger(self):
        """Test max drawdown triggers hard stop"""
        # Setup metrics with excessive drawdown
        metrics = RiskMetrics(current_drawdown_pct=0.15)  # Exceeds 10% limit
        
        # Update metrics
        self.risk_guard.update_metrics(metrics)
        
        # Verify hard stop triggered
        assert self.risk_guard.kill_switch_state == KillSwitchState.HARD_STOP
        assert self.risk_guard.risk_level == RiskLevel.CRITICAL
    
    def test_data_gap_trigger(self):
        """Test data gap triggers hard stop"""
        # Setup metrics with data gap
        metrics = RiskMetrics(data_gap_minutes=5.0)  # Exceeds 2 minute limit
        
        # Update metrics
        self.risk_guard.update_metrics(metrics)
        
        # Verify hard stop triggered
        assert self.risk_guard.kill_switch_state == KillSwitchState.HARD_STOP
        assert self.risk_guard.risk_level == RiskLevel.CRITICAL
    
    def test_exposure_limit_trigger(self):
        """Test exposure limit triggers soft stop"""
        # Setup metrics with excessive exposure
        metrics = RiskMetrics(total_exposure_usd=15000.0)  # Exceeds $10k limit
        
        # Update metrics
        self.risk_guard.update_metrics(metrics)
        
        # Verify soft stop triggered
        assert self.risk_guard.kill_switch_state == KillSwitchState.SOFT_STOP
        assert self.risk_guard.risk_level == RiskLevel.CRITICAL
    
    def test_order_allowed_normal_state(self):
        """Test order allowed in normal state"""
        order = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 1.0,
            'price': 50000.0
        }
        
        allowed, reason = self.risk_guard.check_order_allowed(order)
        assert allowed
        assert "allowed" in reason
    
    def test_order_blocked_emergency_state(self):
        """Test order blocked in emergency state"""
        # Trigger emergency state
        self.risk_guard._trigger_kill_switch(KillSwitchState.EMERGENCY, "Test emergency")
        
        order = {
            'symbol': 'BTC/USD',
            'side': 'buy', 
            'size': 1.0,
            'price': 50000.0
        }
        
        allowed, reason = self.risk_guard.check_order_allowed(order)
        assert not allowed
        assert "Emergency stop" in reason
    
    def test_soft_stop_reduces_only(self):
        """Test soft stop allows only position-reducing orders"""
        # Setup position
        self.risk_guard.update_position('BTC/USD', {'size': 1.0, 'mark_price': 50000.0})
        
        # Trigger soft stop
        self.risk_guard._trigger_kill_switch(KillSwitchState.SOFT_STOP, "Test soft stop")
        
        # Test position-increasing order (should be blocked)
        buy_order = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 1.0,
            'price': 50000.0
        }
        allowed, reason = self.risk_guard.check_order_allowed(buy_order)
        assert not allowed
        assert "position-reducing" in reason
        
        # Test position-reducing order (should be allowed)
        sell_order = {
            'symbol': 'BTC/USD',
            'side': 'sell',
            'size': 0.5,
            'price': 50000.0
        }
        allowed, reason = self.risk_guard.check_order_allowed(sell_order)
        assert allowed
    
    def test_position_size_limit(self):
        """Test position size limit enforcement"""
        # Test oversized order
        large_order = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 1.0,
            'price': 100000.0  # $100k order exceeds $20k limit
        }
        
        allowed, reason = self.risk_guard.check_order_allowed(large_order)
        assert not allowed
        assert "exceeds limit" in reason
    
    def test_data_quality_check(self):
        """Test data quality enforcement"""
        # Setup poor data quality
        metrics = RiskMetrics(data_quality_score=0.5)  # Below 0.8 threshold
        self.risk_guard.update_metrics(metrics)
        
        order = {
            'symbol': 'BTC/USD',
            'side': 'buy',
            'size': 1.0,
            'price': 50000.0
        }
        
        allowed, reason = self.risk_guard.check_order_allowed(order)
        assert not allowed
        assert "quality too low" in reason
    
    def test_manual_kill_switch(self):
        """Test manual kill switch activation"""
        # Trigger manual kill switch
        self.risk_guard.manual_kill_switch(KillSwitchState.HARD_STOP, "Manual test")
        
        assert self.risk_guard.kill_switch_state == KillSwitchState.HARD_STOP
    
    def test_kill_switch_reset(self):
        """Test kill switch reset"""
        # Trigger then reset
        self.risk_guard.manual_kill_switch(KillSwitchState.HARD_STOP, "Test")
        self.risk_guard.reset_kill_switch("Test reset")
        
        assert self.risk_guard.kill_switch_state == KillSwitchState.ACTIVE


class TestExecutionPolicy:
    """Test suite for ExecutionPolicy"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gates = ExecutionGates(
            max_spread_bps=50,
            min_bid_depth_usd=5000.0,
            min_ask_depth_usd=5000.0,
            min_volume_1m_usd=10000.0,
            max_slippage_bps=25
        )
        self.policy = ExecutionPolicy(self.gates)
        
        # Setup mock market data
        self.market_data = MarketData(
            symbol='BTC/USD',
            timestamp=datetime.now(),
            bid=49950.0,
            ask=50050.0,
            mid=50000.0,
            last=50000.0,
            bid_depth_usd=10000.0,
            ask_depth_usd=10000.0,
            total_depth_usd=20000.0,
            volume_1m_usd=50000.0,
            trades_1m=20,
            volume_24h_usd=1000000.0,
            volatility_1h_pct=0.05,
            volatility_24h_pct=0.10,
            spread_bps=20,
            spread_pct=0.002
        )
        self.policy.update_market_data('BTC/USD', self.market_data)
    
    @pytest.mark.asyncio
    async def test_valid_order_passes_gates(self):
        """Test valid order passes all gates"""
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert valid
        assert len(errors) == 0
        assert client_id.startswith('CST_')
    
    @pytest.mark.asyncio
    async def test_wide_spread_blocks_order(self):
        """Test wide spread blocks order"""
        # Update market data with wide spread
        wide_spread_data = self.market_data
        wide_spread_data.spread_bps = 100  # Exceeds 50bps limit
        self.policy.update_market_data('BTC/USD', wide_spread_data)
        
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert not valid
        assert any("Spread too wide" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_insufficient_depth_blocks_order(self):
        """Test insufficient depth blocks order"""
        # Update market data with low depth
        low_depth_data = self.market_data
        low_depth_data.ask_depth_usd = 1000.0  # Below 5000 limit
        self.policy.update_market_data('BTC/USD', low_depth_data)
        
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',  # Buying needs ask depth
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert not valid
        assert any("Insufficient ask depth" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_low_volume_blocks_order(self):
        """Test low volume blocks order"""
        # Update market data with low volume
        low_volume_data = self.market_data
        low_volume_data.volume_1m_usd = 5000.0  # Below 10000 limit
        self.policy.update_market_data('BTC/USD', low_volume_data)
        
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert not valid
        assert any("Insufficient 1m volume" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_duplicate_order_detection(self):
        """Test duplicate order detection"""
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        # First submission should pass
        valid1, errors1, client_id1 = await self.policy.validate_order(request)
        assert valid1
        
        # Second identical submission should be blocked
        valid2, errors2, client_id2 = await self.policy.validate_order(request)
        assert not valid2
        assert any("Duplicate request" in error for error in errors2)
    
    def test_client_order_id_generation(self):
        """Test client order ID generation"""
        request1 = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        request2 = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        # Same request should generate same client order ID
        client_id1 = self.policy.generate_client_order_id(request1)
        client_id2 = self.policy.generate_client_order_id(request2)
        
        assert client_id1 == client_id2
        assert client_id1.startswith('CST_')
    
    def test_slippage_estimation(self):
        """Test slippage estimation"""
        # Test market order slippage
        market_request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.MARKET
        )
        
        slippage = self.policy.estimate_slippage(market_request, self.market_data)
        assert slippage > 0  # Should have positive slippage
        
        # Test limit order slippage
        limit_request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=49900.0  # Below ask
        )
        
        slippage = self.policy.estimate_slippage(limit_request, self.market_data)
        assert slippage < 0  # Should have negative slippage (improvement)
    
    def test_slippage_tracking(self):
        """Test daily slippage tracking"""
        # Create execution result with slippage
        result = ExecutionResult(
            order_id='test_order',
            client_order_id='test_client',
            status=OrderStatus.FILLED,
            slippage_bps=15.0
        )
        
        # Update result
        self.policy.update_order_result('test_order', result)
        
        # Check slippage tracking
        assert self.policy.slippage_tracker.used_slippage_bps == 15.0
        assert self.policy.slippage_tracker.total_executions == 1
        assert self.policy.slippage_tracker.avg_slippage_bps == 15.0
    
    @pytest.mark.asyncio
    async def test_slippage_budget_exhaustion(self):
        """Test slippage budget exhaustion blocks orders"""
        # Exhaust slippage budget
        self.policy.slippage_tracker.used_slippage_bps = 300.0  # Exceeds 200bps budget
        
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert not valid
        assert any("slippage budget exhausted" in error for error in errors)
    
    def test_order_registration_and_tracking(self):
        """Test order registration and tracking"""
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0,
            client_order_id='test_client_123'
        )
        
        # Register order
        order_id = 'test_order_456'
        self.policy.register_order(request, order_id)
        
        # Check tracking
        assert order_id in self.policy.active_orders
        assert request.client_order_id in self.policy.client_order_map
        assert self.policy.client_order_map[request.client_order_id] == order_id
    
    def test_order_result_update(self):
        """Test order result update"""
        # Register order first
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=1.0,
            order_type=OrderType.LIMIT,
            price=50000.0,
            client_order_id='test_client_789'
        )
        
        order_id = 'test_order_789'
        self.policy.register_order(request, order_id)
        
        # Update with result
        result = ExecutionResult(
            order_id=order_id,
            client_order_id=request.client_order_id,
            status=OrderStatus.FILLED,
            filled_size=1.0,
            avg_fill_price=50000.0
        )
        
        self.policy.update_order_result(order_id, result)
        
        # Check result stored and order removed from active
        assert order_id in self.policy.order_history
        assert order_id not in self.policy.active_orders
        assert self.policy.order_history[order_id] == result
    
    @pytest.mark.asyncio
    async def test_oversized_order_blocked(self):
        """Test oversized order is blocked"""
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=2.0,  # 2 BTC * $50k = $100k
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        valid, errors, client_id = await self.policy.validate_order(request)
        assert not valid
        assert any("Order too large" in error for error in errors)


class TestIntegration:
    """Integration tests between RiskGuard and ExecutionPolicy"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.risk_limits = RiskLimits(daily_loss_limit_usd=5000.0)
        self.risk_guard = CentralizedRiskGuard(self.risk_limits)
        
        self.execution_gates = ExecutionGates(max_spread_bps=30)
        self.execution_policy = ExecutionPolicy(self.execution_gates)
        
        # Setup market data
        market_data = MarketData(
            symbol='BTC/USD',
            timestamp=datetime.now(),
            bid=49980.0,
            ask=50020.0,
            mid=50000.0,
            last=50000.0,
            bid_depth_usd=20000.0,
            ask_depth_usd=20000.0,
            total_depth_usd=40000.0,
            volume_1m_usd=100000.0,
            trades_1m=50,
            volume_24h_usd=5000000.0,
            volatility_1h_pct=0.03,
            volatility_24h_pct=0.08,
            spread_bps=8,
            spread_pct=0.0008
        )
        self.execution_policy.update_market_data('BTC/USD', market_data)
    
    @pytest.mark.asyncio
    async def test_integrated_order_flow(self):
        """Test complete order flow through both systems"""
        # Create order request
        request = OrderRequest(
            symbol='BTC/USD',
            side='buy',
            size=0.1,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        # 1. Validate through execution policy
        valid, errors, client_id = await self.execution_policy.validate_order(request)
        assert valid, f"Execution policy failed: {errors}"
        
        # 2. Check through risk guard
        order_dict = {
            'symbol': request.symbol,
            'side': request.side,
            'size': request.size,
            'price': request.price
        }
        allowed, reason = self.risk_guard.check_order_allowed(order_dict)
        assert allowed, f"Risk guard failed: {reason}"
        
        # 3. Register order in execution policy
        order_id = f"order_{int(time.time() * 1000)}"
        self.execution_policy.register_order(request, order_id)
        
        # 4. Simulate successful execution
        result = ExecutionResult(
            order_id=order_id,
            client_order_id=client_id,
            status=OrderStatus.FILLED,
            filled_size=request.size,
            avg_fill_price=50000.0,
            slippage_bps=5.0
        )
        self.execution_policy.update_order_result(order_id, result)
        
        # 5. Verify order is tracked properly
        tracked_result = self.execution_policy.get_order_by_client_id(client_id)
        assert tracked_result is not None
        assert tracked_result.status == OrderStatus.FILLED


if __name__ == "__main__":
    # Run with: pytest tests/test_risk_execution.py -v
    pytest.main([__file__, "-v"])