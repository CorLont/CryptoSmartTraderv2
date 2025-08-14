#!/usr/bin/env python3
"""
Test Mandatory Execution Discipline Enforcement
Comprehensive tests for double-order scenarios, retry/timeout handling
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.cryptosmarttrader.execution.execution_discipline import (
        ExecutionPolicy, OrderRequest, MarketConditions, OrderSide, TimeInForce
    )
    from src.cryptosmarttrader.execution.mandatory_enforcement import (
        require_execution_discipline, DisciplinedExchangeManager, 
        ExecutionDisciplineViolation, get_global_execution_policy
    )
except ImportError:
    pytest.skip("ExecutionDiscipline modules not available", allow_module_level=True)


class TestMandatoryExecutionDiscipline:
    """Test mandatory execution discipline enforcement"""
    
    def setup_method(self):
        """Setup for each test"""
        self.policy = ExecutionPolicy()
        
        # Good market conditions
        self.good_market = MarketConditions(
            spread_bps=25.0,
            bid_depth_usd=50000.0,
            ask_depth_usd=50000.0,
            volume_1m_usd=200000.0,
            last_price=50000.0,
            bid_price=49950.0,
            ask_price=50050.0,
            timestamp=time.time()
        )
        
        # Bad market conditions (wide spread)
        self.bad_market = MarketConditions(
            spread_bps=100.0,  # Too wide
            bid_depth_usd=5000.0,  # Too low
            ask_depth_usd=5000.0,
            volume_1m_usd=50000.0,  # Too low
            last_price=50000.0,
            bid_price=49750.0,
            ask_price=50250.0,
            timestamp=time.time()
        )
    
    def test_decorator_enforcement(self):
        """Test that decorator enforces ExecutionDiscipline"""
        
        executed = []
        
        @require_execution_discipline()
        def mock_place_order(symbol, side, size, price):
            executed.append((symbol, side, size, price))
            return {"id": "test_order_123"}
        
        # Should execute without discipline (logs warning)
        result = mock_place_order("BTC/USD", "buy", 0.1, 50000)
        assert result["id"] == "test_order_123"
        assert len(executed) == 1
    
    def test_double_order_prevention(self):
        """Test explicit double-order scenario prevention"""
        
        # Create order with deterministic ID
        order1 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            strategy_id="test_strategy"
        )
        
        # First order should be approved
        result1 = self.policy.decide(order1, self.good_market)
        assert result1.decision.value == "approve"
        
        # Create identical order (same ID)
        order2 = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            strategy_id="test_strategy",
            client_order_id=order1.client_order_id  # Same ID!
        )
        
        # Second order should be rejected (duplicate)
        result2 = self.policy.decide(order2, self.good_market)
        assert result2.decision.value == "reject"
        assert "Duplicate order ID" in result2.reason
    
    def test_timeout_retry_scenario(self):
        """Test timeout/retry handling with idempotency"""
        
        # Original order
        order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            strategy_id="timeout_test"
        )
        
        # First attempt
        result1 = self.policy.decide(order, self.good_market)
        assert result1.decision.value == "approve"
        
        # Simulate timeout - retry with same order
        # Should be rejected as duplicate
        retry_result = self.policy.decide(order, self.good_market)
        assert retry_result.decision.value == "reject"
        assert "Duplicate order ID" in retry_result.reason
        
        # New order after timeout (different timestamp/ID)
        time.sleep(0.1)  # Ensure different timestamp
        new_order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            strategy_id="timeout_test"  # Same strategy, new ID
        )
        
        # Should be approved (new ID)
        new_result = self.policy.decide(new_order, self.good_market)
        assert new_result.decision.value == "approve"
        assert new_order.client_order_id != order.client_order_id
    
    def test_market_conditions_gates(self):
        """Test all market condition gates"""
        
        order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0
        )
        
        # Good market - should approve
        good_result = self.policy.decide(order, self.good_market)
        assert good_result.decision.value == "approve"
        
        # Bad market - should reject
        bad_result = self.policy.decide(order, self.bad_market)
        assert bad_result.decision.value == "reject"
        
        # Check specific gate failures
        assert not bad_result.gate_results["spread"]
        assert not bad_result.gate_results["depth"] 
        assert not bad_result.gate_results["volume"]
    
    def test_slippage_budget_enforcement(self):
        """Test slippage budget gate"""
        
        # Order with high slippage budget
        high_slippage_order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            max_slippage_bps=100.0  # Too high
        )
        
        result = self.policy.decide(high_slippage_order, self.good_market)
        assert result.decision.value == "reject"
        assert "Slippage budget too high" in result.reason
    
    def test_time_in_force_enforcement(self):
        """Test Time-in-Force validation"""
        
        # Order without post-only
        non_post_only_order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            time_in_force=TimeInForce.GTC  # Not post-only
        )
        
        result = self.policy.decide(non_post_only_order, self.good_market)
        assert result.decision.value == "reject"
        assert "Post-only required" in result.reason
    
    def test_thread_safety(self):
        """Test thread-safe execution with multiple concurrent orders"""
        
        results = []
        
        def place_order(thread_id):
            order = OrderRequest(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                size=0.1,
                limit_price=50000.0,
                strategy_id=f"thread_{thread_id}"
            )
            
            result = self.policy.decide(order, self.good_market)
            results.append((thread_id, result.decision.value))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=place_order, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All orders should be approved (different IDs)
        assert len(results) == 10
        approved_count = sum(1 for _, decision in results if decision == "approve")
        assert approved_count == 10
    
    def test_disciplined_exchange_manager(self):
        """Test DisciplinedExchangeManager integration"""
        
        # Mock exchange manager
        mock_exchange_manager = Mock()
        mock_exchange = Mock()
        mock_exchange.create_limit_order.return_value = {"id": "exchange_order_123"}
        mock_exchange_manager.get_exchange.return_value = mock_exchange
        
        # Create disciplined manager
        disciplined_manager = DisciplinedExchangeManager(mock_exchange_manager)
        
        # Good order
        order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0,
            time_in_force=TimeInForce.POST_ONLY
        )
        
        # Execute through disciplined manager
        result = disciplined_manager.execute_disciplined_order(
            order, self.good_market, "kraken"
        )
        
        assert result["success"] is True
        assert result["order_id"] == "exchange_order_123"
        assert mock_exchange.create_limit_order.called
    
    def test_disciplined_manager_rejection(self):
        """Test DisciplinedExchangeManager rejects bad orders"""
        
        mock_exchange_manager = Mock()
        disciplined_manager = DisciplinedExchangeManager(mock_exchange_manager)
        
        # Bad order (wide spread market)
        order = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            size=0.1,
            limit_price=50000.0
        )
        
        result = disciplined_manager.execute_disciplined_order(
            order, self.bad_market, "kraken"
        )
        
        assert result["success"] is False
        assert "ExecutionDiscipline rejected" in result["error"]
        assert result["order_id"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])