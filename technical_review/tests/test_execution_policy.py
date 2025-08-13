"""
Comprehensive tests for Execution Policy System
Tests tradability gates, slippage enforcement, and order deduplication.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.cryptosmarttrader.execution.execution_policy import (
    ExecutionPolicy, ExecutionParams, MarketConditions, OrderType, 
    TimeInForce, OrderStatus, TradabilityGate, create_market_conditions
)


class TestExecutionPolicy:
    """Test enterprise execution policy system."""
    
    @pytest.fixture
    def execution_policy(self, tmp_path):
        """Create ExecutionPolicy instance for testing."""
        config_path = tmp_path / "test_execution_config.json"
        return ExecutionPolicy(config_path=config_path)
    
    @pytest.fixture
    def good_market_conditions(self):
        """Market conditions that pass all tradability gates."""
        return MarketConditions(
            symbol='BTC-USD',
            bid=50000.0,
            ask=50010.0,  # 2 bps spread
            spread_bps=2.0,
            depth_bid=5000.0,  # $5k depth
            depth_ask=5000.0,
            volume_1m=20000.0,  # $20k volume
            volatility_1m=0.01  # 1% volatility
        )
    
    @pytest.fixture
    def bad_market_conditions(self):
        """Market conditions that fail tradability gates."""
        return MarketConditions(
            symbol='ALTCOIN-USD',
            bid=1.0,
            ask=1.10,  # 1000 bps spread (10%)
            spread_bps=1000.0,
            depth_bid=100.0,   # Low depth
            depth_ask=100.0,
            volume_1m=500.0,   # Low volume
            volatility_1m=0.10  # High volatility (10%)
        )
    
    @pytest.fixture
    def standard_order_params(self):
        """Standard order parameters for testing."""
        return ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            order_type=OrderType.LIMIT,
            limit_price=50010.0,
            max_slippage_bps=30.0
        )
    
    def test_client_order_id_generation(self, execution_policy):
        """Test deterministic client order ID generation."""
        params1 = ExecutionParams(
            symbol='BTC-USD',
            side='buy', 
            quantity=0.1,
            order_type=OrderType.LIMIT,
            limit_price=50000.0
        )
        
        params2 = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.1, 
            order_type=OrderType.LIMIT,
            limit_price=50000.0
        )
        
        # Same parameters should generate same ID (within same minute)
        id1 = execution_policy.generate_client_order_id(params1)
        id2 = execution_policy.generate_client_order_id(params2)
        
        assert id1 == id2
        assert id1.startswith('CST_BTC-USD_BUY_')
        assert len(id1.split('_')[-1]) == 8  # 8-character hash
    
    def test_order_deduplication(self, execution_policy):
        """Test order deduplication mechanism."""
        client_order_id = "CST_BTC-USD_BUY_12345678"
        
        # First submission should be allowed
        is_duplicate, reason = execution_policy.check_order_deduplication(client_order_id)
        assert not is_duplicate
        assert reason is None
        
        # Immediate resubmission should be blocked
        is_duplicate, reason = execution_policy.check_order_deduplication(client_order_id)
        assert is_duplicate
        assert "seconds ago" in reason
        
        # After clearing cache, should be allowed again
        execution_policy.deduplication_cache.clear()
        is_duplicate, reason = execution_policy.check_order_deduplication(client_order_id)
        assert not is_duplicate
    
    def test_tradability_gates_pass(self, execution_policy, good_market_conditions):
        """Test tradability gates with good market conditions."""
        is_tradable, violations = execution_policy.check_tradability_gates(good_market_conditions)
        
        assert is_tradable
        assert len(violations) == 0
    
    def test_tradability_gates_fail(self, execution_policy, bad_market_conditions):
        """Test tradability gates with bad market conditions."""
        is_tradable, violations = execution_policy.check_tradability_gates(bad_market_conditions)
        
        assert not is_tradable
        assert len(violations) > 0
        
        # Check specific violations
        violation_types = [v.value for v in violations]
        assert 'spread_too_wide' in violation_types
        assert 'insufficient_depth' in violation_types
        assert 'low_volume' in violation_types
        assert 'high_volatility' in violation_types
    
    def test_slippage_budget_validation(self, execution_policy):
        """Test slippage budget validation."""
        params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            max_slippage_bps=30.0  # 0.3% budget
        )
        
        conditions = MarketConditions(
            symbol='BTC-USD',
            bid=50000.0,
            ask=50010.0,
            spread_bps=2.0,
            depth_bid=5000.0,
            depth_ask=5000.0,
            volume_1m=20000.0,
            volatility_1m=0.01
        )
        
        # Execution within budget
        within_budget, slippage = execution_policy.validate_slippage_budget(
            params, 50025.0, conditions  # 15 bps slippage
        )
        assert within_budget
        assert abs(slippage - 15.0) < 0.1
        
        # Execution exceeding budget
        within_budget, slippage = execution_policy.validate_slippage_budget(
            params, 50060.0, conditions  # 50 bps slippage
        )
        assert not within_budget
        assert slippage > params.max_slippage_bps
    
    @pytest.mark.asyncio
    async def test_successful_order_execution(self, execution_policy, good_market_conditions, standard_order_params):
        """Test successful order execution."""
        execution = await execution_policy.execute_order(standard_order_params, good_market_conditions)
        
        assert execution.status == OrderStatus.FILLED
        assert execution.filled_quantity == standard_order_params.quantity
        assert execution.average_price > 0
        assert execution.client_order_id is not None
        assert execution.exchange_order_id is not None
        assert execution.execution_time_ms > 0
        assert execution.slippage_bps >= 0  # Should be minimal for good conditions
    
    @pytest.mark.asyncio
    async def test_rejected_order_tradability_gates(self, execution_policy, bad_market_conditions, standard_order_params):
        """Test order rejection due to tradability gates."""
        execution = await execution_policy.execute_order(standard_order_params, bad_market_conditions)
        
        assert execution.status == OrderStatus.REJECTED
        assert execution.filled_quantity == 0.0
        assert execution.exchange_order_id is None
        assert "Tradability gates failed" in execution.error_message
        
        # Check statistics
        stats = execution_policy.get_execution_stats()
        assert stats['rejected_by_gates'] == 1
        assert stats['success_rate_percent'] == 0.0
    
    @pytest.mark.asyncio
    async def test_duplicate_order_rejection(self, execution_policy, good_market_conditions):
        """Test duplicate order rejection."""
        params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            client_order_id='TEST_DUPLICATE_ORDER'
        )
        
        # First execution should succeed
        execution1 = await execution_policy.execute_order(params, good_market_conditions)
        assert execution1.status == OrderStatus.FILLED
        
        # Immediate resubmission should be rejected
        execution2 = await execution_policy.execute_order(params, good_market_conditions)
        assert execution2.status == OrderStatus.REJECTED
        assert "Duplicate order" in execution2.error_message
    
    @pytest.mark.asyncio
    async def test_twap_order_execution(self, execution_policy, good_market_conditions):
        """Test TWAP order execution."""
        params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=1.0,  # Large order for TWAP
            order_type=OrderType.TWAP,
            max_slippage_bps=30.0
        )
        
        execution = await execution_policy.execute_order(params, good_market_conditions)
        
        assert execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert execution.filled_quantity > 0
        assert len(execution.partial_fills) > 1  # Should have multiple slices
        assert execution.exchange_order_id.startswith('TWAP_')
        
        # Check that partial fills are recorded
        total_filled = sum(fill['quantity'] for fill in execution.partial_fills)
        assert abs(total_filled - execution.filled_quantity) < 0.001
    
    @pytest.mark.asyncio
    async def test_iceberg_order_execution(self, execution_policy, good_market_conditions):
        """Test iceberg order execution."""
        params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.5,
            order_type=OrderType.ICEBERG,
            max_slippage_bps=30.0
        )
        
        execution = await execution_policy.execute_order(params, good_market_conditions)
        
        assert execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert execution.filled_quantity > 0
        assert len(execution.partial_fills) > 0  # Should have multiple slices
        assert execution.exchange_order_id.startswith('ICE_')
    
    def test_execution_strategy_optimization(self, execution_policy):
        """Test execution strategy optimization."""
        # Large order in high volume -> TWAP
        large_order_params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=1.0,
            order_type=OrderType.LIMIT
        )
        
        high_volume_conditions = MarketConditions(
            symbol='BTC-USD',
            bid=50000.0,
            ask=50010.0,
            spread_bps=2.0,
            depth_bid=5000.0,
            depth_ask=5000.0,
            volume_1m=100000.0,  # High volume
            volatility_1m=0.01
        )
        
        optimized = execution_policy.calculate_optimal_execution_strategy(
            large_order_params, high_volume_conditions
        )
        assert optimized.order_type == OrderType.TWAP
        
        # Wide spread -> Post-only
        wide_spread_conditions = MarketConditions(
            symbol='BTC-USD',
            bid=50000.0,
            ask=50100.0,  # 200 bps spread
            spread_bps=200.0,
            depth_bid=5000.0,
            depth_ask=5000.0,
            volume_1m=20000.0,
            volatility_1m=0.01
        )
        
        small_order_params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            order_type=OrderType.LIMIT
        )
        
        optimized = execution_policy.calculate_optimal_execution_strategy(
            small_order_params, wide_spread_conditions
        )
        assert optimized.post_only is True
        assert optimized.time_in_force == TimeInForce.POST_ONLY
    
    def test_order_status_tracking(self, execution_policy):
        """Test order status tracking and retrieval."""
        # Initially no orders
        assert len(execution_policy.active_orders) == 0
        assert len(execution_policy.completed_orders) == 0
        
        # Add a completed order manually for testing
        from src.cryptosmarttrader.execution.execution_policy import OrderExecution
        
        test_execution = OrderExecution(
            client_order_id='TEST_ORDER_123',
            exchange_order_id='EX_123',
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            filled_quantity=0.1,
            average_price=50000.0,
            status=OrderStatus.FILLED,
            slippage_bps=5.0,
            fees=5.0,
            execution_time_ms=150.0,
            timestamp=datetime.now()
        )
        
        execution_policy.completed_orders['TEST_ORDER_123'] = test_execution
        
        # Test retrieval
        retrieved = execution_policy.get_order_status('TEST_ORDER_123')
        assert retrieved is not None
        assert retrieved.client_order_id == 'TEST_ORDER_123'
        assert retrieved.status == OrderStatus.FILLED
        
        # Test non-existent order
        not_found = execution_policy.get_order_status('NONEXISTENT')
        assert not_found is None
    
    def test_execution_statistics(self, execution_policy):
        """Test execution statistics tracking."""
        initial_stats = execution_policy.get_execution_stats()
        assert initial_stats['total_orders'] == 0
        assert initial_stats['success_rate_percent'] == 0.0
        
        # Simulate some executions
        from src.cryptosmarttrader.execution.execution_policy import OrderExecution
        
        # Successful execution
        success_execution = OrderExecution(
            client_order_id='SUCCESS_1',
            exchange_order_id='EX_1',
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            filled_quantity=0.1,
            average_price=50000.0,
            status=OrderStatus.FILLED,
            slippage_bps=10.0,
            fees=5.0,
            execution_time_ms=200.0,
            timestamp=datetime.now()
        )
        
        execution_policy._update_execution_stats(success_execution)
        
        stats = execution_policy.get_execution_stats()
        assert stats['total_orders'] == 1
        assert stats['successful_executions'] == 1
        assert stats['success_rate_percent'] == 100.0
        assert stats['average_slippage_bps'] == 10.0
        assert stats['average_execution_time_ms'] == 200.0
    
    def test_order_cancellation(self, execution_policy):
        """Test order cancellation."""
        from src.cryptosmarttrader.execution.execution_policy import OrderExecution
        
        # Add an active order
        active_order = OrderExecution(
            client_order_id='ACTIVE_ORDER',
            exchange_order_id='EX_ACTIVE',
            symbol='BTC-USD',
            side='buy',
            quantity=0.1,
            filled_quantity=0.0,
            average_price=0.0,
            status=OrderStatus.SUBMITTED,
            slippage_bps=0.0,
            fees=0.0,
            execution_time_ms=0.0,
            timestamp=datetime.now()
        )
        
        execution_policy.active_orders['ACTIVE_ORDER'] = active_order
        
        # Test successful cancellation
        success = execution_policy.cancel_order('ACTIVE_ORDER')
        assert success
        assert 'ACTIVE_ORDER' not in execution_policy.active_orders
        assert 'ACTIVE_ORDER' in execution_policy.completed_orders
        assert execution_policy.completed_orders['ACTIVE_ORDER'].status == OrderStatus.CANCELED
        
        # Test cancellation of non-existent order
        fail = execution_policy.cancel_order('NONEXISTENT')
        assert not fail


class TestMarketConditionsUtils:
    """Test market conditions utility functions."""
    
    def test_create_market_conditions(self):
        """Test market conditions creation utility."""
        conditions = create_market_conditions('BTC-USD', 50000.0, 50010.0, 20000.0)
        
        assert conditions.symbol == 'BTC-USD'
        assert conditions.bid == 50000.0
        assert conditions.ask == 50010.0
        assert conditions.volume_1m == 20000.0
        assert conditions.spread_bps == 2.0  # 10/50000 * 10000
        assert conditions.depth_bid == 2000.0  # 10% of volume
        assert conditions.depth_ask == 2000.0


# Integration test scenarios
class TestExecutionScenarios:
    """Test realistic execution scenarios."""
    
    @pytest.fixture
    def execution_system(self, tmp_path):
        """Create execution system for scenario testing."""
        return ExecutionPolicy(config_path=tmp_path / "scenario_config.json")
    
    @pytest.mark.asyncio
    async def test_high_frequency_trading_scenario(self, execution_system):
        """Test high-frequency trading with many small orders."""
        conditions = create_market_conditions('BTC-USD', 50000.0, 50010.0, 50000.0)
        
        # Execute 10 small orders rapidly
        executions = []
        for i in range(10):
            params = ExecutionParams(
                symbol='BTC-USD',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=0.01,  # Small orders
                order_type=OrderType.LIMIT,
                max_slippage_bps=20.0
            )
            
            execution = await execution_system.execute_order(params, conditions)
            executions.append(execution)
        
        # Verify all orders executed successfully
        successful = [e for e in executions if e.status == OrderStatus.FILLED]
        assert len(successful) == 10
        
        # Check statistics
        stats = execution_system.get_execution_stats()
        assert stats['total_orders'] == 10
        assert stats['successful_executions'] == 10
        assert stats['success_rate_percent'] == 100.0
    
    @pytest.mark.asyncio
    async def test_volatile_market_scenario(self, execution_system):
        """Test execution in volatile market conditions."""
        volatile_conditions = MarketConditions(
            symbol='ETH-USD',
            bid=3000.0,
            ask=3010.0,
            spread_bps=33.3,
            depth_bid=2000.0,
            depth_ask=2000.0,
            volume_1m=15000.0,
            volatility_1m=0.08  # 8% volatility (high)
        )
        
        params = ExecutionParams(
            symbol='ETH-USD',
            side='buy',
            quantity=0.5,
            max_slippage_bps=50.0  # Higher budget for volatile conditions
        )
        
        execution = await execution_system.execute_order(params, volatile_conditions)
        
        # Should succeed with optimized parameters
        assert execution.status == OrderStatus.FILLED
        # Slippage budget should be tightened automatically
        assert execution.slippage_bps <= 50.0
    
    @pytest.mark.asyncio
    async def test_large_order_optimization_scenario(self, execution_system):
        """Test large order optimization and execution."""
        high_liquidity_conditions = create_market_conditions('BTC-USD', 50000.0, 50010.0, 200000.0)
        
        large_order_params = ExecutionParams(
            symbol='BTC-USD',
            side='buy',
            quantity=2.0,  # Large order ($100k+)
            order_type=OrderType.LIMIT,
            max_slippage_bps=40.0
        )
        
        execution = await execution_system.execute_order(large_order_params, high_liquidity_conditions)
        
        # Large order should be executed via TWAP
        assert execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
        assert len(execution.partial_fills) > 1  # Multiple slices
        assert execution.exchange_order_id.startswith('TWAP_')


if __name__ == "__main__":
    # Run basic functionality test
    import tempfile
    
    async def test_basic_execution():
        with tempfile.TemporaryDirectory() as tmp_dir:
            policy = ExecutionPolicy(config_path=Path(tmp_dir) / "test_config.json")
            
            print("🚀 Testing Execution Policy System")
            print("=" * 50)
            
            # Test good conditions
            good_conditions = create_market_conditions('BTC-USD', 50000.0, 50010.0, 20000.0)
            
            params = ExecutionParams(
                symbol='BTC-USD',
                side='buy',
                quantity=0.1,
                order_type=OrderType.LIMIT,
                max_slippage_bps=30.0
            )
            
            print("Testing standard order execution...")
            execution = await policy.execute_order(params, good_conditions)
            print(f"Order status: {execution.status.value}")
            print(f"Filled: {execution.filled_quantity}")
            print(f"Slippage: {execution.slippage_bps:.1f} bps")
            
            print("\nTesting TWAP order execution...")
            twap_params = ExecutionParams(
                symbol='BTC-USD',
                side='buy',
                quantity=1.0,
                order_type=OrderType.TWAP
            )
            twap_execution = await policy.execute_order(twap_params, good_conditions)
            print(f"TWAP status: {twap_execution.status.value}")
            print(f"Partial fills: {len(twap_execution.partial_fills)}")
            
            print("\nExecution statistics:")
            stats = policy.get_execution_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print("\n✅ Execution Policy System Test Complete")
    
    asyncio.run(test_basic_execution())