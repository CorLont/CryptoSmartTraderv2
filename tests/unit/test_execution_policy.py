"""Unit tests for execution policy components."""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch
from src.cryptosmarttrader.core.execution_policy import (
    ExecutionPolicy, OrderRequest, OrderType, TimeInForce, ExecutionResult
)


class TestExecutionPolicy:
    """Test ExecutionPolicy execution management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.execution_policy = ExecutionPolicy(
            max_slippage_bps=30,  # 0.3%
            min_liquidity_usd=10000,
            max_order_size_pct=5.0,
            enable_iceberg=True
        )
    
    def test_order_validation_basic(self):
        """Test basic order validation."""
        order = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.1'),
            price=Decimal('50000'),
            time_in_force=TimeInForce.GTC
        )
        
        validation_result = self.execution_policy.validate_order(order)
        
        assert validation_result.is_valid is True
        assert validation_result.errors == []
    
    def test_order_validation_oversized(self):
        """Test validation of oversized orders."""
        large_order = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            order_type=OrderType.MARKET,
            quantity=Decimal('10.0'),  # Very large order
            time_in_force=TimeInForce.IOC
        )
        
        # Mock market data to simulate small market
        with patch.object(self.execution_policy, '_get_market_liquidity') as mock_liquidity:
            mock_liquidity.return_value = 50000  # $50k liquidity
            
            validation_result = self.execution_policy.validate_order(large_order)
            
            assert validation_result.is_valid is False
            assert any('size' in error.lower() for error in validation_result.errors)
    
    def test_slippage_estimation(self):
        """Test slippage estimation for different order sizes."""
        # Small order - low slippage
        small_slippage = self.execution_policy.estimate_slippage(
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('0.01'),
            order_type=OrderType.MARKET
        )
        
        # Large order - higher slippage
        large_slippage = self.execution_policy.estimate_slippage(
            symbol='BTC/USDT',
            side='buy',
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET
        )
        
        assert large_slippage.estimated_slippage_bps > small_slippage.estimated_slippage_bps
        assert small_slippage.estimated_slippage_bps <= self.execution_policy.max_slippage_bps
    
    def test_iceberg_order_splitting(self):
        """Test iceberg order splitting logic."""
        large_order = OrderRequest(
            symbol='ETH/USDT',
            side='sell',
            order_type=OrderType.LIMIT,
            quantity=Decimal('10.0'),
            price=Decimal('3000'),
            time_in_force=TimeInForce.GTC
        )
        
        iceberg_orders = self.execution_policy.create_iceberg_orders(
            large_order,
            max_slice_size=Decimal('2.0')
        )
        
        assert len(iceberg_orders) == 5  # 10.0 / 2.0 = 5 slices
        assert all(order.quantity <= Decimal('2.0') for order in iceberg_orders)
        assert sum(order.quantity for order in iceberg_orders) == large_order.quantity
    
    def test_execution_timing_optimization(self):
        """Test execution timing optimization."""
        order = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.5'),
            price=Decimal('50000'),
            time_in_force=TimeInForce.GTC
        )
        
        # Mock market conditions
        with patch.object(self.execution_policy, '_get_market_conditions') as mock_conditions:
            mock_conditions.return_value = {
                'spread_bps': 5,
                'volume_profile': 'normal',
                'volatility': 0.02
            }
            
            timing_result = self.execution_policy.optimize_execution_timing(order)
            
            assert timing_result.recommended_strategy in ['immediate', 'twap', 'vwap', 'delayed']
            assert timing_result.expected_completion_time is not None
    
    def test_liquidity_filtering(self):
        """Test liquidity-based order filtering."""
        # Mock low liquidity market
        with patch.object(self.execution_policy, '_get_market_liquidity') as mock_liquidity:
            mock_liquidity.return_value = 5000  # Below minimum threshold
            
            order = OrderRequest(
                symbol='SMALL/USDT',
                side='buy',
                order_type=OrderType.MARKET,
                quantity=Decimal('100'),
                time_in_force=TimeInForce.IOC
            )
            
            validation_result = self.execution_policy.validate_order(order)
            
            assert validation_result.is_valid is False
            assert any('liquidity' in error.lower() for error in validation_result.errors)
    
    def test_spread_analysis(self):
        """Test bid-ask spread analysis."""
        spread_analysis = self.execution_policy.analyze_spread(
            symbol='BTC/USDT',
            target_quantity=Decimal('0.1')
        )
        
        assert spread_analysis.current_spread_bps >= 0
        assert spread_analysis.impact_on_spread_bps >= 0
        assert spread_analysis.recommended_order_type in [OrderType.LIMIT, OrderType.MARKET]
    
    def test_order_type_optimization(self):
        """Test order type optimization based on market conditions."""
        # High volatility - should prefer limit orders
        with patch.object(self.execution_policy, '_get_volatility') as mock_vol:
            mock_vol.return_value = 0.08  # High volatility
            
            order_type_result = self.execution_policy.optimize_order_type(
                symbol='BTC/USDT',
                side='buy',
                urgency='normal'
            )
            
            assert order_type_result.recommended_type == OrderType.LIMIT
            assert order_type_result.reasoning is not None
    
    def test_execution_result_tracking(self):
        """Test execution result tracking and analysis."""
        execution_result = ExecutionResult(
            order_id='test-order-123',
            symbol='BTC/USDT',
            executed_quantity=Decimal('0.1'),
            average_price=Decimal('50100'),
            total_fees=Decimal('25.05'),
            execution_time_ms=150,
            slippage_bps=8
        )
        
        self.execution_policy.record_execution_result(execution_result)
        
        # Check execution statistics
        stats = self.execution_policy.get_execution_statistics('BTC/USDT')
        
        assert stats.average_slippage_bps >= 0
        assert stats.average_execution_time_ms >= 0
        assert stats.total_executions >= 1
    
    def test_adaptive_parameters(self):
        """Test adaptive parameter adjustment based on performance."""
        # Simulate poor execution performance
        for i in range(10):
            poor_result = ExecutionResult(
                order_id=f'poor-{i}',
                symbol='BTC/USDT',
                executed_quantity=Decimal('0.1'),
                average_price=Decimal('50000'),
                total_fees=Decimal('25'),
                execution_time_ms=500,  # Slow execution
                slippage_bps=45  # High slippage
            )
            self.execution_policy.record_execution_result(poor_result)
        
        # Should adapt parameters to improve performance
        adapted_params = self.execution_policy.adapt_parameters('BTC/USDT')
        
        assert adapted_params.max_slippage_bps != self.execution_policy.max_slippage_bps
        assert adapted_params.recommended_adjustments is not None
    
    def test_post_only_mode(self):
        """Test post-only execution mode."""
        self.execution_policy.enable_post_only_mode(True)
        
        market_order = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            order_type=OrderType.MARKET,
            quantity=Decimal('0.1'),
            time_in_force=TimeInForce.IOC
        )
        
        # Should convert market order to limit order in post-only mode
        converted_order = self.execution_policy.convert_to_post_only(market_order)
        
        assert converted_order.order_type == OrderType.LIMIT
        assert converted_order.time_in_force == TimeInForce.GTX  # Good Till Crossing
    
    def test_emergency_execution_mode(self):
        """Test emergency execution mode with relaxed constraints."""
        self.execution_policy.enable_emergency_mode(True)
        
        # Should allow higher slippage in emergency mode
        emergency_order = OrderRequest(
            symbol='BTC/USDT',
            side='sell',
            order_type=OrderType.MARKET,
            quantity=Decimal('1.0'),
            time_in_force=TimeInForce.IOC
        )
        
        validation_result = self.execution_policy.validate_order(emergency_order)
        
        # Should be more permissive in emergency mode
        assert validation_result.is_valid is True or len(validation_result.errors) < 3


@pytest.mark.unit
class TestOrderRequest:
    """Test OrderRequest data structure."""
    
    def test_order_creation(self):
        """Test order request creation."""
        order = OrderRequest(
            symbol='ETH/USDT',
            side='buy',
            order_type=OrderType.LIMIT,
            quantity=Decimal('1.0'),
            price=Decimal('3000'),
            time_in_force=TimeInForce.GTC,
            client_order_id='test-123'
        )
        
        assert order.symbol == 'ETH/USDT'
        assert order.side == 'buy'
        assert order.order_type == OrderType.LIMIT
        assert order.quantity == Decimal('1.0')
        assert order.price == Decimal('3000')
        assert order.client_order_id == 'test-123'
    
    def test_order_validation(self):
        """Test order request validation."""
        # Valid order
        valid_order = OrderRequest(
            symbol='BTC/USDT',
            side='buy',
            order_type=OrderType.MARKET,
            quantity=Decimal('0.1'),
            time_in_force=TimeInForce.IOC
        )
        
        assert valid_order.is_valid() is True
        
        # Invalid order - negative quantity
        with pytest.raises(ValueError):
            OrderRequest(
                symbol='BTC/USDT',
                side='buy',
                order_type=OrderType.MARKET,
                quantity=Decimal('-0.1'),
                time_in_force=TimeInForce.IOC
            )
    
    def test_order_serialization(self):
        """Test order serialization to dict."""
        order = OrderRequest(
            symbol='BTC/USDT',
            side='sell',
            order_type=OrderType.LIMIT,
            quantity=Decimal('0.5'),
            price=Decimal('55000'),
            time_in_force=TimeInForce.GTC
        )
        
        order_dict = order.to_dict()
        
        assert isinstance(order_dict, dict)
        assert order_dict['symbol'] == 'BTC/USDT'
        assert order_dict['side'] == 'sell'
        assert order_dict['quantity'] == '0.5'
        assert order_dict['price'] == '55000'


@pytest.mark.unit
class TestExecutionResult:
    """Test ExecutionResult data structure."""
    
    def test_execution_result_creation(self):
        """Test execution result creation."""
        result = ExecutionResult(
            order_id='exec-123',
            symbol='BTC/USDT',
            executed_quantity=Decimal('0.1'),
            average_price=Decimal('50000'),
            total_fees=Decimal('25'),
            execution_time_ms=100,
            slippage_bps=5
        )
        
        assert result.order_id == 'exec-123'
        assert result.symbol == 'BTC/USDT'
        assert result.executed_quantity == Decimal('0.1')
        assert result.slippage_bps == 5
    
    def test_execution_quality_metrics(self):
        """Test execution quality metrics calculation."""
        result = ExecutionResult(
            order_id='quality-test',
            symbol='ETH/USDT',
            executed_quantity=Decimal('1.0'),
            average_price=Decimal('3050'),
            total_fees=Decimal('15.25'),
            execution_time_ms=75,
            slippage_bps=12
        )
        
        quality_score = result.calculate_quality_score()
        
        assert 0 <= quality_score <= 100
        assert isinstance(quality_score, float)