"""
Test suite voor CentralRiskGuard - comprehensive risk validation testing
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.cryptosmarttrader.risk.central_risk_guard import (
    CentralRiskGuard, 
    OrderRequest, 
    RiskLimits, 
    PortfolioState, 
    RiskDecision
)


class TestCentralRiskGuard:
    """Test central risk guard functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_risk_limits.json"
        
        # Initialize risk guard with test config
        self.risk_guard = CentralRiskGuard(str(self.config_path))
        
        # Test order
        self.test_order = OrderRequest(
            symbol="BTC/USD",
            side="buy", 
            size=1.0,
            price=50000.0,
            client_order_id="test_order_123"
        )
        
        # Test market data
        self.test_market_data = {
            'timestamp': 1234567890,
            'price': 50000.0,
            'volume': 1000.0,
            'spread': 0.5
        }
    
    def test_risk_guard_initialization(self):
        """Test risk guard initializes correctly"""
        assert self.risk_guard is not None
        assert isinstance(self.risk_guard.limits, RiskLimits)
        assert isinstance(self.risk_guard.portfolio_state, PortfolioState)
        assert self.risk_guard.evaluation_count == 0
    
    def test_kill_switch_blocks_all_orders(self):
        """Test kill switch blocks all orders"""
        # Activate kill switch
        self.risk_guard.activate_kill_switch("Test activation")
        
        # Try to place order
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            self.test_order, self.test_market_data
        )
        
        assert decision == RiskDecision.EMERGENCY_STOP
        assert "KILL_SWITCH_ACTIVE" in reason
        assert adjusted_size is None
    
    def test_data_quality_rejection(self):
        """Test rejection due to poor data quality"""
        # Test with missing data
        bad_market_data = {'price': 50000.0}  # Missing volume and spread
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            self.test_order, bad_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "DATA_QUALITY_FAIL" in reason
    
    def test_daily_loss_limit_rejection(self):
        """Test rejection due to daily loss limits"""
        # Set portfolio with large daily loss
        portfolio_state = PortfolioState(
            total_value_usd=100000.0,
            daily_pnl_usd=-6000.0  # Exceeds $5000 limit
        )
        self.risk_guard.update_portfolio_state(portfolio_state)
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            self.test_order, self.test_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "DAILY_LOSS_LIMIT" in reason
    
    def test_position_count_limit_rejection(self):
        """Test rejection due to position count limits"""
        # Set portfolio with max positions
        positions = {f"COIN{i}/USD": {"value_usd": 1000} for i in range(10)}
        portfolio_state = PortfolioState(
            position_count=10,
            positions=positions
        )
        self.risk_guard.update_portfolio_state(portfolio_state)
        
        # Try to open new position
        new_order = OrderRequest(
            symbol="NEW/USD",
            side="buy",
            size=1.0,
            price=1000.0
        )
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            new_order, self.test_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "POSITION_LIMIT" in reason
    
    def test_exposure_limit_size_reduction(self):
        """Test size reduction due to exposure limits"""
        # Set portfolio near exposure limit
        portfolio_state = PortfolioState(
            total_exposure_usd=80000.0  # Close to $100k limit
        )
        self.risk_guard.update_portfolio_state(portfolio_state)
        
        # Large order that would exceed limit
        large_order = OrderRequest(
            symbol="BTC/USD",
            side="buy",
            size=1.0,
            price=50000.0  # Would add $50k exposure, exceeding limit
        )
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            large_order, self.test_market_data
        )
        
        assert decision == RiskDecision.REDUCE_SIZE
        assert "EXPOSURE_LIMIT" in reason
        assert adjusted_size is not None
        assert adjusted_size < large_order.size
    
    def test_single_position_size_reduction(self):
        """Test size reduction due to single position limits"""
        # Large order exceeding single position limit
        large_order = OrderRequest(
            symbol="BTC/USD",
            side="buy",
            size=2.0,
            price=50000.0  # $100k position > $50k limit
        )
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            large_order, self.test_market_data
        )
        
        assert decision == RiskDecision.REDUCE_SIZE
        assert "POSITION_SIZE_LIMIT" in reason
        assert adjusted_size is not None
        assert adjusted_size < large_order.size
    
    def test_correlation_limit_rejection(self):
        """Test rejection due to correlation limits"""
        # Set portfolio with high BTC exposure
        positions = {
            "BTC/USD": {"value_usd": 60000},
            "BTC/EUR": {"value_usd": 20000}
        }
        portfolio_state = PortfolioState(
            total_value_usd=100000.0,
            positions=positions
        )
        self.risk_guard.update_portfolio_state(portfolio_state)
        
        # Try to add more BTC exposure
        btc_order = OrderRequest(
            symbol="BTC/GBP",
            side="buy",
            size=0.2,
            price=50000.0
        )
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            btc_order, self.test_market_data
        )
        
        assert decision == RiskDecision.REJECT
        assert "CORRELATION_LIMIT" in reason
    
    def test_successful_order_approval(self):
        """Test successful order approval when all gates pass"""
        # Clean portfolio state
        portfolio_state = PortfolioState(
            total_value_usd=100000.0,
            daily_pnl_usd=0.0,
            position_count=2,
            total_exposure_usd=10000.0
        )
        self.risk_guard.update_portfolio_state(portfolio_state)
        
        # Small reasonable order
        small_order = OrderRequest(
            symbol="ETH/USD",
            side="buy",
            size=10.0,
            price=2000.0  # $20k position
        )
        
        decision, reason, adjusted_size = self.risk_guard.evaluate_order(
            small_order, self.test_market_data
        )
        
        assert decision == RiskDecision.APPROVE
        assert "ALL_RISK_GATES_PASSED" in reason
        assert adjusted_size is None
    
    def test_risk_metrics_tracking(self):
        """Test risk metrics are tracked correctly"""
        initial_metrics = self.risk_guard.get_risk_metrics()
        assert initial_metrics['evaluation_count'] == 0
        
        # Process some orders
        self.risk_guard.evaluate_order(self.test_order, self.test_market_data)
        
        # Activate kill switch to force rejection
        self.risk_guard.activate_kill_switch("Test")
        self.risk_guard.evaluate_order(self.test_order, self.test_market_data)
        
        final_metrics = self.risk_guard.get_risk_metrics()
        assert final_metrics['evaluation_count'] == 2
        assert final_metrics['rejection_count'] == 1
        assert final_metrics['rejection_rate'] == 0.5
        assert final_metrics['kill_switch_active'] is True
    
    def test_config_persistence(self):
        """Test risk limits config persistence"""
        # Modify limits
        original_daily_limit = self.risk_guard.limits.max_daily_loss_usd
        self.risk_guard.limits.max_daily_loss_usd = 10000.0
        self.risk_guard._save_risk_limits(self.risk_guard.limits)
        
        # Create new instance - should load saved config
        new_risk_guard = CentralRiskGuard(str(self.config_path))
        assert new_risk_guard.limits.max_daily_loss_usd == 10000.0
        assert new_risk_guard.limits.max_daily_loss_usd != original_daily_limit
    
    def test_audit_log_creation(self):
        """Test audit log is created and populated"""
        # Process order to generate log entry
        self.risk_guard.evaluate_order(self.test_order, self.test_market_data)
        
        # Check audit log exists and has content
        assert self.risk_guard.audit_log_path.exists()
        
        with open(self.risk_guard.audit_log_path, 'r') as f:
            log_content = f.read().strip()
            assert log_content != ""
            
            # Parse last log entry
            last_entry = json.loads(log_content.split('\n')[-1])
            assert last_entry['symbol'] == self.test_order.symbol
            assert last_entry['decision'] in [d.value for d in RiskDecision]
    
    def test_performance_requirements(self):
        """Test risk evaluation performance < 10ms"""
        import time
        
        # Warm up
        for _ in range(5):
            self.risk_guard.evaluate_order(self.test_order, self.test_market_data)
        
        # Measure performance
        start_time = time.time()
        for _ in range(100):
            self.risk_guard.evaluate_order(self.test_order, self.test_market_data)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / 100 * 1000
        assert avg_time_ms < 10.0, f"Risk evaluation too slow: {avg_time_ms:.1f}ms > 10ms"
    
    def test_error_handling_fails_safe(self):
        """Test error handling fails safe (rejects on error)"""
        # Mock a method to raise exception
        with patch.object(self.risk_guard, '_check_data_quality', side_effect=Exception("Test error")):
            decision, reason, adjusted_size = self.risk_guard.evaluate_order(
                self.test_order, self.test_market_data
            )
            
            assert decision == RiskDecision.REJECT
            assert "RISK_EVALUATION_ERROR" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])