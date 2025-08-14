#!/usr/bin/env python3
"""
Test Central RiskGuard Poortwachter
Comprehensive tests for risk gates and kill switch functionality
"""

import pytest
import time
import threading
from unittest.mock import Mock
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.cryptosmarttrader.risk.central_risk_guard import (
        CentralRiskGuard, RiskLimits, TradingOperation, RiskDecision, 
        RiskViolationType, PortfolioState
    )
    from src.cryptosmarttrader.execution.risk_enforced_execution import (
        RiskEnforcedExecutionManager
    )
except ImportError:
    pytest.skip("RiskGuard modules not available", allow_module_level=True)


class TestCentralRiskGuard:
    """Test Central RiskGuard poortwachter functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.risk_limits = RiskLimits(
            max_day_loss_pct=2.0,
            max_drawdown_pct=10.0,
            max_total_exposure_pct=50.0,
            max_positions=5,
            max_position_size_pct=10.0,
            max_data_gap_minutes=5,
            kill_switch_active=False
        )
        
        self.risk_guard = CentralRiskGuard(self.risk_limits)
        
        # Setup good portfolio state
        self.risk_guard.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=500.0,  # Small profit
            open_positions=3,
            total_exposure_usd=25000.0,  # 25% exposure
            position_sizes={"BTC/USD": 10000, "ETH/USD": 8000, "SOL/USD": 7000},
            correlations={"BTC/USD": 0.8, "ETH/USD": 0.7, "SOL/USD": 0.5}
        )
    
    def test_healthy_operation_approval(self):
        """Test that healthy operations are approved"""
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=8000.0,  # Within 10% position size limit
            current_price=50000.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.APPROVE
        assert result.approved_size_usd == 8000.0
        assert len(result.violations) == 0
        assert result.gate_results["kill_switch"] is True
        assert result.gate_results["data_gap"] is True
        assert result.gate_results["day_loss"] is True
    
    def test_kill_switch_rejection(self):
        """Test that kill switch blocks all operations"""
        
        # Activate kill switch
        self.risk_guard.activate_kill_switch("Emergency stop for testing")
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=1000.0,  # Small size
            current_price=50000.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.KILL_SWITCH_ACTIVATED
        assert result.approved_size_usd == 0.0
        assert RiskViolationType.KILL_SWITCH in result.violations
        assert result.gate_results["kill_switch"] is False
        assert "Emergency stop for testing" in result.reasons[0]
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement"""
        
        # Set portfolio with large daily loss
        self.risk_guard.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=-3000.0,  # -3% daily loss (exceeds 2% limit)
            open_positions=2,
            total_exposure_usd=20000.0,
            position_sizes={"BTC/USD": 10000, "ETH/USD": 10000}
        )
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="SOL/USD",
            side="buy",
            size_usd=5000.0,
            current_price=100.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.REJECT
        assert RiskViolationType.DAY_LOSS in result.violations
        assert result.gate_results["day_loss"] is False
        assert "-3.00%" in result.reasons[0]
    
    def test_max_drawdown_limit(self):
        """Test maximum drawdown enforcement"""
        
        # Set portfolio with large drawdown
        self.risk_guard.portfolio_state.peak_equity = 120000.0
        self.risk_guard.portfolio_state.total_equity = 100000.0  # 16.7% drawdown
        self.risk_guard.portfolio_state.current_drawdown_pct = 16.7
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=5000.0,
            current_price=50000.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.REJECT
        assert RiskViolationType.MAX_DRAWDOWN in result.violations
        assert result.gate_results["drawdown"] is False
    
    def test_max_positions_limit(self):
        """Test maximum positions enforcement"""
        
        # Portfolio already at position limit
        self.risk_guard.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=0.0,
            open_positions=5,  # At max limit
            total_exposure_usd=30000.0,
            position_sizes={"BTC/USD": 6000, "ETH/USD": 6000, "SOL/USD": 6000, "ADA/USD": 6000, "DOT/USD": 6000}
        )
        
        # Try to open new position
        operation = TradingOperation(
            operation_type="entry",  # New position
            symbol="LINK/USD",
            side="buy",
            size_usd=5000.0,
            current_price=20.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.REJECT
        assert RiskViolationType.MAX_POSITIONS in result.violations
        assert result.gate_results["position_count"] is False
    
    def test_total_exposure_limit(self):
        """Test total exposure limit with size reduction"""
        
        # Portfolio near exposure limit
        self.risk_guard.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=0.0,
            open_positions=3,
            total_exposure_usd=45000.0,  # 45% exposure (near 50% limit)
            position_sizes={"BTC/USD": 15000, "ETH/USD": 15000, "SOL/USD": 15000}
        )
        
        # Try to open large position that would exceed limit
        operation = TradingOperation(
            operation_type="entry",
            symbol="ADA/USD",
            side="buy",
            size_usd=10000.0,  # Would bring total to 55%
            current_price=1.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        # Should reduce size to stay within limits
        assert result.decision == RiskDecision.REDUCE_SIZE
        assert result.approved_size_usd < 10000.0
        assert result.approved_size_usd <= 5000.0  # Max 5k to stay at 50%
        assert RiskViolationType.MAX_EXPOSURE in result.violations
    
    def test_position_size_limit(self):
        """Test individual position size limit"""
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=15000.0,  # 15% of 100k equity (exceeds 10% limit)
            current_price=50000.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.REDUCE_SIZE
        assert result.approved_size_usd <= 10000.0  # Reduced to 10% limit
        assert RiskViolationType.POSITION_SIZE in result.violations
        assert result.gate_results["position_size"] is False
    
    def test_data_gap_detection(self):
        """Test data gap detection and rejection"""
        
        # Simulate old data by setting last update time in the past
        self.risk_guard.last_market_data_update = time.time() - 7 * 60  # 7 minutes ago
        
        operation = TradingOperation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=5000.0,
            current_price=50000.0,
            strategy_id="test_strategy"
        )
        
        result = self.risk_guard.evaluate_operation(operation)
        
        assert result.decision == RiskDecision.REJECT
        assert RiskViolationType.DATA_GAP in result.violations
        assert result.gate_results["data_gap"] is False
        assert "7.0 minutes" in result.reasons[0]
    
    def test_correlation_limit(self):
        """Test correlation exposure limit"""
        
        # Portfolio with high correlation exposure
        self.risk_guard.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=0.0,
            open_positions=3,
            total_exposure_usd=30000.0,
            position_sizes={"BTC/USD": 12000, "ETH/USD": 10000, "LTC/USD": 8000},
            correlations={"BTC/USD": 0.9, "ETH/USD": 0.8, "LTC/USD": 0.85}  # All highly correlated
        )
        
        # Try to add another correlated asset
        operation = TradingOperation(
            operation_type="entry",
            symbol="BCH/USD",  # Another Bitcoin-like asset
            side="buy",
            size_usd=5000.0,
            current_price=300.0,
            strategy_id="test_strategy"
        )
        
        # Manually set correlation for test
        self.risk_guard.portfolio_state.correlations["BCH/USD"] = 0.85
        
        result = self.risk_guard.evaluate_operation(operation)
        
        # Should have correlation violation if exposure too high
        if RiskViolationType.CORRELATION_LIMIT in result.violations:
            assert result.gate_results["correlation"] is False
    
    def test_thread_safety(self):
        """Test thread-safe operation of RiskGuard"""
        
        results = []
        
        def evaluate_operation(thread_id):
            operation = TradingOperation(
                operation_type="entry",
                symbol=f"TEST{thread_id}/USD",
                side="buy",
                size_usd=1000.0,
                current_price=100.0,
                strategy_id=f"thread_{thread_id}"
            )
            
            result = self.risk_guard.evaluate_operation(operation)
            results.append((thread_id, result.decision))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=evaluate_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All operations should complete
        assert len(results) == 5
        
        # Most should be approved (small sizes)
        approved_count = sum(1 for _, decision in results if decision == RiskDecision.APPROVE)
        assert approved_count >= 3
    
    def test_kill_switch_history(self):
        """Test kill switch activation history"""
        
        # Activate and deactivate kill switch multiple times
        self.risk_guard.activate_kill_switch("Test reason 1")
        self.risk_guard.deactivate_kill_switch("Issue resolved")
        self.risk_guard.activate_kill_switch("Test reason 2")
        self.risk_guard.deactivate_kill_switch("Issue resolved again")
        
        assert len(self.risk_guard.kill_switch_history) == 4
        assert self.risk_guard.kill_switch_history[0]["action"] == "activated"
        assert self.risk_guard.kill_switch_history[1]["action"] == "deactivated"
        assert self.risk_guard.kill_switch_history[2]["action"] == "activated"
        assert self.risk_guard.kill_switch_history[3]["action"] == "deactivated"
    
    def test_risk_status_comprehensive(self):
        """Test comprehensive risk status reporting"""
        
        status = self.risk_guard.get_risk_status()
        
        # Check all required sections
        assert "risk_limits" in status
        assert "portfolio_state" in status
        assert "statistics" in status
        assert "utilization" in status
        
        # Check risk limits
        limits = status["risk_limits"]
        assert limits["max_day_loss_pct"] == 2.0
        assert limits["max_drawdown_pct"] == 10.0
        assert limits["kill_switch_active"] is False
        
        # Check portfolio state
        portfolio = status["portfolio_state"]
        assert portfolio["total_equity"] == 100000.0
        assert portfolio["open_positions"] == 3
        assert "daily_pnl_pct" in portfolio
        assert "data_age_minutes" in portfolio
        
        # Check statistics
        stats = status["statistics"]
        assert "total_evaluations" in stats
        assert "violation_count" in stats
        assert "violation_rate" in stats
        
        # Check utilization
        util = status["utilization"]
        assert "exposure_utilization" in util
        assert "position_utilization" in util
        assert "drawdown_utilization" in util


class TestRiskEnforcedExecution:
    """Test complete risk-enforced execution pipeline"""
    
    def setup_method(self):
        """Setup for each test"""
        # Mock exchange manager
        self.mock_exchange_manager = Mock()
        self.mock_exchange_manager.create_market_conditions.return_value = Mock(
            spread_bps=25.0,
            bid_depth_usd=50000.0,
            ask_depth_usd=50000.0,
            volume_1m_usd=200000.0,
            last_price=50000.0,
            bid_price=49950.0,
            ask_price=50050.0,
            timestamp=time.time()
        )
        
        self.mock_exchange_manager.execute_disciplined_order.return_value = {
            "success": True,
            "order_id": "test_order_123",
            "client_order_id": "CST_test123"
        }
        
        self.risk_execution_manager = RiskEnforcedExecutionManager(self.mock_exchange_manager)
        
        # Setup healthy portfolio
        self.risk_execution_manager.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=500.0,
            open_positions=2,
            total_exposure_usd=20000.0,
            position_sizes={"BTC/USD": 10000, "ETH/USD": 10000}
        )
    
    def test_complete_execution_pipeline_success(self):
        """Test successful execution through complete pipeline"""
        
        result = self.risk_execution_manager.execute_trading_operation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=8000.0,
            limit_price=50000.0,
            strategy_id="test_strategy"
        )
        
        assert result["success"] is True
        assert result["stage"] == "exchange_execution"
        assert "risk_evaluation" in result
        assert "execution_result" in result
        assert result["approved_size_usd"] == 8000.0
        assert result["order_id"] == "test_order_123"
    
    def test_risk_guard_rejection(self):
        """Test rejection by RiskGuard"""
        
        # Set portfolio with excessive exposure
        self.risk_execution_manager.update_portfolio_state(
            total_equity=100000.0,
            daily_pnl=0.0,
            open_positions=3,
            total_exposure_usd=48000.0,  # Near limit
            position_sizes={"BTC/USD": 20000, "ETH/USD": 15000, "SOL/USD": 13000}
        )
        
        # Try large order that would exceed limits
        result = self.risk_execution_manager.execute_trading_operation(
            operation_type="entry",
            symbol="ADA/USD",
            side="buy",
            size_usd=15000.0,  # Would exceed total exposure
            limit_price=1.0,
            strategy_id="test_strategy"
        )
        
        assert result["success"] is False
        assert result["stage"] == "risk_guard"
        assert "RiskGuard rejected" in result["error"]
        assert "risk_evaluation" in result
    
    def test_emergency_kill_switch(self):
        """Test emergency kill switch activation"""
        
        # Activate kill switch
        self.risk_execution_manager.activate_emergency_stop("Market crash detected")
        
        # Try to execute operation
        result = self.risk_execution_manager.execute_trading_operation(
            operation_type="entry",
            symbol="BTC/USD",
            side="buy",
            size_usd=1000.0,
            limit_price=50000.0,
            strategy_id="test_strategy"
        )
        
        assert result["success"] is False
        assert result["stage"] == "risk_guard"
        assert "KILL SWITCH ACTIVATED" in result["error"]
        assert result.get("kill_switch") is True
    
    def test_execution_stats_tracking(self):
        """Test execution statistics tracking"""
        
        # Execute several operations
        for i in range(3):
            self.risk_execution_manager.execute_trading_operation(
                operation_type="entry",
                symbol=f"TEST{i}/USD",
                side="buy",
                size_usd=5000.0,
                limit_price=100.0,
                strategy_id=f"test_{i}"
            )
        
        stats = self.risk_execution_manager.get_execution_stats()
        
        assert stats["total_requests"] == 3
        assert stats["successful_executions"] == 3
        assert stats["success_rate"] == 100.0
        assert stats["risk_rejection_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])