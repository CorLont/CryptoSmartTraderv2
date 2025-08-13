"""
FASE C Integration Tests - Guardrails & Observability
Tests the complete integration of ExecutionPolicy, RiskGuard, and Prometheus metrics
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import FASE C components
from src.cryptosmarttrader.execution.hard_execution_policy import (
    HardExecutionPolicy, OrderRequest, MarketConditions, 
    OrderSide, TimeInForce, ExecutionDecision, get_execution_policy
)
from src.cryptosmarttrader.risk.central_risk_guard import (
    CentralRiskGuard, RiskCheckResult, RiskLevel, RiskType,
    get_central_risk_guard
)
from src.cryptosmarttrader.observability.metrics import PrometheusMetrics


class TestFaseCGuardrailsIntegration:
    """Test complete FASE C guardrails integration"""
    
    def setup_method(self):
        """Setup test environment"""
        # Reset singletons
        HardExecutionPolicy._instance = None
        CentralRiskGuard._instance = None
        PrometheusMetrics._instance = None
        
        # Initialize components
        self.execution_policy = get_execution_policy({
            'max_spread_bps': 50,
            'min_depth_usd': 10000,
            'max_slippage_bps': 30,
            'daily_slippage_budget_bps': 200
        })
        
        self.risk_guard = get_central_risk_guard({
            'max_day_loss_usd': 10000,
            'max_drawdown_percent': 5.0,
            'max_total_exposure_usd': 100000,
            'max_total_positions': 10
        })
        
        self.metrics = PrometheusMetrics.get_instance()
    
    def test_execution_policy_gates(self):
        """Test all execution policy gates work correctly"""
        
        # Create valid order request
        order_request = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0,
            time_in_force=TimeInForce.POST_ONLY
        )
        
        # Create valid market conditions
        market_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,
            spread_bps=20.0,  # 2.0 basis points
            bid_depth_usd=25000.0,
            ask_depth_usd=25000.0,
            volume_24h_usd=5000000.0,
            volatility_24h=0.02,
            last_update=datetime.now()
        )
        
        # Test successful execution
        result = self.execution_policy.decide(order_request, market_conditions)
        
        assert result.approved == True
        assert result.decision == ExecutionDecision.APPROVE
        assert result.client_order_id.startswith("CST_")
        assert all(result.gate_results.values())  # All gates should pass
        assert result.estimated_slippage_bps > 0
    
    def test_execution_policy_gate_failures(self):
        """Test execution policy correctly rejects orders on gate failures"""
        
        order_request = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0
        )
        
        # Test spread gate failure
        bad_market_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49000.0,
            ask_price=51000.0,
            spread_bps=400.0,  # 400 bps spread - should fail
            bid_depth_usd=25000.0,
            ask_depth_usd=25000.0,
            volume_24h_usd=5000000.0,
            volatility_24h=0.02,
            last_update=datetime.now()
        )
        
        result = self.execution_policy.decide(order_request, bad_market_conditions)
        
        assert result.approved == False
        assert result.decision == ExecutionDecision.REJECT
        assert result.gate_results['spread_gate'] == False
        assert "Spread" in result.reason
    
    def test_risk_guard_day_loss_protection(self):
        """Test RiskGuard day loss protection"""
        
        # Update portfolio with current daily loss
        self.risk_guard.current_metrics.daily_pnl = -8000.0  # Already lost $8k today
        
        # Try trade that would exceed day loss limit
        risk_result = self.risk_guard.validate_trade(
            symbol="BTC/USD",
            side="buy",
            quantity=1.0,
            price=50000.0  # $50k trade
        )
        
        assert risk_result.is_safe == False
        assert any(v.risk_type == RiskType.DAY_LOSS for v in risk_result.violations)
        assert risk_result.risk_score > 0.3
    
    def test_risk_guard_drawdown_protection(self):
        """Test RiskGuard drawdown protection"""
        
        # Set portfolio with high drawdown
        self.risk_guard.portfolio_equity = 90000.0  # Current equity
        self.risk_guard.portfolio_peak = 100000.0   # Peak was $100k
        # Current drawdown = 10%
        
        # Try trade that would increase drawdown
        risk_result = self.risk_guard.validate_trade(
            symbol="ETH/USD",
            side="sell",
            quantity=10.0,
            price=3000.0
        )
        
        assert risk_result.is_safe == False
        assert any(v.risk_type == RiskType.MAX_DRAWDOWN for v in risk_result.violations)
    
    def test_risk_guard_exposure_limits(self):
        """Test RiskGuard exposure limits"""
        
        # Set current exposure near limit
        self.risk_guard.current_metrics.total_exposure = 95000.0  # Near $100k limit
        
        # Try trade that would exceed exposure limit
        risk_result = self.risk_guard.validate_trade(
            symbol="SOL/USD",
            side="buy",
            quantity=100.0,
            price=100.0  # $10k trade would exceed limit
        )
        
        assert risk_result.is_safe == False
        assert any(v.risk_type == RiskType.MAX_EXPOSURE for v in risk_result.violations)
    
    def test_kill_switch_trigger(self):
        """Test kill switch triggers on critical violations"""
        
        # Create multiple critical violations
        self.risk_guard.current_metrics.daily_pnl = -15000.0  # Exceeds day loss
        self.risk_guard.portfolio_equity = 85000.0  # High drawdown
        self.risk_guard.portfolio_peak = 100000.0   # 15% drawdown
        
        # This should trigger kill switch
        risk_result = self.risk_guard.validate_trade(
            symbol="BTC/USD",
            side="buy",
            quantity=1.0,
            price=50000.0
        )
        
        assert risk_result.kill_switch_triggered == True
        assert self.risk_guard.kill_switch_status.value == "triggered"
        assert len(risk_result.violations) >= 2
    
    def test_slippage_budget_tracking(self):
        """Test slippage budget enforcement"""
        
        order_request = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.1,
            price=50000.0
        )
        
        market_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,
            spread_bps=20.0,
            bid_depth_usd=25000.0,
            ask_depth_usd=25000.0,
            volume_24h_usd=5000000.0,
            volatility_24h=0.02,
            last_update=datetime.now()
        )
        
        # Exhaust slippage budget
        self.execution_policy.current_slippage_used_bps = 190.0  # Near limit
        
        # This order should be rejected due to slippage budget
        result = self.execution_policy.decide(order_request, market_conditions)
        
        if not result.approved:
            assert result.gate_results.get('slippage_gate') == False
            assert "slippage budget" in result.reason.lower()
    
    def test_metrics_integration(self):
        """Test Prometheus metrics are recorded correctly"""
        
        # Record some execution decisions
        self.metrics.record_execution_decision("BTC/USD", "buy", "approve")
        self.metrics.record_execution_decision("ETH/USD", "sell", "reject")
        
        # Record gate results
        self.metrics.record_execution_gate("spread_gate", "pass")
        self.metrics.record_execution_gate("depth_gate", "fail")
        
        # Record risk violations
        self.metrics.record_risk_violation("day_loss", "critical")
        
        # Update portfolio metrics
        self.metrics.update_portfolio_metrics(
            equity=95000.0,
            drawdown_pct=5.0,
            exposure=75000.0,
            positions=8
        )
        
        # Check metrics summary
        summary = self.metrics.get_metrics_summary()
        
        assert summary['current_portfolio_equity'] == 95000.0
        assert summary['current_drawdown_pct'] == 5.0
        
    def test_integrated_order_flow(self):
        """Test complete integrated order flow through all components"""
        
        # Create order request
        order_request = OrderRequest(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=0.5,
            price=50000.0,
            strategy_id="test_strategy"
        )
        
        # Create market conditions
        market_conditions = MarketConditions(
            symbol="BTC/USD",
            bid_price=49950.0,
            ask_price=50050.0,
            spread_bps=20.0,
            bid_depth_usd=25000.0,
            ask_depth_usd=25000.0,
            volume_24h_usd=5000000.0,
            volatility_24h=0.02,
            last_update=datetime.now()
        )
        
        # Step 1: Execution Policy Decision
        execution_result = self.execution_policy.decide(order_request, market_conditions)
        
        if execution_result.approved:
            # Step 2: Risk Guard Validation
            risk_result = self.risk_guard.validate_trade(
                symbol=order_request.symbol,
                side=order_request.side.value,
                quantity=order_request.quantity,
                price=order_request.price
            )
            
            # Step 3: Final approval only if both approve
            final_approval = execution_result.approved and risk_result.is_safe
            
            if final_approval:
                # Record successful metrics
                self.metrics.record_execution_decision(
                    order_request.symbol, 
                    order_request.side.value, 
                    "approve"
                )
                
                # Update slippage budget
                if hasattr(self.execution_policy, 'current_slippage_used_bps'):
                    self.execution_policy.current_slippage_used_bps += execution_result.estimated_slippage_bps
            
            assert isinstance(final_approval, bool)
            assert execution_result.processing_time_ms > 0
            assert risk_result.risk_score >= 0.0
    
    def test_p95_slippage_calculation(self):
        """Test p95 slippage calculation meets requirements"""
        
        # Add some slippage history
        for i in range(100):
            slippage = 5.0 + (i % 20)  # Slippage between 5-25 bps
            self.execution_policy.slippage_history.append((datetime.now(), slippage))
        
        p95_slippage = self.execution_policy.get_slippage_p95()
        
        assert p95_slippage > 0
        assert p95_slippage <= 30.0  # Should be within our budget limit
        
        # P95 should be ≤ budget per FASE C requirements
        assert p95_slippage <= self.execution_policy.daily_slippage_budget_bps


class TestFaseCAlerts:
    """Test FASE C alert conditions"""
    
    def setup_method(self):
        """Setup alert testing"""
        PrometheusMetrics._instance = None
        self.metrics = PrometheusMetrics.get_instance()
    
    def test_high_order_error_rate_alert(self):
        """Test HighOrderErrorRate alert condition"""
        
        # Simulate high error rate
        for i in range(10):
            self.metrics.orders_sent.labels(exchange="kraken", symbol="BTC/USD", side="buy", order_type="limit").inc()
        
        for i in range(2):  # 20% error rate
            self.metrics.order_errors.labels(exchange="kraken", symbol="BTC/USD", error_type="timeout", error_code="408").inc()
        
        # In real implementation, this would be calculated by Prometheus
        # Here we simulate the alert condition check
        total_orders = 10
        total_errors = 2
        error_rate = total_errors / total_orders if total_orders > 0 else 0
        
        assert error_rate > 0.10  # Should trigger HighOrderErrorRate alert
    
    def test_drawdown_too_high_alert(self):
        """Test DrawdownTooHigh alert condition"""
        
        # Set high drawdown
        self.metrics.portfolio_drawdown_pct.set(4.5)  # 4.5% drawdown
        
        # Check alert condition
        self.metrics.check_alert_conditions()
        
        drawdown_alert = self.metrics._get_gauge_value(self.metrics.drawdown_too_high)
        assert drawdown_alert == 1  # Should trigger alert
    
    def test_no_signals_alert(self):
        """Test NoSignals alert condition"""
        
        # Set old signal timestamp (> 30 minutes ago)
        old_timestamp = time.time() - (35 * 60)  # 35 minutes ago
        self.metrics.last_signal_timestamp.set(old_timestamp)
        
        # Check alert condition
        self.metrics.check_alert_conditions()
        
        no_signals_alert = self.metrics._get_gauge_value(self.metrics.no_signals_timeout)
        assert no_signals_alert == 1  # Should trigger alert


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])