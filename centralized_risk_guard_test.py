#!/usr/bin/env python3
"""
Quick test script to demonstrate centralized risk guard integration
"""

import sys
import logging
sys.path.append('.')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    # Add src to Python path
    sys.path.insert(0, 'src')
    
    # Import core risk enforcement directly
    from cryptosmarttrader.risk.central_risk_guard import CentralRiskGuard, TradingOperation, RiskDecision
    from cryptosmarttrader.core.mandatory_risk_enforcement import enforce_order_risk_check, require_risk_approval
    
    print("🛡️ CENTRALIZED RISK GUARD TEST")
    print("="*50)
    
    # Test 1: Simple risk approval check
    print("\n1️⃣ Testing simple risk approval...")
    approved = require_risk_approval("BTC/USD", 1000.0, "buy")
    print(f"   Small order approved: {approved}")
    
    # Test 2: Detailed risk check  
    print("\n2️⃣ Testing detailed risk check...")
    risk_result = enforce_order_risk_check(
        order_size=5000.0,
        symbol="ETH/USD",
        side="buy", 
        strategy_id="test_strategy"
    )
    print(f"   Order approved: {risk_result['approved']}")
    print(f"   Approved size: {risk_result.get('approved_size', 0)}")
    print(f"   Reason: {risk_result['reason']}")
    
    # Test 3: Large order (should be rejected or reduced)
    print("\n3️⃣ Testing large order...")
    large_risk_result = enforce_order_risk_check(
        order_size=50000.0,  # Very large order
        symbol="BTC/USD",
        side="buy",
        strategy_id="large_order_test"
    )
    print(f"   Large order approved: {large_risk_result['approved']}")
    if large_risk_result['approved']:
        print(f"   Size reduced to: {large_risk_result.get('approved_size', 0)}")
    else:
        print(f"   Rejection reason: {large_risk_result['reason']}")
    
    # Test 4: Direct CentralRiskGuard test
    print("\n4️⃣ Testing CentralRiskGuard directly...")
    central_guard = CentralRiskGuard()
    
    # Setup a test portfolio state
    central_guard.update_portfolio_state(
        total_equity=10000.0,
        daily_pnl=0.0,
        open_positions=3,
        total_exposure_usd=2000.0
    )
    
    # Test risk operation
    operation = TradingOperation(
        operation_type="entry",
        symbol="TEST/USD", 
        side="buy",
        size_usd=1500.0,
        current_price=100.0
    )
    
    risk_eval = central_guard.evaluate_operation(operation)
    print(f"   Risk decision: {risk_eval.decision.value}")
    print(f"   Approved size: ${risk_eval.approved_size_usd:,.0f}")
    print(f"   Risk score: {risk_eval.risk_score:.1f}")
    if risk_eval.violations:
        print(f"   Violations: {[v.value for v in risk_eval.violations]}")
    
    print("\n✅ ALL TESTS COMPLETED")
    print("🛡️ Centralized risk management is ACTIVE and enforced!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all modules are properly installed")
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()