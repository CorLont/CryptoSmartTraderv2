#!/usr/bin/env python3
"""
Eenvoudige test van de recommendation ledger functionaliteit
"""

import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Set up paths
sys.path.append('src')

from cryptosmarttrader.trading.recommendation_ledger import (
    RecommendationLedger, TradingSide, SignalScores, ExitReason
)

def test_basic_recommendation_flow():
    """Test de basisfunctionaliteit van recommendation logging"""
    
    print("🧪 Testing Basic Recommendation Ledger Flow")
    print("=" * 50)
    
    # Maak test directory
    Path("test_data").mkdir(exist_ok=True)
    
    # Initialize ledger
    ledger = RecommendationLedger("test_data/simple_test_recommendations.json")
    print("✅ Ledger initialized")
    
    # Test 1: Log een recommendation
    signal_scores = SignalScores(
        momentum_score=0.75,
        mean_revert_score=0.40,
        sentiment_score=0.60,
        whale_score=0.80,
        combined_score=0.68,
        confidence=0.85
    )
    
    features_snapshot = {
        "price_usd": 45000.0,
        "volume_24h_usd": 2000000000,
        "market_cap_usd": 850000000000,
        "rsi_14": 58.5,
        "spread_bps": 15.0,
        "depth_usd": 1500000
    }
    
    rec_id = ledger.log_recommendation(
        symbol="BTC/USDT",
        side=TradingSide.BUY,
        signal_scores=signal_scores,
        features_snapshot=features_snapshot,
        expected_return_bps=200,
        risk_budget_bps=300,
        slippage_budget_bps=25,
        market_regime="BULL_TREND",
        volatility_percentile=45.5,
        liquidity_score=0.95
    )
    
    print(f"✅ Recommendation logged: {rec_id}")
    
    # Test 2: Log nog een recommendation (SELL)
    rec_id_2 = ledger.log_recommendation(
        symbol="ETH/USDT",
        side=TradingSide.SELL,
        signal_scores=SignalScores(
            momentum_score=0.30,
            sentiment_score=0.25,
            whale_score=0.40,
            combined_score=0.35,
            confidence=0.70
        ),
        features_snapshot={
            "price_usd": 3200.0,
            "volume_24h_usd": 1500000000,
            "rsi_14": 72.3
        },
        expected_return_bps=150,
        risk_budget_bps=250,
        slippage_budget_bps=30
    )
    
    print(f"✅ Second recommendation logged: {rec_id_2}")
    
    # Test 3: Check active recommendations
    active_recs = ledger.get_active_recommendations()
    print(f"📊 Active recommendations: {len(active_recs)}")
    
    for rec in active_recs:
        print(f"   {rec.symbol} {rec.side.value} | Score: {rec.signal_scores.combined_score:.3f}")
    
    # Test 4: Simulate entry execution
    success = ledger.update_entry_execution(rec_id, 45100.0, 0.1)
    print(f"✅ Entry execution logged: {success}")
    
    # Test 5: Simulate profitable exit
    success = ledger.close_recommendation(
        rec_id, 
        45800.0, 
        ExitReason.TAKE_PROFIT,
        datetime.now(timezone.utc) + timedelta(hours=3)
    )
    print(f"✅ Exit execution logged: {success}")
    
    # Test 6: Simulate losing trade for second recommendation
    ledger.update_entry_execution(rec_id_2, 3180.0, 0.5)
    ledger.close_recommendation(
        rec_id_2,
        3250.0,  # Loss for SELL position
        ExitReason.STOP_LOSS,
        datetime.now(timezone.utc) + timedelta(hours=1)
    )
    print("✅ Second trade completed (loss)")
    
    # Test 7: Check analytics
    analytics = ledger.get_performance_analytics()
    print("\n📈 Performance Analytics:")
    print(json.dumps(analytics, indent=2))
    
    # Test 8: Check training labels
    labels_df = ledger.generate_training_labels()
    print(f"\n📊 Training labels generated: {len(labels_df)} samples")
    
    if not labels_df.empty:
        print("Sample labels:")
        for _, row in labels_df.iterrows():
            print(f"   {row['symbol']}: PnL={row.get('realized_pnl_bps', 'N/A')}bps | "
                  f"Profitable={row.get('label_profitable', 'N/A')}")
    
    # Test 9: Check file persistence
    if ledger.ledger_path.exists():
        file_size = ledger.ledger_path.stat().st_size
        print(f"\n📁 Ledger file created: {file_size} bytes")
        
        # Count records
        with open(ledger.ledger_path, 'r') as f:
            lines = [line for line in f if line.strip()]
        print(f"📝 Records in file: {len(lines)}")
    
    print("\n✅ All tests completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Recommendation logging with full context")
    print("• Entry/exit execution tracking") 
    print("• Performance analytics generation")
    print("• ML training label creation")
    print("• JSON file persistence")
    
    return True

if __name__ == "__main__":
    test_basic_recommendation_flow()