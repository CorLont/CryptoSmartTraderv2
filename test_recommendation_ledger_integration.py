#!/usr/bin/env python3
"""
Test script voor recommendation ledger integratie met practical coin pipeline

Test de complete flow:
1. Pipeline run met recommendation logging
2. Ledger data persistence 
3. Performance analytics generatie
4. Training label extraction
"""

import asyncio
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Set up paths
import sys
sys.path.append('src')

from cryptosmarttrader.alpha.practical_coin_pipeline import (
    PracticalCoinPipeline, CoinCandidate, Regime, UniverseFilters
)
from cryptosmarttrader.trading.recommendation_ledger import (
    RecommendationLedger, TradingSide, SignalScores, ExitReason,
    get_recommendation_ledger
)
from cryptosmarttrader.alpha.enhanced_signal_generators import generate_sample_market_data


async def test_recommendation_ledger_integration():
    """Test complete recommendation ledger integration"""
    
    print("🔍 Testing Recommendation Ledger Integration")
    print("=" * 60)
    
    # Test 1: Initialize ledger
    print("\n1️⃣ Testing ledger initialization...")
    ledger = RecommendationLedger("test_data/test_recommendations.json")
    
    # Ensure test directory exists
    Path("test_data").mkdir(exist_ok=True)
    
    # Test 2: Initialize pipeline with recommendation logging
    print("\n2️⃣ Testing pipeline initialization...")
    universe_filters = UniverseFilters(
        min_volume_24h_usd=5_000_000,  # Lower for testing
        max_spread_bps=100,
        min_depth_usd=100_000
    )
    
    pipeline = PracticalCoinPipeline(
        universe_filters=universe_filters,
        max_positions=5,
        enable_whale_detection=True,
        enable_sentiment_monitoring=True
    )
    
    # Test 3: Generate test market data
    print("\n3️⃣ Generating test market data...")
    market_data = generate_sample_market_data()
    
    # Test 4: Run pipeline with recommendation logging
    print("\n4️⃣ Running pipeline with recommendation logging...")
    try:
        results = await pipeline.run_pipeline(market_data)
        print(f"✅ Pipeline completed with {len(results)} recommendations")
        
        # Check if recommendations were logged
        for result in results:
            if hasattr(result, 'recommendation_id') and result.recommendation_id:
                print(f"   📋 {result.symbol}: {result.recommendation_id}")
            else:
                print(f"   ⚠️  {result.symbol}: No recommendation ID")
                
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return False
    
    # Test 5: Check ledger data
    print("\n5️⃣ Testing ledger data persistence...")
    active_recs = ledger.get_active_recommendations()
    print(f"📊 Active recommendations: {len(active_recs)}")
    
    if active_recs:
        for rec in active_recs[:3]:  # Show first 3
            print(f"   🎯 {rec.symbol} {rec.side.value} | Score: {rec.signal_scores.combined_score:.3f}")
    
    # Test 6: Performance analytics
    print("\n6️⃣ Testing performance analytics...")
    analytics = ledger.get_performance_analytics()
    print(f"📈 Analytics: {json.dumps(analytics, indent=2)}")
    
    # Test 7: Simulate some entries and exits for testing
    print("\n7️⃣ Testing entry/exit simulation...")
    simulation_count = 0
    
    for rec in active_recs[:2]:  # Test with first 2 recommendations
        # Simulate entry
        entry_price = 45000.0 + (simulation_count * 1000)  # Varied prices
        success = ledger.update_entry_execution(
            rec.recommendation_id,
            entry_price,
            0.1,
            datetime.now(timezone.utc)
        )
        
        if success:
            print(f"   ✅ Entry logged: {rec.symbol} @ ${entry_price}")
            
            # Simulate exit after small profit/loss
            profit_factor = 1.02 if simulation_count % 2 == 0 else 0.98
            exit_price = entry_price * profit_factor
            exit_reason = ExitReason.TAKE_PROFIT if profit_factor > 1 else ExitReason.STOP_LOSS
            
            success = ledger.close_recommendation(
                rec.recommendation_id,
                exit_price,
                exit_reason,
                datetime.now(timezone.utc) + timedelta(hours=2)
            )
            
            if success:
                print(f"   ✅ Exit logged: {rec.symbol} @ ${exit_price} ({exit_reason.value})")
            
        simulation_count += 1
    
    # Test 8: Updated analytics after trades
    print("\n8️⃣ Testing analytics after simulated trades...")
    updated_analytics = ledger.get_performance_analytics()
    print(f"📈 Updated Analytics:")
    for key, value in updated_analytics.items():
        if isinstance(value, (int, float)):
            print(f"   {key}: {value}")
        elif isinstance(value, dict):
            print(f"   {key}: {json.dumps(value, indent=4)}")
    
    # Test 9: Training labels generation
    print("\n9️⃣ Testing training labels generation...")
    labels_df = ledger.generate_training_labels()
    
    if not labels_df.empty:
        print(f"📊 Training labels generated: {len(labels_df)} samples")
        print(f"   Columns: {list(labels_df.columns)}")
        
        # Show sample labels
        if len(labels_df) > 0:
            print("   Sample labels:")
            for _, row in labels_df.head(2).iterrows():
                print(f"     Symbol: {row['symbol']} | "
                      f"PnL: {row['realized_pnl_bps']}bps | "
                      f"Labels: P={row['label_profitable']} S={row['label_significant']} M={row['label_multiclass']}")
    else:
        print("📊 No training labels (no completed trades)")
    
    # Test 10: File persistence check
    print("\n🔟 Testing file persistence...")
    if ledger.ledger_path.exists():
        file_size = ledger.ledger_path.stat().st_size
        print(f"📁 Ledger file: {ledger.ledger_path} ({file_size} bytes)")
        
        # Read and validate JSON format
        try:
            with open(ledger.ledger_path, 'r') as f:
                line_count = sum(1 for line in f if line.strip())
            print(f"📝 Recommendation records: {line_count}")
        except Exception as e:
            print(f"❌ File read error: {e}")
    else:
        print("❌ Ledger file not found")
    
    print("\n" + "=" * 60)
    print("✅ Recommendation Ledger Integration Test Complete")
    
    return True


async def test_ledger_performance():
    """Test ledger performance with multiple recommendations"""
    
    print("\n🚀 Testing Ledger Performance")
    print("-" * 40)
    
    ledger = RecommendationLedger("test_data/performance_test.json")
    
    # Create multiple test recommendations
    test_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "DOT/USDT"]
    recommendation_ids = []
    
    start_time = datetime.now()
    
    # Bulk recommendation logging
    for i, symbol in enumerate(test_symbols * 5):  # 25 recommendations total
        signal_scores = SignalScores(
            momentum_score=0.5 + (i % 10) * 0.05,
            sentiment_score=0.3 + (i % 8) * 0.05,
            whale_score=0.4 + (i % 6) * 0.1,
            combined_score=0.6 + (i % 12) * 0.03,
            confidence=0.7 + (i % 5) * 0.05
        )
        
        features = {
            "price_usd": 45000 + (i * 1000),
            "volume_24h": 1000000000 + (i * 50000000),
            "test_iteration": i
        }
        
        rec_id = ledger.log_recommendation(
            symbol=symbol,
            side=TradingSide.BUY if i % 2 == 0 else TradingSide.SELL,
            signal_scores=signal_scores,
            features_snapshot=features,
            expected_return_bps=100 + (i * 10),
            risk_budget_bps=150 + (i * 5),
            slippage_budget_bps=20 + (i * 2)
        )
        
        recommendation_ids.append(rec_id)
    
    log_time = datetime.now() - start_time
    print(f"⏱️  Logged {len(recommendation_ids)} recommendations in {log_time.total_seconds():.2f}s")
    
    # Test analytics performance
    start_time = datetime.now()
    analytics = ledger.get_performance_analytics()
    analytics_time = datetime.now() - start_time
    
    print(f"📊 Generated analytics in {analytics_time.total_seconds():.3f}s")
    print(f"   Total recommendations: {analytics['total_recommendations']}")
    print(f"   Active recommendations: {analytics['active_recommendations']}")
    
    # Test history retrieval
    start_time = datetime.now()
    history_df = ledger.get_recommendation_history(days_back=1)
    history_time = datetime.now() - start_time
    
    print(f"📈 Retrieved history in {history_time.total_seconds():.3f}s")
    print(f"   History records: {len(history_df)}")
    
    print("✅ Performance test complete")


if __name__ == "__main__":
    print("🧪 Recommendation Ledger Integration Test Suite")
    print("=" * 60)
    
    # Run main integration test
    loop = asyncio.get_event_loop()
    success = loop.run_until_complete(test_recommendation_ledger_integration())
    
    if success:
        # Run performance test
        loop.run_until_complete(test_ledger_performance())
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print("1. Review test_data/test_recommendations.json for logged data")
        print("2. Integrate with actual trading execution")
        print("3. Set up automated performance monitoring")
        print("4. Configure ML training pipeline with generated labels")
    else:
        print("\n❌ Tests failed - check logs for details")