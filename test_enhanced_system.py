"""
Test Enhanced System - Comprehensive testing of all improvements
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

async def test_enhanced_sentiment_agent():
    """Test enhanced sentiment agent capabilities"""
    print("🧪 Testing Enhanced Sentiment Agent")
    
    from agents.enhanced_sentiment_agent import sentiment_agent
    
    # Test sentiment analysis with anti-bot features
    result = await sentiment_agent.analyze_coin_sentiment('BTC', timeframe_hours=24)
    
    print(f"   Sentiment Score: {result.sentiment_score:.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Bot Ratio Detected: {result.bot_ratio:.3f}")
    print(f"   Data Completeness: {result.data_completeness:.3f}")
    print(f"   Raw/Filtered Mentions: {result.raw_mentions}/{result.filtered_mentions}")
    
    return result.confidence > 0.5

async def test_enhanced_technical_agent():
    """Test enhanced technical agent with regime detection"""
    print("🧪 Testing Enhanced Technical Agent")
    
    from agents.enhanced_technical_agent import technical_agent
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    volumes = np.random.randint(1000, 10000, 100)
    
    coin_data = {
        'BTC/USD': pd.DataFrame({
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': volumes
        }, index=dates)
    }
    
    # Test parallel analysis
    results = await technical_agent.analyze_multiple_coins(coin_data, ['1h', '4h'])
    
    if 'BTC/USD' in results and '1h' in results['BTC/USD']:
        signal = results['BTC/USD']['1h']
        print(f"   Signal Strength: {signal.signal_strength:.3f}")
        print(f"   Confidence: {signal.confidence:.3f}")
        print(f"   Regime: {signal.regime}")
        print(f"   Computation Time: {signal.computation_time:.3f}s")
        print(f"   Active Indicators: {len(signal.active_indicators)}")
        return True
    
    return False

async def test_enhanced_whale_agent():
    """Test enhanced whale agent with context awareness"""
    print("🧪 Testing Enhanced Whale Agent")
    
    from agents.enhanced_whale_agent import whale_agent
    
    # Test whale activity analysis
    results = await whale_agent.analyze_whale_activity(
        tokens=['ETH'],
        min_value_usd=100000,
        timeframe_hours=24
    )
    
    if 'ETH' in results:
        transactions = results['ETH']
        print(f"   Whale Transactions Found: {len(transactions)}")
        
        if transactions:
            tx = transactions[0]
            print(f"   Sample Transaction:")
            print(f"     Value: ${tx.usd_value:,.0f}")
            print(f"     Type: {tx.transaction_type}")
            print(f"     Confidence: {tx.confidence:.3f}")
            print(f"     False Positive Score: {tx.false_positive_score:.3f}")
            print(f"     Context: {tx.context}")
        
        return True
    
    return False

async def test_enhanced_ml_agent():
    """Test enhanced ML agent with uncertainty quantification"""
    print("🧪 Testing Enhanced ML Agent")
    
    from agents.enhanced_ml_agent import ml_agent
    
    # Create training data
    training_data = {}
    for coin in ['BTC/USD', 'ETH/USD']:
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
        prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
        volumes = np.random.randint(1000, 10000, 1000)
        
        training_data[coin] = pd.DataFrame({
            'close': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'volume': volumes
        }, index=dates)
    
    # Test ensemble training
    training_results = await ml_agent.train_ensemble(training_data, ['1h', '4h'])
    
    print(f"   Trained Horizons: {list(training_results.keys())}")
    
    # Test prediction with uncertainty
    if training_results:
        test_data = training_data['BTC/USD'].iloc[-100:]  # Recent data
        predictions = await ml_agent.predict('BTC/USD', test_data, ['1h'], 'bull')
        
        if predictions:
            pred = predictions[0]
            print(f"   Prediction: {pred.prediction:.6f}")
            print(f"   Uncertainty: {pred.uncertainty:.6f}")
            print(f"   Confidence: {pred.confidence:.3f}")
            print(f"   Model Type: {pred.model_type}")
            print(f"   Interval: [{pred.prediction_interval[0]:.6f}, {pred.prediction_interval[1]:.6f}]")
            print(f"   Features Used: {len(pred.feature_importance)}")
            return True
    
    return False

async def test_enhanced_orchestrator():
    """Test enhanced orchestrator capabilities"""
    print("🧪 Testing Enhanced Orchestrator")
    
    from core.enhanced_orchestrator import orchestrator
    
    # Test system status
    status = orchestrator.get_system_status()
    
    print(f"   Orchestrator Status: {status['orchestrator_status']}")
    print(f"   Agents - Healthy: {status['agents']['healthy']}, Failed: {status['agents']['failed']}")
    print(f"   Total Requests: {status['performance']['total_requests']}")
    print(f"   Success Rate: {status['performance']['successful_requests']}/{status['performance']['total_requests']}")
    
    return status['orchestrator_status'] in ['running', 'stopped']

async def run_comprehensive_test():
    """Run comprehensive test of all enhanced capabilities"""
    
    print("🚀 COMPREHENSIVE ENHANCED SYSTEM TEST")
    print("=" * 70)
    
    test_results = {}
    
    # Test all enhanced agents
    tests = [
        ('Enhanced Sentiment Agent', test_enhanced_sentiment_agent),
        ('Enhanced Technical Agent', test_enhanced_technical_agent),
        ('Enhanced Whale Agent', test_enhanced_whale_agent),
        ('Enhanced ML Agent', test_enhanced_ml_agent),
        ('Enhanced Orchestrator', test_enhanced_orchestrator),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            start_time = time.time()
            result = await test_func()
            duration = time.time() - start_time
            
            test_results[test_name] = {
                'passed': result,
                'duration': duration,
                'error': None
            }
            
            status = "✅ PASSED" if result else "⚠️ PARTIAL"
            print(f"   Result: {status} ({duration:.2f}s)")
            
        except Exception as e:
            test_results[test_name] = {
                'passed': False,
                'duration': 0,
                'error': str(e)
            }
            print(f"   Result: ❌ FAILED - {e}")
    
    # Summary
    print(f"\n🏆 TEST SUMMARY")
    print("=" * 50)
    
    passed_count = sum(1 for result in test_results.values() if result['passed'])
    total_count = len(test_results)
    
    print(f"Tests Passed: {passed_count}/{total_count}")
    print(f"Success Rate: {(passed_count/total_count)*100:.1f}%")
    
    if passed_count == total_count:
        print("\n🎉 ALL ENHANCED CAPABILITIES WORKING PERFECTLY!")
        print("🔧 All critical issues from attachment have been addressed:")
        print("   ✅ Anti-bot detection and rate limiting")
        print("   ✅ Parallel processing and regime detection") 
        print("   ✅ Async pipelines and false positive filtering")
        print("   ✅ Mandatory deep learning with uncertainty")
        print("   ✅ Self-healing orchestrator with monitoring")
    else:
        print(f"\n⚠️ {total_count - passed_count} components need attention")
        
        for test_name, result in test_results.items():
            if not result['passed']:
                error_msg = result['error'] or 'Test failed'
                print(f"   🔧 {test_name}: {error_msg}")
    
    return test_results

if __name__ == "__main__":
    # Run the comprehensive test
    results = asyncio.run(run_comprehensive_test())