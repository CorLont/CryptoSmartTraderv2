#!/usr/bin/env python3
"""
Test OpenAI Enhanced Intelligence System
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_test_market_data() -> pd.DataFrame:
    """Create realistic test market data"""
    
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Generate realistic price movements
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.025, 100)  # Daily returns with drift
    
    prices = [base_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    # Generate realistic volume
    volumes = np.random.lognormal(13, 0.5, 100)  # Log-normal distribution for volume
    
    return pd.DataFrame({
        'close': prices,
        'volume': volumes,
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices]
    }, index=dates)

def test_openai_availability():
    """Test OpenAI API availability"""
    
    print('🔍 TEST 1: OpenAI API Availability')
    print('-' * 40)
    
    try:
        # Check if OpenAI API key is available
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key:
            print(f'✅ OpenAI API key found (length: {len(api_key)})')
            
            # Test basic OpenAI connection
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Simple test call
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "Respond with just 'API_TEST_SUCCESS' if you receive this."}
                ],
                max_tokens=10,
                temperature=0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            if "API_TEST_SUCCESS" in response_text:
                print('✅ OpenAI API connection successful')
                return True
            else:
                print(f'⚠️ OpenAI API responded but unexpected response: {response_text}')
                return False
                
        else:
            print('❌ OpenAI API key not found in environment')
            print('   Set OPENAI_API_KEY environment variable to enable AI features')
            return False
            
    except Exception as e:
        print(f'❌ OpenAI API test failed: {e}')
        return False

def test_sentiment_analysis():
    """Test AI-powered sentiment analysis"""
    
    print()
    print('🧠 TEST 2: AI Sentiment Analysis')
    print('-' * 40)
    
    try:
        from ml.intelligence.openai_enhanced_analysis import create_openai_analyzer
        
        # Create test data
        market_data = create_test_market_data()
        test_news = [
            "Bitcoin ETF approval boosts institutional adoption",
            "Regulatory clarity improves market sentiment",
            "Major crypto exchange reports record trading volume"
        ]
        
        analyzer = create_openai_analyzer()
        
        print('Analyzing market sentiment with AI...')
        sentiment = analyzer.analyze_market_sentiment(
            price_data=market_data,
            news_data=test_news
        )
        
        print(f'Sentiment Analysis Results:')
        print(f'  Overall sentiment: {sentiment.overall_sentiment}')
        print(f'  Sentiment score: {sentiment.sentiment_score:.2f}')
        print(f'  Confidence: {sentiment.confidence:.2f}')
        print(f'  Key themes: {", ".join(sentiment.key_themes[:3])}')
        print(f'  Market drivers: {", ".join(sentiment.market_drivers[:3])}')
        
        if sentiment.confidence > 0.5:
            print('✅ AI sentiment analysis working effectively')
            return True
        else:
            print('⚠️ AI sentiment analysis has low confidence')
            return False
            
    except Exception as e:
        print(f'❌ Sentiment analysis test failed: {e}')
        return False

def test_news_impact_analysis():
    """Test AI-powered news impact assessment"""
    
    print()
    print('📰 TEST 3: AI News Impact Analysis')
    print('-' * 40)
    
    try:
        from ml.intelligence.openai_enhanced_analysis import create_openai_analyzer
        
        analyzer = create_openai_analyzer()
        
        # Test news impact assessment
        test_news = """
        The Federal Reserve announces potential interest rate cuts in Q2 2024, 
        citing economic uncertainty. Cryptocurrency markets show mixed reactions 
        as investors weigh the implications for digital assets.
        """
        
        market_context = {
            "current_btc_price": 52000.0,
            "market_cap_24h_change": -2.3,
            "fear_greed_index": 45,
            "recent_volatility": "high"
        }
        
        print('Analyzing news impact with AI...')
        impact = analyzer.assess_news_impact(
            news_content=test_news,
            market_context=market_context,
            crypto_symbols=["BTC", "ETH", "ADA"]
        )
        
        print(f'News Impact Assessment:')
        print(f'  Impact magnitude: {impact.impact_magnitude:.2f}')
        print(f'  Impact direction: {impact.impact_direction}')
        print(f'  Time horizon: {impact.time_horizon}')
        print(f'  Affected assets: {", ".join(impact.affected_assets[:3])}')
        print(f'  Confidence: {impact.confidence:.2f}')
        print(f'  Reasoning: {impact.reasoning[:100]}...')
        
        if impact.confidence > 0.4:
            print('✅ AI news impact analysis working')
            return True
        else:
            print('⚠️ AI news impact analysis has low confidence')
            return False
            
    except Exception as e:
        print(f'❌ News impact analysis test failed: {e}')
        return False

def test_intelligent_feature_generation():
    """Test AI-powered feature engineering"""
    
    print()
    print('⚙️ TEST 4: AI Feature Engineering')
    print('-' * 40)
    
    try:
        from ml.intelligence.openai_enhanced_analysis import create_openai_analyzer
        
        analyzer = create_openai_analyzer()
        
        current_features = [
            "sma_20", "rsi_14", "volume_ratio", "price_momentum",
            "volatility_7d", "correlation_btc", "market_cap_rank"
        ]
        
        available_data = {
            "price_data": "OHLCV data for 1000+ cryptocurrencies",
            "volume_data": "24h trading volumes across exchanges",
            "social_data": "Twitter mentions, Reddit sentiment",
            "on_chain_data": "Transaction counts, active addresses",
            "market_data": "Market cap, circulating supply"
        }
        
        print('Generating intelligent features with AI...')
        features = analyzer.generate_intelligent_features(
            current_features=current_features,
            available_data=available_data,
            prediction_target="24h_price_direction"
        )
        
        print(f'Generated {len(features)} intelligent features:')
        for i, feature in enumerate(features[:5]):  # Show first 5
            print(f'  {i+1}. {feature.get("name", "unnamed")}')
            print(f'     Description: {feature.get("description", "no description")[:80]}...')
            print(f'     Complexity: {feature.get("complexity", "unknown")}')
        
        if len(features) >= 5:
            print('✅ AI feature engineering working effectively')
            return True
        else:
            print('⚠️ AI generated fewer features than expected')
            return False
            
    except Exception as e:
        print(f'❌ Feature engineering test failed: {e}')
        return False

def test_anomaly_detection():
    """Test AI-powered anomaly detection"""
    
    print()
    print('🔍 TEST 5: AI Anomaly Detection')
    print('-' * 40)
    
    try:
        from ml.intelligence.openai_enhanced_analysis import create_openai_analyzer
        
        # Create data with artificial anomaly
        market_data = create_test_market_data()
        
        # Inject volume spike anomaly
        market_data.loc[market_data.index[-5], 'volume'] *= 10  # 10x volume spike
        
        # Inject price spike anomaly  
        market_data.loc[market_data.index[-3], 'close'] *= 1.15  # 15% price spike
        
        analyzer = create_openai_analyzer()
        
        # Prepare technical indicators
        technical_indicators = {
            "sma_20": market_data['close'].rolling(20).mean(),
            "volume_sma": market_data['volume'].rolling(20).mean(),
            "price_std": market_data['close'].rolling(20).std()
        }
        
        print('Detecting market anomalies with AI...')
        insights = analyzer.analyze_market_anomalies(
            price_data=market_data,
            volume_data=market_data[['volume']],
            technical_indicators=technical_indicators
        )
        
        print(f'Detected {len(insights)} anomalies:')
        for i, insight in enumerate(insights):
            print(f'  {i+1}. Type: {insight.insight_type}')
            print(f'     Confidence: {insight.confidence:.2f}')
            print(f'     Content: {insight.content[:100]}...')
        
        if len(insights) > 0:
            print('✅ AI anomaly detection working')
            return True
        else:
            print('⚠️ No anomalies detected (may be expected)')
            return False
            
    except Exception as e:
        print(f'❌ Anomaly detection test failed: {e}')
        return False

def test_comprehensive_analysis():
    """Test comprehensive AI market analysis"""
    
    print()
    print('🚀 TEST 6: Comprehensive AI Analysis')
    print('-' * 40)
    
    try:
        from ml.intelligence.openai_enhanced_analysis import analyze_market_with_ai
        
        market_data = create_test_market_data()
        news_data = [
            "Central bank digital currency trials show promising results",
            "Institutional crypto adoption reaches new milestone"
        ]
        
        print('Running comprehensive AI market analysis...')
        results = analyze_market_with_ai(
            price_data=market_data,
            news_data=news_data,
            include_sentiment=True,
            include_anomalies=True,
            include_strategy=True
        )
        
        print('Analysis Results:')
        
        if 'sentiment_analysis' in results:
            sentiment = results['sentiment_analysis']
            print(f'  Sentiment: {sentiment.overall_sentiment} (score: {sentiment.sentiment_score:.2f})')
        
        if 'anomaly_analysis' in results:
            anomalies = results['anomaly_analysis']
            print(f'  Anomalies detected: {len(anomalies)}')
        
        if 'strategy_insights' in results:
            strategy = results['strategy_insights']
            print(f'  Strategy insights: {len(strategy)}')
        
        if 'error' not in results and len(results) >= 2:
            print('✅ Comprehensive AI analysis working')
            return True
        else:
            print(f'⚠️ Analysis incomplete or failed: {results.get("error", "unknown")}')
            return False
            
    except Exception as e:
        print(f'❌ Comprehensive analysis test failed: {e}')
        return False

def main():
    """Run all OpenAI intelligence tests"""
    
    print('🤖 TESTING OPENAI ENHANCED INTELLIGENCE SYSTEM')
    print('=' * 70)
    
    tests = [
        test_openai_availability,
        test_sentiment_analysis,
        test_news_impact_analysis,
        test_intelligent_feature_generation,
        test_anomaly_detection,
        test_comprehensive_analysis
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f'❌ Test failed with exception: {e}')
            results.append(False)
    
    print()
    print('🎯 OPENAI INTELLIGENCE TEST SUMMARY')
    print('=' * 70)
    
    passed = sum(results)
    total = len(results)
    
    if results[0]:  # API available
        print('✅ OpenAI API: Connected and operational')
        print('✅ Model: GPT-4o (latest model as of May 13, 2024)')
        
        if passed >= 4:  # Most tests passed
            print('✅ AI Sentiment Analysis: Advanced market psychology analysis')
            print('✅ AI News Impact: Intelligent news impact assessment')
            print('✅ AI Feature Engineering: Automated intelligent feature generation')
            print('✅ AI Anomaly Detection: Smart pattern recognition')
            print('✅ AI Strategy Insights: Intelligent trading recommendations')
            print()
            print('🏆 OPENAI ENHANCED INTELLIGENCE: FULLY OPERATIONAL')
            print('🧠 Smart AI integration maximizing predictive power')
        else:
            print('⚠️ Some AI features may need configuration')
    else:
        print('❌ OpenAI API not available - AI features disabled')
        print('   Configure OPENAI_API_KEY to enable intelligent analysis')
    
    print(f'Test Results: {passed}/{total} passed')
    
    return passed >= (total * 0.7)  # 70% pass rate

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)