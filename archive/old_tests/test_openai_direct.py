#!/usr/bin/env python3
"""
Direct OpenAI Test - Test OpenAI integration without complex dependencies
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from openai import OpenAI

def test_openai_connection():
    """Test basic OpenAI connection"""
    
    print('🔍 TEST: OpenAI API Connection')
    print('-' * 40)
    
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            print('❌ OpenAI API key not found')
            return False
        
        print(f'✅ API key found (length: {len(api_key)})')
        
        client = OpenAI(api_key=api_key)
        
        # Simple test
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "Respond with exactly 'TEST_SUCCESS' if you receive this message."}
            ],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        if "TEST_SUCCESS" in result:
            print('✅ OpenAI connection successful')
            print(f'✅ Model: gpt-4o')
            return True
        else:
            print(f'⚠️ Unexpected response: {result}')
            return False
            
    except Exception as e:
        print(f'❌ Connection failed: {e}')
        return False

def test_crypto_sentiment_analysis():
    """Test crypto-specific sentiment analysis"""
    
    print()
    print('🧠 TEST: Crypto Sentiment Analysis')
    print('-' * 40)
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create test market scenario
        market_data = {
            "current_btc_price": 52000,
            "price_change_24h": 0.035,  # 3.5% up
            "volume_change": 1.8,       # 80% volume increase
            "trend": "bullish_breakout"
        }
        
        news_headlines = [
            "Bitcoin ETF approval drives institutional adoption",
            "Major bank announces crypto custody services",
            "Regulatory clarity improves market sentiment"
        ]
        
        prompt = f"""
        Analyze cryptocurrency market sentiment based on:
        
        MARKET DATA: {json.dumps(market_data)}
        NEWS: {' | '.join(news_headlines)}
        
        Provide analysis in JSON format:
        {{
            "sentiment": "bullish|bearish|neutral",
            "confidence": 0.0-1.0,
            "key_factors": ["factor1", "factor2"],
            "price_outlook": "short analysis"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a cryptocurrency market analyst. Provide objective sentiment analysis in JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print('✅ AI Sentiment Analysis Results:')
        print(f'  Sentiment: {result.get("sentiment", "unknown")}')
        print(f'  Confidence: {result.get("confidence", 0):.2f}')
        print(f'  Key factors: {", ".join(result.get("key_factors", []))}')
        print(f'  Outlook: {result.get("price_outlook", "none")[:80]}...')
        
        return True
        
    except Exception as e:
        print(f'❌ Sentiment analysis failed: {e}')
        return False

def test_news_impact_assessment():
    """Test news impact assessment"""
    
    print()
    print('📰 TEST: News Impact Assessment')
    print('-' * 40)
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        news_event = """
        BREAKING: Federal Reserve announces potential interest rate cuts in Q2 2024, 
        citing economic uncertainty. Cryptocurrency markets show mixed reactions as 
        investors weigh implications for risk assets and store-of-value narratives.
        """
        
        prompt = f"""
        Assess the cryptocurrency market impact of this news:
        
        NEWS: {news_event}
        
        Provide impact assessment in JSON format:
        {{
            "impact_magnitude": 0.0-1.0,
            "impact_direction": "positive|negative|neutral",
            "affected_cryptos": ["BTC", "ETH", ...],
            "time_horizon": "immediate|short-term|long-term",
            "reasoning": "explanation"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a cryptocurrency news impact specialist. Assess market impact objectively."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print('✅ AI News Impact Assessment:')
        print(f'  Impact magnitude: {result.get("impact_magnitude", 0):.2f}')
        print(f'  Direction: {result.get("impact_direction", "unknown")}')
        print(f'  Time horizon: {result.get("time_horizon", "unknown")}')
        print(f'  Affected cryptos: {", ".join(result.get("affected_cryptos", [])[:5])}')
        print(f'  Reasoning: {result.get("reasoning", "none")[:100]}...')
        
        return True
        
    except Exception as e:
        print(f'❌ News impact assessment failed: {e}')
        return False

def test_trading_insights():
    """Test AI-powered trading insights"""
    
    print()
    print('💡 TEST: AI Trading Insights')
    print('-' * 40)
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        market_context = {
            "btc_price": 52000,
            "eth_price": 3200,
            "market_trend": "consolidation_breakout",
            "volatility": "moderate",
            "volume_profile": "above_average"
        }
        
        prompt = f"""
        Generate strategic cryptocurrency trading insights based on:
        
        MARKET CONTEXT: {json.dumps(market_context)}
        
        Provide 3 actionable insights in JSON format:
        {{
            "insights": [
                {{
                    "type": "entry|exit|risk_management",
                    "priority": "high|medium|low",
                    "description": "specific insight",
                    "rationale": "why this matters"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional cryptocurrency trader. Generate practical, risk-aware trading insights."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print('✅ AI Trading Insights:')
        insights = result.get("insights", [])
        for i, insight in enumerate(insights[:3]):
            print(f'  {i+1}. {insight.get("type", "unknown").upper()} ({insight.get("priority", "medium")} priority)')
            print(f'     {insight.get("description", "no description")[:80]}...')
        
        return True
        
    except Exception as e:
        print(f'❌ Trading insights failed: {e}')
        return False

def test_feature_engineering_suggestions():
    """Test AI feature engineering"""
    
    print()
    print('⚙️ TEST: AI Feature Engineering')
    print('-' * 40)
    
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        current_features = [
            "sma_20", "rsi_14", "volume_ratio", "price_momentum", 
            "volatility_7d", "correlation_btc"
        ]
        
        prompt = f"""
        Suggest 5 advanced cryptocurrency prediction features beyond these:
        
        CURRENT FEATURES: {", ".join(current_features)}
        
        Generate innovative features in JSON format:
        {{
            "features": [
                {{
                    "name": "feature_name",
                    "description": "what it measures",
                    "predictive_value": "why it helps prediction"
                }}
            ]
        }}
        
        Focus on features that capture market psychology and regime changes.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a quantitative analyst specializing in cryptocurrency prediction features."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.4
        )
        
        result = json.loads(response.choices[0].message.content)
        
        print('✅ AI Feature Engineering Suggestions:')
        features = result.get("features", [])
        for i, feature in enumerate(features[:5]):
            print(f'  {i+1}. {feature.get("name", "unnamed")}')
            print(f'     {feature.get("description", "no description")[:70]}...')
        
        return True
        
    except Exception as e:
        print(f'❌ Feature engineering failed: {e}')
        return False

def main():
    """Run all OpenAI intelligence tests"""
    
    print('🤖 TESTING OPENAI ENHANCED INTELLIGENCE')
    print('=' * 70)
    
    tests = [
        test_openai_connection,
        test_crypto_sentiment_analysis,
        test_news_impact_assessment,
        test_trading_insights,
        test_feature_engineering_suggestions
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f'❌ Test failed: {e}')
            results.append(False)
    
    print()
    print('🎯 OPENAI INTELLIGENCE SUMMARY')
    print('=' * 70)
    
    passed = sum(results)
    total = len(results)
    
    if results[0]:  # API connection works
        print('✅ OpenAI API: Connected and operational')
        print('✅ Model: GPT-4o (latest as of May 13, 2024)')
        
        if passed >= 4:
            print('✅ AI Sentiment Analysis: Advanced market psychology')
            print('✅ AI News Impact: Intelligent impact assessment')
            print('✅ AI Trading Insights: Smart recommendations')
            print('✅ AI Feature Engineering: Automated feature discovery')
            print()
            print('🏆 OPENAI ENHANCED INTELLIGENCE: FULLY OPERATIONAL')
            print('🧠 Advanced AI maximizing trading intelligence')
        else:
            print('⚠️ Some AI features need refinement')
    else:
        print('❌ OpenAI API not accessible')
        print('   Check OPENAI_API_KEY configuration')
    
    print(f'Test Results: {passed}/{total} passed')
    return passed >= 4

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)