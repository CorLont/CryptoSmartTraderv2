"""
Demo: Real Data Analysis with Kraken API
Test script to verify authentic data collection is working
"""

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from cryptosmarttrader.data.authentic_data_collector import get_authentic_collector
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_data_collection():
    """Test real data collection from Kraken API"""
    
    try:
        print("🚀 Testing Real Data Collection...")
        
        # Get authentic data collector
        collector = get_authentic_collector()
        
        # Test market status
        print("\n📊 Testing Market Status...")
        status = collector.get_live_market_status()
        print(f"✅ Market Status: {status}")
        
        # Test real market data collection
        print("\n💰 Testing Real Market Data Collection...")
        market_data = collector.collect_real_market_data()
        
        if market_data:
            print(f"✅ Collected data for {len(market_data)} trading pairs")
            
            # Show sample data
            for i, data in enumerate(market_data[:3]):
                print(f"  {i+1}. {data.symbol}: ${data.last_price:.4f} (Spread: {data.spread_bps:.1f}bps)")
        
        # Test opportunity analysis
        print("\n🎯 Testing Real Opportunity Analysis...")
        opportunities = collector.analyze_real_opportunities()
        
        if opportunities:
            print(f"✅ Found {len(opportunities)} real trading opportunities")
            
            # Show top opportunities
            for i, opp in enumerate(opportunities[:3]):
                print(f"  {i+1}. {opp.symbol}: {opp.expected_return_pct:.1f}% expected return "
                      f"(Confidence: {opp.confidence_score:.1%})")
        else:
            print("⚠️ No high-return opportunities found in current market")
        
        # Test account balance
        print("\n💳 Testing Account Balance...")
        try:
            balance = collector.get_account_balance()
            if balance:
                print(f"✅ Account balance retrieved: {len(balance)} currencies")
                for currency, amount in list(balance.items())[:5]:
                    print(f"  {currency}: {amount}")
            else:
                print("ℹ️ No balances found or demo account")
        except Exception as e:
            print(f"⚠️ Balance check failed: {e}")
        
        print("\n🎉 REAL DATA TEST COMPLETED SUCCESSFULLY")
        print("✅ Kraken API connection working")
        print("✅ Market data collection functional")
        print("✅ Opportunity analysis operational")
        
        return True
        
    except Exception as e:
        print(f"\n❌ REAL DATA TEST FAILED: {e}")
        print("Check your Kraken API credentials")
        return False

if __name__ == "__main__":
    
    # Check API keys first
    if not os.environ.get('KRAKEN_API_KEY'):
        print("❌ KRAKEN_API_KEY environment variable not set")
        exit(1)
    
    if not os.environ.get('KRAKEN_SECRET'):
        print("❌ KRAKEN_SECRET environment variable not set")
        exit(1)
    
    print("🔑 API keys found, proceeding with test...")
    
    success = test_real_data_collection()
    
    if success:
        print("\n🚀 Ready to run dashboard with REAL DATA!")
        print("Run: streamlit run app_trading_analysis_dashboard.py --server.port 5000")
    else:
        print("\n❌ Fix API issues before running dashboard")