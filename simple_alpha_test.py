#!/usr/bin/env python3
"""
Simple Alpha Motor Test - Direct validation van core functionaliteit
"""

import asyncio
import logging
import numpy as np

# Setup logging with debug for alpha motor
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('src.cryptosmarttrader.alpha.coin_picker_alpha_motor').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Direct import test
try:
    from src.cryptosmarttrader.alpha.coin_picker_alpha_motor import get_alpha_motor
    logger.info("✅ Alpha Motor import successful")
except Exception as e:
    logger.error(f"❌ Alpha Motor import failed: {e}")
    exit(1)

def create_test_market_data():
    """Create minimal test market data"""
    
    # Sample coins with realistic data
    coins = []
    
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
    
    for i, symbol in enumerate(test_symbols):
        coin = {
            'symbol': symbol,
            'market_cap_usd': 10_000_000_000 - i * 1_000_000_000,  # Decreasing market cap
            'volume_24h_usd': 100_000_000 - i * 10_000_000,       # Decreasing volume
            'spread_bps': 10 + i * 2,                             # 10-18 bps spread
            'depth_1pct_usd': 500_000 + i * 100_000,             # 500K-900K depth
            
            # Technical indicators (random but realistic)
            'rsi_14': 30 + i * 10,                               # 30, 40, 50, 60, 70
            'price_change_24h_pct': (-2 + i) / 100,             # -2%, -1%, 0%, 1%, 2%
            'volume_7d_avg': (100_000_000 - i * 10_000_000) * 0.9,
            
            # Funding data
            'funding_rate_8h_pct': (-0.1 + i * 0.05) / 100,     # -0.1% to 0.1%
            'oi_change_24h_pct': -10 + i * 5,                   # -10% to 10%
            
            # Sentiment data
            'social_mentions_24h': 1000 - i * 100,              # Decreasing mentions
            'sentiment_score': 0.3 + i * 0.1,                   # 0.3 to 0.7
        }
        coins.append(coin)
    
    return {'coins': coins}

async def test_alpha_motor():
    """Test basic alpha motor functionality"""
    
    logger.info("🧪 Testing Alpha Motor Core Functionality")
    logger.info("=" * 50)
    
    # Initialize alpha motor
    alpha_motor = get_alpha_motor()
    logger.info("✅ Alpha Motor initialized")
    
    # Create test data
    market_data = create_test_market_data()
    logger.info(f"✅ Test market data created ({len(market_data['coins'])} coins)")
    
    # Run alpha cycle
    logger.info("🔄 Running alpha generation cycle...")
    try:
        positions = await alpha_motor.run_alpha_cycle(market_data)
        logger.info(f"✅ Alpha cycle completed - {len(positions)} positions generated")
        
        # Display results
        if positions:
            logger.info("📊 Generated Positions:")
            for i, pos in enumerate(positions):
                logger.info(f"   {i+1}. {pos.symbol:12} | Weight: {pos.final_weight:.2%} | Alpha: {pos.total_score:.3f}")
            
            # Summary stats
            total_weight = sum(p.final_weight for p in positions)
            avg_alpha = sum(p.total_score for p in positions) / len(positions)
            logger.info(f"📈 Portfolio Summary:")
            logger.info(f"   Total Weight: {total_weight:.1%}")
            logger.info(f"   Average Alpha: {avg_alpha:.3f}")
            logger.info(f"   Positions: {len(positions)}")
            
            return True
        else:
            logger.warning("⚠️ No positions generated")
            return False
            
    except Exception as e:
        logger.error(f"❌ Alpha cycle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    
    print("🚀 Simple Alpha Motor Test")
    print("=" * 30)
    
    success = await test_alpha_motor()
    
    if success:
        print("\n✅ Alpha Motor test PASSED")
        print("🎯 Core coin-picking functionality validated")
    else:
        print("\n❌ Alpha Motor test FAILED")
        print("⚠️ Check implementation and data")

if __name__ == "__main__":
    asyncio.run(main())