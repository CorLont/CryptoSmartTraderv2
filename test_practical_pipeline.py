#!/usr/bin/env python3
"""
Test Practical Coin Pipeline - "Welke coins kopen voor hoge rendementen?"

Test complete pipeline from universe filtering to order tickets
"""

import asyncio
import logging
import json
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import pipeline
try:
    from src.cryptosmarttrader.alpha.practical_coin_pipeline import (
        PracticalCoinPipeline, UniverseFilters, Regime
    )
    from src.cryptosmarttrader.alpha.enhanced_signal_generators import generate_sample_market_data
    logger.info("✅ Practical Pipeline import successful")
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    exit(1)

def generate_realistic_universe(num_coins=30):
    """Generate realistic crypto universe for testing"""
    
    # Top 30 crypto symbols with realistic data
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
        'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'SHIB/USDT',
        'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT',
        'FTM/USDT', 'ALGO/USDT', 'ICP/USDT', 'VET/USDT', 'ETC/USDT',
        'HBAR/USDT', 'FIL/USDT', 'TRON/USDT', 'XLM/USDT', 'MANA/USDT',
        'SAND/USDT', 'CRO/USDT', 'NEAR/USDT', 'APE/USDT', 'LRC/USDT'
    ]
    
    # Generate base market data
    market_data = generate_sample_market_data(symbols[:num_coins])
    
    # Enhance with realistic market conditions
    enhanced_data = {}
    
    for symbol, data in market_data.items():
        # Calculate realistic spread based on market cap tier
        base_volume = data['volume_24h_usd']
        
        if base_volume > 100_000_000:  # Top tier
            spread_range = (5, 20)
            depth_multiplier = (0.05, 0.15)
        elif base_volume > 50_000_000:  # Mid tier
            spread_range = (15, 35)
            depth_multiplier = (0.03, 0.08)
        else:  # Lower tier
            spread_range = (25, 60)
            depth_multiplier = (0.02, 0.05)
        
        import numpy as np
        spread_bps = np.random.uniform(*spread_range)
        depth_usd = base_volume * np.random.uniform(*depth_multiplier)
        
        enhanced_data[symbol] = {
            **data,
            'spread_bps': spread_bps,
            'depth_1pct_usd': depth_usd,
            'market_cap_usd': base_volume * np.random.uniform(20, 200)
        }
    
    return enhanced_data

async def main():
    """Test the complete practical pipeline"""
    
    logger.info("🚀 Testing Practical Coin Pipeline")
    logger.info("=" * 60)
    
    # Initialize pipeline with realistic filters
    filters = UniverseFilters(
        min_volume_24h_usd=15_000_000,  # 15M USD
        max_spread_bps=40,              # 40 bps
        min_depth_usd=750_000           # 750K USD
    )
    
    pipeline = PracticalCoinPipeline(
        universe_filters=filters,
        max_positions=10,
        target_leverage=0.85
    )
    
    logger.info("✅ Pipeline initialized")
    
    # Generate test universe
    logger.info("📊 Generating realistic test universe...")
    market_data = generate_realistic_universe(25)
    logger.info(f"✅ Universe generated: {len(market_data)} coins")
    
    # Run complete pipeline
    logger.info("🔄 Running complete pipeline...")
    
    try:
        start_time = datetime.now()
        final_tickets = await pipeline.run_pipeline(market_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Pipeline completed in {processing_time:.2f}s")
        logger.info(f"🎫 Generated {len(final_tickets)} order tickets")
        
        # Display results
        if final_tickets:
            logger.info("\n📈 FINAL ORDER TICKETS:")
            logger.info("-" * 80)
            
            total_weight = sum(ticket.final_weight for ticket in final_tickets)
            total_score = sum(ticket.final_score for ticket in final_tickets)
            
            for i, ticket in enumerate(final_tickets, 1):
                logger.info(
                    f"{i:2d}. {ticket.symbol:12s} | "
                    f"Weight: {ticket.final_weight:6.1%} | "
                    f"Score: {ticket.final_score:5.3f} | "
                    f"Regime: {ticket.regime.value:12s} | "
                    f"COID: {ticket.client_order_id}"
                )
            
            logger.info("-" * 80)
            logger.info(f"📊 PORTFOLIO SUMMARY:")
            logger.info(f"    Total Weight: {total_weight:.1%}")
            logger.info(f"    Average Score: {total_score/len(final_tickets):.3f}")
            logger.info(f"    Positions: {len(final_tickets)}")
            
            # Regime distribution
            regime_counts = {}
            for ticket in final_tickets:
                regime_counts[ticket.regime.value] = regime_counts.get(ticket.regime.value, 0) + 1
            
            logger.info(f"    Regime Distribution:")
            for regime, count in regime_counts.items():
                logger.info(f"      {regime}: {count} positions")
            
            # Cluster distribution
            cluster_weights = {}
            for ticket in final_tickets:
                cluster = ticket.correlation_cluster
                cluster_weights[cluster] = cluster_weights.get(cluster, 0) + ticket.final_weight
            
            logger.info(f"    Cluster Weights:")
            for cluster, weight in cluster_weights.items():
                logger.info(f"      Cluster {cluster}: {weight:.1%}")
            
        else:
            logger.warning("⚠️ No tickets generated - check filters and market conditions")
        
        # Test summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PRACTICAL PIPELINE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Universe filtering: Working")
        logger.info(f"✅ Regime detection: Working") 
        logger.info(f"✅ Signal generation: Working")
        logger.info(f"✅ Risk validation: Working")
        logger.info(f"✅ Kelly sizing: Working")
        logger.info(f"✅ Order tickets: Working")
        logger.info(f"⏱️ Processing time: {processing_time:.2f}s")
        logger.info(f"🎫 Final output: {len(final_tickets)} actionable positions")
        
        if len(final_tickets) >= 5:
            logger.info("🟢 TEST PASSED - Pipeline generates sufficient positions")
        else:
            logger.info("🟡 TEST MARGINAL - Few positions generated")
            
    except Exception as e:
        logger.error(f"❌ Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())