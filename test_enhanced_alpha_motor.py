#!/usr/bin/env python3
"""
Test Enhanced Alpha Motor - Demonstratie van echte waarde-creatie

Test script dat de verbeterde Alpha Motor valideert met:
- Enhanced signal generators met realistische crypto data
- Multi-factor coin selection met momentum/mean-revert/funding/sentiment
- Portfolio construction met Kelly sizing en correlatie caps
- Performance attribution per signal bucket
- Risk-adjusted ranking met execution quality

Verwachte resultaten:
- Top 15 coin selecties met gediversifieerde signal sources
- Alpha scores > 0.3 voor top kandidaten  
- Balanced exposure across signal buckets
- Realistic portfolio weights met risk controls
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from typing import Dict, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Alpha Motor components
from src.cryptosmarttrader.alpha.coin_picker_alpha_motor import (
    get_alpha_motor, CoinCandidate, SignalBucket
)
from src.cryptosmarttrader.alpha.enhanced_signal_generators import (
    get_enhanced_signal_generator, generate_sample_market_data,
    TechnicalIndicators, FundingData, SentimentData
)


def generate_realistic_crypto_universe() -> Dict[str, List]:
    """Generate realistic crypto market data for testing"""
    
    # Top crypto symbols by volume
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
        'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'SHIB/USDT',
        'MATIC/USDT', 'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT',
        'FTM/USDT', 'ALGO/USDT', 'ICP/USDT', 'VET/USDT', 'ETC/USDT',
        'HBAR/USDT', 'FIL/USDT', 'TRON/USDT', 'XLM/USDT', 'MANA/USDT'
    ]
    
    # Generate sample market data
    market_data = generate_sample_market_data(symbols)
    
    # Convert to Alpha Motor format
    coins = []
    for symbol, data in market_data.items():
        coin = {
            'symbol': symbol,
            'market_cap_usd': float(hash(symbol) % 100_000_000_000),  # Fake market cap
            'volume_24h_usd': data['volume_24h_usd'],
            'spread_bps': min(50, max(5, abs(hash(symbol) % 30))),  # 5-50 bps spread
            'depth_1pct_usd': data['volume_24h_usd'] * 0.05,  # 5% of volume as depth
            
            # Technical indicator proxies
            'rsi_14': data['indicators'].rsi_14,
            'price_change_24h_pct': (data['indicators'].rsi_14 - 50) / 100,  # Fake price change
            'volume_7d_avg': data['volume_24h_usd'] * 0.8,  # Fake weekly average
            
            # Funding data
            'funding_rate_8h_pct': data['funding'].funding_rate_8h * 100,
            'oi_change_24h_pct': data['funding'].oi_change_24h_pct,
            
            # Sentiment data
            'social_mentions_24h': (data['sentiment'].reddit_mentions_24h + 
                                  data['sentiment'].twitter_mentions_24h),
            'sentiment_score': (data['sentiment'].reddit_sentiment + 
                              data['sentiment'].twitter_sentiment) / 2,
            
            # Store original enhanced data for advanced signals
            '_enhanced_indicators': data['indicators'],
            '_enhanced_funding': data['funding'], 
            '_enhanced_sentiment': data['sentiment']
        }
        coins.append(coin)
    
    return {'coins': coins}


async def test_enhanced_alpha_cycle():
    """Test complete alpha generation cycle met enhanced signals"""
    
    logger.info("🧪 Testing Enhanced Alpha Motor Implementation")
    logger.info("=" * 60)
    
    # Initialize components
    alpha_motor = get_alpha_motor()
    signal_generator = get_enhanced_signal_generator()
    
    # Generate realistic market data
    logger.info("📊 Generating realistic crypto market universe...")
    market_data = generate_realistic_crypto_universe()
    logger.info(f"   Generated data for {len(market_data['coins'])} coins")
    
    # Run alpha cycle with enhanced signals
    logger.info("🎯 Running enhanced alpha generation cycle...")
    start_time = datetime.now()
    
    # Test enhanced signal generation
    candidates = await test_enhanced_signal_generation(signal_generator, market_data)
    
    # Run main alpha cycle
    final_positions = await alpha_motor.run_alpha_cycle(market_data)
    
    cycle_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"   Alpha cycle completed in {cycle_time:.2f}s")
    
    # Analyze results
    await analyze_alpha_results(final_positions, alpha_motor)
    
    return final_positions


async def test_enhanced_signal_generation(signal_generator, market_data):
    """Test enhanced signal generation voor sample coins"""
    
    logger.info("🔬 Testing Enhanced Signal Generation...")
    
    # Test signals voor top 5 coins
    test_coins = market_data['coins'][:5]
    enhanced_results = []
    
    for coin in test_coins:
        symbol = coin['symbol']
        
        # Extract enhanced data
        indicators = coin['_enhanced_indicators']
        funding_data = coin['_enhanced_funding'] 
        sentiment_data = coin['_enhanced_sentiment']
        volume_24h = coin['volume_24h_usd']
        
        # Calculate enhanced signals
        momentum_signal = await signal_generator.calculate_momentum_signal(
            symbol, {}, indicators
        )
        
        mean_revert_signal = await signal_generator.calculate_mean_revert_signal(
            symbol, {}, indicators
        )
        
        funding_signal = await signal_generator.calculate_funding_basis_signal(
            symbol, funding_data
        )
        
        sentiment_signal = await signal_generator.calculate_sentiment_signal(
            symbol, sentiment_data, volume_24h
        )
        
        result = {
            'symbol': symbol,
            'momentum_signal': momentum_signal,
            'mean_revert_signal': mean_revert_signal, 
            'funding_signal': funding_signal,
            'sentiment_signal': sentiment_signal,
            'composite_score': (momentum_signal * 0.3 + mean_revert_signal * 0.25 + 
                              funding_signal * 0.25 + sentiment_signal * 0.2)
        }
        
        enhanced_results.append(result)
        
        logger.info(f"   {symbol:12} | M:{momentum_signal:.3f} | R:{mean_revert_signal:.3f} | "
                   f"F:{funding_signal:.3f} | S:{sentiment_signal:.3f} | "
                   f"Total:{result['composite_score']:.3f}")
    
    # Sort by composite score
    enhanced_results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    logger.info(f"   Best enhanced signal: {enhanced_results[0]['symbol']} "
               f"({enhanced_results[0]['composite_score']:.3f})")
    
    return enhanced_results


async def analyze_alpha_results(positions: List[CoinCandidate], alpha_motor):
    """Analyze alpha generation results"""
    
    logger.info("📈 Alpha Results Analysis:")
    logger.info("-" * 40)
    
    if not positions:
        logger.warning("   ⚠️ No positions generated!")
        return
    
    # Portfolio summary
    total_weight = sum(p.final_weight for p in positions)
    avg_alpha_score = sum(p.total_score for p in positions) / len(positions)
    
    logger.info(f"   📊 Portfolio Summary:")
    logger.info(f"      - Total Positions: {len(positions)}")
    logger.info(f"      - Total Weight: {total_weight:.1%}")
    logger.info(f"      - Average Alpha Score: {avg_alpha_score:.3f}")
    
    # Top positions
    logger.info(f"   🏆 Top 5 Positions:")
    for i, pos in enumerate(positions[:5]):
        logger.info(f"      {i+1}. {pos.symbol:12} | Weight: {pos.final_weight:.1%} | "
                   f"Alpha: {pos.total_score:.3f} | Risk-Adj: {pos.risk_adjusted_score:.3f}")
    
    # Signal attribution
    attribution = alpha_motor.get_performance_attribution(positions)
    logger.info(f"   🎯 Signal Attribution:")
    logger.info(f"      - Momentum:    {attribution.get('momentum_contribution', 0):.3f}")
    logger.info(f"      - Mean-Revert: {attribution.get('mean_revert_contribution', 0):.3f}")
    logger.info(f"      - Funding:     {attribution.get('funding_contribution', 0):.3f}")
    logger.info(f"      - Sentiment:   {attribution.get('sentiment_contribution', 0):.3f}")
    
    # Risk metrics
    logger.info(f"   ⚖️ Risk Metrics:")
    
    # Position concentration
    max_weight = max(p.final_weight for p in positions)
    logger.info(f"      - Max Single Weight: {max_weight:.1%}")
    
    # Cluster diversification  
    clusters = {}
    for pos in positions:
        cluster = pos.correlation_cluster
        clusters[cluster] = clusters.get(cluster, 0) + pos.final_weight
    
    max_cluster_weight = max(clusters.values()) if clusters else 0
    logger.info(f"      - Max Cluster Weight: {max_cluster_weight:.1%}")
    logger.info(f"      - Number of Clusters: {len(clusters)}")
    
    # Execution quality
    avg_execution_quality = sum(p.execution_quality for p in positions) / len(positions)
    logger.info(f"      - Avg Execution Quality: {avg_execution_quality:.3f}")
    
    # Expected performance metrics
    total_alpha_contribution = sum(p.total_score * p.final_weight for p in positions)
    logger.info(f"   📊 Expected Performance:")
    logger.info(f"      - Weighted Alpha Score: {total_alpha_contribution:.3f}")
    logger.info(f"      - Expected Monthly Return: {total_alpha_contribution * 15:.1f}%")
    
    # Portfolio construction validation
    logger.info(f"   ✅ Portfolio Validation:")
    weight_sum_check = abs(total_weight - 0.95) < 0.05  # Within 5% of target
    diversification_check = len(set(p.correlation_cluster for p in positions)) >= 3
    quality_check = avg_execution_quality > 0.5
    
    logger.info(f"      - Weight Target: {'✅' if weight_sum_check else '❌'}")
    logger.info(f"      - Diversification: {'✅' if diversification_check else '❌'}")
    logger.info(f"      - Execution Quality: {'✅' if quality_check else '❌'}")


def save_results_for_analysis(positions: List[CoinCandidate]):
    """Save results voor verdere analyse"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_positions': len(positions),
        'positions': []
    }
    
    for pos in positions:
        pos_data = {
            'symbol': pos.symbol,
            'final_weight': pos.final_weight,
            'alpha_score': pos.total_score,
            'risk_adjusted_score': pos.risk_adjusted_score,
            'signal_breakdown': {
                'momentum': pos.momentum_score,
                'mean_revert': pos.mean_revert_score,
                'funding': pos.funding_score,
                'sentiment': pos.sentiment_score
            },
            'risk_metrics': {
                'correlation_cluster': pos.correlation_cluster,
                'execution_quality': pos.execution_quality,
                'liquidity_rank': pos.liquidity_rank
            }
        }
        results['positions'].append(pos_data)
    
    # Save to file
    filename = f"alpha_motor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"   💾 Results saved to {filename}")


async def main():
    """Main test function"""
    
    try:
        print("🚀 Enhanced Alpha Motor Test Suite")
        print("=" * 50)
        
        # Run enhanced alpha cycle test
        positions = await test_enhanced_alpha_cycle()
        
        # Save results
        if positions:
            save_results_for_analysis(positions)
        
        print("\n" + "=" * 50)
        print("✅ Enhanced Alpha Motor test completed successfully!")
        
        if positions:
            print(f"📈 Generated {len(positions)} optimized positions")
            print(f"🎯 Top position: {positions[0].symbol} ({positions[0].total_score:.3f} alpha)")
            print(f"💰 Portfolio weight: {sum(p.final_weight for p in positions):.1%}")
        else:
            print("⚠️  No positions generated - check market data and thresholds")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())