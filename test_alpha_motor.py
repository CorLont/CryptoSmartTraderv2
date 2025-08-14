#!/usr/bin/env python3
"""
Test Alpha Motor - Simpele test van coin picking functionaliteit
"""

import asyncio
import json
from src.cryptosmarttrader.alpha.coin_picker_alpha_motor import get_alpha_motor
from src.cryptosmarttrader.alpha.market_data_simulator import MarketDataSimulator


async def test_alpha_motor():
    """Test de alpha motor met gesimuleerde data"""
    
    print("🎯 Testing Alpha Motor...")
    
    # 1. Generate market data
    print("\n📊 Generating market data...")
    simulator = MarketDataSimulator()
    market_data = simulator.generate_market_snapshot()
    market_data = simulator.add_signal_scenarios(market_data)
    
    print(f"  - Generated {len(market_data['coins'])} coins")
    print(f"  - Total market cap: ${market_data['market_summary']['total_market_cap']:,.0f}")
    
    # 2. Run alpha motor
    print("\n🧠 Running alpha motor...")
    alpha_motor = get_alpha_motor()
    
    try:
        positions = await alpha_motor.run_alpha_cycle(market_data)
        print(f"  - Selected {len(positions)} positions")
        
        if positions:
            # 3. Show results
            print("\n📈 Top Positions:")
            for i, pos in enumerate(positions[:10]):
                print(f"  {i+1:2d}. {pos.symbol:10s} - "
                      f"Weight: {pos.final_weight:6.1%} | "
                      f"Score: {pos.total_score:.3f} | "
                      f"Cap: ${pos.market_cap_usd:,.0f}")
            
            # 4. Performance attribution
            print("\n⚖️ Performance Attribution:")
            attribution = alpha_motor.get_performance_attribution(positions)
            for factor, contribution in attribution.items():
                print(f"  {factor:25s}: {contribution:.3f}")
                
            # 5. Portfolio metrics
            total_weight = sum(p.final_weight for p in positions)
            avg_score = sum(p.total_score * p.final_weight for p in positions) / total_weight if total_weight > 0 else 0
            avg_mcap = sum(p.market_cap_usd * p.final_weight for p in positions) / total_weight if total_weight > 0 else 0
            
            print(f"\n📊 Portfolio Summary:")
            print(f"  Total Allocation: {total_weight:.1%}")
            print(f"  Average Score: {avg_score:.3f}")
            print(f"  Weighted Market Cap: ${avg_mcap:,.0f}")
            print(f"  Position Count: {len(positions)}")
            
        else:
            print("❌ No positions generated")
            
    except Exception as e:
        print(f"❌ Alpha motor error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_alpha_motor())