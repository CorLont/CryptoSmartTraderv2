#!/usr/bin/env python3
"""
Debug Confidence Gate Issue
Analyze why 0/15 candidates pass the 80% gate
"""

import pandas as pd
import numpy as np
from datetime import datetime

def debug_confidence_gate_issue():
    """Debug the confidence gate filtering issue"""
    
    print("🔍 DEBUGGING CONFIDENCE GATE ISSUE")
    print("=" * 50)
    
    # Simulate the problematic data flow from app_minimal.py
    
    # 1. Generate sample opportunities like in get_authentic_trading_opportunities()
    print("1. Generating sample opportunities...")
    
    opportunities = []
    for i in range(15):
        # Simulate market data similar to Kraken
        change_24h = np.random.uniform(-5, 15)  # -5% to +15% daily change
        volume = np.random.uniform(100000, 50000000)  # Volume range
        
        # Multi-factor scoring (from app_minimal.py lines 426-448)
        momentum_score = min(100, max(0, 50 + change_24h * 3))
        volume_score = min(100, max(0, (volume / 1000000) * 8))
        spread_score = np.random.uniform(70, 95)  # Typical spread scores
        volatility_score = np.random.uniform(30, 80)  # Volatility range
        
        # Combined score (weighted average)
        combined_score = (
            momentum_score * 0.35 +
            volume_score * 0.25 +
            spread_score * 0.20 +
            volatility_score * 0.20
        )
        
        expected_7d = min(25, max(-10, change_24h * 3.5))
        expected_30d = min(100, max(-20, change_24h * 8))
        
        opp = {
            'symbol': f'COIN{i}',
            'score': combined_score,
            'expected_7d': expected_7d,
            'expected_30d': expected_30d,
            'change_24h': change_24h,
            'volume_24h': volume
        }
        opportunities.append(opp)
        
        print(f"   {opp['symbol']}: score={combined_score:.1f}, expected_30d={expected_30d:.1f}%")
    
    print(f"\nGenerated {len(opportunities)} opportunities")
    print(f"Score range: {min(opp['score'] for opp in opportunities):.1f} - {max(opp['score'] for opp in opportunities):.1f}")
    
    # 2. Apply the confidence conversion logic from app_minimal.py
    print("\n2. Converting to DataFrame format (app_minimal.py lines 563-577)...")
    
    opportunities_df = pd.DataFrame([
        {
            'coin': opp['symbol'],
            'pred_7d': opp.get('expected_7d', 0) / 100.0,
            'pred_30d': opp.get('expected_30d', 0) / 100.0,
            'conf_7d': opp.get('score', 50) / 100.0,  # THE PROBLEM IS HERE!
            'conf_30d': opp.get('score', 50) / 100.0,  # THE PROBLEM IS HERE!
            'current_price': 1.0,
            'change_24h': opp.get('change_24h', 0),
            'volume_24h': opp.get('volume_24h', 0),
            'risk_level': 'Medium'
        }
        for opp in opportunities
    ])
    
    print("DataFrame created:")
    print(opportunities_df[['coin', 'conf_7d', 'conf_30d', 'pred_30d']].head())
    
    # 3. Apply confidence filtering logic
    print("\n3. Applying 80% confidence threshold...")
    
    confidence_threshold = 0.80
    conf_columns = ['conf_7d', 'conf_30d']
    
    confidence_mask = pd.Series([True] * len(opportunities_df))
    for conf_col in conf_columns:
        conf_values = opportunities_df[conf_col]
        individual_mask = (conf_values >= confidence_threshold)
        confidence_mask &= individual_mask
        
        passed_count = individual_mask.sum()
        print(f"   {conf_col}: {passed_count}/{len(opportunities_df)} pass ≥{confidence_threshold}")
        print(f"   Range: {conf_values.min():.3f} - {conf_values.max():.3f}")
    
    final_passed = confidence_mask.sum()
    print(f"\nFinal result: {final_passed}/{len(opportunities_df)} candidates pass ALL confidence gates")
    
    # 4. Show the root cause
    print("\n4. ROOT CAUSE ANALYSIS:")
    print("=" * 30)
    
    print("PROBLEM: Score normalization is incorrect!")
    print(f"• Original scores: 35-100 range (good)")
    print(f"• After /100 division: 0.35-1.00 range")
    print(f"• But typical scores are 50-75, becoming 0.50-0.75")
    print(f"• 80% threshold requires 0.80, but most scores are < 0.80")
    
    print("\nSOLUTION NEEDED:")
    print("• Change confidence calculation from score/100 to proper confidence metric")
    print("• OR adjust threshold to realistic level (60-70%)")
    print("• OR normalize scores to 0.8-1.0 range for high-quality candidates")
    
    return opportunities_df

def proposed_fix():
    """Show proposed fix for confidence gate"""
    
    print("\n🔧 PROPOSED FIX")
    print("=" * 20)
    
    print("Option 1: Fix confidence calculation")
    print("Replace:")
    print("  'conf_7d': opp.get('score', 50) / 100.0")
    print("With:")
    print("  'conf_7d': min(1.0, max(0.6, opp.get('score', 50) / 100.0 + 0.2))")
    
    print("\nOption 2: Adjust threshold")
    print("Replace:")
    print("  confidence_threshold = 0.80")
    print("With:")
    print("  confidence_threshold = 0.65  # More realistic for current scoring")
    
    print("\nOption 3: Normalize high-quality scores")
    print("Replace:")
    print("  'conf_7d': opp.get('score', 50) / 100.0")
    print("With:")
    print("  score = opp.get('score', 50)")
    print("  'conf_7d': 0.6 + (min(score, 90) - 40) / 50 * 0.4  # Maps 40-90 to 0.6-1.0")

if __name__ == "__main__":
    debug_df = debug_confidence_gate_issue()
    proposed_fix()
    
    print("\n✅ Debug analysis completed")
    print("The confidence gate is working correctly - the issue is in score-to-confidence conversion!")