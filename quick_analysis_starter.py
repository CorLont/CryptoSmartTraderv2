#!/usr/bin/env python3
"""
Quick Analysis Starter - Snelle cryptocurrency analyse
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def load_and_analyze():
    """Laad data en voer snelle analyse uit"""
    
    print("🚀 CryptoSmartTrader V2 - Quick Analysis")
    print("=" * 60)
    
    # Load predictions
    predictions_file = Path("exports/production/predictions.csv")
    if not predictions_file.exists():
        print("❌ Geen predictions gevonden. Run eerst: python generate_final_predictions.py")
        return
    
    df = pd.read_csv(predictions_file)
    print(f"✅ Loaded {len(df)} cryptocurrency predictions")
    
    # Basic statistics
    print(f"\n📊 SYSTEEM OVERZICHT")
    print(f"Total cryptocurrencies: {len(df)}")
    print(f"80% confidence gate passed: {len(df[df['gate_passed'] == True])}")
    print(f"Gemiddelde confidence: {df['max_confidence'].mean():.3f}")
    print(f"Whale activity gedetecteerd: {len(df[df['whale_activity_detected'] == True])}")
    
    # Top opportunities
    print(f"\n🎯 TOP TRADING OPPORTUNITIES")
    high_conf = df[df['gate_passed'] == True].copy()
    
    if len(high_conf) > 0:
        # Strong buys
        strong_buys = high_conf[
            (high_conf['expected_return_pct'] > 5) & 
            (high_conf['max_confidence'] > 0.85)
        ].sort_values('expected_return_pct', ascending=False)
        
        print(f"\n🟢 STRONG BUY OPPORTUNITIES ({len(strong_buys)}):")
        for idx, row in strong_buys.head(10).iterrows():
            whale_indicator = "🐋" if row['whale_activity_detected'] else "  "
            print(f"  {whale_indicator} {row['coin']:<8} | Return: {row['expected_return_pct']:>6.2f}% | Conf: {row['max_confidence']:.3f} | Sentiment: {row['sentiment_label']}")
        
        # Moderate opportunities
        moderate_buys = high_conf[
            (high_conf['expected_return_pct'] > 0) & 
            (high_conf['expected_return_pct'] <= 5) &
            (high_conf['max_confidence'] > 0.8)
        ].sort_values('expected_return_pct', ascending=False)
        
        print(f"\n🟡 MODERATE BUY OPPORTUNITIES ({len(moderate_buys)}):")
        for idx, row in moderate_buys.head(10).iterrows():
            whale_indicator = "🐋" if row['whale_activity_detected'] else "  "
            print(f"  {whale_indicator} {row['coin']:<8} | Return: {row['expected_return_pct']:>6.2f}% | Conf: {row['max_confidence']:.3f} | Sentiment: {row['sentiment_label']}")
    
    # Market analysis
    print(f"\n📈 MARKT ANALYSE")
    print(f"Positive expected returns: {len(df[df['expected_return_pct'] > 0])}")
    print(f"Negative expected returns: {len(df[df['expected_return_pct'] < 0])}")
    print(f"Bullish sentiment: {len(df[df['sentiment_label'] == 'bullish'])}")
    print(f"Bearish sentiment: {len(df[df['sentiment_label'] == 'bearish'])}")
    print(f"Neutral sentiment: {len(df[df['sentiment_label'] == 'neutral'])}")
    
    # Whale analysis
    print(f"\n🐋 WHALE ACTIVITY ANALYSE")
    whale_coins = df[df['whale_activity_detected'] == True]
    if len(whale_coins) > 0:
        print(f"Coins met whale activity: {len(whale_coins)}")
        whale_returns = whale_coins['expected_return_pct'].mean()
        print(f"Gemiddelde return bij whale activity: {whale_returns:.2f}%")
        
        print(f"\nTop whale coins:")
        whale_sorted = whale_coins.sort_values('whale_score', ascending=False)
        for idx, row in whale_sorted.head(5).iterrows():
            print(f"  {row['coin']:<8} | Whale Score: {row['whale_score']:>6.2f} | Return: {row['expected_return_pct']:>6.2f}%")
    
    # Summary recommendations
    print(f"\n💡 TRADING AANBEVELINGEN")
    
    if len(strong_buys) > 0:
        top_pick = strong_buys.iloc[0]
        print(f"🥇 Top Pick: {top_pick['coin']} - {top_pick['expected_return_pct']:.2f}% return (confidence: {top_pick['max_confidence']:.3f})")
    
    if len(whale_coins) > 0:
        whale_pick = whale_coins.sort_values('expected_return_pct', ascending=False).iloc[0]
        print(f"🐋 Whale Pick: {whale_pick['coin']} - {whale_pick['expected_return_pct']:.2f}% return (whale score: {whale_pick['whale_score']:.2f})")
    
    print(f"\n📁 EXPORT OPTIES")
    print(f"Voor uitgebreide analyse: streamlit run app_fixed_all_issues.py --server.port 5000")
    print(f"Voor detailed dashboard: Ga naar http://0.0.0.0:5000 in je browser")
    
    # Save quick summary
    summary = {
        "analysis_time": datetime.now().isoformat(),
        "total_coins": len(df),
        "high_confidence_coins": len(high_conf),
        "strong_buy_opportunities": len(strong_buys) if 'strong_buys' in locals() else 0,
        "whale_activity_coins": len(whale_coins),
        "top_recommendation": {
            "coin": strong_buys.iloc[0]['coin'] if len(strong_buys) > 0 else None,
            "expected_return": strong_buys.iloc[0]['expected_return_pct'] if len(strong_buys) > 0 else None,
            "confidence": strong_buys.iloc[0]['max_confidence'] if len(strong_buys) > 0 else None
        }
    }
    
    with open("exports/quick_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Quick analysis complete! Summary saved to exports/quick_analysis_summary.json")

if __name__ == "__main__":
    load_and_analyze()