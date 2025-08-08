#!/usr/bin/env python3
"""
Strict 80% Confidence Gate for Orchestration
Ultra-tight filtering - only highest confidence predictions pass through
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

def strict_toplist(pred_df: pd.DataFrame, conf_col: str = "conf_720h", 
                  pred_col: str = "pred_720h", thr: float = 0.80) -> pd.DataFrame:
    """
    Ultra-strict 80% confidence gate filtering
    Only predictions with >= 80% confidence pass through
    """
    
    # Apply strict filtering
    df = pred_df[(pred_df[conf_col] >= thr) & pred_df[pred_col].notna()]
    
    # Sort by prediction strength (descending)
    return df.sort_values(pred_col, ascending=False).reset_index(drop=True)

def apply_strict_gate_orchestration(all_predictions: Dict[str, pd.DataFrame], 
                                   confidence_threshold: float = 0.80) -> Dict[str, Any]:
    """
    Apply strict confidence gate across all prediction horizons in orchestration
    """
    
    gate_results = {
        'timestamp': datetime.now().isoformat(),
        'confidence_threshold': confidence_threshold,
        'horizon_results': {},
        'total_passed': 0,
        'total_candidates': 0,
        'gate_status': 'CLOSED'
    }
    
    horizons = ['1h', '24h', '168h', '720h']  # 1H, 24H, 7D, 30D
    
    for horizon in horizons:
        if horizon in all_predictions:
            pred_df = all_predictions[horizon]
            
            # Apply strict gate for this horizon
            conf_col = f"conf_{horizon}"
            pred_col = f"pred_{horizon}"
            
            if conf_col in pred_df.columns and pred_col in pred_df.columns:
                strict_filtered = strict_toplist(pred_df, conf_col, pred_col, confidence_threshold)
                
                horizon_result = {
                    'total_candidates': len(pred_df),
                    'passed_count': len(strict_filtered),
                    'pass_rate': len(strict_filtered) / len(pred_df) if len(pred_df) > 0 else 0.0,
                    'top_opportunities': strict_filtered.head(5).to_dict('records') if len(strict_filtered) > 0 else []
                }
                
                gate_results['horizon_results'][horizon] = horizon_result
                gate_results['total_passed'] += len(strict_filtered)
                gate_results['total_candidates'] += len(pred_df)
    
    # Overall gate decision
    if gate_results['total_passed'] > 0:
        gate_results['gate_status'] = 'OPEN'
        gate_results['overall_pass_rate'] = gate_results['total_passed'] / max(gate_results['total_candidates'], 1)
    
    return gate_results

def get_strict_opportunities(gate_results: Dict[str, Any], min_opportunities: int = 1) -> List[Dict[str, Any]]:
    """
    Extract strict opportunities that passed the 80% gate
    """
    
    opportunities = []
    
    if gate_results['gate_status'] == 'OPEN':
        for horizon, results in gate_results['horizon_results'].items():
            for opp in results['top_opportunities']:
                opportunity = {
                    'symbol': opp.get('symbol', 'UNKNOWN'),
                    'horizon': horizon,
                    'confidence': opp.get(f'conf_{horizon}', 0.0),
                    'prediction': opp.get(f'pred_{horizon}', 0.0),
                    'passed_strict_gate': True,
                    'gate_timestamp': gate_results['timestamp']
                }
                opportunities.append(opportunity)
    
    # Sort by confidence descending
    opportunities.sort(key=lambda x: x['confidence'], reverse=True)
    
    return opportunities[:min_opportunities] if len(opportunities) >= min_opportunities else []

if __name__ == "__main__":
    # Test strict gate with sample data
    import numpy as np
    
    print("🔒 TESTING STRICT 80% CONFIDENCE GATE")
    print("=" * 50)
    
    # Create sample prediction data
    np.random.seed(42)
    n_coins = 100
    
    sample_data = {
        'symbol': [f'COIN{i}' for i in range(n_coins)],
        'pred_720h': np.random.normal(0.1, 0.3, n_coins),  # Return predictions
        'conf_720h': np.random.beta(2, 3, n_coins),  # Confidence scores
        'pred_168h': np.random.normal(0.05, 0.2, n_coins),
        'conf_168h': np.random.beta(2, 3, n_coins)
    }
    
    pred_df = pd.DataFrame(sample_data)
    
    print(f"📊 Total candidates: {len(pred_df)}")
    print(f"   Mean confidence 720h: {pred_df['conf_720h'].mean():.3f}")
    print(f"   Mean confidence 168h: {pred_df['conf_168h'].mean():.3f}")
    
    # Apply strict 80% gate
    strict_720h = strict_toplist(pred_df, 'conf_720h', 'pred_720h', 0.80)
    strict_168h = strict_toplist(pred_df, 'conf_168h', 'pred_168h', 0.80)
    
    print(f"\n🔒 Strict 80% Gate Results:")
    print(f"   720h horizon: {len(strict_720h)}/{len(pred_df)} passed ({len(strict_720h)/len(pred_df)*100:.1f}%)")
    print(f"   168h horizon: {len(strict_168h)}/{len(pred_df)} passed ({len(strict_168h)/len(pred_df)*100:.1f}%)")
    
    if len(strict_720h) > 0:
        print(f"\n📈 Top 720h opportunities:")
        for i, row in strict_720h.head(3).iterrows():
            print(f"   {row['symbol']}: {row['pred_720h']:.3f} return, {row['conf_720h']:.3f} confidence")
    
    # Test orchestration gate
    all_preds = {
        '720h': pred_df,
        '168h': pred_df
    }
    
    gate_results = apply_strict_gate_orchestration(all_preds, 0.80)
    
    print(f"\n🎯 Orchestration Gate Status: {gate_results['gate_status']}")
    print(f"   Total passed: {gate_results['total_passed']}/{gate_results['total_candidates']}")
    
    opportunities = get_strict_opportunities(gate_results)
    print(f"   High-confidence opportunities: {len(opportunities)}")
    
    print("\n✅ Strict gate test completed")