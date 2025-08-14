#!/usr/bin/env python3
"""
Eenvoudige demonstratie van evaluation system functies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

def demo_evaluation_metrics():
    """Demo van evaluation metrics op synthetic data"""
    
    print("🧪 Evaluation System Demonstration")
    print("=" * 50)
    
    # Create synthetic recommendation data
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic trading data
    data = []
    for i in range(n_samples):
        # Signal scores
        momentum_score = np.random.beta(2, 2)
        sentiment_score = np.random.beta(1.5, 2)
        whale_score = np.random.beta(2, 3)
        combined_score = (momentum_score + sentiment_score + whale_score) / 3
        confidence = np.random.beta(3, 2)
        
        # Add signal bias to returns
        signal_bias = (combined_score - 0.5) * 0.02  # 2% max bias
        base_return = np.random.normal(signal_bias, 0.03) * 10000  # bps
        
        # Generate entry and exit
        entry_price = 45000 + np.random.normal(0, 500)
        side = np.random.choice(["BUY", "SELL"])
        
        # Adjust returns for side
        if side == "SELL":
            base_return = -base_return
        
        exit_price = entry_price * (1 + base_return / 10000)
        
        data.append({
            'recommendation_id': f"SIM_{i:03d}",
            'symbol': np.random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT"]),
            'side': side,
            'combined_score': combined_score,
            'confidence': confidence,
            'momentum_score': momentum_score,
            'sentiment_score': sentiment_score,
            'whale_score': whale_score,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'realized_pnl_bps': base_return,
            'label_profitable': 1 if base_return > 0 else 0,
            'holding_hours': np.random.uniform(1, 48),
            'slippage_bps': np.random.uniform(5, 25),
            'ts_signal': datetime.now(timezone.utc) - timedelta(days=np.random.uniform(0, 30))
        })
    
    df = pd.DataFrame(data)
    
    print(f"📊 Generated {len(df)} synthetic recommendations")
    print(f"📈 Hit Rate: {df['label_profitable'].mean():.1%}")
    print(f"💰 Avg PnL: {df['realized_pnl_bps'].mean():.1f} bps")
    
    # Demo 1: Basic Metrics
    print("\n" + "=" * 50)
    print("📈 1. Basic Performance Metrics")
    
    basic_metrics = calculate_basic_metrics(df)
    for key, value in basic_metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Demo 2: Precision@K
    print("\n" + "=" * 50)
    print("🎯 2. Precision@K Analysis")
    
    precision_metrics = calculate_precision_at_k(df)
    for key, value in precision_metrics.items():
        print(f"   {key}: {value:.3f}")
    
    # Demo 3: Risk Metrics
    print("\n" + "=" * 50)
    print("🛡️ 3. Risk-Adjusted Metrics")
    
    risk_metrics = calculate_risk_metrics(df)
    for key, value in risk_metrics.items():
        print(f"   {key}: {value:.3f}")
    
    # Demo 4: Signal Attribution
    print("\n" + "=" * 50)
    print("🔍 4. Signal Attribution Analysis")
    
    attribution = calculate_signal_attribution(df)
    for key, value in attribution.items():
        print(f"   {key}: {value:.3f}")
    
    # Demo 5: Calibration
    print("\n" + "=" * 50)
    print("🎛️ 5. Model Calibration Analysis")
    
    calibration = calculate_calibration_metrics(df)
    for key, value in calibration.items():
        print(f"   {key}: {value:.4f}")
    
    # Demo 6: Forward-Looking Labels
    print("\n" + "=" * 50)
    print("🏷️ 6. Forward-Looking Label Generation")
    
    labels = generate_forward_labels(df, horizon_hours=24, target_return_bps=50)
    print(f"   Labels Generated: {len(labels)}")
    print(f"   Positive Rate: {sum(l['label_binary'] for l in labels) / len(labels):.1%}")
    print(f"   Avg Forward Return: {np.mean([l['forward_return_bps'] for l in labels]):.1f} bps")
    
    # Save results
    output_dir = Path("data/evaluation_demo")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    demo_results = {
        'timestamp': datetime.now().isoformat(),
        'samples': len(df),
        'basic_metrics': basic_metrics,
        'precision_metrics': precision_metrics,
        'risk_metrics': risk_metrics,
        'signal_attribution': attribution,
        'calibration': calibration,
        'forward_labels_sample': labels[:5]  # First 5 labels
    }
    
    results_file = output_dir / "evaluation_demo_results.json"
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2, default=str)
    
    print(f"\n📁 Results saved to: {results_file}")
    print("\n✅ Evaluation system demonstration completed!")
    
    return demo_results

def calculate_basic_metrics(df):
    """Calculate basic performance metrics"""
    
    return {
        'total_trades': len(df),
        'hit_rate': df['label_profitable'].mean(),
        'avg_pnl_bps': df['realized_pnl_bps'].mean(),
        'total_pnl_bps': df['realized_pnl_bps'].sum(),
        'best_trade_bps': df['realized_pnl_bps'].max(),
        'worst_trade_bps': df['realized_pnl_bps'].min(),
        'volatility_bps': df['realized_pnl_bps'].std(),
        'avg_holding_hours': df['holding_hours'].mean()
    }

def calculate_precision_at_k(df):
    """Calculate precision@K metrics"""
    
    # Sort by combined score
    df_sorted = df.sort_values('combined_score', ascending=False)
    
    metrics = {}
    for k in [5, 10, 20]:
        if len(df_sorted) >= k:
            top_k = df_sorted.head(k)
            metrics[f'precision_at_{k}'] = top_k['label_profitable'].mean()
            metrics[f'avg_return_top_{k}_bps'] = top_k['realized_pnl_bps'].mean()
    
    return metrics

def calculate_risk_metrics(df):
    """Calculate risk-adjusted metrics"""
    
    returns = df['realized_pnl_bps']
    
    return {
        'sharpe_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0,
        'sortino_ratio': calculate_sortino_ratio(returns),
        'max_drawdown_bps': calculate_max_drawdown(returns),
        'information_ratio': returns.mean() / returns.std() if returns.std() > 0 else 0
    }

def calculate_signal_attribution(df):
    """Calculate signal attribution"""
    
    attribution = {}
    
    signals = ['momentum_score', 'sentiment_score', 'whale_score', 'combined_score']
    
    for signal in signals:
        if signal in df.columns:
            correlation = df[signal].corr(df['realized_pnl_bps'])
            attribution[f'{signal}_correlation'] = correlation
    
    return attribution

def calculate_calibration_metrics(df):
    """Calculate calibration metrics"""
    
    from sklearn.metrics import brier_score_loss
    
    y_true = df['label_profitable'].values
    y_prob = df['confidence'].values
    
    return {
        'brier_score': brier_score_loss(y_true, y_prob),
        'avg_confidence': y_prob.mean(),
        'confidence_std': y_prob.std()
    }

def generate_forward_labels(df, horizon_hours=24, target_return_bps=50):
    """Generate forward-looking labels"""
    
    labels = []
    
    for _, row in df.head(10).iterrows():  # Demo with first 10
        # Simulate forward return
        signal_bias = (row['combined_score'] - 0.5) * 0.01
        random_component = np.random.normal(0, 0.02)
        forward_return_bps = (signal_bias + random_component) * 10000
        
        # Adjust for side
        if row['side'] == "SELL":
            forward_return_bps = -forward_return_bps
        
        max_favorable = abs(forward_return_bps) * 1.2
        
        labels.append({
            'recommendation_id': row['recommendation_id'],
            'horizon_hours': horizon_hours,
            'forward_return_bps': forward_return_bps,
            'max_favorable_return_bps': max_favorable,
            'label_binary': 1 if max_favorable >= target_return_bps else 0,
            'target_return_bps': target_return_bps
        })
    
    return labels

def calculate_sortino_ratio(returns):
    """Calculate Sortino ratio"""
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf') if returns.mean() > 0 else 0
    return returns.mean() / downside_returns.std()

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = returns.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    return drawdown.min()

if __name__ == "__main__":
    demo_evaluation_metrics()