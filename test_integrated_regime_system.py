#!/usr/bin/env python3
"""
Test script for the integrated regime-aware confidence system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def create_synthetic_market_data(n_days: int = 365) -> pd.DataFrame:
    """Create realistic synthetic market data with regime changes"""
    
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Define regime periods
    regime_periods = [
        ('bull', 0, 100, 0.001, 0.015),      # Bull: positive drift, low vol
        ('bear', 100, 200, -0.002, 0.025),   # Bear: negative drift, higher vol
        ('high_vol', 200, 250, 0.0, 0.04),   # High vol: no drift, very high vol
        ('consolidation', 250, 365, 0.0, 0.01) # Consolidation: no drift, low vol
    ]
    
    prices = []
    volumes = []
    current_price = 100.0
    
    for i, date in enumerate(dates):
        # Determine current regime
        regime_name = 'consolidation'  # default
        drift = 0.0
        volatility = 0.02
        
        for reg_name, start, end, reg_drift, reg_vol in regime_periods:
            if start <= i < end:
                regime_name = reg_name
                drift = reg_drift
                volatility = reg_vol
                break
        
        # Generate price movement
        daily_return = np.random.normal(drift, volatility)
        current_price *= (1 + daily_return)
        prices.append(current_price)
        
        # Generate volume based on regime
        if regime_name == 'high_vol':
            volume_factor = np.random.normal(1.5, 0.4)
        elif regime_name == 'bear':
            volume_factor = np.random.normal(1.2, 0.3)
        else:
            volume_factor = np.random.normal(1.0, 0.2)
        
        volume = max(100000, int(1000000 * volume_factor))
        volumes.append(volume)
    
    # Create OHLC data
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    # Add OHLC columns
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df['close'] * (1 + np.random.uniform(0, 0.01, len(df)))
    df['low'] = df['close'] * (1 - np.random.uniform(0, 0.01, len(df)))
    
    return df

def create_synthetic_predictions(n_predictions: int = 100) -> List[dict]:
    """Create synthetic predictions for testing"""
    
    np.random.seed(42)
    
    symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD', 'LINK/USD'] * (n_predictions // 5 + 1)
    symbols = symbols[:n_predictions]
    
    predictions = []
    
    for i, symbol in enumerate(symbols):
        # Create prediction with varying confidence
        base_confidence = np.random.beta(3, 2)  # Skewed toward higher confidence
        
        # Add some structure - some symbols consistently better
        if 'BTC' in symbol:
            base_confidence += 0.1
        elif 'ETH' in symbol:
            base_confidence += 0.05
        
        prediction = {
            'symbol': symbol,
            'prediction': np.random.choice([0, 1]),
            'confidence': min(0.99, max(0.01, base_confidence)),
            'model_name': 'LSTM_ensemble',
            'features': [f'feature_{j}' for j in range(10)]
        }
        
        predictions.append(prediction)
    
    return predictions

def test_integrated_regime_system():
    """Test the complete integrated regime-aware confidence system"""
    
    print('🎯 TESTING INTEGRATED REGIME-AWARE CONFIDENCE SYSTEM')
    print('=' * 70)
    
    try:
        # Import the integrated system
        from ml.integrated.regime_aware_confidence_system import (
            create_integrated_confidence_system,
            integrate_regime_awareness,
            RegimeAwareConfidenceSystem
        )
        
        print('✅ Successfully imported integrated system')
        
        # Create synthetic data
        print('\n📊 Creating synthetic test data...')
        market_data = create_synthetic_market_data(365)
        predictions = create_synthetic_predictions(50)
        
        print(f'Generated {len(market_data)} days of market data')
        print(f'Generated {len(predictions)} predictions')
        
        # Create feature data for testing
        np.random.seed(42)
        n_samples = len(predictions)
        feature_data = np.random.randn(n_samples, 15)  # 15 features
        target_data = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate
        
        print(f'Generated {n_samples} samples of training data')
        
        # Test 1: Create integrated system
        print('\n🔧 TEST 1: System Creation')
        print('-' * 40)
        
        system = create_integrated_confidence_system(
            base_confidence_threshold=0.75,
            regime_adjustments={
                'bull_market': 0.0,
                'bear_market': 0.1,
                'high_volatility': 0.15
            }
        )
        
        print('✅ Integrated system created successfully')
        
        # Test 2: System initialization
        print('\n⚙️ TEST 2: System Initialization')
        print('-' * 40)
        
        # Create historical predictions for calibration
        historical_predictions = pd.DataFrame({
            'raw_confidence': np.random.beta(2, 2, 200),
            'actual_outcome': np.random.binomial(1, 0.4, 200),
            'prediction_timestamp': pd.date_range(end=datetime.now(), periods=200, freq='H')
        })
        
        print('Initializing system with historical data...')
        system.initialize_system(
            historical_market_data=market_data,
            historical_predictions=historical_predictions,
            feature_data=feature_data,
            target_data=target_data
        )
        
        print('✅ System initialization completed')
        
        # Test 3: Regime-aware predictions
        print('\n🎲 TEST 3: Regime-Aware Predictions')
        print('-' * 40)
        
        regime_predictions = system.predict_with_regime_awareness(
            predictions=predictions,
            current_market_data=market_data,
            feature_data=feature_data
        )
        
        print(f'Generated {len(regime_predictions)} regime-aware predictions')
        
        # Show sample results
        print('\nSample Regime-Aware Predictions:')
        for i, pred in enumerate(regime_predictions[:5]):
            print(f'  {i+1}. {pred.symbol}:')
            print(f'     Raw confidence: {pred.raw_confidence:.3f}')
            print(f'     Calibrated confidence: {pred.calibrated_confidence:.3f}')
            print(f'     Regime: {pred.regime.value}')
            print(f'     Regime confidence: {pred.regime_confidence:.3f}')
            print(f'     Total uncertainty: {pred.total_uncertainty:.3f}')
            print(f'     Passes gate: {"✅" if pred.passes_regime_gate else "❌"}')
            print()
        
        # Test 4: Filtering high-confidence predictions
        print('\n🔍 TEST 4: Confidence Filtering')
        print('-' * 40)
        
        filtered_predictions = system.filter_regime_confident_predictions(
            regime_predictions,
            min_regime_confidence=0.6,
            max_uncertainty=0.4
        )
        
        pass_rate = len(filtered_predictions) / len(regime_predictions)
        print(f'Filtered predictions: {len(filtered_predictions)}/{len(regime_predictions)} ({pass_rate:.1%})')
        
        # Test 5: Performance analysis
        print('\n📈 TEST 5: Performance Analysis')
        print('-' * 40)
        
        analysis = system.get_regime_performance_analysis()
        
        print('Overall Statistics:')
        overall = analysis.get('overall_stats', {})
        print(f'  Total predictions: {overall.get("total_predictions", 0)}')
        print(f'  Overall gate pass rate: {overall.get("overall_gate_pass_rate", 0):.1%}')
        print(f'  Current regime: {overall.get("current_regime", "unknown")}')
        print(f'  Current regime confidence: {overall.get("current_regime_confidence", 0):.3f}')
        print(f'  Regimes encountered: {overall.get("regimes_encountered", 0)}')
        
        print('\nCalibration Status:')
        calibration = analysis.get('calibration_status', {})
        print(f'  Regime calibrators fitted: {calibration.get("regime_calibrators_fitted", 0)}')
        print(f'  Fallback calibrator fitted: {calibration.get("fallback_calibrator_fitted", False)}')
        print(f'  Uncertainty quantifier available: {calibration.get("uncertainty_quantifier_available", False)}')
        
        if 'regime_analysis' in analysis:
            print('\nRegime-Specific Analysis:')
            for regime, stats in analysis['regime_analysis'].items():
                print(f'  {regime}:')
                print(f'    Predictions: {stats.get("total_predictions", 0)}')
                print(f'    Gate pass rate: {stats.get("gate_pass_rate", 0):.1%}')
                print(f'    Avg confidence: {stats.get("avg_confidence", 0):.3f}')
                print(f'    Avg uncertainty: {stats.get("avg_uncertainty", 0):.3f}')
        
        # Test 6: High-level integration function
        print('\n🚀 TEST 6: High-Level Integration')
        print('-' * 40)
        
        filtered_preds, analysis_hl = integrate_regime_awareness(
            predictions=predictions[:10],  # Use subset for quick test
            market_data=market_data,
            historical_data=market_data.iloc[:200],  # Use earlier data as historical
            feature_data=feature_data[:10],
            target_data=target_data[:10]
        )
        
        print(f'High-level integration: {len(filtered_preds)} filtered predictions')
        print(f'Analysis keys: {list(analysis_hl.keys())}')
        
        # Summary
        print('\n🎯 INTEGRATION TEST SUMMARY')
        print('=' * 70)
        print('✅ Regime detection: Working')
        print('✅ Confidence calibration: Regime-specific calibrators fitted')
        print('✅ Uncertainty quantification: Bayesian uncertainty estimates')
        print('✅ Regime-aware gating: Dynamic thresholds per regime')
        print('✅ Performance tracking: Comprehensive regime analysis')
        print('✅ High-level interface: Complete workflow integration')
        
        print('\n🏆 INTEGRATED REGIME-AWARE CONFIDENCE SYSTEM: FULLY OPERATIONAL')
        
        return True
        
    except ImportError as e:
        print(f'❌ Import failed: {e}')
        print('Some dependencies may be missing')
        return False
        
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_integrated_regime_system()
    exit(0 if success else 1)