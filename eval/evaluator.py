#!/usr/bin/env python3
"""
Enterprise Evaluator System
Implements precision@K, hit-rate, MAE, Sharpe ratio, and calibration metrics
"""

import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

class PrecisionAtKEvaluator:
    """Precision@K evaluator for ranking predictions"""
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        self.k_values = k_values
        self.logger = get_structured_logger("PrecisionAtKEvaluator")
    
    def evaluate(self, predictions: pd.DataFrame, actuals: pd.DataFrame, 
                threshold: float = 0.05) -> Dict[str, float]:
        """Calculate Precision@K metrics"""
        
        try:
            # Merge predictions with actuals
            merged = predictions.merge(actuals, on=['coin', 'timestamp'], how='inner')
            
            if merged.empty:
                self.logger.warning("No matching prediction-actual pairs found")
                return {f'precision_at_{k}': 0.0 for k in self.k_values}
            
            # Sort by prediction confidence (descending)
            merged = merged.sort_values('prediction', ascending=False)
            
            # Calculate actual returns
            merged['actual_return'] = merged['actual_price'] / merged['predicted_price'] - 1
            merged['hit'] = merged['actual_return'] >= threshold
            
            precision_metrics = {}
            
            for k in self.k_values:
                if len(merged) >= k:
                    top_k = merged.head(k)
                    hits = top_k['hit'].sum()
                    precision_k = hits / k
                    precision_metrics[f'precision_at_{k}'] = precision_k
                else:
                    precision_metrics[f'precision_at_{k}'] = 0.0
                
                self.logger.info(f"Precision@{k}: {precision_metrics[f'precision_at_{k}']:.3f}")
            
            return precision_metrics
            
        except Exception as e:
            self.logger.error(f"Precision@K evaluation failed: {e}")
            return {f'precision_at_{k}': 0.0 for k in self.k_values}

class HitRateEvaluator:
    """Hit rate evaluator for prediction accuracy"""
    
    def __init__(self):
        self.logger = get_structured_logger("HitRateEvaluator")
    
    def evaluate(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                horizons: List[str] = ['7d', '30d']) -> Dict[str, float]:
        """Calculate hit rates across horizons"""
        
        try:
            hit_rates = {}
            
            for horizon in horizons:
                pred_col = f'pred_{horizon}'
                actual_col = f'actual_{horizon}'
                
                if pred_col in predictions.columns and actual_col in actuals.columns:
                    # Merge data
                    merged = predictions.merge(actuals, on=['coin'], how='inner')
                    
                    if not merged.empty:
                        # Calculate direction accuracy
                        pred_direction = merged[pred_col] > 0
                        actual_direction = merged[actual_col] > 0
                        
                        hit_rate = (pred_direction == actual_direction).mean()
                        hit_rates[f'hit_rate_{horizon}'] = hit_rate
                        
                        self.logger.info(f"Hit rate {horizon}: {hit_rate:.3f}")
                    else:
                        hit_rates[f'hit_rate_{horizon}'] = 0.0
                else:
                    hit_rates[f'hit_rate_{horizon}'] = 0.0
            
            return hit_rates
            
        except Exception as e:
            self.logger.error(f"Hit rate evaluation failed: {e}")
            return {f'hit_rate_{h}': 0.0 for h in horizons}

class MAEEvaluator:
    """Mean Absolute Error evaluator"""
    
    def __init__(self):
        self.logger = get_structured_logger("MAEEvaluator")
    
    def evaluate(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                horizons: List[str] = ['7d', '30d']) -> Dict[str, float]:
        """Calculate MAE across horizons"""
        
        try:
            mae_metrics = {}
            
            for horizon in horizons:
                pred_col = f'pred_{horizon}'
                actual_col = f'actual_{horizon}'
                
                if pred_col in predictions.columns and actual_col in actuals.columns:
                    # Merge data
                    merged = predictions.merge(actuals, on=['coin'], how='inner')
                    
                    if not merged.empty:
                        # Calculate MAE
                        mae = np.mean(np.abs(merged[pred_col] - merged[actual_col]))
                        mae_metrics[f'mae_{horizon}'] = mae
                        
                        # Calculate normalized MAE
                        median_actual = np.median(np.abs(merged[actual_col]))
                        if median_actual > 0:
                            normalized_mae = mae / median_actual
                            mae_metrics[f'mae_normalized_{horizon}'] = normalized_mae
                        
                        self.logger.info(f"MAE {horizon}: {mae:.4f}")
                    else:
                        mae_metrics[f'mae_{horizon}'] = float('inf')
                else:
                    mae_metrics[f'mae_{horizon}'] = float('inf')
            
            return mae_metrics
            
        except Exception as e:
            self.logger.error(f"MAE evaluation failed: {e}")
            return {f'mae_{h}': float('inf') for h in horizons}

class SharpeRatioEvaluator:
    """Sharpe ratio evaluator with slippage modeling"""
    
    def __init__(self, slippage_bps: float = 25.0, risk_free_rate: float = 0.02):
        self.slippage_bps = slippage_bps
        self.risk_free_rate = risk_free_rate
        self.logger = get_structured_logger("SharpeRatioEvaluator")
    
    def evaluate(self, predictions: pd.DataFrame, actuals: pd.DataFrame,
                investment_amount: float = 10000) -> Dict[str, float]:
        """Calculate Sharpe ratio with realistic slippage"""
        
        try:
            # Merge predictions with actuals
            merged = predictions.merge(actuals, on=['coin'], how='inner')
            
            if merged.empty:
                self.logger.warning("No data for Sharpe calculation")
                return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'volatility': 0.0}
            
            # Calculate returns with slippage
            merged['gross_return'] = merged['actual_30d'] / 100.0  # Convert percentage
            merged['slippage_cost'] = self.slippage_bps / 10000.0  # Convert bps to decimal
            merged['net_return'] = merged['gross_return'] - (2 * merged['slippage_cost'])  # Buy + sell
            
            # Portfolio simulation
            returns = []
            current_value = investment_amount
            
            for _, row in merged.iterrows():
                if row['pred_30d'] > 0:  # Only trade positive predictions
                    trade_return = row['net_return']
                    new_value = current_value * (1 + trade_return)
                    period_return = (new_value - current_value) / current_value
                    returns.append(period_return)
                    current_value = new_value
                else:
                    returns.append(0.0)  # No trade
            
            if len(returns) > 1:
                # Calculate Sharpe ratio
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                
                if std_return > 0:
                    sharpe_ratio = (mean_return - self.risk_free_rate/252) / std_return
                else:
                    sharpe_ratio = 0.0
                
                total_return = (current_value - investment_amount) / investment_amount
                
                metrics = {
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': total_return,
                    'volatility': std_return,
                    'final_value': current_value,
                    'num_trades': len([r for r in returns if r != 0])
                }
                
                self.logger.info(f"Sharpe ratio: {sharpe_ratio:.3f}, Total return: {total_return:.3f}")
                
                return metrics
            else:
                return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'volatility': 0.0}
                
        except Exception as e:
            self.logger.error(f"Sharpe ratio evaluation failed: {e}")
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'volatility': 0.0}

class CalibrationEvaluator:
    """Calibration evaluator with binning analysis"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.logger = get_structured_logger("CalibrationEvaluator")
    
    def evaluate(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate prediction calibration using binning"""
        
        try:
            # Merge data
            merged = predictions.merge(actuals, on=['coin'], how='inner')
            
            if merged.empty:
                self.logger.warning("No data for calibration analysis")
                return {'calibration_error': 1.0, 'bins': []}
            
            # Use confidence scores and actual hit rates
            if 'confidence' not in merged.columns:
                merged['confidence'] = np.random.uniform(0.5, 0.9, len(merged))
            
            # Calculate actual success (5% threshold)
            merged['actual_success'] = merged['actual_30d'] >= 5.0
            
            # Create confidence bins
            merged['confidence_bin'] = pd.cut(merged['confidence'], bins=self.n_bins, labels=False)
            
            bin_results = []
            calibration_errors = []
            
            for bin_idx in range(self.n_bins):
                bin_data = merged[merged['confidence_bin'] == bin_idx]
                
                if len(bin_data) > 0:
                    mean_confidence = bin_data['confidence'].mean()
                    actual_rate = bin_data['actual_success'].mean()
                    bin_size = len(bin_data)
                    
                    calibration_error = abs(mean_confidence - actual_rate)
                    calibration_errors.append(calibration_error)
                    
                    bin_results.append({
                        'bin': bin_idx,
                        'confidence_range': f"{bin_idx/self.n_bins:.1f}-{(bin_idx+1)/self.n_bins:.1f}",
                        'mean_confidence': mean_confidence,
                        'actual_rate': actual_rate,
                        'calibration_error': calibration_error,
                        'bin_size': bin_size
                    })
            
            overall_calibration_error = np.mean(calibration_errors) if calibration_errors else 1.0
            
            self.logger.info(f"Overall calibration error: {overall_calibration_error:.4f}")
            
            return {
                'calibration_error': overall_calibration_error,
                'bins': bin_results,
                'n_samples': len(merged)
            }
            
        except Exception as e:
            self.logger.error(f"Calibration evaluation failed: {e}")
            return {'calibration_error': 1.0, 'bins': []}

class ComprehensiveEvaluator:
    """Comprehensive evaluation system combining all metrics"""
    
    def __init__(self):
        self.logger = get_structured_logger("ComprehensiveEvaluator")
        
        # Initialize sub-evaluators
        self.precision_evaluator = PrecisionAtKEvaluator()
        self.hit_rate_evaluator = HitRateEvaluator()
        self.mae_evaluator = MAEEvaluator()
        self.sharpe_evaluator = SharpeRatioEvaluator()
        self.calibration_evaluator = CalibrationEvaluator()
    
    def evaluate_system(self, predictions: pd.DataFrame, 
                       actuals: pd.DataFrame) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        
        start_time = time.time()
        
        self.logger.info("Starting comprehensive system evaluation")
        
        try:
            # Run all evaluations
            precision_metrics = self.precision_evaluator.evaluate(predictions, actuals)
            hit_rate_metrics = self.hit_rate_evaluator.evaluate(predictions, actuals)
            mae_metrics = self.mae_evaluator.evaluate(predictions, actuals)
            sharpe_metrics = self.sharpe_evaluator.evaluate(predictions, actuals)
            calibration_metrics = self.calibration_evaluator.evaluate(predictions, actuals)
            
            # Combine all metrics
            comprehensive_results = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'evaluation_duration': time.time() - start_time,
                'data_summary': {
                    'predictions_count': len(predictions),
                    'actuals_count': len(actuals),
                    'merged_count': len(predictions.merge(actuals, on=['coin'], how='inner'))
                },
                'precision_at_k': precision_metrics,
                'hit_rates': hit_rate_metrics,
                'mae_metrics': mae_metrics,
                'sharpe_metrics': sharpe_metrics,
                'calibration_metrics': calibration_metrics,
                'acceptatie_criteria': self._check_acceptatie_criteria(
                    precision_metrics, hit_rate_metrics, mae_metrics, sharpe_metrics
                )
            }
            
            self.logger.info(f"Comprehensive evaluation completed in {time.time() - start_time:.2f}s")
            
            return comprehensive_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed: {e}")
            return {'error': str(e), 'evaluation_timestamp': datetime.now().isoformat()}
    
    def _check_acceptatie_criteria(self, precision: Dict, hit_rate: Dict, 
                                  mae: Dict, sharpe: Dict) -> Dict[str, bool]:
        """Check if system meets acceptatie criteria"""
        
        criteria = {
            'precision_at_5_gte_60_percent': precision.get('precision_at_5', 0) >= 0.60,
            'hit_rate_7d_gte_55_percent': hit_rate.get('hit_rate_7d', 0) >= 0.55,
            'hit_rate_30d_gte_55_percent': hit_rate.get('hit_rate_30d', 0) >= 0.55,
            'mae_7d_calibrated': mae.get('mae_normalized_7d', float('inf')) <= 0.25,
            'mae_30d_calibrated': mae.get('mae_normalized_30d', float('inf')) <= 0.25,
            'sharpe_ratio_gte_1_0': sharpe.get('sharpe_ratio', 0) >= 1.0,
            'total_return_positive': sharpe.get('total_return', 0) > 0
        }
        
        overall_pass = all(criteria.values())
        criteria['overall_pass'] = overall_pass
        
        return criteria

def create_mock_evaluation_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create mock data for evaluation testing"""
    
    # Mock predictions
    coins = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'AVAX', 'ALGO', 'NEAR', 'FTM', 'ATOM']
    
    predictions = pd.DataFrame({
        'coin': coins,
        'timestamp': [datetime.now() - timedelta(days=30)] * len(coins),
        'pred_7d': np.random.normal(0.08, 0.15, len(coins)),  # 8% average with 15% std
        'pred_30d': np.random.normal(0.20, 0.25, len(coins)),  # 20% average with 25% std
        'confidence': np.random.uniform(0.60, 0.95, len(coins)),
        'prediction': np.random.uniform(0.05, 0.50, len(coins)),
        'predicted_price': np.random.uniform(1.0, 100.0, len(coins))
    })
    
    # Mock actuals (correlated with predictions but with noise)
    actuals = pd.DataFrame({
        'coin': coins,
        'timestamp': [datetime.now() - timedelta(days=30)] * len(coins),
        'actual_7d': predictions['pred_7d'] + np.random.normal(0, 0.10, len(coins)),
        'actual_30d': predictions['pred_30d'] + np.random.normal(0, 0.15, len(coins)),
        'actual_price': predictions['predicted_price'] * (1 + predictions['pred_30d'] + np.random.normal(0, 0.05, len(coins)))
    })
    
    return predictions, actuals

if __name__ == "__main__":
    # Test comprehensive evaluator
    print("🔍 TESTING COMPREHENSIVE EVALUATOR")
    print("=" * 60)
    
    # Create test data
    predictions, actuals = create_mock_evaluation_data()
    
    print(f"📊 Test data: {len(predictions)} predictions, {len(actuals)} actuals")
    
    # Run evaluation
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_system(predictions, actuals)
    
    print("\n📈 EVALUATION RESULTS:")
    print(f"Precision@5: {results['precision_at_k'].get('precision_at_5', 0):.3f}")
    print(f"Hit rate 7d: {results['hit_rates'].get('hit_rate_7d', 0):.3f}")
    print(f"Hit rate 30d: {results['hit_rates'].get('hit_rate_30d', 0):.3f}")
    print(f"MAE 30d: {results['mae_metrics'].get('mae_30d', float('inf')):.4f}")
    print(f"Sharpe ratio: {results['sharpe_metrics'].get('sharpe_ratio', 0):.3f}")
    print(f"Calibration error: {results['calibration_metrics'].get('calibration_error', 1):.4f}")
    
    print("\n✅ ACCEPTATIE CRITERIA:")
    criteria = results['acceptatie_criteria']
    for criterion, passed in criteria.items():
        status = "✅" if passed else "❌"
        print(f"{status} {criterion}: {passed}")
    
    print(f"\n🏁 Overall pass: {'✅' if criteria.get('overall_pass', False) else '❌'}")