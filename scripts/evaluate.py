#!/usr/bin/env python3
"""
Evaluation pipeline with calibration report and coverage audit
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import sys
from typing import Dict, List
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.calibration import calibration_curve

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_predictions():
    """Load latest predictions"""
    pred_file = Path("exports/production/predictions.parquet")
    
    if not pred_file.exists():
        pred_file = Path("exports/production/predictions.csv")
        if pred_file.exists():
            return pd.read_csv(pred_file)
        else:
            logger.error("No predictions file found")
            sys.exit(1)
    
    return pd.read_parquet(pred_file)

def generate_calibration_report(predictions_df: pd.DataFrame) -> Dict:
    """Generate calibration report with reliability bins"""
    
    calibration_report = {
        'timestamp': datetime.now().isoformat(),
        'reliability_bins': {},
        'overall_calibration': {}
    }
    
    # For each horizon, check calibration if we have historical validation data
    for horizon in ['1h', '24h', '168h', '720h']:
        confidence_col = f'confidence_{horizon}'
        direction_col = f'predicted_direction_{horizon}'
        
        if confidence_col not in predictions_df.columns:
            continue
        
        confidences = predictions_df[confidence_col].dropna()
        directions = predictions_df[direction_col].dropna()
        
        if len(confidences) == 0:
            continue
        
        # Create reliability bins
        bins = np.arange(50, 101, 10)  # 50-60, 60-70, ..., 90-100
        bin_report = {}
        
        for i in range(len(bins)-1):
            bin_start = bins[i]
            bin_end = bins[i+1]
            
            mask = (confidences >= bin_start) & (confidences < bin_end)
            bin_confidences = confidences[mask]
            bin_directions = directions[mask]
            
            if len(bin_confidences) > 0:
                avg_confidence = bin_confidences.mean()
                # For calibration, we would need actual outcomes, so this is a placeholder structure
                bin_report[f"{bin_start}-{bin_end}%"] = {
                    'avg_confidence': avg_confidence,
                    'count': len(bin_confidences),
                    'positive_predictions': bin_directions.sum() if len(bin_directions) > 0 else 0
                }
        
        calibration_report['reliability_bins'][horizon] = bin_report
        
        # Overall statistics for this horizon
        calibration_report['overall_calibration'][horizon] = {
            'mean_confidence': confidences.mean(),
            'std_confidence': confidences.std(),
            'min_confidence': confidences.min(),
            'max_confidence': confidences.max(),
            'samples': len(confidences)
        }
    
    return calibration_report

def generate_coverage_audit() -> Dict:
    """Generate coverage audit - Kraken vs processed"""
    
    coverage_audit = {
        'timestamp': datetime.now().isoformat(),
        'kraken_coverage': {},
        'processed_coverage': {},
        'coverage_gaps': []
    }
    
    # Check Kraken data availability
    kraken_data_file = Path("data/raw/kraken_tickers.json")
    if kraken_data_file.exists():
        try:
            with open(kraken_data_file, 'r') as f:
                kraken_data = json.load(f)
            
            kraken_symbols = set()
            if isinstance(kraken_data, dict) and 'tickers' in kraken_data:
                kraken_symbols = set(kraken_data['tickers'].keys())
            elif isinstance(kraken_data, dict):
                kraken_symbols = set(kraken_data.keys())
            
            coverage_audit['kraken_coverage'] = {
                'total_symbols': len(kraken_symbols),
                'symbols': sorted(list(kraken_symbols))[:50],  # First 50 for brevity
                'status': 'available'
            }
            
        except Exception as e:
            logger.warning(f"Error reading Kraken data: {e}")
            coverage_audit['kraken_coverage'] = {'status': 'error', 'error': str(e)}
    else:
        coverage_audit['kraken_coverage'] = {'status': 'missing'}
    
    # Check processed data
    processed_file = Path("data/processed/features.csv")
    if processed_file.exists():
        try:
            processed_df = pd.read_csv(processed_file)
            processed_coins = set(processed_df['coin'].unique()) if 'coin' in processed_df.columns else set()
            
            coverage_audit['processed_coverage'] = {
                'total_coins': len(processed_coins),
                'coins': sorted(list(processed_coins))[:50],  # First 50 for brevity
                'status': 'available'
            }
            
            # Compare coverage
            if 'kraken_coverage' in coverage_audit and coverage_audit['kraken_coverage'].get('status') == 'available':
                kraken_symbols = set(coverage_audit['kraken_coverage'].get('symbols', []))
                
                # Find gaps (symbols in Kraken but not processed)
                missing_from_processed = kraken_symbols - processed_coins
                coverage_audit['coverage_gaps'] = sorted(list(missing_from_processed))[:20]  # First 20 gaps
                
                coverage_ratio = len(processed_coins) / max(len(kraken_symbols), 1)
                coverage_audit['coverage_ratio'] = coverage_ratio
                
        except Exception as e:
            logger.warning(f"Error reading processed data: {e}")
            coverage_audit['processed_coverage'] = {'status': 'error', 'error': str(e)}
    else:
        coverage_audit['processed_coverage'] = {'status': 'missing'}
    
    return coverage_audit

def evaluate_predictions(predictions_df: pd.DataFrame) -> Dict:
    """Evaluate prediction quality"""
    
    evaluation = {
        'timestamp': datetime.now().isoformat(),
        'prediction_stats': {},
        'confidence_distribution': {},
        'horizon_analysis': {}
    }
    
    # Overall stats
    evaluation['prediction_stats'] = {
        'total_predictions': len(predictions_df),
        'unique_coins': predictions_df['coin'].nunique() if 'coin' in predictions_df.columns else 0,
        'avg_confidence': predictions_df[[col for col in predictions_df.columns if col.startswith('confidence_')]].mean().mean()
    }
    
    # Confidence distribution
    confidence_cols = [col for col in predictions_df.columns if col.startswith('confidence_')]
    if confidence_cols:
        all_confidences = []
        for col in confidence_cols:
            all_confidences.extend(predictions_df[col].dropna().tolist())
        
        if all_confidences:
            evaluation['confidence_distribution'] = {
                'mean': np.mean(all_confidences),
                'std': np.std(all_confidences),
                'min': np.min(all_confidences),
                'max': np.max(all_confidences),
                'percentiles': {
                    '50': np.percentile(all_confidences, 50),
                    '80': np.percentile(all_confidences, 80),
                    '90': np.percentile(all_confidences, 90),
                    '95': np.percentile(all_confidences, 95)
                },
                'high_confidence_rate': (np.array(all_confidences) >= 80).mean()
            }
    
    # Per-horizon analysis
    for horizon in ['1h', '24h', '168h', '720h']:
        horizon_data = {}
        
        conf_col = f'confidence_{horizon}'
        ret_col = f'predicted_return_{horizon}'
        dir_col = f'predicted_direction_{horizon}'
        
        if conf_col in predictions_df.columns:
            conf_data = predictions_df[conf_col].dropna()
            horizon_data['confidence'] = {
                'mean': conf_data.mean(),
                'count': len(conf_data),
                'high_conf_count': (conf_data >= 80).sum()
            }
        
        if ret_col in predictions_df.columns:
            ret_data = predictions_df[ret_col].dropna()
            horizon_data['returns'] = {
                'mean_predicted': ret_data.mean(),
                'std_predicted': ret_data.std(),
                'positive_predictions': (ret_data > 0).sum()
            }
        
        if dir_col in predictions_df.columns:
            dir_data = predictions_df[dir_col].dropna()
            horizon_data['directions'] = {
                'bullish_rate': dir_data.mean(),
                'total_directions': len(dir_data)
            }
        
        evaluation['horizon_analysis'][horizon] = horizon_data
    
    return evaluation

def save_daily_report(calibration_report: Dict, coverage_audit: Dict, evaluation: Dict):
    """Save reports to logs/daily/latest.json"""
    
    daily_logs_dir = Path("logs/daily")
    daily_logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all reports
    daily_report = {
        'timestamp': datetime.now().isoformat(),
        'calibration_report': calibration_report,
        'coverage_audit': coverage_audit,
        'evaluation': evaluation
    }
    
    # Save as latest
    latest_file = daily_logs_dir / "latest.json"
    with open(latest_file, 'w') as f:
        json.dump(daily_report, f, indent=2)
    
    # Also save timestamped version
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = daily_logs_dir / f"report_{date_str}.json"
    with open(timestamped_file, 'w') as f:
        json.dump(daily_report, f, indent=2)
    
    logger.info(f"Saved daily report to {latest_file}")
    logger.info(f"Saved timestamped report to {timestamped_file}")
    
    return latest_file

def main():
    """Main evaluation pipeline"""
    logger.info("Starting evaluation pipeline")
    
    # Load predictions
    predictions_df = load_predictions()
    logger.info(f"Loaded {len(predictions_df)} predictions")
    
    # Generate calibration report
    logger.info("Generating calibration report...")
    calibration_report = generate_calibration_report(predictions_df)
    
    # Generate coverage audit
    logger.info("Generating coverage audit...")
    coverage_audit = generate_coverage_audit()
    
    # Evaluate predictions
    logger.info("Evaluating predictions...")
    evaluation = evaluate_predictions(predictions_df)
    
    # Save daily report
    report_file = save_daily_report(calibration_report, coverage_audit, evaluation)
    
    # Print summary
    logger.info("=== Evaluation Summary ===")
    logger.info(f"Total predictions: {len(predictions_df)}")
    
    if 'confidence_distribution' in evaluation and evaluation['confidence_distribution']:
        conf_dist = evaluation['confidence_distribution']
        logger.info(f"Mean confidence: {conf_dist['mean']:.1f}%")
        logger.info(f"High confidence rate: {conf_dist['high_confidence_rate']*100:.1f}%")
    
    if coverage_audit['processed_coverage'].get('status') == 'available':
        logger.info(f"Processed coins: {coverage_audit['processed_coverage']['total_coins']}")
    
    logger.info(f"Report saved to: {report_file}")
    
    logger.info("Evaluation pipeline completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())