#!/usr/bin/env python3
"""
Fixed confidence gate - consistent met generate_final_predictions.py
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Union

logger = logging.getLogger(__name__)

def calculate_ensemble_confidence(sigma: float) -> float:
    """
    Consistent confidence berekening - gebaseerd op generate_final_predictions.py
    confidence = 1/(1 + sigma) waar sigma = ensemble std
    """
    return 1.0 / (1.0 + max(sigma, 0.01))  # Prevent division by zero

def normalize_score_to_confidence(score: float, min_score: float = 0, max_score: float = 100) -> float:
    """
    Score normalisatie naar confidence [0,1]
    Gebruikt voor backwards compatibility met oude score systemen
    """
    normalized = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))

def apply_consistent_confidence_gate(predictions: Union[List[Dict], pd.DataFrame], 
                                   threshold: float = 0.80,
                                   confidence_method: str = 'ensemble') -> Union[List[Dict], pd.DataFrame]:
    """
    Fixed confidence gate - consistent tussen batch en UI
    
    Args:
        predictions: Predictions data
        threshold: Confidence threshold (0.80 = 80%)
        confidence_method: 'ensemble' (sigma-based) or 'score' (score-based)
    """
    logger.info(f"Applying {confidence_method} confidence gate with threshold {threshold}")
    
    if isinstance(predictions, pd.DataFrame):
        return _apply_gate_dataframe(predictions, threshold, confidence_method)
    elif isinstance(predictions, list):
        return _apply_gate_list(predictions, threshold, confidence_method)
    else:
        logger.error(f"Unsupported predictions type: {type(predictions)}")
        return predictions

def _apply_gate_dataframe(df: pd.DataFrame, threshold: float, method: str) -> pd.DataFrame:
    """Apply gate to DataFrame"""
    if df.empty:
        return df
    
    original_count = len(df)
    
    # Find confidence columns
    conf_cols = [col for col in df.columns if 'confidence' in col.lower()]
    
    if not conf_cols:
        logger.warning("No confidence columns found in DataFrame")
        return df
    
    # Calculate max confidence per row
    if method == 'ensemble':
        # Use existing confidence values (assumed to be sigma-based)
        max_confidences = df[conf_cols].max(axis=1)
    else:  # score method
        # Convert scores to confidence
        score_cols = [col for col in df.columns if 'score' in col.lower()]
        if score_cols:
            max_scores = df[score_cols].max(axis=1)
            max_confidences = max_scores.apply(lambda x: normalize_score_to_confidence(x))
        else:
            max_confidences = df[conf_cols].max(axis=1)
    
    # Apply threshold
    passed_mask = max_confidences >= threshold
    filtered_df = df[passed_mask].copy()  # Explicit copy to avoid SettingWithCopyWarning
    
    # Add metadata
    filtered_df.loc[:, 'gate_confidence'] = max_confidences[passed_mask]
    filtered_df.loc[:, 'gate_passed'] = True
    
    passed_count = len(filtered_df)
    rejection_rate = (original_count - passed_count) / original_count if original_count > 0 else 0
    
    logger.info(f"Gate results: {passed_count}/{original_count} passed ({rejection_rate*100:.1f}% rejected)")
    
    return filtered_df

def _apply_gate_list(predictions: List[Dict], threshold: float, method: str) -> List[Dict]:
    """Apply gate to list of predictions"""
    if not predictions:
        return predictions
    
    original_count = len(predictions)
    filtered_predictions = []
    
    for pred in predictions:
        # Find confidence values
        conf_values = [v for k, v in pred.items() if 'confidence' in k.lower()]
        
        if not conf_values:
            logger.warning(f"No confidence values found for prediction: {pred.get('coin', 'unknown')}")
            continue
        
        # Calculate confidence
        if method == 'ensemble':
            max_confidence = max(conf_values)
        else:  # score method
            score_values = [v for k, v in pred.items() if 'score' in k.lower()]
            if score_values:
                max_score = max(score_values)
                max_confidence = normalize_score_to_confidence(max_score)
            else:
                max_confidence = max(conf_values)
        
        # Apply threshold
        if max_confidence >= threshold:
            pred_copy = pred.copy()
            pred_copy['gate_confidence'] = max_confidence
            pred_copy['gate_passed'] = True
            filtered_predictions.append(pred_copy)
    
    passed_count = len(filtered_predictions)
    rejection_rate = (original_count - passed_count) / original_count if original_count > 0 else 0
    
    logger.info(f"Gate results: {passed_count}/{original_count} passed ({rejection_rate*100:.1f}% rejected)")
    
    return filtered_predictions

def validate_confidence_consistency(batch_predictions: str, ui_predictions: List[Dict]) -> Dict:
    """
    Validate consistency tussen batch en UI confidence values
    """
    issues = []
    
    try:
        # Load batch predictions
        if batch_predictions.endswith('.csv'):
            batch_df = pd.read_csv(batch_predictions)
        else:
            import json
            with open(batch_predictions, 'r') as f:
                batch_data = json.load(f)
            batch_df = pd.DataFrame(batch_data)
        
        # Compare confidence methods
        batch_conf_cols = [col for col in batch_df.columns if 'confidence' in col.lower()]
        ui_conf_keys = set()
        for pred in ui_predictions:
            ui_conf_keys.update([k for k in pred.keys() if 'confidence' in k.lower()])
        
        if set(batch_conf_cols) != ui_conf_keys:
            issues.append("Confidence column mismatch between batch and UI")
        
        # Check value ranges
        batch_conf_ranges = {}
        for col in batch_conf_cols:
            batch_conf_ranges[col] = (batch_df[col].min(), batch_df[col].max())
        
        ui_conf_ranges = {}
        for key in ui_conf_keys:
            values = [pred[key] for pred in ui_predictions if key in pred]
            if values:
                ui_conf_ranges[key] = (min(values), max(values))
        
        for key in batch_conf_ranges:
            if key in ui_conf_ranges:
                batch_range = batch_conf_ranges[key]
                ui_range = ui_conf_ranges[key]
                
                # Check if ranges are significantly different
                if abs(batch_range[0] - ui_range[0]) > 0.1 or abs(batch_range[1] - ui_range[1]) > 0.1:
                    issues.append(f"Confidence range mismatch for {key}: batch {batch_range} vs UI {ui_range}")
        
    except Exception as e:
        issues.append(f"Validation failed: {e}")
    
    return {
        'consistent': len(issues) == 0,
        'issues': issues
    }

def test_confidence_gate_fixes():
    """Test the fixed confidence gate"""
    print("🧪 Testing Fixed Confidence Gate")
    print("-" * 40)
    
    # Test data
    test_predictions = [
        {'coin': 'BTC', 'confidence_24h': 0.85, 'score': 85},
        {'coin': 'ETH', 'confidence_24h': 0.75, 'score': 75},
        {'coin': 'ADA', 'confidence_24h': 0.90, 'score': 90},
    ]
    
    # Test ensemble method
    print("Testing ensemble method:")
    filtered_ensemble = apply_consistent_confidence_gate(test_predictions, 0.80, 'ensemble')
    print(f"Ensemble result: {len(filtered_ensemble)} predictions passed")
    
    # Test score method
    print("\nTesting score method:")
    filtered_score = apply_consistent_confidence_gate(test_predictions, 0.80, 'score')
    print(f"Score result: {len(filtered_score)} predictions passed")
    
    print("\n✅ Confidence gate tests completed")

if __name__ == "__main__":
    test_confidence_gate_fixes()