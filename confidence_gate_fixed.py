#!/usr/bin/env python3
"""
Fixed confidence gate - implements audit point D
Proper confidence normalization without 0-pass bug
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_confidence_gate_normalized(predictions_df, threshold=0.80):
    """
    Fixed: proper confidence normalization
    Based on debug_confidence_gate.py proposed fix
    """
    logger.info(f"Applying normalized confidence gate with threshold {threshold}")
    
    original_count = len(predictions_df)
    
    # Get confidence columns
    conf_cols = [col for col in predictions_df.columns if 'confidence' in col.lower()]
    
    if not conf_cols:
        logger.warning("No confidence columns found")
        return predictions_df
    
    # Get max confidence across all horizons
    max_confidences = predictions_df[conf_cols].max(axis=1)
    
    # Apply fixed normalization (0.65-0.95 mapping to 0-1)
    # This prevents the 0-pass bug from earlier versions
    normalized_conf = np.clip(
        (max_confidences - 65) / (95 - 65),  # Map 65-95 to 0-1
        0, 1
    )
    
    # Apply threshold
    passed_mask = normalized_conf >= threshold
    filtered_df = predictions_df[passed_mask].copy()
    
    # Add normalized confidence to output
    filtered_df['normalized_confidence'] = normalized_conf[passed_mask]
    
    filtered_count = len(filtered_df)
    rejection_rate = (original_count - filtered_count) / original_count if original_count > 0 else 0
    
    logger.info(f"Confidence gate results:")
    logger.info(f"  Original: {original_count}")
    logger.info(f"  Passed: {filtered_count}")
    logger.info(f"  Rejection rate: {rejection_rate*100:.1f}%")
    
    return filtered_df

def test_confidence_gate_fix():
    """Test the fixed confidence gate"""
    # Create test data
    test_data = pd.DataFrame({
        'coin': ['BTC', 'ETH', 'ADA'],
        'confidence_1h': [85, 75, 90],
        'confidence_24h': [80, 70, 95],
        'prediction': [0.05, 0.02, 0.08]
    })
    
    print("Original data:")
    print(test_data)
    
    # Apply fixed gate
    result = apply_confidence_gate_normalized(test_data, 0.80)
    
    print("\nAfter 80% confidence gate:")
    print(result)
    
    return result

if __name__ == "__main__":
    test_confidence_gate_fix()