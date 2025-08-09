# orchestration/strict_gate_standalone.py
# Standalone strict backend gate implementation - geen dependencies op andere agents
import pandas as pd
import numpy as np
from pathlib import Path

def apply_strict_gate_orchestration(pred_df: pd.DataFrame, pred_col="pred_720h", conf_col="conf_720h", threshold=0.80):
    """
    Strict backend enforcement van confidence gate
    Returnt lege DataFrame als geen data voldoet aan threshold
    
    Deze functie mag NOOIT fallback data of placeholders gebruiken
    """
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    
    # Drop NaN values - geen fallbacks toegestaan
    clean_df = pred_df.dropna(subset=[pred_col, conf_col])
    
    if clean_df.empty:
        return pd.DataFrame()
    
    # Strict filtering - alleen >= threshold
    filtered = clean_df[clean_df[conf_col] >= threshold]
    
    # Sort by prediction strength
    if not filtered.empty:
        filtered = filtered.sort_values(pred_col, ascending=False).reset_index(drop=True)
    
    return filtered

def strict_toplist_multi_horizon(pred_df: pd.DataFrame, threshold=0.80):
    """
    Multi-horizon strict filtering voor alle timeframes
    """
    if pred_df is None or pred_df.empty:
        return {}
    
    results = {}
    horizons = ['1h', '24h', '168h', '720h']
    
    for h in horizons:
        pred_col = f'pred_{h}'
        conf_col = f'conf_{h}'
        
        if pred_col in pred_df.columns and conf_col in pred_df.columns:
            filtered = apply_strict_gate_orchestration(
                pred_df, pred_col, conf_col, threshold
            )
            results[h] = filtered
        else:
            results[h] = pd.DataFrame()
    
    return results

def get_strict_opportunities_count(pred_df: pd.DataFrame, threshold=0.80):
    """
    Authentic count van opportunities - geen fake numbers
    """
    if pred_df is None or pred_df.empty:
        return 0
    
    # Check alle horizons
    horizons = ['1h', '24h', '168h', '720h'] 
    total_opportunities = 0
    
    for h in horizons:
        pred_col = f'pred_{h}'
        conf_col = f'conf_{h}'
        
        if pred_col in pred_df.columns and conf_col in pred_df.columns:
            clean = pred_df.dropna(subset=[pred_col, conf_col])
            high_conf = clean[clean[conf_col] >= threshold]
            total_opportunities += len(high_conf)
    
    return total_opportunities

def validate_predictions_authentic(pred_df: pd.DataFrame):
    """
    Valideer dat predictions authentiek zijn (geen fake confidence values)
    """
    if pred_df is None or pred_df.empty:
        return False, "No predictions available"
    
    horizons = ['1h', '24h', '168h', '720h']
    required_cols = []
    
    for h in horizons:
        required_cols.extend([f'pred_{h}', f'conf_{h}'])
    
    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for authentic confidence values (ensemble-based should vary)
    for h in horizons:
        conf_col = f'conf_{h}'
        conf_values = pred_df[conf_col].dropna()
        
        if len(conf_values) == 0:
            continue
            
        # Authentic confidence should have variance (not all same value)
        if conf_values.std() < 0.001:  # Too uniform = likely fake
            return False, f"Suspicious uniform confidence in {conf_col}"
        
        # Should be in reasonable range
        if conf_values.min() < 0 or conf_values.max() > 1:
            return False, f"Invalid confidence range in {conf_col}"
    
    return True, "Predictions appear authentic"