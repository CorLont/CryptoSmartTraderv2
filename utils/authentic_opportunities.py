# utils/authentic_opportunities.py
# Authentieke opportunity counting zonder fake numbers (per review feedback)

import pandas as pd
from pathlib import Path

def get_authentic_opportunities_count(threshold=0.80):
    """
    Get AUTHENTIC count van high-confidence opportunities
    Geen fake numbers - alleen wat echt in predictions.csv staat
    
    Per review: "toon lege-staat en géén 'opportunities: 15' zolang deze filter leeg is"
    """
    
    predictions_path = Path("exports/production/predictions.csv")
    
    if not predictions_path.exists():
        return 0  # Geen predictions = geen opportunities
    
    try:
        pred_df = pd.read_csv(predictions_path)
        
        if pred_df.empty:
            return 0
        
        # Check alle horizons voor high confidence predictions
        horizons = ['1h', '24h', '168h', '720h']
        total_opportunities = 0
        
        for h in horizons:
            pred_col = f'pred_{h}'
            conf_col = f'conf_{h}'
            
            if pred_col in pred_df.columns and conf_col in pred_df.columns:
                # Drop NaN values (no fallback allowed)
                clean_df = pred_df.dropna(subset=[pred_col, conf_col])
                
                # Only count predictions >= threshold
                high_conf = clean_df[clean_df[conf_col] >= threshold]
                total_opportunities += len(high_conf)
        
        return total_opportunities
        
    except Exception as e:
        print(f"Error reading predictions: {e}")
        return 0

def get_strict_filtered_predictions(threshold=0.80):
    """
    Get strict filtered predictions voor UI display
    Returns lege DataFrame als geen data voldoet aan threshold
    """
    predictions_path = Path("exports/production/predictions.csv")
    
    if not predictions_path.exists():
        return pd.DataFrame()
    
    try:
        pred_df = pd.read_csv(predictions_path)
        
        if pred_df.empty:
            return pd.DataFrame()
        
        # Multi-horizon filtering
        results = []
        horizons = ['1h', '24h', '168h', '720h']
        
        for h in horizons:
            pred_col = f'pred_{h}'
            conf_col = f'conf_{h}'
            
            if pred_col in pred_df.columns and conf_col in pred_df.columns:
                # Strict filtering: drop NaN, filter by confidence
                clean_df = pred_df.dropna(subset=[pred_col, conf_col])
                filtered = clean_df[clean_df[conf_col] >= threshold]
                
                if not filtered.empty:
                    # Add horizon info en sort by prediction strength
                    filtered = filtered.copy()
                    filtered['horizon'] = h
                    filtered = filtered.sort_values(pred_col, ascending=False)
                    results.append(filtered)
        
        if results:
            combined = pd.concat(results, ignore_index=True)
            return combined.head(50)  # Top 50 max voor performance
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error filtering predictions: {e}")
        return pd.DataFrame()

def validate_opportunities_authentic():
    """
    Valideer dat opportunity counts authentiek zijn
    """
    predictions_path = Path("exports/production/predictions.csv")
    
    if not predictions_path.exists():
        return False, "No predictions file"
    
    try:
        pred_df = pd.read_csv(predictions_path)
        
        if pred_df.empty:
            return False, "Empty predictions"
        
        # Check dat confidence values niet uniform zijn (teken van fake data)
        horizons = ['1h', '24h', '168h', '720h']
        
        for h in horizons:
            conf_col = f'conf_{h}'
            if conf_col in pred_df.columns:
                conf_values = pred_df[conf_col].dropna()
                
                if len(conf_values) > 1:
                    # Authentic ensemble confidence should vary
                    if conf_values.std() < 0.001:
                        return False, f"Suspicious uniform confidence in {conf_col}"
                
                # Should be in valid range
                if (conf_values < 0).any() or (conf_values > 1).any():
                    return False, f"Invalid confidence range in {conf_col}"
        
        return True, "Opportunities appear authentic"
        
    except Exception as e:
        return False, f"Validation error: {e}"