# orchestration/strict_gate_standalone.py
# Standalone strict backend gate implementation - geen dependencies op andere agents
import pandas as pd
from pathlib import Path

# Configureerbare horizon-lijst - gemakkelijk aan te passen bij modelset wijzigingen
DEFAULT_HORIZONS = ['1h', '24h', '168h', '720h']

def get_available_horizons(pred_df: pd.DataFrame) -> list:
    """
    Detecteer beschikbare horizons automatisch uit DataFrame columns
    Voorkomt problemen bij modelset wijzigingen
    """
    if pred_df is None or pred_df.empty:
        return []

    horizons = []
    for col in pred_df.columns:
        if col.startswith('pred_') and col.replace('pred_', '') not in [h.replace('pred_', '') for h in horizons]:
            horizon = col.replace('pred_', '')
            conf_col = f'conf_{horizon}'
            if conf_col in pred_df.columns:
                horizons.append(horizon)

    return horizons

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

    # Sort by prediction strength - altijd consistent reset_index
    if not filtered.empty:
        filtered = filtered.sort_values(by=pred_col, ascending=False).reset_index(drop=True)
    else:
        # Ook lege DataFrame krijgt consistente index
        filtered = filtered.reset_index(drop=True)

    return filtered

def strict_toplist_multi_horizon(pred_df: pd.DataFrame, threshold=0.80):
    """
    Multi-horizon strict filtering voor alle timeframes
    """
    if pred_df is None or pred_df.empty:
        return {}

    results = {}
    # Gebruik automatische horizon detectie in plaats van harde lijst
    horizons = get_available_horizons(pred_df)
    if not horizons:  # Fallback naar default als detectie faalt
        horizons = DEFAULT_HORIZONS

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

    # Check alle beschikbare horizons dynamisch
    horizons = get_available_horizons(pred_df)
    if not horizons:  # Fallback naar default als detectie faalt
        horizons = DEFAULT_HORIZONS
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

    # Gebruik dynamische horizon detectie
    horizons = get_available_horizons(pred_df)
    if not horizons:  # Fallback naar default als detectie faalt
        horizons = DEFAULT_HORIZONS
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

        # Minder agressieve authenticiteitscheck - voorkom false positives
        # Bij kleine sets of sterke kalibratie kan std laag zijn
        if len(conf_values) > 10 and conf_values.std() < 0.0001:  # Nog steeds detectie van echte fake data
            return False, f"Suspicious uniform confidence in {conf_col} (std: {conf_values.std():.6f})"

        # Should be in reasonable range
        if conf_values.min() < 0 or conf_values.max() > 1:
            return False, f"Invalid confidence range in {conf_col}"

    return True, "Predictions appear authentic"
