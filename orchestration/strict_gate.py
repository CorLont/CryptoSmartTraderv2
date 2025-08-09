# orchestration/strict_gate.py - Server-side strict 80% confidence enforcement
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def strict_filter(df: pd.DataFrame, pred="pred_720h", conf="conf_720h", thr=0.80):
    """
    Server-side strict confidence filtering - no exceptions
    Returns empty DataFrame if no candidates meet threshold
    """
    # Input validation
    if df is None or df.empty:
        logger.warning("strict_filter: Empty input DataFrame")
        return pd.DataFrame()
    
    if pred not in df.columns or conf not in df.columns:
        logger.error(f"strict_filter: Missing columns {pred} or {conf}")
        return pd.DataFrame()
    
    # Drop rows with NaN predictions or confidence
    df_clean = df.dropna(subset=[pred, conf])
    
    if df_clean.empty:
        logger.warning("strict_filter: All rows had NaN predictions/confidence")
        return pd.DataFrame()
    
    # Apply strict threshold - NO MERCY
    filtered = df_clean[df_clean[conf] >= thr]
    
    if filtered.empty:
        logger.info(f"strict_filter: No candidates met {thr:.0%} confidence threshold")
        return pd.DataFrame()
    
    # Sort by prediction (best first)
    result = filtered.sort_values(pred, ascending=False).reset_index(drop=True)
    
    logger.info(f"strict_filter: {len(result)}/{len(df)} candidates passed {thr:.0%} gate")
    
    return result

def multi_horizon_strict_filter(df: pd.DataFrame, horizons=None, thr=0.80):
    """
    Apply strict filtering across multiple horizons
    Returns candidates that meet threshold on ANY horizon
    """
    if horizons is None:
        horizons = ["1h", "24h", "168h", "720h"]
    
    all_passed = pd.DataFrame()
    
    for h in horizons:
        pred_col = f"pred_{h}"
        conf_col = f"conf_{h}"
        
        if pred_col in df.columns and conf_col in df.columns:
            passed = strict_filter(df, pred=pred_col, conf=conf_col, thr=thr)
            
            if not passed.empty:
                passed['passed_horizon'] = h
                all_passed = pd.concat([all_passed, passed], ignore_index=True)
    
    # Remove duplicates (same coin passing multiple horizons)
    if not all_passed.empty and 'coin' in all_passed.columns:
        all_passed = all_passed.drop_duplicates(subset=['coin'], keep='first')
    
    return all_passed

def enterprise_confidence_gate(df: pd.DataFrame, min_threshold=0.80):
    """
    Enterprise-grade confidence gate - zero tolerance for low confidence
    
    This is the FINAL gate - if it returns empty, UI shows nothing
    No fallbacks, no exceptions, no "soft mode"
    """
    gate_id = f"enterprise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Enterprise confidence gate {gate_id}: Processing {len(df) if df is not None else 0} candidates")
    
    if df is None or df.empty:
        logger.warning(f"Enterprise gate {gate_id}: Empty input - BLOCKING ALL")
        return pd.DataFrame(), {"status": "blocked_empty_input", "gate_id": gate_id}
    
    # Apply multi-horizon strict filtering
    passed = multi_horizon_strict_filter(df, thr=min_threshold)
    
    result = {
        "status": "passed" if not passed.empty else "blocked_low_confidence",
        "gate_id": gate_id,
        "input_count": len(df),
        "passed_count": len(passed),
        "threshold": min_threshold,
        "blocked_count": len(df) - len(passed)
    }
    
    if passed.empty:
        logger.warning(f"Enterprise gate {gate_id}: BLOCKED ALL {len(df)} candidates - none met {min_threshold:.0%} threshold")
    else:
        logger.info(f"Enterprise gate {gate_id}: PASSED {len(passed)}/{len(df)} candidates")
    
    return passed, result