#!/usr/bin/env python3
"""
Create synthetic targets for training - DEMO/TEST PURPOSES ONLY
⚠️ WARNING: This is a temporary testing solution. Production systems should use real price data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List

logger = logging.getLogger(__name__)

def create_synthetic_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic price targets based on current features
    
    ⚠️ WARNING: This generates synthetic targets for testing purposes only.
    Production systems must use authentic market data.
    
    Args:
        df: DataFrame with market features
        
    Returns:
        DataFrame with synthetic target columns added
        
    Raises:
        ValueError: If required features are missing
    """
    
    # Check for required features - explicit validation
    required_features = ["momentum_3d", "momentum_7d", "price_change_24h"]
    missing_features = [col for col in required_features if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing expected features: {missing_features}")
        logger.warning("Using zero defaults for missing features - may impact synthetic target quality")
    
    # Create synthetic price movements based on technical indicators
    np.random.seed(42)  # Reproducible results for testing
    
    for horizon in ['1h', '24h', '168h', '720h']:
        # Convert horizon to hours for scaling
        hours_map = {'1h': 1, '24h': 24, '168h': 168, '720h': 720}
        hours = hours_map[horizon]
        
        # Base return using momentum and volatility - explicit feature handling
        momentum_3d = df['momentum_3d'] if 'momentum_3d' in df.columns else pd.Series(0, index=df.index)
        momentum_7d = df['momentum_7d'] if 'momentum_7d' in df.columns else pd.Series(0, index=df.index)
        price_change_24h = df['price_change_24h'] if 'price_change_24h' in df.columns else pd.Series(0, index=df.index)
        
        base_return = (
            momentum_3d * 0.3 +
            momentum_7d * 0.2 +
            price_change_24h / 100 * 0.5
        )
        
        # Scale by time horizon
        time_scaling = np.log(hours + 1) / np.log(25)  # Normalized scaling
        
        # Add controlled noise
        noise = np.random.normal(0, 0.02 * time_scaling, len(df))
        
        # Synthetic return target
        synthetic_return = base_return * time_scaling + noise
        df[f'target_return_{horizon}'] = synthetic_return
        
        # Direction target (binary)
        df[f'target_direction_{horizon}'] = (synthetic_return > 0.01).astype(int)
    
    logger.info(f"Created synthetic targets for {len(df)} coins across 4 horizons")
    logger.warning("⚠️ SYNTHETIC TARGETS USED - Replace with real price data for production!")
    return df


def validate_synthetic_targets(df: pd.DataFrame) -> bool:
    """
    Validate that synthetic targets were created successfully
    
    Args:
        df: DataFrame to validate
        
    Returns:
        bool: True if all expected targets are present
    """
    expected_targets = []
    for horizon in ['1h', '24h', '168h', '720h']:
        expected_targets.extend([
            f'target_return_{horizon}',
            f'target_direction_{horizon}'
        ])
    
    missing_targets = [col for col in expected_targets if col not in df.columns]
    
    if missing_targets:
        logger.error(f"Missing synthetic targets: {missing_targets}")
        return False
    
    # Check for NaN values
    for target in expected_targets:
        if target in df.columns and df[target].isna().any():
            logger.warning(f"NaN values found in {target}")
    
    logger.info("Synthetic targets validation passed")
    return True