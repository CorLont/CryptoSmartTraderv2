#!/usr/bin/env python3
"""
Active Regime Detection Implementation
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """Market Regime Detection and Classification"""
    
    def __init__(self):
        self.regime_thresholds = {
            'volatility_high': 0.03,
            'momentum_bull': 0.02,
            'momentum_bear': -0.02,
            'volume_spike': 2.0
        }
    
    def detect_regime(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Detect market regime for each prediction"""
        
        regime_predictions = []
        
        for _, pred in predictions_df.iterrows():
            # Get market indicators
            price_change = pred.get('price_change_24h', 0) / 100
            volume_ratio = pred.get('volume_trend_7d', 1)
            volatility = pred.get('volatility_7d', 0.02)
            
            # Classify regime
            regime = self._classify_regime(price_change, volume_ratio, volatility)
            
            # Add regime fields
            pred_dict = pred.to_dict()
            pred_dict['regime'] = regime
            pred_dict['regime_confidence'] = self._calculate_regime_confidence(
                price_change, volume_ratio, volatility, regime
            )
            pred_dict['regime_detected'] = True
            
            regime_predictions.append(pred_dict)
        
        result_df = pd.DataFrame(regime_predictions)
        
        # Log regime distribution
        regime_counts = result_df['regime'].value_counts()
        logger.info(f"Detected regimes: {dict(regime_counts)}")
        
        return result_df
    
    def _classify_regime(self, price_change: float, volume_ratio: float, volatility: float) -> str:
        """Classify market regime based on indicators"""
        
        # High volatility regime
        if volatility > self.regime_thresholds['volatility_high']:
            return 'volatile'
        
        # Bull market regime
        elif price_change > self.regime_thresholds['momentum_bull']:
            if volume_ratio > self.regime_thresholds['volume_spike']:
                return 'bull_strong'
            else:
                return 'bull_weak'
        
        # Bear market regime
        elif price_change < self.regime_thresholds['momentum_bear']:
            if volume_ratio > self.regime_thresholds['volume_spike']:
                return 'bear_strong'
            else:
                return 'bear_weak'
        
        # Sideways market
        else:
            return 'sideways'
    
    def _calculate_regime_confidence(self, price_change: float, volume_ratio: float, 
                                   volatility: float, regime: str) -> float:
        """Calculate confidence in regime classification"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on signal strength
        if regime.startswith('bull') or regime.startswith('bear'):
            momentum_strength = abs(price_change)
            confidence += min(momentum_strength * 10, 0.4)
        
        if 'strong' in regime:
            confidence += 0.2
        
        if regime == 'volatile':
            vol_strength = volatility / self.regime_thresholds['volatility_high']
            confidence += min((vol_strength - 1) * 0.3, 0.3)
        
        return min(1.0, max(0.2, confidence))

class RegimeRouter:
    """Route predictions based on market regime"""
    
    def __init__(self):
        self.regime_adjustments = {
            'bull_strong': 1.2,
            'bull_weak': 1.1,
            'bear_strong': 0.7,
            'bear_weak': 0.8,
            'sideways': 0.9,
            'volatile': 0.6
        }
    
    def route_by_regime(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Adjust predictions based on regime"""
        
        routed_predictions = []
        
        for _, pred in predictions_df.iterrows():
            regime = pred.get('regime', 'sideways')
            adjustment = self.regime_adjustments.get(regime, 1.0)
            
            # Adjust expected return based on regime
            original_return = pred.get('expected_return_pct', 0)
            adjusted_return = original_return * adjustment
            
            # Adjust confidence based on regime certainty
            original_confidence = pred.get('confidence_24h', pred.get('confidence_1h', 70))
            regime_confidence = pred.get('regime_confidence', 0.5)
            adjusted_confidence = original_confidence * (0.8 + regime_confidence * 0.4)
            
            pred_dict = pred.to_dict()
            pred_dict['expected_return_pct'] = adjusted_return
            pred_dict['regime_adjusted_confidence'] = adjusted_confidence
            pred_dict['regime_adjustment_factor'] = adjustment
            pred_dict['regime_routed'] = True
            
            routed_predictions.append(pred_dict)
        
        result_df = pd.DataFrame(routed_predictions)
        logger.info(f"Applied regime routing to {len(result_df)} predictions")
        
        return result_df

def apply_regime_detection(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply regime detection and routing"""
    
    # Detect regimes
    detector = RegimeDetector()
    regime_df = detector.detect_regime(predictions_df)
    
    # Route by regime
    router = RegimeRouter()
    routed_df = router.route_by_regime(regime_df)
    
    return routed_df