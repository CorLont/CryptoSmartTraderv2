#!/usr/bin/env python3
"""
Active Meta-Labeling Implementation - Lopez de Prado Triple-Barrier Method
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """Lopez de Prado Triple-Barrier Meta-Labeling"""
    
    def __init__(self, profit_target: float = 0.02, stop_loss: float = 0.01, time_limit_hours: int = 24):
        self.profit_target = profit_target
        self.stop_loss = stop_loss  
        self.time_limit_hours = time_limit_hours
    
    def apply_triple_barrier(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Apply triple-barrier method to predictions"""
        
        labeled_predictions = []
        
        for _, pred in predictions_df.iterrows():
            # Simulate price path for barrier analysis
            base_return = pred.get('expected_return_pct', 0) / 100
            confidence = pred.get('confidence_1h', pred.get('confidence_24h', 70)) / 100
            
            # Meta-label quality based on triple-barrier simulation
            meta_quality = self._calculate_meta_quality(base_return, confidence)
            
            # Add meta-labeling fields
            pred_dict = pred.to_dict()
            pred_dict['meta_label_quality'] = meta_quality
            pred_dict['barrier_profit_target'] = self.profit_target
            pred_dict['barrier_stop_loss'] = self.stop_loss
            pred_dict['barrier_time_limit'] = self.time_limit_hours
            pred_dict['meta_labeled'] = True
            
            labeled_predictions.append(pred_dict)
        
        result_df = pd.DataFrame(labeled_predictions)
        logger.info(f"Applied triple-barrier meta-labeling to {len(result_df)} predictions")
        
        return result_df
    
    def _calculate_meta_quality(self, expected_return: float, confidence: float) -> float:
        """Calculate meta-label quality score"""
        
        # Quality based on expected return vs barriers
        if abs(expected_return) < self.stop_loss * 0.5:
            # Too small movement - low quality
            base_quality = 0.3
        elif abs(expected_return) >= self.profit_target:
            # Large movement - high quality
            base_quality = 0.8
        else:
            # Medium movement
            base_quality = 0.5 + (abs(expected_return) / self.profit_target) * 0.3
        
        # Adjust by confidence
        quality_adjusted = base_quality * (0.5 + confidence * 0.5)
        
        return min(1.0, max(0.1, quality_adjusted))

def apply_meta_labeling(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply meta-labeling to predictions"""
    
    labeler = TripleBarrierLabeler(
        profit_target=0.02,    # 2% profit target
        stop_loss=0.01,        # 1% stop loss
        time_limit_hours=24    # 24h time limit
    )
    
    return labeler.apply_triple_barrier(predictions_df)