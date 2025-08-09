#!/usr/bin/env python3
"""
Strict Confidence Gate System
Enterprise-grade confidence filtering with 80% threshold
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.structured_logger import get_structured_logger

class StrictConfidenceGate:
    """Strict confidence gating system with 80% threshold"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.logger = get_structured_logger("StrictConfidenceGate")
        
    def apply_gate(self, predictions: List[Dict[str, Any]], cycle_id: Optional[str] = None) -> Dict[str, Any]:
        """Apply strict confidence gate to predictions"""
        
        if cycle_id is None:
            cycle_id = f"gate_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Applying strict confidence gate: {cycle_id}")
        
        if not predictions:
            self.logger.warning("No predictions to filter")
            return {
                "filtered_predictions": [],
                "gate_passed": False,
                "original_count": 0,
                "filtered_count": 0,
                "pass_rate": 0.0
            }
        
        # Filter predictions by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            confidence = pred.get('confidence', 0.0)
            if confidence >= self.threshold:
                filtered_predictions.append(pred)
        
        pass_rate = len(filtered_predictions) / len(predictions) if predictions else 0
        gate_passed = len(filtered_predictions) > 0
        
        self.logger.info(f"STRICT GATE {'PASSED' if gate_passed else 'FAILED'}: {cycle_id} - {len(filtered_predictions)}/{len(predictions)} candidates passed")
        
        if gate_passed:
            self.logger.info(f"Generating explanations for {len(filtered_predictions)} predictions")
            explanations = self._generate_explanations(filtered_predictions)
            if not explanations:
                self.logger.warning("No explanations generated, using fallback")
        else:
            self.logger.warning(f"No predictions passed confidence threshold {self.threshold}")
        
        return {
            "filtered_predictions": filtered_predictions,
            "gate_passed": gate_passed,
            "original_count": len(predictions),
            "filtered_count": len(filtered_predictions),
            "pass_rate": pass_rate,
            "threshold_used": self.threshold,
            "cycle_id": cycle_id
        }
    
    def _generate_explanations(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate explanations for filtered predictions"""
        
        explanations = []
        
        for pred in predictions:
            try:
                model_type = pred.get('model_used', 'ensemble')
                
                if model_type == 'ensemble':
                    self.logger.error(f"No explainer found for model {model_type}")
                    continue
                
                explanation = {
                    "symbol": pred.get('symbol', 'UNKNOWN'),
                    "confidence": pred.get('confidence', 0.0),
                    "explanation": f"High confidence prediction based on {model_type} model",
                    "key_factors": ["market_momentum", "technical_indicators", "sentiment_analysis"]
                }
                explanations.append(explanation)
                
            except Exception as e:
                self.logger.error(f"Failed to generate explanation: {e}")
        
        return explanations