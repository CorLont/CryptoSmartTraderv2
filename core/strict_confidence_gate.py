#!/usr/bin/env python3
"""
Strict 80% Confidence Gate Manager
Enforces strict confidence threshold with empty state handling
"""

import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from core.structured_logger import get_structured_logger
except ImportError:
    try:
        from config.structured_logging import get_structured_logger
    except ImportError:
        import logging
        def get_structured_logger(name):
            return logging.getLogger(name)

class StrictConfidenceGate:
    """Strict confidence gate with 80% threshold enforcement"""
    
    def __init__(self, confidence_threshold: float = 0.80):
        self.confidence_threshold = confidence_threshold
        self.logger = get_structured_logger("StrictConfidenceGate")
        
        # Tracking
        self.gate_applications = []
        self.rejection_counts = {
            'low_confidence': 0,
            'missing_data': 0,
            'invalid_predictions': 0
        }
    
    def apply_gate(self, predictions_df: pd.DataFrame, 
                   gate_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply strict confidence gate to predictions"""
        
        if gate_id is None:
            gate_id = f"gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Applying strict confidence gate: {gate_id}")
            
            if predictions_df.empty:
                self.logger.warning(f"STRICT GATE: No input candidates for {gate_id}")
                return pd.DataFrame(), self._create_gate_report(gate_id, 0, 0, "no_input")
            
            original_count = len(predictions_df)
            
            # Validate required columns
            required_columns = ['coin', 'pred_7d', 'pred_30d']
            confidence_columns = [col for col in predictions_df.columns if col.startswith('conf_')]
            
            if not all(col in predictions_df.columns for col in required_columns):
                self.logger.error(f"Missing required columns in predictions")
                self.rejection_counts['missing_data'] += original_count
                return pd.DataFrame(), self._create_gate_report(gate_id, original_count, 0, "missing_columns")
            
            if not confidence_columns:
                self.logger.error(f"No confidence columns found")
                self.rejection_counts['missing_data'] += original_count
                return pd.DataFrame(), self._create_gate_report(gate_id, original_count, 0, "no_confidence")
            
            # Apply confidence filter - ALL confidence values must be >= threshold
            confidence_mask = pd.Series([True] * len(predictions_df), index=predictions_df.index)
            
            for conf_col in confidence_columns:
                if conf_col in predictions_df.columns:
                    conf_values = pd.to_numeric(predictions_df[conf_col], errors='coerce')
                    confidence_mask &= (conf_values >= self.confidence_threshold)
            
            # Additional validity checks
            validity_mask = pd.Series([True] * len(predictions_df), index=predictions_df.index)
            
            # Check for invalid predictions (NaN, inf, extremely large values)
            for pred_col in ['pred_7d', 'pred_30d']:
                if pred_col in predictions_df.columns:
                    pred_values = pd.to_numeric(predictions_df[pred_col], errors='coerce')
                    validity_mask &= (
                        pred_values.notna() & 
                        np.isfinite(pred_values) & 
                        (np.abs(pred_values) < 10.0)  # No predictions > 1000%
                    )
            
            # Combine all filters
            final_mask = confidence_mask & validity_mask
            
            # Apply filter
            filtered_df = predictions_df[final_mask].copy()
            passed_count = len(filtered_df)
            
            # Calculate rejection reasons
            low_confidence_count = len(predictions_df[~confidence_mask])
            invalid_predictions_count = len(predictions_df[~validity_mask])
            
            self.rejection_counts['low_confidence'] += low_confidence_count
            self.rejection_counts['invalid_predictions'] += invalid_predictions_count
            
            # Sort by pred_30d (descending) as specified
            if not filtered_df.empty and 'pred_30d' in filtered_df.columns:
                filtered_df = filtered_df.sort_values('pred_30d', ascending=False)
            
            # Create gate report - remove inconsistent explanations
            gate_report = self._create_gate_report(
                gate_id, original_count, passed_count, 
                "success" if passed_count > 0 else "no_candidates"
            )
            
            gate_report.update({
                'low_confidence_rejected': low_confidence_count,
                'invalid_predictions_rejected': invalid_predictions_count,
                'confidence_threshold': self.confidence_threshold,
                'processing_time': time.time() - start_time
            })
            
            # Log results
            if passed_count == 0:
                self.logger.warning(f"STRICT GATE CLOSED: {gate_id} - only {passed_count}/{original_count} candidates passed (minimum: 1)")
                self.logger.info("No reliable opportunities available - confidence threshold not met")
            else:
                self.logger.info(f"STRICT GATE PASSED: {gate_id} - {passed_count}/{original_count} candidates passed")
            
            # Track gate application
            self.gate_applications.append(gate_report)
            
            return filtered_df, gate_report
            
        except Exception as e:
            self.logger.error(f"Confidence gate application failed: {e}")
            return pd.DataFrame(), self._create_gate_report(gate_id, 0, 0, "error", str(e))
    
    def _create_gate_report(self, gate_id: str, input_count: int, 
                           output_count: int, status: str, 
                           error: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized gate report"""
        
        return {
            'gate_id': gate_id,
            'timestamp': datetime.now().isoformat(),
            'input_count': input_count,
            'output_count': output_count,
            'status': status,
            'confidence_threshold': self.confidence_threshold,
            'rejection_rate': (input_count - output_count) / max(input_count, 1),
            'error': error
        }
    
    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get comprehensive gate statistics"""
        
        recent_applications = self.gate_applications[-10:]  # Last 10 applications
        
        if not recent_applications:
            return {
                'total_applications': 0,
                'recent_pass_rate': 0.0,
                'average_candidates': 0,
                'rejection_counts': self.rejection_counts
            }
        
        total_input = sum(app['input_count'] for app in recent_applications)
        total_output = sum(app['output_count'] for app in recent_applications)
        
        return {
            'total_applications': len(self.gate_applications),
            'recent_applications': len(recent_applications),
            'recent_pass_rate': total_output / max(total_input, 1),
            'average_input_candidates': total_input / len(recent_applications),
            'average_output_candidates': total_output / len(recent_applications),
            'rejection_counts': self.rejection_counts.copy(),
            'confidence_threshold': self.confidence_threshold,
            'last_application': recent_applications[-1] if recent_applications else None
        }
    
    def log_empty_state(self, context: str = "dashboard"):
        """Log empty state with context"""
        
        self.logger.info(f"No reliable opportunities available in {context} - strict confidence gate active")
        
        # Create empty state log entry
        empty_state_log = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'confidence_threshold': self.confidence_threshold,
            'message': 'no reliable opportunities',
            'gate_statistics': self.get_gate_statistics()
        }
        
        # Save to daily logs
        self._save_empty_state_log(empty_state_log)
    
    def _save_empty_state_log(self, log_entry: Dict[str, Any]):
        """Save empty state log to daily logs"""
        
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            daily_log_dir = Path("logs/daily") / today_str
            daily_log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp_str = datetime.now().strftime("%H%M%S")
            log_file = daily_log_dir / f"empty_state_log_{timestamp_str}.json"
            
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save empty state log: {e}")

# Global gate instance
_strict_confidence_gate: Optional[StrictConfidenceGate] = None

def get_strict_confidence_gate(threshold: float = 0.80) -> StrictConfidenceGate:
    """Get global strict confidence gate instance"""
    global _strict_confidence_gate
    
    if _strict_confidence_gate is None:
        _strict_confidence_gate = StrictConfidenceGate(threshold)
    
    return _strict_confidence_gate

def apply_strict_confidence_filter(predictions_df: pd.DataFrame, 
                                  threshold: float = 0.80,
                                  gate_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply strict confidence filter to predictions"""
    gate = get_strict_confidence_gate(threshold)
    return gate.apply_gate(predictions_df, gate_id)

def log_no_opportunities(context: str = "system"):
    """Log no reliable opportunities message"""
    gate = get_strict_confidence_gate()
    gate.log_empty_state(context)

if __name__ == "__main__":
    # Test strict confidence gate
    print("Testing Strict Confidence Gate")
    
    # Create test data with various confidence levels
    test_data = pd.DataFrame({
        'coin': ['BTC', 'ETH', 'ADA', 'SOL', 'DOT'],
        'pred_7d': [0.15, 0.08, 0.25, 0.12, 0.05],
        'pred_30d': [0.35, 0.20, 0.45, 0.28, 0.15],
        'conf_7d': [0.85, 0.75, 0.90, 0.82, 0.65],  # Two below threshold
        'conf_30d': [0.82, 0.78, 0.88, 0.85, 0.70],  # Three below threshold
        'regime': ['bull', 'sideways', 'bull', 'bull', 'bear']
    })
    
    print(f"Input data: {len(test_data)} candidates")
    print(test_data)
    
    # Apply gate
    filtered_data, report = apply_strict_confidence_filter(test_data)
    
    print(f"\nFiltered data: {len(filtered_data)} candidates")
    print(filtered_data)
    print(f"\nGate report: {report}")
    
    # Test empty case
    empty_filtered, empty_report = apply_strict_confidence_filter(pd.DataFrame())
    print(f"\nEmpty case: {len(empty_filtered)} candidates, status: {empty_report['status']}")