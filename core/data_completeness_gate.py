#!/usr/bin/env python3
"""
Data Completeness Gate
Zero-tolerance validation for incomplete data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime

class DataCompletenessGate:
    """Strict data completeness validation"""
    
    def __init__(self, required_columns: List[str] = None):
        self.required_columns = required_columns or [
            'price', 'volume_24h', 'change_24h', 
            'sent_score', 'rsi_14', 'whale_score'
        ]
        self.rejection_log = []
    
    def validate_completeness(self, df: pd.DataFrame, 
                            min_completeness: float = 0.95) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate data completeness with zero tolerance"""
        
        validation_start = datetime.now()
        original_count = len(df)
        
        if df.empty:
            return df, {"status": "empty", "original_count": 0, "passed_count": 0}
        
        issues = []
        
        # Check required columns exist
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            return pd.DataFrame(), {
                "status": "failed",
                "issues": issues,
                "original_count": original_count,
                "passed_count": 0
            }
        
        # Check completeness per row
        required_data = df[self.required_columns]
        row_completeness = required_data.notna().sum(axis=1) / len(self.required_columns)
        
        # Apply strict filter
        complete_mask = row_completeness >= min_completeness
        filtered_df = df[complete_mask].copy()
        
        passed_count = len(filtered_df)
        rejection_count = original_count - passed_count
        
        # Log rejections
        if rejection_count > 0:
            self.rejection_log.append({
                "timestamp": validation_start.isoformat(),
                "rejected_count": rejection_count,
                "reason": f"Completeness below {min_completeness:.0%}"
            })
        
        # Additional validation checks
        for col in self.required_columns:
            if col in filtered_df.columns:
                # Check for placeholder values
                placeholder_mask = (
                    (filtered_df[col] == 0) |
                    (filtered_df[col] == -999) |
                    (filtered_df[col] == 999) |
                    (filtered_df[col] == -1)
                )
                
                placeholder_count = placeholder_mask.sum()
                if placeholder_count > len(filtered_df) * 0.1:  # >10% placeholders
                    issues.append(f"High placeholder count in {col}: {placeholder_count}")
        
        # Check for realistic value ranges
        if 'price' in filtered_df.columns:
            unrealistic_prices = ((filtered_df['price'] <= 0) | (filtered_df['price'] > 1000000)).sum()
            if unrealistic_prices > 0:
                issues.append(f"Unrealistic prices: {unrealistic_prices}")
        
        if 'volume_24h' in filtered_df.columns:
            zero_volume = (filtered_df['volume_24h'] <= 0).sum()
            if zero_volume > len(filtered_df) * 0.1:
                issues.append(f"High zero volume count: {zero_volume}")
        
        validation_result = {
            "status": "passed" if passed_count > 0 else "failed",
            "original_count": original_count,
            "passed_count": passed_count,
            "rejection_count": rejection_count,
            "rejection_rate": rejection_count / original_count if original_count > 0 else 0,
            "completeness_threshold": min_completeness,
            "issues": issues,
            "validation_duration_ms": (datetime.now() - validation_start).total_seconds() * 1000
        }
        
        return filtered_df, validation_result
    
    def get_rejection_summary(self) -> Dict[str, Any]:
        """Get summary of all rejections"""
        
        if not self.rejection_log:
            return {"total_rejections": 0}
        
        total_rejections = sum(entry["rejected_count"] for entry in self.rejection_log)
        
        return {
            "total_rejections": total_rejections,
            "rejection_events": len(self.rejection_log),
            "latest_rejection": self.rejection_log[-1] if self.rejection_log else None,
            "rejection_history": self.rejection_log[-10:]  # Last 10 events
        }

def create_zero_tolerance_pipeline():
    """Create zero-tolerance data pipeline"""
    
    def pipeline_step(df: pd.DataFrame, step_name: str) -> pd.DataFrame:
        """Pipeline step with completeness validation"""
        
        gate = DataCompletenessGate()
        validated_df, result = gate.validate_completeness(df)
        
        print(f"Pipeline step '{step_name}': {result['passed_count']}/{result['original_count']} passed")
        
        if result['issues']:
            print(f"  Issues: {result['issues']}")
        
        return validated_df
    
    return pipeline_step
