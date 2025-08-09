#!/usr/bin/env python3
"""
Data Completeness Gate - Zero-tolerance policy for incomplete data
"""

import pandas as pd
from typing import Dict, Any

class DataCompletenessGate:
    """Enforce strict data completeness requirements"""
    
    def __init__(self, min_completeness: float = 0.95):
        self.min_completeness = min_completeness
    
    def validate_data_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data meets completeness requirements"""
        
        if data.empty:
            return {
                "passed": False,
                "reason": "Empty dataset",
                "coverage_percentage": 0.0
            }
        
        # Calculate completeness
        total_cells = data.size
        missing_cells = data.isna().sum().sum()
        completeness = (total_cells - missing_cells) / total_cells
        
        passed = completeness >= self.min_completeness
        
        return {
            "passed": passed,
            "coverage_percentage": completeness * 100,
            "missing_cells": missing_cells,
            "total_cells": total_cells,
            "reason": f"Data completeness {completeness:.1%} < {self.min_completeness:.1%}" if not passed else "OK"
        }