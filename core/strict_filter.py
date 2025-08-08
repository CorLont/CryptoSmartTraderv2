#!/usr/bin/env python3
"""
Strict Filter - Kill dummy data in production
Zero tolerance for incomplete or synthetic data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StrictProductionFilter:
    """
    Ultra-strict filter that eliminates any dummy/incomplete data in production
    """
    
    def __init__(self, min_completeness: float = 0.8):
        self.min_completeness = min_completeness
        self.required_features = [
            'price', 'volume_24h', 'market_cap',
            'technical_rsi', 'technical_macd', 'technical_bb_position',
            'sentiment_score', 'whale_activity_score', 'volume_momentum'
        ]
        self.blocked_coins = set()
        self.stats = {
            'total_processed': 0,
            'blocked_incomplete': 0,
            'blocked_dummy': 0,
            'passed_filter': 0
        }
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality with zero tolerance"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'completeness_score': 0.0,
            'blocked_reasons': []
        }
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append('Empty dataframe')
            return validation_results
        
        # Check for required columns
        missing_cols = [col for col in self.required_features if col not in df.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f'Missing columns: {missing_cols}')
            validation_results['blocked_reasons'].append('missing_features')
        
        # Check completeness for each row
        if not missing_cols:
            completeness_mask = df[self.required_features].notna().all(axis=1)
            completeness_score = completeness_mask.mean()
            validation_results['completeness_score'] = completeness_score
            
            if completeness_score < self.min_completeness:
                validation_results['is_valid'] = False
                validation_results['issues'].append(f'Completeness {completeness_score:.2%} < {self.min_completeness:.0%}')
                validation_results['blocked_reasons'].append('low_completeness')
        
        # Detect dummy/placeholder patterns
        dummy_patterns = self._detect_dummy_patterns(df)
        if dummy_patterns:
            validation_results['is_valid'] = False
            validation_results['issues'].extend(dummy_patterns)
            validation_results['blocked_reasons'].append('dummy_data')
        
        # Check for realistic value ranges
        range_issues = self._validate_value_ranges(df)
        if range_issues:
            validation_results['is_valid'] = False
            validation_results['issues'].extend(range_issues)
            validation_results['blocked_reasons'].append('unrealistic_values')
        
        return validation_results
    
    def _detect_dummy_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detect dummy/placeholder data patterns"""
        
        issues = []
        
        # Check for repeated identical values (common in dummy data)
        for col in self.required_features:
            if col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1 and len(df) > 10:  # Less than 10% unique values
                    issues.append(f'{col}: Too many repeated values ({unique_ratio:.1%} unique)')
        
        # Check for obvious placeholder values
        placeholder_checks = {
            'price': [0, 1, 100, 1000],  # Common placeholder prices
            'volume_24h': [0, 1000000],  # Round numbers
            'market_cap': [0, 1000000000],
            'sentiment_score': [0, 0.5, 1],  # Exact neutral values
        }
        
        for col, placeholders in placeholder_checks.items():
            if col in df.columns:
                for placeholder in placeholders:
                    if (df[col] == placeholder).sum() > len(df) * 0.5:  # More than 50%
                        issues.append(f'{col}: Excessive placeholder values ({placeholder})')
        
        # Check for sequential/generated patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns and len(df) > 3:
                diffs = df[col].diff().dropna()
                if len(diffs) > 0 and (diffs == diffs.iloc[0]).sum() > len(diffs) * 0.8:
                    issues.append(f'{col}: Sequential pattern detected (likely generated)')
        
        return issues
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> List[str]:
        """Validate that values are in realistic ranges"""
        
        issues = []
        
        # Define realistic ranges for crypto data
        range_checks = {
            'price': (0.000001, 1000000),  # $0.000001 to $1M
            'volume_24h': (0, 1e12),       # Up to $1T daily volume
            'market_cap': (0, 1e13),       # Up to $10T market cap
            'technical_rsi': (0, 100),     # RSI 0-100
            'sentiment_score': (-1, 1),    # Sentiment -1 to 1
            'whale_activity_score': (0, 1) # Activity 0-1
        }
        
        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    count = out_of_range.sum()
                    issues.append(f'{col}: {count} values out of range [{min_val}, {max_val}]')
        
        return issues
    
    def apply_strict_filter(self, df: pd.DataFrame, coin_column: str = 'coin') -> pd.DataFrame:
        """Apply strict filter and return only valid data"""
        
        if df.empty:
            return df
        
        self.stats['total_processed'] += len(df)
        
        # Apply filter row by row for per-coin validation
        valid_rows = []
        
        if coin_column in df.columns:
            # Group by coin and validate each group
            for coin, group in df.groupby(coin_column):
                validation = self.validate_data_quality(group)
                
                if validation['is_valid']:
                    valid_rows.append(group)
                    self.stats['passed_filter'] += len(group)
                else:
                    # Block this coin
                    self.blocked_coins.add(coin)
                    
                    if 'dummy_data' in validation['blocked_reasons']:
                        self.stats['blocked_dummy'] += len(group)
                    if 'low_completeness' in validation['blocked_reasons']:
                        self.stats['blocked_incomplete'] += len(group)
                    
                    print(f"BLOCKED {coin}: {', '.join(validation['issues'])}")
        else:
            # Validate entire dataframe if no coin column
            validation = self.validate_data_quality(df)
            
            if validation['is_valid']:
                valid_rows.append(df)
                self.stats['passed_filter'] += len(df)
            else:
                self.stats['blocked_incomplete'] += len(df)
                print(f"BLOCKED entire dataset: {', '.join(validation['issues'])}")
        
        # Combine valid rows
        if valid_rows:
            filtered_df = pd.concat(valid_rows, ignore_index=True)
        else:
            filtered_df = pd.DataFrame()
        
        return filtered_df
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        
        pass_rate = (self.stats['passed_filter'] / max(1, self.stats['total_processed'])) * 100
        
        return {
            'total_processed': self.stats['total_processed'],
            'passed_filter': self.stats['passed_filter'],
            'blocked_incomplete': self.stats['blocked_incomplete'],
            'blocked_dummy': self.stats['blocked_dummy'],
            'blocked_coins': list(self.blocked_coins),
            'pass_rate_percent': pass_rate,
            'strict_filter_active': True
        }
    
    def reset_stats(self):
        """Reset filtering statistics"""
        self.stats = {
            'total_processed': 0,
            'blocked_incomplete': 0,
            'blocked_dummy': 0,
            'passed_filter': 0
        }
        self.blocked_coins.clear()

def apply_production_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict production filter to any dataframe"""
    
    strict_filter = StrictProductionFilter()
    filtered_df = strict_filter.apply_strict_filter(df)
    
    stats = strict_filter.get_filter_stats()
    print(f"Strict Filter Results: {stats['passed_filter']}/{stats['total_processed']} passed ({stats['pass_rate_percent']:.1f}%)")
    
    return filtered_df

if __name__ == "__main__":
    print("🚫 TESTING STRICT PRODUCTION FILTER")
    print("=" * 40)
    
    # Create test data with mixed quality
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'coin': ['BTC', 'ETH', 'FAKE1', 'SOL', 'FAKE2'] * 20,
        'price': np.random.uniform(0.1, 70000, 100),
        'volume_24h': np.random.uniform(1000000, 1e10, 100),
        'market_cap': np.random.uniform(1e8, 1e12, 100),
        'technical_rsi': np.random.uniform(20, 80, 100),
        'technical_macd': np.random.uniform(-10, 10, 100),
        'technical_bb_position': np.random.uniform(0, 1, 100),
        'sentiment_score': np.random.uniform(-0.5, 0.5, 100),
        'whale_activity_score': np.random.uniform(0, 1, 100),
        'volume_momentum': np.random.uniform(-0.2, 0.2, 100)
    })
    
    # Inject dummy data for FAKE coins
    fake_mask = test_data['coin'].str.contains('FAKE')
    test_data.loc[fake_mask, 'price'] = 1.0  # Placeholder price
    test_data.loc[fake_mask, 'sentiment_score'] = 0.0  # Neutral placeholder
    test_data.loc[fake_mask, 'volume_24h'] = 1000000  # Round number
    
    # Inject missing data
    test_data.loc[test_data['coin'] == 'FAKE2', 'technical_rsi'] = np.nan
    
    print(f"Test data created: {len(test_data)} rows, {test_data['coin'].nunique()} coins")
    
    # Apply strict filter
    filtered_data = apply_production_filter(test_data)
    
    print(f"Filtered data: {len(filtered_data)} rows, {filtered_data['coin'].nunique() if not filtered_data.empty else 0} coins")
    
    if not filtered_data.empty:
        print(f"Remaining coins: {sorted(filtered_data['coin'].unique())}")
    
    print("✅ Strict production filter testing completed")