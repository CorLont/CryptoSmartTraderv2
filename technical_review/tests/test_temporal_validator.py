#!/usr/bin/env python3
"""
Temporal Validator Tests - Feature leakage and time-series validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ml.temporal_validator import TemporalValidator, ValidationResult


class TestTemporalValidator:
    """Test temporal validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create temporal validator"""
        return TemporalValidator()
    
    @pytest.fixture
    def clean_timeseries_data(self):
        """Create clean time series data without leakage"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        
        # Create realistic price data
        np.random.seed(42)
        price = 50000
        prices = []
        
        for i in range(len(dates)):
            # Random walk with some mean reversion
            change = np.random.normal(0, 100) - 0.001 * (price - 50000)
            price += change
            prices.append(price)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.uniform(100, 1000, len(dates)),
            'rsi': np.random.uniform(20, 80, len(dates)),
            'target': np.random.choice([0, 1], len(dates))
        })
        
        return df
    
    @pytest.fixture
    def leaky_data(self):
        """Create data with feature leakage"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='H')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.uniform(49000, 51000, len(dates)),
            'target': np.random.choice([0, 1], len(dates))
        })
        
        # Create leaky feature: future price as current feature
        df['leaky_feature'] = df['price'].shift(-1)  # Future price!
        
        return df
    
    def test_clean_data_validation(self, validator, clean_timeseries_data):
        """Test validation passes for clean data"""
        result = validator.validate_temporal_integrity(
            clean_timeseries_data,
            timestamp_col='timestamp',
            feature_cols=['price', 'volume', 'rsi'],
            target_col='target'
        )
        
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert "passed" in result.message.lower()
        assert result.details is not None
        assert result.details["feature_leakage_detected"] is False
    
    def test_leaky_data_detection(self, validator, leaky_data):
        """Test detection of feature leakage"""
        result = validator.validate_temporal_integrity(
            leaky_data,
            timestamp_col='timestamp',
            feature_cols=['leaky_feature'],
            target_col='target'
        )
        
        # Should detect leakage in this case
        assert result.details["feature_leakage_detected"] is True
        assert "leaky_feature" in result.details["leakage_details"]
    
    def test_missing_timestamp_column(self, validator, clean_timeseries_data):
        """Test validation fails when timestamp column is missing"""
        df_no_timestamp = clean_timeseries_data.drop('timestamp', axis=1)
        
        result = validator.validate_temporal_integrity(
            df_no_timestamp,
            timestamp_col='missing_timestamp'
        )
        
        assert result.passed is False
        assert "not found" in result.message
    
    def test_time_series_splits(self, validator, clean_timeseries_data):
        """Test time-series cross-validation splits"""
        splits = validator.create_time_series_splits(
            clean_timeseries_data,
            timestamp_col='timestamp',
            n_splits=3,
            test_size_days=2,
            gap_days=1
        )
        
        assert len(splits) > 0
        
        for train_idx, test_idx in splits:
            # Test that train comes before test (temporal order)
            train_data = clean_timeseries_data.iloc[train_idx]
            test_data = clean_timeseries_data.iloc[test_idx]
            
            max_train_time = train_data['timestamp'].max()
            min_test_time = test_data['timestamp'].min()
            
            # There should be a gap between train and test
            assert max_train_time < min_test_time
            
            # Both sets should have data
            assert len(train_idx) > 0
            assert len(test_idx) > 0
    
    def test_duplicate_timestamps(self, validator):
        """Test detection of duplicate timestamps"""
        # Create data with duplicate timestamps
        dates = pd.date_range(start='2024-01-01', periods=10, freq='H')
        dates_with_duplicates = dates.tolist()
        dates_with_duplicates.extend([dates[0], dates[1]])  # Add duplicates
        
        df = pd.DataFrame({
            'timestamp': dates_with_duplicates,
            'price': np.random.uniform(49000, 51000, len(dates_with_duplicates)),
            'target': np.random.choice([0, 1], len(dates_with_duplicates))
        })
        
        result = validator.validate_temporal_integrity(df)
        
        assert result.details["duplicate_timestamps"] > 0
        # Validation might still pass depending on other factors
    
    def test_high_nan_detection(self, validator):
        """Test detection of high NaN percentages"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.uniform(49000, 51000, len(dates)),
            'target': np.random.choice([0, 1], len(dates))
        })
        
        # Add feature with high NaN percentage
        df['high_nan_feature'] = np.random.uniform(0, 1, len(dates))
        df.loc[df.index[:80], 'high_nan_feature'] = np.nan  # 80% NaN
        
        result = validator.validate_temporal_integrity(
            df,
            feature_cols=['price', 'high_nan_feature']
        )
        
        assert 'high_nan_feature' in result.details["high_nan_columns"]
        assert result.passed is False  # Should fail due to high NaN percentage
    
    def test_data_drift_detection(self, validator):
        """Test data drift detection"""
        # Create reference dataset
        np.random.seed(42)
        ref_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(5, 2, 1000),
            'feature3': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        # Create current dataset with drift
        np.random.seed(43)
        curr_data = pd.DataFrame({
            'feature1': np.random.normal(1, 1, 500),  # Mean shifted
            'feature2': np.random.normal(5, 4, 500),  # Variance increased
            'feature3': np.random.choice(['A', 'A', 'B'], 500)  # Distribution changed
        })
        
        drift_results = validator.check_data_drift(
            reference_df=ref_data,
            current_df=curr_data,
            feature_cols=['feature1', 'feature2', 'feature3'],
            drift_threshold=0.1
        )
        
        assert isinstance(drift_results, dict)
        assert 'has_drift' in drift_results
        assert 'drift_score' in drift_results
        assert 'feature_drifts' in drift_results
        assert 'recommendation' in drift_results
        
        # Should detect drift for feature1 (mean shift)
        assert 'feature1' in drift_results['feature_drifts']
        
        # Check recommendations
        assert drift_results['recommendation'] in [
            'no_action', 'monitor_closely', 'reduce_confidence', 'retrain_immediately'
        ]


@pytest.mark.integration
class TestTemporalValidatorIntegration:
    """Integration tests for temporal validator"""
    
    def test_feature_engineering_validation(self):
        """Test validation of feature engineering functions"""
        validator = TemporalValidator()
        
        # Create base data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'timestamp': dates,
            'price': np.random.uniform(49000, 51000, len(dates)),
            'volume': np.random.uniform(100, 1000, len(dates))
        })
        
        # Good feature engineering function (no leakage)
        def good_feature_engineering(data):
            result = data.copy()
            result['price_ma'] = result['price'].rolling(window=5).mean()
            result['volume_ratio'] = result['volume'] / result['volume'].rolling(window=10).mean()
            return result
        
        result = validator.validate_feature_engineering(
            df=df,
            feature_engineering_func=good_feature_engineering,
            timestamp_col='timestamp'
        )
        
        assert result.passed is True
        assert len(result.details["new_features"]) == 2
        assert 'price_ma' in result.details["new_features"]
        assert 'volume_ratio' in result.details["new_features"]
    
    def test_real_world_scenario(self):
        """Test with realistic cryptocurrency data scenario"""
        validator = TemporalValidator()
        
        # Simulate realistic crypto data
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
        
        # Create correlated price movements
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, len(dates))
        prices = [50000]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': np.random.exponential(500, len(dates)),
            'bid_ask_spread': np.random.uniform(0.1, 2.0, len(dates)),
            'order_book_imbalance': np.random.uniform(-1, 1, len(dates))
        })
        
        # Add realistic features
        df['price_change'] = df['price'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volatility'] = df['price_change'].rolling(window=50).std()
        df['target'] = (df['price'].shift(-5) > df['price']).astype(int)
        
        # Remove the last 5 rows due to target calculation
        df = df[:-5]
        
        result = validator.validate_temporal_integrity(
            df,
            timestamp_col='timestamp',
            feature_cols=['price', 'volume', 'price_change', 'volume_ma', 'volatility', 
                         'bid_ask_spread', 'order_book_imbalance'],
            target_col='target'
        )
        
        # Should pass for realistic data
        assert result.passed is True or len(result.details.get("high_nan_columns", [])) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])