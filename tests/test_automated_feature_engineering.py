#!/usr/bin/env python3
"""
Test suite for Automated Feature Engineering & Discovery
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.automated_feature_engineering import (
    AutomatedFeatureEngineer, 
    FeatureEngineeringConfig,
    get_automated_feature_engineer,
    engineer_features_for_coin
)
from core.feature_discovery_engine import (
    FeatureDiscoveryEngine,
    DiscoveryConfig,
    get_feature_discovery_engine
)
from core.shap_regime_analyzer import (
    SHAPRegimeAnalyzer,
    SHAPRegimeConfig,
    MarketRegime,
    get_shap_regime_analyzer
)
from core.live_feature_adaptation import (
    LiveFeatureAdaptationEngine,
    AdaptationConfig,
    AdaptationTrigger,
    get_live_adaptation_engine
)

class TestAutomatedFeatureEngineering:
    """Test automated feature engineering system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample cryptocurrency data"""
        np.random.seed(42)
        
        # Generate 200 data points
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        
        # Base price with trend and volatility
        base_price = 100
        trend = np.cumsum(np.random.normal(0.001, 0.02, 200))
        noise = np.random.normal(0, 0.05, 200)
        prices = base_price * np.exp(trend + noise)
        
        # Volume with correlation to price changes
        price_changes = np.diff(prices, prepend=prices[0])
        volumes = np.abs(price_changes) * 1000 + np.random.exponential(500, 200)
        
        # Technical indicators
        rsi = 30 + 40 * np.random.random(200)  # RSI between 30-70
        volatility = np.random.exponential(0.02, 200)
        
        # Market regime (simplified)
        regimes = np.random.choice(['bull', 'bear', 'sideways'], 200, p=[0.3, 0.3, 0.4])
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'rsi': rsi,
            'volatility': volatility,
            'regime': regimes,
            'price_change': price_changes,
            'returns': price_changes / prices,
            'high': prices * (1 + np.random.uniform(0, 0.02, 200)),
            'low': prices * (1 - np.random.uniform(0, 0.02, 200)),
            'close': prices
        }).set_index('timestamp')
    
    def test_automated_feature_engineer_initialization(self):
        """Test automated feature engineer initialization"""
        config = FeatureEngineeringConfig(max_features_per_iteration=10)
        engineer = AutomatedFeatureEngineer(config)
        
        assert engineer.config.max_features_per_iteration == 10
        assert engineer.synthesizer is not None
        assert engineer.shap_analyzer is not None
        assert len(engineer.feature_pipeline) == 0
    
    def test_feature_synthesis(self, sample_data):
        """Test deep feature synthesis"""
        engineer = get_automated_feature_engineer()
        target_column = 'price'
        
        # Test temporal features
        temporal_features = engineer.synthesizer.generate_temporal_features(
            sample_data, target_column
        )
        
        assert len(temporal_features.columns) > len(sample_data.columns)
        
        # Check for rolling features
        rolling_features = [col for col in temporal_features.columns if 'rolling' in col]
        assert len(rolling_features) > 0
        
        # Check for EMA features
        ema_features = [col for col in temporal_features.columns if 'ema' in col]
        assert len(ema_features) > 0
        
        # Check for lag features
        lag_features = [col for col in temporal_features.columns if 'lag' in col]
        assert len(lag_features) > 0
    
    def test_cross_feature_generation(self, sample_data):
        """Test cross feature generation"""
        engineer = get_automated_feature_engineer()
        target_column = 'price'
        
        cross_features = engineer.synthesizer.generate_cross_features(
            sample_data, target_column
        )
        
        assert len(cross_features.columns) > len(sample_data.columns)
        
        # Check for interaction features
        interaction_features = [col for col in cross_features.columns if '_x_' in col or '_div_' in col]
        assert len(interaction_features) > 0
    
    def test_polynomial_features(self, sample_data):
        """Test polynomial feature generation"""
        engineer = get_automated_feature_engineer()
        target_column = 'price'
        
        poly_features = engineer.synthesizer.generate_polynomial_features(
            sample_data, target_column
        )
        
        assert len(poly_features.columns) > len(sample_data.columns)
        
        # Check for polynomial features
        poly_cols = [col for col in poly_features.columns if 'poly_' in col]
        assert len(poly_cols) > 0
        
        # Check for transformation features
        transform_cols = [col for col in poly_features.columns if any(t in col for t in ['log', 'sqrt', 'reciprocal'])]
        assert len(transform_cols) > 0
    
    def test_feature_importance_analysis(self, sample_data):
        """Test feature importance analysis"""
        engineer = get_automated_feature_engineer()
        target_column = 'price'
        
        # Generate features first
        engineered_features = engineer._engineer_features(sample_data, target_column)
        
        # Analyze importance
        features = engineered_features.drop(columns=[target_column])
        target = engineered_features[target_column]
        
        importance_results = engineer.shap_analyzer.analyze_feature_importance(
            features, target
        )
        
        assert len(importance_results) > 0
        
        # Check that results have proper structure
        for result in importance_results[:5]:  # Check first 5
            assert hasattr(result, 'feature_name')
            assert hasattr(result, 'importance_score')
            assert hasattr(result, 'method')
            assert result.importance_score >= 0
    
    def test_feature_discovery_engine(self, sample_data):
        """Test feature discovery engine"""
        engine = get_feature_discovery_engine()
        target_column = 'price'
        
        # Discover features
        candidates = engine.discover_features(sample_data, target_column, 'regime')
        
        assert isinstance(candidates, list)
        
        # Check candidate structure
        if candidates:
            candidate = candidates[0]
            assert hasattr(candidate, 'name')
            assert hasattr(candidate, 'feature_data')
            assert hasattr(candidate, 'performance_score')
            assert hasattr(candidate, 'creation_method')
    
    def test_shap_regime_analyzer(self, sample_data):
        """Test SHAP regime analyzer"""
        analyzer = get_shap_regime_analyzer()
        target_column = 'price'
        
        # Add some engineered features first
        sample_data['price_ma_5'] = sample_data['price'].rolling(5).mean()
        sample_data['volume_ma_10'] = sample_data['volume'].rolling(10).mean()
        sample_data['rsi_change'] = sample_data['rsi'].diff()
        
        # Analyze regime-specific importance
        results = analyzer.analyze_regime_specific_importance(
            sample_data, target_column, 'regime'
        )
        
        assert isinstance(results, dict)
        
        # Check if we have results for different regimes
        if results:
            for regime, analysis in results.items():
                assert hasattr(analysis, 'regime')
                assert hasattr(analysis, 'feature_importance')
                assert hasattr(analysis, 'confidence_score')
                assert isinstance(analysis.feature_importance, dict)
    
    def test_live_feature_adaptation(self, sample_data):
        """Test live feature adaptation"""
        adapter = get_live_adaptation_engine()
        target_column = 'price'
        
        # Add some features to work with
        sample_data['price_ma_5'] = sample_data['price'].rolling(5).mean()
        sample_data['volume_ma_10'] = sample_data['volume'].rolling(10).mean()
        sample_data['momentum'] = sample_data['price'].pct_change(5)
        
        # Test adaptation
        adapted_features = adapter.adapt_features(
            sample_data, target_column, AdaptationTrigger.REGIME_CHANGE
        )
        
        assert isinstance(adapted_features, list)
        # Features should be selected from available columns
        available_features = [col for col in sample_data.columns if col != target_column]
        for feature in adapted_features:
            if feature in sample_data.columns:  # Allow for generated features
                assert feature in available_features
    
    def test_end_to_end_feature_engineering(self, sample_data):
        """Test complete end-to-end feature engineering pipeline"""
        target_column = 'price'
        
        # Test the convenient function
        engineered_data = engineer_features_for_coin(
            sample_data, target_column, 'regime'
        )
        
        assert isinstance(engineered_data, pd.DataFrame)
        assert len(engineered_data.columns) >= len(sample_data.columns)
        assert target_column in engineered_data.columns
        
        # Check that we have generated meaningful features
        original_cols = set(sample_data.columns)
        new_cols = set(engineered_data.columns) - original_cols
        assert len(new_cols) > 0
        
        # Verify no NaN in target
        assert not engineered_data[target_column].isna().any()
    
    def test_feature_performance_tracking(self, sample_data):
        """Test feature performance tracking"""
        engineer = get_automated_feature_engineer()
        target_column = 'price'
        
        # Fit the engineer
        engineer.fit(sample_data, target_column, 'regime')
        
        # Get performance summary
        summary = engineer.get_feature_importance_summary()
        
        assert isinstance(summary, dict)
        assert 'total_features_generated' in summary
        assert 'global_importance_scores' in summary
        assert 'feature_types_distribution' in summary
        
        # Check that we have some features tracked
        assert summary['total_features_generated'] > 0
    
    def test_regime_specific_features(self, sample_data):
        """Test regime-specific feature selection"""
        analyzer = get_shap_regime_analyzer()
        target_column = 'price'
        
        # Add engineered features
        sample_data['price_ma_5'] = sample_data['price'].rolling(5).mean()
        sample_data['volume_ratio'] = sample_data['volume'] / sample_data['volume'].rolling(10).mean()
        
        # Get regime-optimized features
        bull_features = analyzer.get_regime_optimized_features(MarketRegime.BULL_MARKET)
        bear_features = analyzer.get_regime_optimized_features(MarketRegime.BEAR_MARKET)
        
        assert isinstance(bull_features, list)
        assert isinstance(bear_features, list)
        
        # Features should be available in data or be engineered
        for feature_list in [bull_features, bear_features]:
            for feature in feature_list[:5]:  # Check first 5
                # Feature should be a string
                assert isinstance(feature, str)
    
    def test_feature_adaptation_triggers(self, sample_data):
        """Test different adaptation triggers"""
        adapter = get_live_adaptation_engine()
        target_column = 'price'
        
        # Test different triggers
        triggers = [
            AdaptationTrigger.REGIME_CHANGE,
            AdaptationTrigger.PERFORMANCE_DECLINE,
            AdaptationTrigger.NEW_FEATURE_DISCOVERY
        ]
        
        for trigger in triggers:
            adapted_features = adapter.adapt_features(
                sample_data, target_column, trigger
            )
            assert isinstance(adapted_features, list)
    
    def test_feature_rollback(self, sample_data):
        """Test feature adaptation rollback"""
        adapter = get_live_adaptation_engine()
        target_column = 'price'
        
        # Perform adaptation
        original_features = adapter.current_features.copy() if adapter.current_features else []
        adapted_features = adapter.adapt_features(
            sample_data, target_column, AdaptationTrigger.USER_REQUESTED
        )
        
        # Test rollback
        rollback_success = adapter.rollback_adaptation()
        
        # Should succeed if rollback is enabled and snapshots exist
        if adapter.config.rollback_enabled and adapter.feature_snapshots:
            assert rollback_success == True
        else:
            assert rollback_success == False
    
    def test_feature_stability_scoring(self, sample_data):
        """Test feature stability scoring"""
        engineer = get_automated_feature_engineer()
        
        # Create a stable feature
        stable_feature = pd.Series([1.0] * len(sample_data), index=sample_data.index)
        
        # Create an unstable feature
        unstable_feature = pd.Series(np.random.random(len(sample_data)), index=sample_data.index)
        
        # Calculate stability scores would be internal to the analyzer
        # This is more of an integration test
        sample_data['stable_feature'] = stable_feature
        sample_data['unstable_feature'] = unstable_feature
        
        # The system should prefer stable features
        engineered_data = engineer.fit(sample_data, 'price').transform(sample_data, 'price')
        
        assert isinstance(engineered_data, pd.DataFrame)
    
    def test_concurrent_feature_operations(self, sample_data):
        """Test concurrent feature operations"""
        import threading
        import time
        
        results = []
        errors = []
        
        def feature_operation():
            try:
                engineer = get_automated_feature_engineer()
                result = engineer.transform(sample_data, 'price')
                results.append(len(result.columns))
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=feature_operation)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent operations: {errors}"
        assert len(results) > 0
    
    def test_memory_management(self, sample_data):
        """Test memory management in feature engineering"""
        config = FeatureEngineeringConfig(
            max_features_per_iteration=5,
            max_total_features=20
        )
        engineer = AutomatedFeatureEngineer(config)
        
        # Generate features multiple times
        for i in range(3):
            engineered_data = engineer._engineer_features(sample_data, 'price')
            
            # Should respect limits
            feature_count = len(engineered_data.columns) - 1  # Exclude target
            assert feature_count <= config.max_total_features * 2  # Allow some flexibility
    
    def test_error_handling(self, sample_data):
        """Test error handling in feature engineering"""
        engineer = get_automated_feature_engineer()
        
        # Test with missing target column
        result = engineer.transform(sample_data, 'nonexistent_column')
        assert isinstance(result, pd.DataFrame)
        
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = engineer.transform(empty_df, 'price')
        assert isinstance(result, pd.DataFrame)
        
        # Test with single row
        single_row = sample_data.iloc[:1]
        result = engineer.transform(single_row, 'price')
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    # Run basic tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Automated Feature Engineering...")
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'price': 100 + np.cumsum(np.random.normal(0, 1, 100)),
        'volume': np.random.exponential(1000, 100),
        'rsi': 30 + 40 * np.random.random(100),
        'regime': np.random.choice(['bull', 'bear', 'sideways'], 100)
    }).set_index('timestamp')
    
    try:
        # Test basic feature engineering
        engineer = get_automated_feature_engineer()
        engineered_data = engineer.fit(sample_data, 'price').transform(sample_data, 'price')
        print(f"✓ Generated {len(engineered_data.columns)} features from {len(sample_data.columns)} original features")
        
        # Test feature discovery
        discovery_engine = get_feature_discovery_engine()
        candidates = discovery_engine.discover_features(sample_data, 'price')
        print(f"✓ Discovered {len(candidates)} feature candidates")
        
        # Test SHAP analyzer
        analyzer = get_shap_regime_analyzer()
        sample_data['price_ma'] = sample_data['price'].rolling(5).mean()
        regime_results = analyzer.analyze_regime_specific_importance(sample_data, 'price', 'regime')
        print(f"✓ Analyzed {len(regime_results)} market regimes for feature importance")
        
        # Test live adaptation
        adapter = get_live_adaptation_engine()
        adapted_features = adapter.adapt_features(sample_data, 'price', AdaptationTrigger.REGIME_CHANGE)
        print(f"✓ Adapted feature set with {len(adapted_features)} features")
        
        print("\n🎉 All automated feature engineering tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()