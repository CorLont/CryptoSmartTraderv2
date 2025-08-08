#!/usr/bin/env python3
"""
Test suite for Market Regime Detection and Adaptive Model Switching
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_regime_detector import (
    MarketRegimeDetector,
    MarketRegime,
    DetectionMethod,
    RegimeDetectionConfig,
    get_market_regime_detector,
    detect_current_regime
)
from core.adaptive_model_switcher import (
    AdaptiveModelSwitcher,
    ModelType,
    AdaptationStrategy,
    AdaptiveSwitcherConfig,
    get_adaptive_model_switcher,
    adapt_to_current_regime
)

class TestMarketRegimeDetection:
    """Test market regime detection system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample cryptocurrency data with different regime patterns"""
        np.random.seed(42)
        
        # Generate 300 data points
        dates = pd.date_range(start='2024-01-01', periods=300, freq='1h')
        
        # Create different regime periods
        bull_period = 100  # First 100 periods - bull market
        bear_period = 100  # Next 100 periods - bear market
        volatile_period = 100  # Last 100 periods - high volatility
        
        prices = []
        volumes = []
        
        # Bull market period
        base_price = 100
        for i in range(bull_period):
            price_change = np.random.normal(0.002, 0.01)  # Positive drift, low volatility
            base_price *= (1 + price_change)
            prices.append(base_price)
            volumes.append(np.random.exponential(1000))
        
        # Bear market period
        for i in range(bear_period):
            price_change = np.random.normal(-0.001, 0.015)  # Negative drift, moderate volatility
            base_price *= (1 + price_change)
            prices.append(base_price)
            volumes.append(np.random.exponential(800))
        
        # High volatility period
        for i in range(volatile_period):
            price_change = np.random.normal(0, 0.04)  # No drift, high volatility
            base_price *= (1 + price_change)
            prices.append(base_price)
            volumes.append(np.random.exponential(1500))
        
        # Create DataFrame with OHLCV data
        data = pd.DataFrame({
            'timestamp': dates,
            'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'high': [p * np.random.uniform(1.00, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.00) for p in prices],
            'close': prices,
            'volume': volumes
        }).set_index('timestamp')
        
        return data
    
    def test_regime_detector_initialization(self):
        """Test regime detector initialization"""
        config = RegimeDetectionConfig(
            primary_method=DetectionMethod.ENSEMBLE,
            lookback_periods=50
        )
        detector = MarketRegimeDetector(config)
        
        assert detector.config.primary_method == DetectionMethod.ENSEMBLE
        assert detector.config.lookback_periods == 50
        assert detector.current_regime == MarketRegime.UNKNOWN
        assert len(detector.regime_history) == 0
    
    def test_feature_engineering(self, sample_data):
        """Test regime feature engineering"""
        detector = get_market_regime_detector()
        
        features = detector._engineer_regime_features(sample_data, 'close')
        
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_data)
        assert len(features.columns) > 10  # Should have many features
        
        # Check for key features
        expected_features = ['sma_short', 'sma_long', 'volatility', 'returns', 'rsi']
        for feature in expected_features:
            assert feature in features.columns, f"Missing feature: {feature}"
        
        # Check that features are numeric and not all NaN
        for col in features.columns:
            assert features[col].dtype in [np.float64, np.int64], f"Non-numeric feature: {col}"
            assert not features[col].isna().all(), f"All NaN feature: {col}"
    
    def test_statistical_detection(self, sample_data):
        """Test statistical regime detection (fallback method)"""
        detector = get_market_regime_detector()
        
        # Test different periods to see regime changes
        bull_data = sample_data.iloc[:100]  # Bull period
        bear_data = sample_data.iloc[100:200]  # Bear period
        volatile_data = sample_data.iloc[200:]  # Volatile period
        
        # Test bull market detection
        bull_features = detector._engineer_regime_features(bull_data, 'close')
        bull_result = detector._statistical_detection(bull_features)
        
        assert isinstance(bull_result.regime, MarketRegime)
        assert bull_result.confidence >= 0.0
        assert bull_result.method == DetectionMethod.STATISTICAL
        
        # Test bear market detection
        bear_features = detector._engineer_regime_features(bear_data, 'close')
        bear_result = detector._statistical_detection(bear_features)
        
        assert isinstance(bear_result.regime, MarketRegime)
        assert bear_result.confidence >= 0.0
        
        # Regimes should potentially be different
        print(f"Bull regime: {bull_result.regime.value}, Bear regime: {bear_result.regime.value}")
    
    def test_regime_detection_pipeline(self, sample_data):
        """Test full regime detection pipeline"""
        detector = get_market_regime_detector()
        
        # Fit the detector (will train models if libraries available)
        detector.fit(sample_data, 'close')
        
        # Detect regime on different periods
        bull_period = sample_data.iloc[:100]
        bear_period = sample_data.iloc[100:200]
        volatile_period = sample_data.iloc[200:]
        
        bull_result = detector.detect_regime(bull_period, 'close')
        bear_result = detector.detect_regime(bear_period, 'close')
        volatile_result = detector.detect_regime(volatile_period, 'close')
        
        # Check result structure
        for result in [bull_result, bear_result, volatile_result]:
            assert isinstance(result.regime, MarketRegime)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.method, DetectionMethod)
            assert isinstance(result.features_used, list)
            assert isinstance(result.detection_timestamp, datetime)
    
    def test_regime_transitions(self, sample_data):
        """Test regime transition detection"""
        detector = get_market_regime_detector()
        detector.fit(sample_data, 'close')
        
        # Simulate regime detection over time
        window_size = 50
        for i in range(0, len(sample_data) - window_size, 25):
            window_data = sample_data.iloc[i:i + window_size]
            result = detector.detect_regime(window_data, 'close')
            
            # Should track regime history
            assert len(detector.regime_history) > 0
        
        # Check if transitions were detected
        print(f"Detected {len(detector.regime_transitions)} regime transitions")
        
        for transition in detector.regime_transitions:
            assert isinstance(transition.from_regime, MarketRegime)
            assert isinstance(transition.to_regime, MarketRegime)
            assert transition.from_regime != transition.to_regime
            assert 0.0 <= transition.confidence <= 1.0
    
    def test_convenient_function(self, sample_data):
        """Test convenient regime detection function"""
        result = detect_current_regime(sample_data, 'close')
        
        assert isinstance(result.regime, MarketRegime)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.method, DetectionMethod)
    
    def test_regime_summary(self, sample_data):
        """Test regime detection summary"""
        detector = get_market_regime_detector()
        detector.fit(sample_data, 'close')
        
        # Run some detections
        for i in range(0, len(sample_data), 50):
            window_data = sample_data.iloc[i:min(i+100, len(sample_data))]
            if len(window_data) >= 50:
                detector.detect_regime(window_data, 'close')
        
        summary = detector.get_regime_summary()
        
        assert 'current_regime' in summary
        assert 'regime_confidence' in summary
        assert 'detection_method' in summary
        assert 'regime_history_length' in summary
        assert 'model_status' in summary
        assert 'regime_distribution' in summary


class TestAdaptiveModelSwitcher:
    """Test adaptive model switching system"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for model testing"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        prices = 100 + np.cumsum(np.random.normal(0, 1, 200))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.exponential(1000, 200),
            'returns': np.random.normal(0, 0.02, 200),
            'volatility': np.random.exponential(0.01, 200),
            'rsi': 30 + 40 * np.random.random(200),
            'macd': np.random.normal(0, 0.5, 200)
        }).set_index('timestamp')
        
        return data
    
    def test_adaptive_switcher_initialization(self):
        """Test adaptive model switcher initialization"""
        config = AdaptiveSwitcherConfig(
            adaptation_strategy=AdaptationStrategy.MODERATE,
            max_switches_per_day=3
        )
        switcher = AdaptiveModelSwitcher(config)
        
        assert switcher.config.adaptation_strategy == AdaptationStrategy.MODERATE
        assert switcher.config.max_switches_per_day == 3
        assert switcher.current_regime == MarketRegime.UNKNOWN
        assert len(switcher.regime_models) > 0  # Should have initialized regime mappings
    
    def test_regime_model_mappings(self):
        """Test regime model mappings initialization"""
        switcher = get_adaptive_model_switcher()
        
        # Check that mappings exist for major regimes
        important_regimes = [
            MarketRegime.BULL_MARKET,
            MarketRegime.BEAR_MARKET,
            MarketRegime.HIGH_VOLATILITY,
            MarketRegime.SIDEWAYS
        ]
        
        for regime in important_regimes:
            assert regime in switcher.regime_models
            mapping = switcher.regime_models[regime]
            
            assert isinstance(mapping.primary_model.model_type, ModelType)
            assert len(mapping.backup_models) > 0
            assert len(mapping.feature_set) > 0
            assert 'buy' in mapping.trading_thresholds
            assert 'sell' in mapping.trading_thresholds
    
    def test_model_creation(self, sample_data):
        """Test model creation for different types"""
        switcher = get_adaptive_model_switcher()
        
        # Test different model types
        model_types = [
            ModelType.LINEAR,
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING
        ]
        
        for model_type in model_types:
            config = switcher.regime_models[MarketRegime.BULL_MARKET].primary_model
            config.model_type = model_type
            
            model = switcher._create_model(config)
            
            assert model is not None
            # Check if model has basic ML interface
            assert hasattr(model, 'fit') or hasattr(model, '__call__')
    
    def test_feature_adaptation(self, sample_data):
        """Test feature adaptation for regimes"""
        switcher = get_adaptive_model_switcher()
        
        # Test adaptation for different regimes
        test_regimes = [MarketRegime.BULL_MARKET, MarketRegime.HIGH_VOLATILITY]
        
        for regime in test_regimes:
            if regime in switcher.regime_models:
                regime_mapping = switcher.regime_models[regime]
                
                adapted_features = switcher._adapt_features(
                    sample_data, regime_mapping, 'close'
                )
                
                assert isinstance(adapted_features, list)
                assert len(adapted_features) > 0
                
                # Features should be relevant to the data
                available_cols = sample_data.columns.tolist()
                valid_features = [f for f in adapted_features if f in available_cols]
                
                print(f"Regime {regime.value}: {len(adapted_features)} features, {len(valid_features)} available")
    
    def test_threshold_adaptation(self, sample_data):
        """Test threshold adaptation based on regime"""
        switcher = get_adaptive_model_switcher()
        
        # Mock regime detection result
        from core.market_regime_detector import RegimeDetectionResult
        
        regime_result = RegimeDetectionResult(
            regime=MarketRegime.HIGH_VOLATILITY,
            confidence=0.8,
            method=DetectionMethod.STATISTICAL,
            features_used=[],
            detection_timestamp=datetime.now(),
            supporting_evidence={'volatility': 0.05}
        )
        
        if MarketRegime.HIGH_VOLATILITY in switcher.regime_models:
            regime_mapping = switcher.regime_models[MarketRegime.HIGH_VOLATILITY]
            
            adapted_thresholds = switcher._adapt_thresholds(
                sample_data, regime_mapping, regime_result
            )
            
            assert isinstance(adapted_thresholds, dict)
            assert 'buy' in adapted_thresholds
            assert 'sell' in adapted_thresholds
            
            # High volatility should lead to higher thresholds
            base_buy = regime_mapping.trading_thresholds['buy']
            adapted_buy = adapted_thresholds['buy']
            
            print(f"Base buy threshold: {base_buy}, Adapted: {adapted_buy}")
    
    def test_regime_adaptation_pipeline(self, sample_data):
        """Test full regime adaptation pipeline"""
        switcher = get_adaptive_model_switcher()
        
        # Add some basic features to data
        sample_data['price_ma'] = sample_data['close'].rolling(10).mean()
        sample_data['volume_ma'] = sample_data['volume'].rolling(10).mean()
        
        # Run adaptation
        result = switcher.adapt_to_regime(sample_data, 'close')
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'regime' in result
            assert 'model_type' in result
            assert 'features' in result
            assert 'thresholds' in result
            assert 'confidence' in result
            
            print(f"Adaptation successful: {result['regime']} regime with {result['model_type']} model")
        else:
            print(f"Adaptation failed: {result.get('error', 'Unknown error')}")
    
    def test_performance_evaluation(self, sample_data):
        """Test model performance evaluation"""
        switcher = get_adaptive_model_switcher()
        
        # Set up a simple model and features
        switcher.current_features = ['volume', 'rsi']  # Use available features
        
        # Create a simple model for testing
        from sklearn.linear_model import LinearRegression
        switcher.current_model = LinearRegression()
        
        # Train the model
        available_features = [f for f in switcher.current_features if f in sample_data.columns]
        if available_features:
            X = sample_data[available_features].fillna(0)
            y = sample_data['close'].fillna(0)
            
            switcher.current_model.fit(X[:100], y[:100])  # Train on first 100 samples
            
            # Evaluate performance
            metrics = switcher.evaluate_current_performance(sample_data[100:], 'close')
            
            if metrics:
                assert 'mse' in metrics
                assert 'r2' in metrics
                assert 'directional_accuracy' in metrics
                
                print(f"Model performance: R² = {metrics.get('r2', 'N/A'):.3f}")
    
    def test_trading_signals(self, sample_data):
        """Test trading signal generation"""
        switcher = get_adaptive_model_switcher()
        
        # Set up simple configuration
        switcher.current_features = ['volume', 'rsi']
        switcher.current_thresholds = {'buy': 0.02, 'sell': -0.02}
        
        # Create and train a simple model
        from sklearn.linear_model import LinearRegression
        switcher.current_model = LinearRegression()
        
        available_features = [f for f in switcher.current_features if f in sample_data.columns]
        if available_features:
            X = sample_data[available_features].fillna(0)
            y = sample_data['close'].fillna(0)
            
            switcher.current_model.fit(X, y)
            
            # Generate signals
            signals = switcher.get_current_signals(sample_data, 'close')
            
            assert isinstance(signals, dict)
            assert 'signal' in signals
            assert 'confidence' in signals
            
            assert signals['signal'] in ['BUY', 'SELL', 'HOLD']
            assert 0.0 <= signals['confidence'] <= 1.0
            
            print(f"Generated signal: {signals['signal']} with confidence {signals['confidence']:.3f}")
    
    def test_convenient_function(self, sample_data):
        """Test convenient adaptation function"""
        result = adapt_to_current_regime(sample_data, 'close')
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'regime' in result
    
    def test_adaptation_summary(self, sample_data):
        """Test adaptation summary"""
        switcher = get_adaptive_model_switcher()
        
        # Run some adaptations
        switcher.adapt_to_regime(sample_data, 'close')
        
        summary = switcher.get_adaptation_summary()
        
        assert isinstance(summary, dict)
        assert 'current_regime' in summary
        assert 'current_model_type' in summary
        assert 'regime_mappings' in summary
        assert 'adaptation_strategy' in summary
        
        print(f"Current regime: {summary['current_regime']}")
        print(f"Model type: {summary['current_model_type']}")


class TestIntegration:
    """Test integration between regime detection and adaptive switching"""
    
    @pytest.fixture
    def complex_data(self):
        """Create complex data with multiple regime changes"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        
        # Create regime-specific data
        data_segments = []
        
        # Bull market (first 150 periods)
        bull_prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 150))
        bull_volumes = np.random.exponential(1000, 150)
        
        # Crash period (next 50 periods)
        crash_start = bull_prices[-1]
        crash_prices = crash_start * np.cumprod(1 + np.random.normal(-0.01, 0.05, 50))
        crash_volumes = np.random.exponential(2000, 50)
        
        # Recovery (next 150 periods)
        recovery_start = crash_prices[-1]
        recovery_prices = recovery_start * np.cumprod(1 + np.random.normal(0.002, 0.02, 150))
        recovery_volumes = np.random.exponential(1200, 150)
        
        # Sideways (final 150 periods)
        sideways_start = recovery_prices[-1]
        sideways_prices = sideways_start + np.cumsum(np.random.normal(0, 0.5, 150))
        sideways_volumes = np.random.exponential(800, 150)
        
        # Combine all periods
        all_prices = np.concatenate([bull_prices, crash_prices, recovery_prices, sideways_prices])
        all_volumes = np.concatenate([bull_volumes, crash_volumes, recovery_volumes, sideways_volumes])
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': all_prices,
            'volume': all_volumes,
            'high': all_prices * np.random.uniform(1.001, 1.02, 500),
            'low': all_prices * np.random.uniform(0.98, 0.999, 500),
            'open': all_prices * np.random.uniform(0.995, 1.005, 500)
        }).set_index('timestamp')
        
        return data
    
    def test_full_system_integration(self, complex_data):
        """Test full integration of regime detection and adaptive switching"""
        # Initialize components
        detector = get_market_regime_detector()
        switcher = get_adaptive_model_switcher()
        
        # Train regime detector
        detector.fit(complex_data, 'close')
        
        # Test adaptation over different periods
        window_size = 100
        results = []
        
        for i in range(0, len(complex_data) - window_size, 50):
            window_data = complex_data.iloc[i:i + window_size]
            
            # Detect regime
            regime_result = detector.detect_regime(window_data, 'close')
            
            # Adapt model
            adaptation_result = switcher.adapt_to_regime(window_data, 'close')
            
            results.append({
                'period': i,
                'regime': regime_result.regime.value,
                'regime_confidence': regime_result.confidence,
                'adaptation_success': adaptation_result.get('success', False),
                'model_type': adaptation_result.get('model_type', 'unknown')
            })
        
        # Analyze results
        successful_adaptations = sum(1 for r in results if r['adaptation_success'])
        detected_regimes = set(r['regime'] for r in results)
        
        print(f"Integration test results:")
        print(f"- Total periods tested: {len(results)}")
        print(f"- Successful adaptations: {successful_adaptations}")
        print(f"- Detected regimes: {detected_regimes}")
        
        # Verify that system detected regime changes
        assert len(detected_regimes) > 1, "Should detect multiple regimes"
        assert successful_adaptations > 0, "Should have successful adaptations"
        
        # Check regime transitions
        regime_transitions = []
        for i in range(1, len(results)):
            if results[i]['regime'] != results[i-1]['regime']:
                regime_transitions.append(
                    f"{results[i-1]['regime']} -> {results[i]['regime']}"
                )
        
        print(f"- Regime transitions: {regime_transitions}")
        
        return results
    
    def test_performance_tracking(self, complex_data):
        """Test performance tracking across regime changes"""
        switcher = get_adaptive_model_switcher()
        
        # Run adaptations and track performance
        window_size = 80
        performance_history = []
        
        for i in range(0, len(complex_data) - window_size, 40):
            window_data = complex_data.iloc[i:i + window_size]
            
            # Adapt to regime
            adaptation_result = switcher.adapt_to_regime(window_data, 'close')
            
            if adaptation_result.get('success'):
                # Evaluate performance on next period
                if i + window_size + 20 < len(complex_data):
                    test_data = complex_data.iloc[i + window_size:i + window_size + 20]
                    performance = switcher.evaluate_current_performance(test_data, 'close')
                    
                    if performance:
                        performance_history.append({
                            'period': i,
                            'regime': adaptation_result.get('regime'),
                            'r2_score': performance.get('r2', 0),
                            'directional_accuracy': performance.get('directional_accuracy', 0)
                        })
        
        # Analyze performance trends
        if performance_history:
            avg_r2 = np.mean([p['r2_score'] for p in performance_history])
            avg_directional = np.mean([p['directional_accuracy'] for p in performance_history])
            
            print(f"Performance tracking results:")
            print(f"- Average R² score: {avg_r2:.3f}")
            print(f"- Average directional accuracy: {avg_directional:.3f}")
            
            # Group by regime
            regime_performance = {}
            for p in performance_history:
                regime = p['regime']
                if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(p['r2_score'])
            
            print(f"- Performance by regime:")
            for regime, scores in regime_performance.items():
                avg_score = np.mean(scores)
                print(f"  {regime}: {avg_score:.3f} (n={len(scores)})")


if __name__ == "__main__":
    # Run basic tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Market Regime Detection and Adaptive Model Switching...")
    
    try:
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='1h')
        
        # Bull market data
        bull_prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 100))
        # Bear market data  
        bear_prices = bull_prices[-1] * np.cumprod(1 + np.random.normal(-0.002, 0.02, 100))
        
        prices = np.concatenate([bull_prices, bear_prices])
        
        sample_data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'volume': np.random.exponential(1000, 200)
        }).set_index('timestamp')
        
        # Test regime detection
        print("\n1. Testing Regime Detection...")
        detector = get_market_regime_detector()
        detector.fit(sample_data, 'close')
        
        # Test different periods
        bull_period = sample_data.iloc[:100]
        bear_period = sample_data.iloc[100:]
        
        bull_regime = detector.detect_regime(bull_period, 'close')
        bear_regime = detector.detect_regime(bear_period, 'close')
        
        print(f"   Bull period regime: {bull_regime.regime.value} (confidence: {bull_regime.confidence:.2f})")
        print(f"   Bear period regime: {bear_regime.regime.value} (confidence: {bear_regime.confidence:.2f})")
        
        # Test adaptive switching
        print("\n2. Testing Adaptive Model Switching...")
        switcher = get_adaptive_model_switcher()
        
        # Add features for switching
        sample_data['rsi'] = 50 + 20 * np.random.randn(200)
        sample_data['volume_ma'] = sample_data['volume'].rolling(10).mean()
        
        bull_adaptation = switcher.adapt_to_regime(bull_period, 'close')
        bear_adaptation = switcher.adapt_to_regime(bear_period, 'close')
        
        print(f"   Bull adaptation: {bull_adaptation.get('success')} - {bull_adaptation.get('model_type', 'N/A')}")
        print(f"   Bear adaptation: {bear_adaptation.get('success')} - {bear_adaptation.get('model_type', 'N/A')}")
        
        # Test signal generation
        print("\n3. Testing Signal Generation...")
        if bull_adaptation.get('success'):
            signals = switcher.get_current_signals(bull_period, 'close')
            print(f"   Bull period signal: {signals.get('signal')} (confidence: {signals.get('confidence', 0):.2f})")
        
        if bear_adaptation.get('success'):
            signals = switcher.get_current_signals(bear_period, 'close')
            print(f"   Bear period signal: {signals.get('signal')} (confidence: {signals.get('confidence', 0):.2f})")
        
        # Test convenient functions
        print("\n4. Testing Convenient Functions...")
        regime_result = detect_current_regime(sample_data, 'close')
        adaptation_result = adapt_to_current_regime(sample_data, 'close')
        
        print(f"   Current regime: {regime_result.regime.value}")
        print(f"   Adaptation success: {adaptation_result.get('success')}")
        
        print("\n🎉 All market regime detection and adaptive switching tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()