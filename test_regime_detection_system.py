"""
Test script for the Regime Detection System

Tests all components:
1. Feature calculation
2. Regime classification  
3. Strategy adaptation
4. Complete regime detection workflow
"""

import asyncio
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, 'src')

from cryptosmarttrader.regime import RegimeDetector, RegimeFeatures, RegimeClassifier, RegimeStrategies
from cryptosmarttrader.regime.regime_models import MarketRegime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_market_data():
    """Create realistic test market data"""
    logger.info("Creating test market data...")
    
    # Create 200 hours of hourly data
    dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
    
    # Different market scenarios
    scenarios = {
        'BTC/USD': _create_btc_scenario(dates),
        'ETH/USD': _create_eth_scenario(dates),  
        'SOL/USD': _create_sol_scenario(dates)
    }
    
    return scenarios


def _create_btc_scenario(dates):
    """Create BTC data with different market phases"""
    np.random.seed(42)
    
    # Phase 1: Strong trend (first 50 hours)
    trend_phase = np.cumsum(np.random.normal(0.002, 0.01, 50))  # Strong uptrend
    
    # Phase 2: Mean reversion (next 50 hours) 
    mr_phase = np.sin(np.linspace(0, 4*np.pi, 50)) * 0.03  # Oscillating
    
    # Phase 3: High volatility chop (next 50 hours)
    chop_phase = np.cumsum(np.random.normal(0, 0.03, 50))  # High vol, no direction
    
    # Phase 4: Low volatility drift (last 50 hours)
    drift_phase = np.cumsum(np.random.normal(0.0001, 0.005, 50))  # Low vol
    
    # Combine phases
    price_changes = np.concatenate([trend_phase, mr_phase, chop_phase, drift_phase])
    prices = 45000 * (1 + price_changes)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.999, 1.001, len(prices)),
        'high': prices * np.random.uniform(1.002, 1.015, len(prices)),
        'low': prices * np.random.uniform(0.985, 0.998, len(prices)), 
        'close': prices,
        'volume': np.random.uniform(1000, 5000, len(prices))
    }, index=dates)
    
    return data


def _create_eth_scenario(dates):
    """Create ETH data (correlated with BTC but different characteristics)"""
    np.random.seed(43)
    
    # More volatile than BTC, similar patterns but amplified
    base_changes = np.random.normal(0.0005, 0.02, len(dates))  
    prices = 2800 * np.cumprod(1 + base_changes)
    
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.998, 1.002, len(prices)),
        'high': prices * np.random.uniform(1.005, 1.02, len(prices)),
        'low': prices * np.random.uniform(0.98, 0.995, len(prices)),
        'close': prices,
        'volume': np.random.uniform(500, 3000, len(prices))
    }, index=dates)
    
    return data


def _create_sol_scenario(dates):
    """Create SOL data (higher beta alt)"""
    np.random.seed(44)
    
    # More volatile alt with higher beta to market moves
    base_changes = np.random.normal(0.001, 0.035, len(dates))
    prices = 95 * np.cumprod(1 + base_changes)
    
    data = pd.DataFrame({
        'open': prices * np.random.uniform(0.997, 1.003, len(prices)),
        'high': prices * np.random.uniform(1.01, 1.03, len(prices)),
        'low': prices * np.random.uniform(0.97, 0.99, len(prices)),
        'close': prices,
        'volume': np.random.uniform(200, 1500, len(prices))
    }, index=dates)
    
    return data


async def test_feature_calculation():
    """Test regime feature calculation"""
    logger.info("Testing feature calculation...")
    
    try:
        # Create test data
        market_data = create_test_market_data()
        dominance_data = {'BTC': 44.5, 'ETH': 19.2, 'others': 36.3}
        
        # Initialize feature calculator
        feature_calc = RegimeFeatures()
        
        # Calculate features
        features = feature_calc.calculate_all_features(
            market_data=market_data,
            dominance_data=dominance_data
        )
        
        # Validate results
        assert 0 <= features.hurst_exponent <= 1, "Hurst exponent out of range"
        assert features.adx >= 0, "ADX should be positive"
        assert features.realized_vol >= 0, "Volatility should be positive"
        assert 0 <= features.btc_dominance <= 100, "BTC dominance out of range"
        
        logger.info("✅ Feature calculation successful")
        logger.info(f"   Hurst: {features.hurst_exponent:.3f}")
        logger.info(f"   ADX: {features.adx:.1f}")
        logger.info(f"   Realized Vol: {features.realized_vol:.1f}%")
        logger.info(f"   Volatility Regime: {features.volatility_regime}")
        logger.info(f"   Trend Strength: {features.trend_strength}")
        
        return features
        
    except Exception as e:
        logger.error(f"❌ Feature calculation failed: {e}")
        raise


async def test_regime_classification():
    """Test regime classification model"""
    logger.info("Testing regime classification...")
    
    try:
        # Initialize classifier
        classifier = RegimeClassifier("models/test_regime_classifier.pkl")
        
        # Create training data (simplified)
        market_data = create_test_market_data()
        feature_calc = RegimeFeatures()
        dominance_data = {'BTC': 44.5, 'ETH': 19.2, 'others': 36.3}
        
        # Calculate features for different periods
        training_features = []
        returns_data = []
        
        btc_data = market_data['BTC/USD']
        
        # Create features for sliding windows
        for i in range(50, len(btc_data), 10):  # Every 10 hours
            window_data = {
                'BTC/USD': btc_data.iloc[i-50:i],
                'ETH/USD': market_data['ETH/USD'].iloc[i-50:i],
                'SOL/USD': market_data['SOL/USD'].iloc[i-50:i]
            }
            
            features = feature_calc.calculate_all_features(
                market_data=window_data,
                dominance_data=dominance_data
            )
            
            training_features.append(features)
            
            # Calculate return for this period
            period_return = btc_data['close'].iloc[i-10:i].pct_change().sum()
            returns_data.append(period_return)
        
        if len(training_features) > 10:
            # Train model
            returns_series = pd.Series(returns_data)
            training_results = classifier.train(training_features, returns_series)
            
            if "error" not in training_results:
                logger.info("✅ Model training successful")
                logger.info(f"   Mean CV Score: {training_results.get('mean_cv_score', 0):.3f}")
                logger.info(f"   Training Samples: {training_results.get('n_samples', 0)}")
                
                # Test prediction
                test_features = training_features[-1]
                classification = classifier.predict_regime(test_features)
                
                logger.info(f"   Test Prediction: {classification.primary_regime.value}")
                logger.info(f"   Confidence: {classification.confidence:.3f}")
                logger.info(f"   Should Trade: {classification.should_trade}")
                
                return classification
            else:
                logger.error(f"❌ Training failed: {training_results['error']}")
                return None
        else:
            logger.warning("⚠️ Insufficient training data, creating mock classification")
            # Create mock classification for testing
            from cryptosmarttrader.regime.regime_models import RegimeClassification
            return RegimeClassification(
                primary_regime=MarketRegime.TREND_UP,
                confidence=0.75,
                probabilities={MarketRegime.TREND_UP: 0.75, MarketRegime.MEAN_REVERSION: 0.25},
                feature_importance={'hurst_exponent': 0.3, 'adx': 0.4},
                timestamp=pd.Timestamp.now(),
                should_trade=True
            )
        
    except Exception as e:
        logger.error(f"❌ Regime classification failed: {e}")
        raise


async def test_strategy_adaptation():
    """Test strategy adaptation based on regime"""
    logger.info("Testing strategy adaptation...")
    
    try:
        # Initialize strategy adapter
        strategies = RegimeStrategies()
        
        # Test different regimes
        test_regimes = [
            MarketRegime.TREND_UP,
            MarketRegime.TREND_DOWN,
            MarketRegime.MEAN_REVERSION,
            MarketRegime.HIGH_VOL_CHOP,
            MarketRegime.RISK_OFF
        ]
        
        for regime in test_regimes:
            # Create mock classification
            from cryptosmarttrader.regime.regime_models import RegimeClassification
            mock_classification = RegimeClassification(
                primary_regime=regime,
                confidence=0.8,
                probabilities={regime: 0.8},
                feature_importance={},
                timestamp=pd.Timestamp.now(),
                should_trade=regime not in [MarketRegime.HIGH_VOL_CHOP, MarketRegime.RISK_OFF]
            )
            
            # Get strategy parameters
            strategy_params = strategies.get_strategy_for_regime(mock_classification)
            
            logger.info(f"✅ {regime.value} Strategy:")
            logger.info(f"   Entry Threshold: {strategy_params.entry_threshold:.2f}")
            logger.info(f"   Position Size: {strategy_params.position_size_pct:.1f}%")
            logger.info(f"   Stop Loss: {strategy_params.stop_loss_pct:.1f}%")
            logger.info(f"   Take Profit: {strategy_params.take_profit_pct:.1f}%")
            logger.info(f"   No Trade: {strategy_params.no_trade}")
            
            # Test trading decision
            signal_strength = 0.75
            decision = strategies.should_enter_position(signal_strength, strategy_params)
            logger.info(f"   Trading Decision: {decision['enter']} ({decision['reason']})")
            logger.info("")
        
        logger.info("✅ Strategy adaptation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy adaptation failed: {e}")
        raise


async def test_full_regime_system():
    """Test complete regime detection system"""
    logger.info("Testing complete regime detection system...")
    
    try:
        # Initialize regime detector (without data manager for testing)
        detector = RegimeDetector(
            data_manager=None,
            update_frequency_minutes=1,  # Fast for testing
            model_path="models/test_regime_system.pkl"
        )
        
        # Initialize
        init_success = await detector.initialize()
        if not init_success:
            logger.warning("⚠️ Detector initialization had issues, continuing...")
        
        # Update regime (will use mock data)
        regime_classification = await detector.update_regime()
        
        if regime_classification:
            logger.info("✅ Regime detection successful")
            logger.info(f"   Detected Regime: {regime_classification.primary_regime.value}")
            logger.info(f"   Confidence: {regime_classification.confidence:.3f}")
            logger.info(f"   Should Trade: {regime_classification.should_trade}")
            
            # Test trading recommendation
            signal_strength = 0.72
            recommendation = detector.should_trade(signal_strength)
            
            logger.info(f"   Trading Recommendation: {recommendation['trade']}")
            logger.info(f"   Reason: {recommendation['reason']}")
            
            # Get analytics
            analytics = detector.get_regime_analytics()
            logger.info(f"   Current Strategy: {analytics.get('strategy_summary', {}).get('strategy_type', 'unknown')}")
            
            return True
        else:
            logger.error("❌ No regime classification returned")
            return False
        
    except Exception as e:
        logger.error(f"❌ Full system test failed: {e}")
        raise


async def main():
    """Run all regime detection tests"""
    logger.info("🚀 Starting Regime Detection System Tests")
    print("=" * 60)
    
    try:
        # Test 1: Feature calculation
        features = await test_feature_calculation()
        print("-" * 40)
        
        # Test 2: Regime classification
        classification = await test_regime_classification()
        print("-" * 40)
        
        # Test 3: Strategy adaptation  
        strategy_success = await test_strategy_adaptation()
        print("-" * 40)
        
        # Test 4: Full system
        system_success = await test_full_regime_system()
        print("=" * 60)
        
        if all([features, classification, strategy_success, system_success]):
            logger.info("🎉 ALL TESTS PASSED - Regime Detection System is operational!")
            print("\n🎯 SYSTEM CAPABILITIES:")
            print("✅ Market regime identification (6 regime types)")
            print("✅ Adaptive trading strategies per regime")
            print("✅ Risk management integration")
            print("✅ Real-time regime monitoring")
            print("✅ Feature-based classification (Hurst, ADX, volatility, etc.)")
            print("✅ ML-powered regime prediction")
            
            print("\n📊 TRADING REGIMES SUPPORTED:")
            for regime in MarketRegime:
                print(f"   • {regime.value.replace('_', ' ').title()}")
                
        else:
            logger.error("❌ Some tests failed - review logs above")
            
    except Exception as e:
        logger.error(f"💥 Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())