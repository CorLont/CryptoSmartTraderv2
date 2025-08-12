"""
Regime Detection & Strategy Switching Test

Test regime detection accuracy and strategy switching performance
with focus on Sharpe ratio improvement and turnover reduction.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.ml.regime_detection import RegimeDetector, MarketRegime, RegimeFeatures
from cryptosmarttrader.ml.strategy_switcher import StrategySwitcher, StrategyType

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegimeStrategyTester:
    """
    Comprehensive test suite for regime detection and strategy switching
    """
    
    def __init__(self):
        self.test_results = {}
    
    def generate_test_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate synthetic market data with different regime patterns"""
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1H')
        
        # Create different market regimes
        price = 50000.0
        prices = [price]
        volumes = []
        regime_labels = []
        
        for i in range(1, periods):
            # Determine regime based on period
            if i < 200:  # Trend up
                trend = 0.0005
                volatility = 0.01
                regime = MarketRegime.TREND_UP
            elif i < 400:  # Trend down  
                trend = -0.0003
                volatility = 0.012
                regime = MarketRegime.TREND_DOWN
            elif i < 600:  # Mean reversion
                trend = 0.0001 if price < 48000 else -0.0001
                volatility = 0.008
                regime = MarketRegime.MEAN_REVERSION
            elif i < 800:  # Chop
                trend = np.random.choice([-0.0001, 0.0001])
                volatility = 0.006
                regime = MarketRegime.CHOP
            else:  # Breakout
                trend = 0.0008 if i < 850 else 0.0002
                volatility = 0.015
                regime = MarketRegime.BREAKOUT
            
            # Generate price movement
            random_move = np.random.normal(trend, volatility)
            price = price * (1 + random_move)
            prices.append(price)
            
            # Generate volume
            volume = np.random.lognormal(10, 0.5) * (1 + abs(random_move) * 2)
            volumes.append(volume)
            
            regime_labels.append(regime)
        
        # Create OHLC data
        df = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': [np.random.lognormal(10, 0.5)] + volumes
        })
        
        # Generate OHLC from close prices
        df['open'] = df['close'].shift(1)
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.005, len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.005, len(df)))
        
        # Clean first row
        df = df.dropna().reset_index(drop=True)
        
        # Add true regime labels for testing
        df['true_regime'] = regime_labels[:len(df)]
        
        return df
    
    def test_regime_detection_accuracy(self) -> bool:
        """Test regime detection accuracy against known regimes"""
        
        logger.info("Testing regime detection accuracy...")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(1000)
            
            # Initialize detector
            detector = RegimeDetector(lookback_periods=50)
            
            correct_predictions = 0
            total_predictions = 0
            regime_accuracies = {}
            
            # Test detection on sliding windows
            for i in range(100, len(test_data), 50):
                window_data = test_data.iloc[i-100:i]
                true_regime = test_data.iloc[i]['true_regime']
                
                # Detect regime
                classification = detector.detect_regime(window_data)
                predicted_regime = classification.regime
                
                # Check accuracy
                if predicted_regime == true_regime:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Track per-regime accuracy
                if true_regime not in regime_accuracies:
                    regime_accuracies[true_regime] = {'correct': 0, 'total': 0}
                
                regime_accuracies[true_regime]['total'] += 1
                if predicted_regime == true_regime:
                    regime_accuracies[true_regime]['correct'] += 1
            
            # Calculate overall accuracy
            overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            logger.info(f"Overall regime detection accuracy: {overall_accuracy:.2%}")
            
            # Per-regime accuracy
            for regime, stats in regime_accuracies.items():
                regime_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                logger.info(f"{regime.value} accuracy: {regime_accuracy:.2%} ({stats['correct']}/{stats['total']})")
            
            # Test passes if overall accuracy > 40% (reasonable for complex regime detection)
            success = overall_accuracy > 0.4
            
            self.test_results['regime_detection_accuracy'] = {
                'success': success,
                'overall_accuracy': overall_accuracy,
                'regime_accuracies': {r.value: s['correct']/s['total'] for r, s in regime_accuracies.items()}
            }
            
            return success
            
        except Exception as e:
            logger.error(f"Regime detection test failed: {e}")
            return False
    
    def test_strategy_switching_logic(self) -> bool:
        """Test strategy switching based on regime changes"""
        
        logger.info("Testing strategy switching logic...")
        
        try:
            # Initialize components
            detector = RegimeDetector()
            switcher = StrategySwitcher()
            
            test_data = self.generate_test_data(500)
            
            strategy_switches = 0
            regime_durations = []
            strategy_consistency = 0
            
            last_regime = None
            last_strategy = None
            regime_start_time = None
            
            # Test strategy switching
            for i in range(100, len(test_data), 20):
                window_data = test_data.iloc[i-100:i]
                
                # Detect regime
                classification = detector.detect_regime(window_data)
                
                # Switch strategy
                switch_success = switcher.switch_strategy(classification)
                current_strategy = switcher.active_strategy
                
                if switch_success and current_strategy:
                    # Track regime changes
                    if last_regime and last_regime != classification.regime:
                        if regime_start_time:
                            duration = i - regime_start_time
                            regime_durations.append(duration)
                        regime_start_time = i
                        strategy_switches += 1
                    elif not last_regime:
                        regime_start_time = i
                    
                    # Check strategy consistency within regime
                    if (last_strategy and 
                        last_regime == classification.regime and 
                        current_strategy.regime == last_strategy.regime):
                        strategy_consistency += 1
                    
                    last_regime = classification.regime
                    last_strategy = current_strategy
            
            # Calculate metrics
            avg_regime_duration = np.mean(regime_durations) if regime_durations else 0
            consistency_rate = strategy_consistency / max(1, strategy_switches)
            
            logger.info(f"Strategy switches: {strategy_switches}")
            logger.info(f"Average regime duration: {avg_regime_duration:.1f} periods")
            logger.info(f"Strategy consistency rate: {consistency_rate:.2%}")
            
            # Test current strategy parameters
            current_params = switcher.get_current_parameters()
            if current_params:
                logger.info(f"Current strategy: {current_params.strategy_type.value}")
                logger.info(f"Position size: {current_params.base_position_size:.2f}")
                logger.info(f"Stop loss: {current_params.stop_loss_pct:.2f}%")
                logger.info(f"Take profit: {current_params.take_profit_pct:.2f}%")
            
            success = strategy_switches > 5 and consistency_rate > 0.7
            
            self.test_results['strategy_switching'] = {
                'success': success,
                'strategy_switches': strategy_switches,
                'avg_regime_duration': avg_regime_duration,
                'consistency_rate': consistency_rate
            }
            
            return success
            
        except Exception as e:
            logger.error(f"Strategy switching test failed: {e}")
            return False
    
    def test_regime_specific_parameters(self) -> bool:
        """Test that different regimes have appropriately different parameters"""
        
        logger.info("Testing regime-specific parameter differentiation...")
        
        try:
            switcher = StrategySwitcher()
            
            # Test parameters for different regimes
            regime_params = {}
            
            for regime in MarketRegime:
                if regime in switcher.regime_strategies:
                    strategy = switcher.regime_strategies[regime]
                    params = strategy.primary_strategy
                    
                    regime_params[regime.value] = {
                        'position_size': params.base_position_size,
                        'stop_loss': params.stop_loss_pct,
                        'take_profit': params.take_profit_pct,
                        'max_trades': params.max_daily_trades,
                        'min_hold': params.hold_time_min_minutes,
                        'strategy_type': params.strategy_type.value
                    }
            
            # Log parameter differences
            logger.info("Regime-specific parameters:")
            for regime, params in regime_params.items():
                logger.info(f"{regime}: Size={params['position_size']:.2f}, "
                          f"Stop={params['stop_loss']:.1f}%, "
                          f"TP={params['take_profit']:.1f}%, "
                          f"Strategy={params['strategy_type']}")
            
            # Verify key differences
            differences_found = 0
            
            # Check that trending regimes have different parameters than chop
            if 'chop' in regime_params and 'trend_up' in regime_params:
                chop_params = regime_params['chop']
                trend_params = regime_params['trend_up']
                
                # Chop should have smaller position sizes
                if chop_params['position_size'] < trend_params['position_size']:
                    differences_found += 1
                
                # Chop should have fewer trades
                if chop_params['max_trades'] < trend_params['max_trades']:
                    differences_found += 1
                
                # Different strategy types
                if chop_params['strategy_type'] != trend_params['strategy_type']:
                    differences_found += 1
            
            # Check mean reversion vs trend following differences
            if 'mean_reversion' in regime_params and 'trend_up' in regime_params:
                mr_params = regime_params['mean_reversion']
                trend_params = regime_params['trend_up']
                
                # Different hold times
                if mr_params['min_hold'] != trend_params['min_hold']:
                    differences_found += 1
            
            success = differences_found >= 3
            
            logger.info(f"Parameter differences found: {differences_found}/4")
            
            self.test_results['parameter_differentiation'] = {
                'success': success,
                'differences_found': differences_found,
                'regime_parameters': regime_params
            }
            
            return success
            
        except Exception as e:
            logger.error(f"Parameter differentiation test failed: {e}")
            return False
    
    def test_performance_tracking(self) -> bool:
        """Test performance tracking and Sharpe ratio calculation"""
        
        logger.info("Testing performance tracking...")
        
        try:
            switcher = StrategySwitcher()
            
            # Simulate trades for different regimes
            test_trades = [
                (MarketRegime.TREND_UP, 100, 30, True),
                (MarketRegime.TREND_UP, 50, 45, True),
                (MarketRegime.TREND_UP, -30, 20, False),
                (MarketRegime.CHOP, 20, 10, True),
                (MarketRegime.CHOP, -15, 5, False),
                (MarketRegime.CHOP, 10, 8, True),
                (MarketRegime.MEAN_REVERSION, 80, 25, True),
                (MarketRegime.MEAN_REVERSION, -20, 15, False),
            ]
            
            # Update performance for each trade
            for regime, pnl, duration, win in test_trades:
                switcher.update_strategy_performance(regime, pnl, duration, win)
            
            # Get performance summary
            performance_summary = switcher.get_regime_performance_summary()
            
            logger.info("Performance Summary by Regime:")
            for regime, stats in performance_summary.items():
                if stats['trades'] > 0:
                    logger.info(f"{regime}: {stats['trades']} trades, "
                              f"Win Rate: {stats['win_rate']:.1f}%, "
                              f"Total PnL: {stats['total_pnl']:.0f}, "
                              f"Sharpe: {stats['sharpe_ratio']:.2f}")
            
            # Check that performance is being tracked
            tracked_regimes = sum(1 for stats in performance_summary.values() if stats['trades'] > 0)
            
            # Verify Sharpe calculations
            trend_up_stats = performance_summary.get('trend_up', {})
            sharpe_calculated = 'sharpe_ratio' in trend_up_stats
            
            success = tracked_regimes >= 3 and sharpe_calculated
            
            self.test_results['performance_tracking'] = {
                'success': success,
                'tracked_regimes': tracked_regimes,
                'performance_summary': performance_summary
            }
            
            return success
            
        except Exception as e:
            logger.error(f"Performance tracking test failed: {e}")
            return False
    
    def test_turnover_reduction_in_chop(self) -> bool:
        """Test that turnover is reduced in choppy markets"""
        
        logger.info("Testing turnover reduction in chop regime...")
        
        try:
            switcher = StrategySwitcher()
            
            # Get chop and trend strategies
            chop_strategy = switcher.regime_strategies.get(MarketRegime.CHOP)
            trend_strategy = switcher.regime_strategies.get(MarketRegime.TREND_UP)
            
            if not chop_strategy or not trend_strategy:
                logger.error("Required strategies not found")
                return False
            
            chop_params = chop_strategy.primary_strategy
            trend_params = trend_strategy.primary_strategy
            
            # Compare trading activity parameters
            logger.info("Turnover comparison:")
            logger.info(f"Chop max daily trades: {chop_params.max_daily_trades}")
            logger.info(f"Trend max daily trades: {trend_params.max_daily_trades}")
            logger.info(f"Chop position size: {chop_params.base_position_size}")
            logger.info(f"Trend position size: {trend_params.base_position_size}")
            logger.info(f"Chop signal threshold: {chop_params.entry_signal_strength}")
            logger.info(f"Trend signal threshold: {trend_params.entry_signal_strength}")
            
            # Check turnover reduction mechanisms
            turnover_checks = 0
            
            # Lower daily trade limit in chop
            if chop_params.max_daily_trades < trend_params.max_daily_trades:
                turnover_checks += 1
                logger.info("✓ Chop has lower daily trade limit")
            
            # Smaller position sizes in chop
            if chop_params.base_position_size < trend_params.base_position_size:
                turnover_checks += 1
                logger.info("✓ Chop has smaller position sizes")
            
            # Higher signal threshold in chop (more selective)
            if chop_params.entry_signal_strength > trend_params.entry_signal_strength:
                turnover_checks += 1
                logger.info("✓ Chop has higher signal threshold")
            
            # Shorter hold times in chop (but fewer trades)
            if chop_params.max_concurrent_positions < trend_params.max_concurrent_positions:
                turnover_checks += 1
                logger.info("✓ Chop has fewer concurrent positions")
            
            success = turnover_checks >= 3
            
            logger.info(f"Turnover reduction checks passed: {turnover_checks}/4")
            
            self.test_results['turnover_reduction'] = {
                'success': success,
                'turnover_checks': turnover_checks,
                'chop_daily_trades': chop_params.max_daily_trades,
                'trend_daily_trades': trend_params.max_daily_trades,
                'chop_position_size': chop_params.base_position_size,
                'trend_position_size': trend_params.base_position_size
            }
            
            return success
            
        except Exception as e:
            logger.error(f"Turnover reduction test failed: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run all regime and strategy tests"""
        
        logger.info("=" * 60)
        logger.info("🧪 REGIME DETECTION & STRATEGY SWITCHING TESTS")
        logger.info("=" * 60)
        
        tests = [
            ("Regime Detection Accuracy", self.test_regime_detection_accuracy),
            ("Strategy Switching Logic", self.test_strategy_switching_logic),
            ("Parameter Differentiation", self.test_regime_specific_parameters),
            ("Performance Tracking", self.test_performance_tracking),
            ("Turnover Reduction in Chop", self.test_turnover_reduction_in_chop)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 {test_name}")
            try:
                success = test_func()
                if success:
                    logger.info(f"✅ {test_name} - PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} failed with exception: {e}")
        
        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("🏁 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, _ in tests:
            result = "✅ PASSED" if self.test_results.get(test_name.lower().replace(' ', '_'), {}).get('success', False) else "❌ FAILED"
            logger.info(f"{test_name:<35} {result}")
        
        logger.info("=" * 60)
        logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - REGIME SYSTEM READY")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        
        # Key metrics summary
        logger.info("\n📊 KEY PERFORMANCE METRICS:")
        if 'regime_detection_accuracy' in self.test_results:
            acc = self.test_results['regime_detection_accuracy']['overall_accuracy']
            logger.info(f"• Regime Detection Accuracy: {acc:.1%}")
        
        if 'turnover_reduction' in self.test_results:
            tr = self.test_results['turnover_reduction']
            logger.info(f"• Chop Daily Trades: {tr.get('chop_daily_trades', 0)}")
            logger.info(f"• Trend Daily Trades: {tr.get('trend_daily_trades', 0)}")
        
        return passed_tests == total_tests

def main():
    """Run regime and strategy tests"""
    
    tester = RegimeStrategyTester()
    success = tester.run_comprehensive_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)