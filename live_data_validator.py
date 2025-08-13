#!/usr/bin/env python3
"""
Live Data Validator - Zero-tolerance authentic data validation for CryptoSmartTrader V2
Implements strict data integrity policy with real-time Kraken API validation
"""

import ccxt
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    timestamp: datetime
    symbol: str
    price_valid: bool
    volume_valid: bool
    spread_reasonable: bool
    timestamp_fresh: bool
    no_nulls: bool
    overall_score: float
    
    def is_enterprise_grade(self) -> bool:
        """Check if data meets enterprise quality standards (95%+)"""
        return self.overall_score >= 0.95

class LiveDataValidator:
    """Enterprise-grade live data validation system"""
    
    def __init__(self):
        """Initialize with zero-tolerance policy enforcement"""
        self.logger = logger
        self.api_key = os.environ.get('KRAKEN_API_KEY')
        self.secret = os.environ.get('KRAKEN_SECRET')
        
        if not self.api_key or not self.secret:
            raise ValueError("KRAKEN_API_KEY and KRAKEN_SECRET must be set for live data validation")
        
        # Initialize exchange with production settings
        self.exchange = ccxt.kraken({
            'apiKey': self.api_key,
            'secret': self.secret,
            'sandbox': False,  # Production data only
            'enableRateLimit': True,
            'timeout': 10000,  # 10 second timeout
        })
        
        self.logger.info("Live Data Validator initialized with zero-tolerance policy")
    
    def validate_market_data(self, symbols: List[str]) -> Dict[str, DataQualityMetrics]:
        """
        Validate live market data against enterprise quality standards
        
        Args:
            symbols: List of trading pairs to validate
            
        Returns:
            Dict mapping symbols to quality metrics
        """
        results = {}
        
        try:
            # Load markets to verify symbol availability
            markets = self.exchange.load_markets()
            self.logger.info(f"Loaded {len(markets)} markets from Kraken")
            
            for symbol in symbols:
                if symbol not in markets:
                    self.logger.warning(f"Symbol {symbol} not available in markets")
                    continue
                
                try:
                    # Fetch live ticker data
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # Validate data quality
                    metrics = self._assess_data_quality(symbol, ticker)
                    results[symbol] = metrics
                    
                    status = "✅ PASS" if metrics.is_enterprise_grade() else "❌ FAIL"
                    self.logger.info(f"{symbol}: {status} (Score: {metrics.overall_score:.1%})")
                    
                except Exception as e:
                    self.logger.error(f"Failed to validate {symbol}: {e}")
                    results[symbol] = self._create_failed_metrics(symbol, str(e))
        
        except Exception as e:
            self.logger.error(f"Market data validation failed: {e}")
            
        return results
    
    def _assess_data_quality(self, symbol: str, ticker: Dict[str, Any]) -> DataQualityMetrics:
        """Assess individual ticker data quality"""
        now = datetime.now()
        
        # Quality checks according to zero-tolerance policy
        checks = {
            'price_valid': ticker.get('last', 0) > 0,
            'volume_valid': ticker.get('baseVolume', 0) >= 0,
            'spread_reasonable': self._check_spread_reasonable(ticker),
            'timestamp_fresh': self._check_timestamp_fresh(ticker, now),
            'no_nulls': all(v is not None for v in [
                ticker.get('last'), ticker.get('baseVolume'), 
                ticker.get('bid'), ticker.get('ask'), ticker.get('timestamp')
            ])
        }
        
        # Calculate overall score
        score = sum(checks.values()) / len(checks)
        
        return DataQualityMetrics(
            timestamp=now,
            symbol=symbol,
            price_valid=checks['price_valid'],
            volume_valid=checks['volume_valid'],
            spread_reasonable=checks['spread_reasonable'],
            timestamp_fresh=checks['timestamp_fresh'],
            no_nulls=checks['no_nulls'],
            overall_score=score
        )
    
    def _check_spread_reasonable(self, ticker: Dict[str, Any]) -> bool:
        """Check if bid-ask spread is reasonable (< 5%)"""
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        
        if not bid or not ask or bid <= 0 or ask <= 0:
            return False
            
        spread_pct = (ask - bid) / bid
        return spread_pct < 0.05  # 5% maximum spread
    
    def _check_timestamp_fresh(self, ticker: Dict[str, Any], now: datetime) -> bool:
        """Check if timestamp is recent (within 5 minutes)"""
        timestamp = ticker.get('timestamp')
        if not timestamp:
            return False
            
        ticker_time = datetime.fromtimestamp(timestamp / 1000)
        age_minutes = (now - ticker_time).total_seconds() / 60
        
        return age_minutes < 5  # Must be within 5 minutes
    
    def _create_failed_metrics(self, symbol: str, error: str) -> DataQualityMetrics:
        """Create metrics for failed validation"""
        return DataQualityMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            price_valid=False,
            volume_valid=False,
            spread_reasonable=False,
            timestamp_fresh=False,
            no_nulls=False,
            overall_score=0.0
        )
    
    def generate_quality_report(self, validation_results: Dict[str, DataQualityMetrics]) -> Dict[str, Any]:
        """Generate comprehensive quality assessment report"""
        if not validation_results:
            return {
                'status': 'FAILED',
                'message': 'No validation results available',
                'zero_tolerance_passed': False
            }
        
        # Calculate aggregate metrics
        total_symbols = len(validation_results)
        passed_symbols = sum(1 for metrics in validation_results.values() 
                           if metrics.is_enterprise_grade())
        
        overall_score = sum(metrics.overall_score for metrics in validation_results.values()) / total_symbols
        zero_tolerance_passed = overall_score >= 0.95
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': total_symbols,
            'passed_symbols': passed_symbols,
            'failed_symbols': total_symbols - passed_symbols,
            'overall_score': overall_score,
            'zero_tolerance_passed': zero_tolerance_passed,
            'status': 'PASSED' if zero_tolerance_passed else 'FAILED',
            'details': {
                symbol: {
                    'score': metrics.overall_score,
                    'checks_passed': sum([
                        metrics.price_valid,
                        metrics.volume_valid, 
                        metrics.spread_reasonable,
                        metrics.timestamp_fresh,
                        metrics.no_nulls
                    ]),
                    'enterprise_grade': metrics.is_enterprise_grade()
                }
                for symbol, metrics in validation_results.items()
            }
        }

def main():
    """Run live data validation demo"""
    print("🔗 CRYPTOSMARTTRADER V2 - LIVE DATA VALIDATION")
    print("=" * 50)
    
    try:
        validator = LiveDataValidator()
        
        # Test major cryptocurrency pairs
        test_symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'XRP/USD', 'DOT/USD']
        
        print(f"Testing {len(test_symbols)} major cryptocurrency pairs...")
        
        # Run validation
        results = validator.validate_market_data(test_symbols)
        
        # Generate report
        report = validator.generate_quality_report(results)
        
        # Display results
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"Symbols tested: {report['total_symbols']}")
        print(f"Passed: {report['passed_symbols']}")
        print(f"Failed: {report['failed_symbols']}")
        print(f"Overall Score: {report['overall_score']:.1%}")
        print(f"Zero-Tolerance Policy: {'✅ PASSED' if report['zero_tolerance_passed'] else '❌ FAILED'}")
        
        # Save report
        report_path = f"data/live_data_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('data', exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved: {report_path}")
        
        return report['zero_tolerance_passed']
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)