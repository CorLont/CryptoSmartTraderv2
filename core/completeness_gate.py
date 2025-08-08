#!/usr/bin/env python3
"""
Data Completeness Gate
Skip coins with insufficient data quality - zero tolerance for incomplete data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json
from pathlib import Path

class CompletenessGate:
    """
    Data completeness gate - hard block coins with missing critical data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'min_completeness_threshold': 0.8,
            'required_fields': {
                'price_data': ['open', 'high', 'low', 'close', 'volume'],
                'technical_indicators': ['rsi', 'macd', 'bollinger_bands'],
                'sentiment_data': ['sentiment_score', 'social_volume'],
                'on_chain_data': ['active_addresses', 'transaction_volume']
            },
            'critical_fields': ['close', 'volume', 'sentiment_score'],
            'zero_tolerance_mode': True,
            'log_blocked_coins': True
        }
        
        if config:
            self.config.update(config)
        
        self.blocked_coins = set()
        self.completeness_reports = {}
    
    def check_coin_completeness(self, symbol: str, coin_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check completeness for a single coin
        """
        
        completeness_report = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'overall_completeness': 0.0,
            'field_completeness': {},
            'critical_fields_complete': False,
            'passes_gate': False,
            'blocked_reasons': []
        }
        
        total_fields = 0
        complete_fields = 0
        
        # Check each data category
        for category, required_fields in self.config['required_fields'].items():
            category_data = coin_data.get(category, {})
            
            for field in required_fields:
                total_fields += 1
                
                field_complete = self._check_field_completeness(category_data, field)
                completeness_report['field_completeness'][f"{category}.{field}"] = field_complete
                
                if field_complete:
                    complete_fields += 1
                elif field in self.config['critical_fields']:
                    completeness_report['blocked_reasons'].append(f"Critical field missing: {category}.{field}")
        
        # Calculate overall completeness
        if total_fields > 0:
            completeness_report['overall_completeness'] = complete_fields / total_fields
        
        # Check critical fields
        critical_complete = all(
            completeness_report['field_completeness'].get(f"{cat}.{field}", False)
            for cat, fields in self.config['required_fields'].items()
            for field in fields
            if field in self.config['critical_fields']
        )
        
        completeness_report['critical_fields_complete'] = critical_complete
        
        # Gate decision
        passes_completeness = completeness_report['overall_completeness'] >= self.config['min_completeness_threshold']
        passes_critical = critical_complete or not self.config['zero_tolerance_mode']
        
        completeness_report['passes_gate'] = passes_completeness and passes_critical
        
        if not completeness_report['passes_gate']:
            if not passes_completeness:
                completeness_report['blocked_reasons'].append(
                    f"Overall completeness {completeness_report['overall_completeness']:.2f} below threshold {self.config['min_completeness_threshold']}"
                )
            
            self.blocked_coins.add(symbol)
            
            if self.config['log_blocked_coins']:
                self._log_blocked_coin(symbol, completeness_report)
        
        self.completeness_reports[symbol] = completeness_report
        
        return completeness_report
    
    def _check_field_completeness(self, data: Dict[str, Any], field: str) -> bool:
        """
        Check if a specific field is complete
        """
        
        if field not in data:
            return False
        
        value = data[field]
        
        # Check for None, NaN, empty values
        if value is None:
            return False
        
        if isinstance(value, (int, float)) and np.isnan(value):
            return False
        
        if isinstance(value, str) and not value.strip():
            return False
        
        if isinstance(value, (list, dict)) and len(value) == 0:
            return False
        
        return True
    
    def filter_complete_coins(self, all_coins_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filter out coins that don't pass completeness gate
        """
        
        filtered_coins = {}
        gate_summary = {
            'total_coins': len(all_coins_data),
            'passed_coins': 0,
            'blocked_coins': 0,
            'pass_rate': 0.0,
            'blocked_symbols': []
        }
        
        for symbol, coin_data in all_coins_data.items():
            completeness_report = self.check_coin_completeness(symbol, coin_data)
            
            if completeness_report['passes_gate']:
                filtered_coins[symbol] = coin_data
                gate_summary['passed_coins'] += 1
            else:
                gate_summary['blocked_coins'] += 1
                gate_summary['blocked_symbols'].append(symbol)
        
        gate_summary['pass_rate'] = gate_summary['passed_coins'] / max(gate_summary['total_coins'], 1)
        
        # Log gate summary
        self._log_gate_summary(gate_summary)
        
        return filtered_coins
    
    def get_completeness_statistics(self) -> Dict[str, Any]:
        """
        Get completeness statistics across all checked coins
        """
        
        if not self.completeness_reports:
            return {'error': 'No coins have been checked'}
        
        completeness_scores = [
            report['overall_completeness'] 
            for report in self.completeness_reports.values()
        ]
        
        field_completeness = {}
        for report in self.completeness_reports.values():
            for field, complete in report['field_completeness'].items():
                if field not in field_completeness:
                    field_completeness[field] = []
                field_completeness[field].append(complete)
        
        # Calculate field-level statistics
        field_stats = {}
        for field, values in field_completeness.items():
            field_stats[field] = {
                'completeness_rate': sum(values) / len(values),
                'total_coins': len(values),
                'complete_coins': sum(values)
            }
        
        statistics = {
            'total_coins_checked': len(self.completeness_reports),
            'blocked_coins_count': len(self.blocked_coins),
            'pass_rate': 1 - (len(self.blocked_coins) / len(self.completeness_reports)),
            'completeness_distribution': {
                'mean': np.mean(completeness_scores),
                'median': np.median(completeness_scores),
                'std': np.std(completeness_scores),
                'min': np.min(completeness_scores),
                'max': np.max(completeness_scores)
            },
            'field_statistics': field_stats,
            'blocked_coins': list(self.blocked_coins)
        }
        
        return statistics
    
    def _log_blocked_coin(self, symbol: str, report: Dict[str, Any]):
        """Log blocked coin for audit trail"""
        
        log_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        blocked_log = log_dir / "blocked_coins.jsonl"
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'completeness': report['overall_completeness'],
            'critical_fields_complete': report['critical_fields_complete'],
            'blocked_reasons': report['blocked_reasons']
        }
        
        with open(blocked_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def _log_gate_summary(self, summary: Dict[str, Any]):
        """Log gate summary"""
        
        log_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        summary_log = log_dir / "completeness_gate_summary.json"
        
        summary_with_timestamp = {
            'timestamp': datetime.now().isoformat(),
            **summary
        }
        
        with open(summary_log, 'w') as f:
            json.dump(summary_with_timestamp, f, indent=2)

class DataQualityMonitor:
    """
    Monitor data quality trends over time
    """
    
    def __init__(self):
        self.quality_history = []
    
    def add_quality_report(self, gate_stats: Dict[str, Any]):
        """Add quality report to history"""
        
        quality_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'pass_rate': gate_stats['pass_rate'],
            'total_coins': gate_stats['total_coins_checked'],
            'blocked_coins': gate_stats['blocked_coins_count'],
            'mean_completeness': gate_stats['completeness_distribution']['mean']
        }
        
        self.quality_history.append(quality_snapshot)
        
        # Keep only last 30 days
        if len(self.quality_history) > 30:
            self.quality_history = self.quality_history[-30:]
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Get data quality trends"""
        
        if len(self.quality_history) < 2:
            return {'error': 'Insufficient history for trends'}
        
        recent = self.quality_history[-1]
        previous = self.quality_history[-2]
        
        trends = {
            'pass_rate_trend': recent['pass_rate'] - previous['pass_rate'],
            'completeness_trend': recent['mean_completeness'] - previous['mean_completeness'],
            'total_coins_trend': recent['total_coins'] - previous['total_coins'],
            'current_quality': recent,
            'history_points': len(self.quality_history)
        }
        
        return trends

# Global instances
_completeness_gate = None
_quality_monitor = None

def get_completeness_gate() -> CompletenessGate:
    """Get singleton completeness gate"""
    global _completeness_gate
    
    if _completeness_gate is None:
        _completeness_gate = CompletenessGate()
    
    return _completeness_gate

def get_quality_monitor() -> DataQualityMonitor:
    """Get singleton quality monitor"""
    global _quality_monitor
    
    if _quality_monitor is None:
        _quality_monitor = DataQualityMonitor()
    
    return _quality_monitor

if __name__ == "__main__":
    print("🚪 TESTING COMPLETENESS GATE")
    print("=" * 50)
    
    # Create test data
    test_coins = {
        'BTC': {
            'price_data': {
                'open': 45000,
                'high': 46000,
                'low': 44000,
                'close': 45500,
                'volume': 1000000
            },
            'technical_indicators': {
                'rsi': 65.5,
                'macd': 150.2,
                'bollinger_bands': [44000, 45500, 47000]
            },
            'sentiment_data': {
                'sentiment_score': 0.75,
                'social_volume': 15000
            },
            'on_chain_data': {
                'active_addresses': 950000,
                'transaction_volume': 50000
            }
        },
        'ETH': {
            'price_data': {
                'open': 3000,
                'high': 3100,
                'low': 2950,
                'close': 3050,
                'volume': 800000
            },
            'technical_indicators': {
                'rsi': 58.2,
                'macd': None,  # Missing critical data
                'bollinger_bands': [2900, 3050, 3200]
            },
            'sentiment_data': {
                'sentiment_score': 0.68,
                'social_volume': 12000
            },
            'on_chain_data': {
                'active_addresses': 650000,
                # Missing transaction_volume
            }
        },
        'ADA': {
            'price_data': {
                'open': 0.5,
                'high': 0.52,
                'low': 0.48,
                'close': None,  # Missing critical field
                'volume': 500000
            },
            'technical_indicators': {
                'rsi': 45.0,
                'macd': 0.01,
                'bollinger_bands': [0.48, 0.51, 0.54]
            },
            'sentiment_data': {
                'sentiment_score': None,  # Missing critical field
                'social_volume': 8000
            },
            'on_chain_data': {
                'active_addresses': 300000,
                'transaction_volume': 25000
            }
        }
    }
    
    # Test completeness gate
    gate = get_completeness_gate()
    
    print("🔍 Checking individual coins...")
    
    for symbol, data in test_coins.items():
        report = gate.check_coin_completeness(symbol, data)
        print(f"\n   {symbol}:")
        print(f"      Completeness: {report['overall_completeness']:.2f}")
        print(f"      Critical fields: {'✓' if report['critical_fields_complete'] else '✗'}")
        print(f"      Passes gate: {'✓' if report['passes_gate'] else '✗'}")
        
        if report['blocked_reasons']:
            print(f"      Blocked reasons: {', '.join(report['blocked_reasons'])}")
    
    # Test filtering
    print(f"\n🚪 Testing completeness filtering...")
    
    filtered_coins = gate.filter_complete_coins(test_coins)
    
    print(f"   Original coins: {len(test_coins)}")
    print(f"   Filtered coins: {len(filtered_coins)}")
    print(f"   Passed coins: {list(filtered_coins.keys())}")
    print(f"   Blocked coins: {list(gate.blocked_coins)}")
    
    # Test statistics
    print(f"\n📊 Completeness statistics...")
    
    stats = gate.get_completeness_statistics()
    
    print(f"   Total checked: {stats['total_coins_checked']}")
    print(f"   Pass rate: {stats['pass_rate']:.2f}")
    print(f"   Mean completeness: {stats['completeness_distribution']['mean']:.3f}")
    print(f"   Blocked count: {stats['blocked_coins_count']}")
    
    # Test quality monitoring
    print(f"\n📈 Testing quality monitoring...")
    
    monitor = get_quality_monitor()
    monitor.add_quality_report(stats)
    
    # Simulate another day with different quality
    stats2 = stats.copy()
    stats2['pass_rate'] = 0.75
    stats2['completeness_distribution']['mean'] = 0.82
    monitor.add_quality_report(stats2)
    
    trends = monitor.get_quality_trends()
    
    if 'error' not in trends:
        print(f"   Pass rate trend: {trends['pass_rate_trend']:+.3f}")
        print(f"   Completeness trend: {trends['completeness_trend']:+.3f}")
    
    print("\n✅ Completeness gate testing completed")