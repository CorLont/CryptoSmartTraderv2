#!/usr/bin/env python3
"""
Enterprise Data Quality Manager for CryptoSmartTrader V2
Implements strict no-fallback data integrity with per-coin completeness tracking
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from core.logging_manager import get_logger
from core.secrets_manager import get_secrets_manager, SecretRedactor
from config.settings import get_settings

class DataQualityStatus(str, Enum):
    """Data quality status levels"""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MISSING = "missing"
    STALE = "stale"
    CORRUPTED = "corrupted"
    BLACKLISTED = "blacklisted"

class FallbackMode(str, Enum):
    """Fallback handling modes"""
    STRICT_NO_FALLBACK = "strict_no_fallback"  # Production mode - never use synthetic data
    DEVELOPMENT_ONLY = "development_only"      # Allow fallback only in development
    DISABLED = "disabled"                      # Never use any fallback

@dataclass
class CoinDataQuality:
    """Per-coin data quality tracking"""
    symbol: str
    exchange: str
    last_update: datetime
    completeness_score: float  # 0.0 to 1.0
    missing_features: Set[str] = field(default_factory=set)
    data_sources: Dict[str, datetime] = field(default_factory=dict)
    quality_status: DataQualityStatus = DataQualityStatus.INCOMPLETE
    consecutive_failures: int = 0
    blacklisted_until: Optional[datetime] = None
    
    def is_complete(self, required_features: Set[str]) -> bool:
        """Check if coin has all required features"""
        return len(self.missing_features.intersection(required_features)) == 0
    
    def is_stale(self, max_age_minutes: int = 30) -> bool:
        """Check if data is stale"""
        return (datetime.now() - self.last_update).total_seconds() > (max_age_minutes * 60)
    
    def is_blacklisted(self) -> bool:
        """Check if coin is currently blacklisted"""
        if self.blacklisted_until is None:
            return False
        return datetime.now() < self.blacklisted_until

@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    timestamp: datetime
    total_coins: int
    complete_coins: int
    incomplete_coins: int
    missing_coins: int
    blacklisted_coins: int
    overall_completeness: float
    exchange_completeness: Dict[str, float]
    feature_completeness: Dict[str, float]
    quality_issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_coins': self.total_coins,
            'complete_coins': self.complete_coins,
            'incomplete_coins': self.incomplete_coins,
            'missing_coins': self.missing_coins,
            'blacklisted_coins': self.blacklisted_coins,
            'overall_completeness': self.overall_completeness,
            'exchange_completeness': self.exchange_completeness,
            'feature_completeness': self.feature_completeness,
            'quality_issues': self.quality_issues
        }

class DataQualityManager:
    """Enterprise data quality management with strict no-fallback enforcement"""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger()
        self.secrets_manager = get_secrets_manager()
        
        # Quality tracking
        self.coin_quality: Dict[str, CoinDataQuality] = {}
        self.required_features = {
            'price', 'volume', 'bid', 'ask', 'high', 'low', 'open', 'close'
        }
        self.optional_features = {
            'market_cap', 'circulating_supply', 'total_supply'
        }
        
        # Configuration
        self.fallback_mode = FallbackMode.STRICT_NO_FALLBACK
        self.min_completeness_threshold = 0.95  # 95% completeness required
        self.max_consecutive_failures = 3
        self.blacklist_duration_hours = 24
        self.staleness_threshold_minutes = 30
        
        # Quality metrics
        self.quality_history: List[DataQualityReport] = []
        self.last_quality_check = datetime.now()
        
        # Initialize from environment
        self._configure_from_environment()
        
        self.logger.info(
            "Data Quality Manager initialized",
            extra={
                'fallback_mode': self.fallback_mode.value,
                'min_completeness_threshold': self.min_completeness_threshold,
                'required_features_count': len(self.required_features),
                'staleness_threshold_minutes': self.staleness_threshold_minutes
            }
        )
    
    def _configure_from_environment(self):
        """Configure from environment variables"""
        environment = self.settings.environment.lower()
        
        # Set fallback mode based on environment
        if environment == 'production':
            self.fallback_mode = FallbackMode.STRICT_NO_FALLBACK
            self.min_completeness_threshold = 0.98  # Stricter in production
        elif environment == 'development':
            self.fallback_mode = FallbackMode.DEVELOPMENT_ONLY
            self.min_completeness_threshold = 0.90  # More lenient in dev
        else:
            self.fallback_mode = FallbackMode.DISABLED
    
    def validate_coin_data(
        self, 
        symbol: str, 
        exchange: str, 
        data: Dict[str, Any]
    ) -> Tuple[bool, CoinDataQuality]:
        """
        Validate coin data quality with strict no-fallback enforcement
        Returns (is_valid, quality_info)
        """
        coin_key = f"{exchange}:{symbol}"
        
        # Get or create quality tracking
        if coin_key not in self.coin_quality:
            self.coin_quality[coin_key] = CoinDataQuality(
                symbol=symbol,
                exchange=exchange,
                last_update=datetime.now(),
                completeness_score=0.0
            )
        
        quality = self.coin_quality[coin_key]
        
        # Check if blacklisted
        if quality.is_blacklisted():
            self.logger.warning(
                f"Coin {symbol} on {exchange} is blacklisted",
                extra={
                    'symbol': symbol,
                    'exchange': exchange,
                    'blacklisted_until': quality.blacklisted_until.isoformat()
                }
            )
            return False, quality
        
        # Validate data presence and quality
        missing_features = set()
        present_features = set()
        
        for feature in self.required_features:
            if feature not in data or data[feature] is None:
                missing_features.add(feature)
            else:
                # Validate data quality (not synthetic/placeholder)
                if self._is_synthetic_data(data[feature], feature):
                    missing_features.add(feature)
                    self.logger.warning(
                        f"Synthetic data detected for {feature}",
                        extra={
                            'symbol': symbol,
                            'exchange': exchange,
                            'feature': feature,
                            'value': SecretRedactor.redact_secrets(str(data[feature]))
                        }
                    )
                else:
                    present_features.add(feature)
        
        # Calculate completeness score
        total_required = len(self.required_features)
        complete_required = len(present_features)
        completeness_score = complete_required / total_required if total_required > 0 else 0.0
        
        # Update quality tracking
        quality.missing_features = missing_features
        quality.completeness_score = completeness_score
        quality.last_update = datetime.now()
        quality.data_sources[exchange] = datetime.now()
        
        # Determine quality status
        if completeness_score >= self.min_completeness_threshold:
            quality.quality_status = DataQualityStatus.COMPLETE
            quality.consecutive_failures = 0
        elif completeness_score >= 0.5:
            quality.quality_status = DataQualityStatus.INCOMPLETE
            quality.consecutive_failures += 1
        else:
            quality.quality_status = DataQualityStatus.MISSING
            quality.consecutive_failures += 1
        
        # Apply blacklisting for persistent failures
        if quality.consecutive_failures >= self.max_consecutive_failures:
            quality.blacklisted_until = datetime.now() + timedelta(hours=self.blacklist_duration_hours)
            quality.quality_status = DataQualityStatus.BLACKLISTED
            
            self.logger.error(
                f"Blacklisting coin {symbol} on {exchange} due to persistent quality issues",
                extra={
                    'symbol': symbol,
                    'exchange': exchange,
                    'consecutive_failures': quality.consecutive_failures,
                    'completeness_score': completeness_score,
                    'missing_features': list(missing_features),
                    'blacklisted_until': quality.blacklisted_until.isoformat()
                }
            )
            return False, quality
        
        # Strict validation for production
        is_valid = self._validate_strict_mode(quality, symbol, exchange)
        
        # Log quality metrics
        self.logger.info(
            f"Data quality validation for {symbol}",
            extra={
                'symbol': symbol,
                'exchange': exchange,
                'completeness_score': completeness_score,
                'quality_status': quality.quality_status.value,
                'is_valid': is_valid,
                'missing_features': list(missing_features),
                'consecutive_failures': quality.consecutive_failures
            }
        )
        
        return is_valid, quality
    
    def _is_synthetic_data(self, value: Any, feature: str) -> bool:
        """Detect synthetic/placeholder/forward-filled data"""
        if value is None:
            return True
        
        # Check for common synthetic patterns
        if isinstance(value, (int, float)):
            # Check for obvious placeholder values
            if value == 0 or value == -1 or value == 999999:
                return True
            
            # Check for repeated decimal patterns (forward-fill indicator)
            str_value = str(value)
            if len(str_value) > 6:
                # Check for repeated patterns like 12.121212 or 0.000000
                decimal_part = str_value.split('.')[-1] if '.' in str_value else ''
                if len(decimal_part) > 4:
                    if len(set(decimal_part)) <= 2:  # Too uniform
                        return True
        
        elif isinstance(value, str):
            # Check for placeholder strings
            synthetic_indicators = [
                'null', 'none', 'n/a', 'placeholder', 'synthetic',
                'forward_fill', 'interpolated', 'estimated'
            ]
            if value.lower() in synthetic_indicators:
                return True
        
        return False
    
    def _validate_strict_mode(self, quality: CoinDataQuality, symbol: str, exchange: str) -> bool:
        """Apply strict validation based on fallback mode"""
        
        if self.fallback_mode == FallbackMode.STRICT_NO_FALLBACK:
            # Production mode: Require perfect data
            if quality.completeness_score < self.min_completeness_threshold:
                self.logger.warning(
                    f"Rejecting {symbol} in strict mode - insufficient completeness",
                    extra={
                        'symbol': symbol,
                        'exchange': exchange,
                        'completeness_score': quality.completeness_score,
                        'threshold': self.min_completeness_threshold,
                        'fallback_mode': self.fallback_mode.value
                    }
                )
                return False
            
            # Check for staleness
            if quality.is_stale(self.staleness_threshold_minutes):
                self.logger.warning(
                    f"Rejecting {symbol} in strict mode - data is stale",
                    extra={
                        'symbol': symbol,
                        'exchange': exchange,
                        'last_update': quality.last_update.isoformat(),
                        'staleness_threshold_minutes': self.staleness_threshold_minutes
                    }
                )
                return False
            
            return True
        
        elif self.fallback_mode == FallbackMode.DEVELOPMENT_ONLY:
            # Development mode: More lenient but still track issues
            if quality.completeness_score < 0.5:  # Minimum 50% in dev
                return False
            return True
        
        else:  # DISABLED
            # Never allow any data quality issues
            return quality.completeness_score == 1.0
    
    def filter_quality_coins(
        self, 
        market_data: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], DataQualityReport]:
        """
        Filter market data to only include coins meeting quality standards
        Returns (filtered_data, quality_report)
        """
        filtered_data = {}
        quality_issues = []
        
        complete_coins = 0
        incomplete_coins = 0
        missing_coins = 0
        blacklisted_coins = 0
        total_coins = 0
        
        exchange_stats = {}
        feature_stats = {}
        
        for exchange, exchange_data in market_data.items():
            if exchange not in exchange_stats:
                exchange_stats[exchange] = {'total': 0, 'valid': 0}
            
            filtered_exchange_data = {}
            
            for symbol, coin_data in exchange_data.items():
                total_coins += 1
                exchange_stats[exchange]['total'] += 1
                
                # Validate coin data
                is_valid, quality = self.validate_coin_data(symbol, exchange, coin_data)
                
                if is_valid:
                    filtered_exchange_data[symbol] = coin_data
                    exchange_stats[exchange]['valid'] += 1
                    
                    if quality.quality_status == DataQualityStatus.COMPLETE:
                        complete_coins += 1
                    else:
                        incomplete_coins += 1
                else:
                    # Track why coin was rejected
                    if quality.is_blacklisted():
                        blacklisted_coins += 1
                        quality_issues.append(f"{symbol} on {exchange}: blacklisted")
                    elif quality.quality_status == DataQualityStatus.MISSING:
                        missing_coins += 1
                        quality_issues.append(f"{symbol} on {exchange}: missing data")
                    else:
                        incomplete_coins += 1
                        quality_issues.append(
                            f"{symbol} on {exchange}: incomplete "
                            f"(score: {quality.completeness_score:.2f})"
                        )
                    
                    # Track missing features
                    for feature in quality.missing_features:
                        if feature not in feature_stats:
                            feature_stats[feature] = {'total': 0, 'missing': 0}
                        feature_stats[feature]['total'] += 1
                        feature_stats[feature]['missing'] += 1
            
            if filtered_exchange_data:
                filtered_data[exchange] = filtered_exchange_data
        
        # Calculate completeness metrics
        overall_completeness = complete_coins / total_coins if total_coins > 0 else 0.0
        
        exchange_completeness = {
            exchange: stats['valid'] / stats['total'] if stats['total'] > 0 else 0.0
            for exchange, stats in exchange_stats.items()
        }
        
        feature_completeness = {
            feature: 1.0 - (stats['missing'] / stats['total']) if stats['total'] > 0 else 1.0
            for feature, stats in feature_stats.items()
        }
        
        # Generate quality report
        quality_report = DataQualityReport(
            timestamp=datetime.now(),
            total_coins=total_coins,
            complete_coins=complete_coins,
            incomplete_coins=incomplete_coins,
            missing_coins=missing_coins,
            blacklisted_coins=blacklisted_coins,
            overall_completeness=overall_completeness,
            exchange_completeness=exchange_completeness,
            feature_completeness=feature_completeness,
            quality_issues=quality_issues[:50]  # Limit to first 50 issues
        )
        
        # Store quality history
        self.quality_history.append(quality_report)
        if len(self.quality_history) > 100:  # Keep last 100 reports
            self.quality_history = self.quality_history[-100:]
        
        # Log comprehensive quality report
        self.logger.info(
            "Data quality filtering completed",
            extra=quality_report.to_dict()
        )
        
        # Alert on low quality
        if overall_completeness < 0.7:  # Less than 70% complete
            self.logger.error(
                "LOW DATA QUALITY ALERT",
                extra={
                    'overall_completeness': overall_completeness,
                    'total_coins': total_coins,
                    'complete_coins': complete_coins,
                    'quality_issues_count': len(quality_issues),
                    'fallback_mode': self.fallback_mode.value
                }
            )
        
        return filtered_data, quality_report
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current data quality summary"""
        if not self.quality_history:
            return {'status': 'no_data', 'message': 'No quality data available'}
        
        latest_report = self.quality_history[-1]
        
        # Calculate trends
        if len(self.quality_history) > 1:
            prev_report = self.quality_history[-2]
            completeness_trend = (
                latest_report.overall_completeness - prev_report.overall_completeness
            )
        else:
            completeness_trend = 0.0
        
        # Get problematic coins
        problematic_coins = [
            coin_key for coin_key, quality in self.coin_quality.items()
            if quality.quality_status in [DataQualityStatus.INCOMPLETE, DataQualityStatus.MISSING]
        ]
        
        # Get blacklisted coins
        blacklisted_coins = [
            coin_key for coin_key, quality in self.coin_quality.items()
            if quality.is_blacklisted()
        ]
        
        return {
            'timestamp': latest_report.timestamp.isoformat(),
            'overall_completeness': latest_report.overall_completeness,
            'completeness_trend': completeness_trend,
            'total_coins_tracked': len(self.coin_quality),
            'complete_coins': latest_report.complete_coins,
            'problematic_coins_count': len(problematic_coins),
            'blacklisted_coins_count': len(blacklisted_coins),
            'fallback_mode': self.fallback_mode.value,
            'min_completeness_threshold': self.min_completeness_threshold,
            'quality_issues_sample': latest_report.quality_issues[:10],
            'exchange_performance': latest_report.exchange_completeness,
            'feature_completeness': latest_report.feature_completeness
        }
    
    def export_quality_report(self, file_path: Path) -> bool:
        """Export detailed quality report to file"""
        try:
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'configuration': {
                    'fallback_mode': self.fallback_mode.value,
                    'min_completeness_threshold': self.min_completeness_threshold,
                    'staleness_threshold_minutes': self.staleness_threshold_minutes,
                    'required_features': list(self.required_features),
                    'optional_features': list(self.optional_features)
                },
                'current_summary': self.get_quality_summary(),
                'coin_details': {
                    coin_key: {
                        'symbol': quality.symbol,
                        'exchange': quality.exchange,
                        'completeness_score': quality.completeness_score,
                        'quality_status': quality.quality_status.value,
                        'missing_features': list(quality.missing_features),
                        'consecutive_failures': quality.consecutive_failures,
                        'last_update': quality.last_update.isoformat(),
                        'is_blacklisted': quality.is_blacklisted(),
                        'blacklisted_until': quality.blacklisted_until.isoformat() if quality.blacklisted_until else None
                    }
                    for coin_key, quality in self.coin_quality.items()
                },
                'quality_history': [report.to_dict() for report in self.quality_history[-10:]]
            }
            
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(
                f"Quality report exported to {file_path}",
                extra={'file_path': str(file_path), 'coins_tracked': len(self.coin_quality)}
            )
            return True
            
        except Exception as e:
            self.logger.error(
                f"Failed to export quality report: {e}",
                extra={'file_path': str(file_path), 'error': str(e)}
            )
            return False
    
    def reset_blacklist(self, symbol: str = None, exchange: str = None):
        """Reset blacklist for specific coin or all coins"""
        if symbol and exchange:
            coin_key = f"{exchange}:{symbol}"
            if coin_key in self.coin_quality:
                self.coin_quality[coin_key].blacklisted_until = None
                self.coin_quality[coin_key].consecutive_failures = 0
                self.coin_quality[coin_key].quality_status = DataQualityStatus.INCOMPLETE
                
                self.logger.info(
                    f"Blacklist reset for {symbol} on {exchange}",
                    extra={'symbol': symbol, 'exchange': exchange}
                )
        else:
            # Reset all blacklists
            reset_count = 0
            for quality in self.coin_quality.values():
                if quality.is_blacklisted():
                    quality.blacklisted_until = None
                    quality.consecutive_failures = 0
                    quality.quality_status = DataQualityStatus.INCOMPLETE
                    reset_count += 1
            
            self.logger.info(
                f"All blacklists reset - {reset_count} coins affected",
                extra={'reset_count': reset_count}
            )

# Global instance
_data_quality_manager = None

def get_data_quality_manager() -> DataQualityManager:
    """Get global data quality manager instance"""
    global _data_quality_manager
    if _data_quality_manager is None:
        _data_quality_manager = DataQualityManager()
    return _data_quality_manager