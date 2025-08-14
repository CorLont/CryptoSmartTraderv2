#!/usr/bin/env python3
"""
Data Quality Validator
Real-time validatie van data quality, completeness en integrity
"""

import logging
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from .enterprise_data_ingestion import DataResponse


class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityMetric(Enum):
    """Data quality metrics"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"


@dataclass
class ValidationIssue:
    """Data validation issue"""
    metric: QualityMetric
    severity: ValidationSeverity
    message: str
    source: str
    symbol: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Data quality assessment report"""
    source: str
    symbol: Optional[str]
    timestamp: datetime
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    issues: List[ValidationIssue]
    data_points: int
    assessment_duration_ms: float


class DataQualityValidator:
    """Enterprise data quality validation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            QualityMetric.COMPLETENESS: config.get('completeness_threshold', 0.95),
            QualityMetric.ACCURACY: config.get('accuracy_threshold', 0.98),
            QualityMetric.CONSISTENCY: config.get('consistency_threshold', 0.90),
            QualityMetric.TIMELINESS: config.get('timeliness_threshold', 0.95),
            QualityMetric.VALIDITY: config.get('validity_threshold', 0.99),
            QualityMetric.UNIQUENESS: config.get('uniqueness_threshold', 0.98)
        }
        
        # Historical data for analysis
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timestamp_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Validation metrics
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'critical_issues': 0,
            'avg_quality_score': 0.0
        }
        
        # Known symbols for validation
        self.known_symbols = set(config.get('tracked_symbols', []))
        self.symbol_patterns = config.get('symbol_patterns', {})
        
        # Price bounds for outlier detection
        self.price_bounds = config.get('price_bounds', {
            'BTC/USD': (1000, 200000),
            'ETH/USD': (10, 10000),
            'default': (0.0001, 100000)
        })
    
    async def validate_response(self, response: DataResponse) -> QualityReport:
        """Validate data response comprehensively"""
        start_time = datetime.now()
        issues = []
        metric_scores = {}
        
        try:
            # Skip validation for failed responses
            if response.status != "success" or not response.data:
                return self._create_failed_report(response, "No data to validate")
            
            # Route to appropriate validator based on data type
            if isinstance(response.data, dict):
                if 'symbol' in response.data and 'last' in response.data:
                    # Ticker data
                    issues, metric_scores = await self._validate_ticker_data(response)
                elif 'bids' in response.data and 'asks' in response.data:
                    # Order book data
                    issues, metric_scores = await self._validate_orderbook_data(response)
                elif isinstance(response.data, dict) and len(response.data) > 1:
                    # Multiple tickers
                    issues, metric_scores = await self._validate_multiple_tickers(response)
                else:
                    issues, metric_scores = await self._validate_generic_dict(response)
                    
            elif isinstance(response.data, list):
                if response.data and isinstance(response.data[0], list) and len(response.data[0]) >= 6:
                    # OHLCV data
                    issues, metric_scores = await self._validate_ohlcv_data(response)
                elif response.data and isinstance(response.data[0], dict):
                    # Trades data
                    issues, metric_scores = await self._validate_trades_data(response)
                else:
                    issues, metric_scores = await self._validate_generic_list(response)
            else:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Unsupported data type: {type(response.data)}",
                    source=response.source
                ))
                metric_scores = {metric: 0.0 for metric in QualityMetric}
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(metric_scores)
            
            # Update historical data for future validations
            self._update_historical_data(response)
            
            # Update validation statistics
            self._update_validation_stats(overall_score, issues)
            
            # Create quality report
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return QualityReport(
                source=response.source,
                symbol=self._extract_symbol(response.data),
                timestamp=response.timestamp,
                overall_score=overall_score,
                metric_scores=metric_scores,
                issues=issues,
                data_points=self._count_data_points(response.data),
                assessment_duration_ms=duration_ms
            )
            
        except Exception as e:
            self.logger.error(f"Validation error for {response.source}: {e}")
            return self._create_failed_report(response, f"Validation failed: {str(e)}")
    
    async def _validate_ticker_data(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate single ticker data"""
        issues = []
        scores = {}
        data = response.data
        
        # Completeness validation
        required_fields = ['symbol', 'last', 'timestamp']
        optional_fields = ['bid', 'ask', 'high', 'low', 'volume', 'percentage']
        
        completeness_score = self._validate_completeness(
            data, required_fields, optional_fields, issues, response.source
        )
        scores[QualityMetric.COMPLETENESS] = completeness_score
        
        # Validity validation
        validity_score = self._validate_ticker_validity(data, issues, response.source)
        scores[QualityMetric.VALIDITY] = validity_score
        
        # Accuracy validation (price outlier detection)
        accuracy_score = self._validate_price_accuracy(
            data.get('symbol'), data.get('last'), issues, response.source
        )
        scores[QualityMetric.ACCURACY] = accuracy_score
        
        # Timeliness validation
        timeliness_score = self._validate_timeliness(
            data.get('timestamp'), issues, response.source
        )
        scores[QualityMetric.TIMELINESS] = timeliness_score
        
        # Consistency validation
        consistency_score = self._validate_ticker_consistency(data, issues, response.source)
        scores[QualityMetric.CONSISTENCY] = consistency_score
        
        return issues, scores
    
    async def _validate_orderbook_data(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate order book data"""
        issues = []
        scores = {}
        data = response.data
        
        # Completeness validation
        required_fields = ['bids', 'asks']
        completeness_score = self._validate_completeness(
            data, required_fields, [], issues, response.source
        )
        scores[QualityMetric.COMPLETENESS] = completeness_score
        
        # Validity validation
        validity_score = self._validate_orderbook_validity(data, issues, response.source)
        scores[QualityMetric.VALIDITY] = validity_score
        
        # Consistency validation (spread, bid/ask ordering)
        consistency_score = self._validate_orderbook_consistency(data, issues, response.source)
        scores[QualityMetric.CONSISTENCY] = consistency_score
        
        # Accuracy validation
        accuracy_score = self._validate_orderbook_accuracy(data, issues, response.source)
        scores[QualityMetric.ACCURACY] = accuracy_score
        
        # Default scores for less relevant metrics
        scores[QualityMetric.TIMELINESS] = 1.0
        scores[QualityMetric.UNIQUENESS] = 1.0
        
        return issues, scores
    
    async def _validate_ohlcv_data(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate OHLCV data"""
        issues = []
        scores = {}
        data = response.data
        
        if not data:
            scores = {metric: 0.0 for metric in QualityMetric}
            issues.append(ValidationIssue(
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                message="Empty OHLCV data",
                source=response.source
            ))
            return issues, scores
        
        # Completeness validation
        completeness_score = self._validate_ohlcv_completeness(data, issues, response.source)
        scores[QualityMetric.COMPLETENESS] = completeness_score
        
        # Validity validation
        validity_score = self._validate_ohlcv_validity(data, issues, response.source)
        scores[QualityMetric.VALIDITY] = validity_score
        
        # Consistency validation (OHLC relationships)
        consistency_score = self._validate_ohlcv_consistency(data, issues, response.source)
        scores[QualityMetric.CONSISTENCY] = consistency_score
        
        # Timeliness validation (timestamp ordering)
        timeliness_score = self._validate_ohlcv_timeliness(data, issues, response.source)
        scores[QualityMetric.TIMELINESS] = timeliness_score
        
        # Uniqueness validation (no duplicate timestamps)
        uniqueness_score = self._validate_ohlcv_uniqueness(data, issues, response.source)
        scores[QualityMetric.UNIQUENESS] = uniqueness_score
        
        # Accuracy validation (price reasonableness)
        accuracy_score = self._validate_ohlcv_accuracy(data, issues, response.source)
        scores[QualityMetric.ACCURACY] = accuracy_score
        
        return issues, scores
    
    def _validate_completeness(
        self, 
        data: Dict, 
        required_fields: List[str], 
        optional_fields: List[str],
        issues: List[ValidationIssue],
        source: str
    ) -> float:
        """Validate data completeness"""
        total_fields = len(required_fields) + len(optional_fields)
        present_fields = 0
        missing_required = []
        
        for field in required_fields:
            if field in data and data[field] is not None:
                present_fields += 1
            else:
                missing_required.append(field)
        
        for field in optional_fields:
            if field in data and data[field] is not None:
                present_fields += 1
        
        # Critical issue for missing required fields
        if missing_required:
            issues.append(ValidationIssue(
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing required fields: {missing_required}",
                source=source,
                details={'missing_fields': missing_required}
            ))
        
        return present_fields / total_fields if total_fields > 0 else 0.0
    
    def _validate_ticker_validity(
        self, 
        data: Dict, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate ticker data validity"""
        validity_score = 1.0
        
        # Price validation
        price = data.get('last')
        if price is not None:
            if not isinstance(price, (int, float)) or price <= 0:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid price: {price}",
                    source=source,
                    symbol=data.get('symbol')
                ))
                validity_score -= 0.3
        
        # Volume validation
        volume = data.get('volume')
        if volume is not None:
            if not isinstance(volume, (int, float)) or volume < 0:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Invalid volume: {volume}",
                    source=source,
                    symbol=data.get('symbol')
                ))
                validity_score -= 0.1
        
        # Symbol validation
        symbol = data.get('symbol')
        if symbol and not self._is_valid_symbol(symbol):
            issues.append(ValidationIssue(
                metric=QualityMetric.VALIDITY,
                severity=ValidationSeverity.WARNING,
                message=f"Unexpected symbol format: {symbol}",
                source=source,
                symbol=symbol
            ))
            validity_score -= 0.1
        
        return max(0.0, validity_score)
    
    def _validate_price_accuracy(
        self, 
        symbol: Optional[str], 
        price: Optional[float], 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate price accuracy using outlier detection"""
        if not symbol or price is None:
            return 1.0
        
        # Check against known price bounds
        bounds = self.price_bounds.get(symbol, self.price_bounds['default'])
        if not (bounds[0] <= price <= bounds[1]):
            issues.append(ValidationIssue(
                metric=QualityMetric.ACCURACY,
                severity=ValidationSeverity.ERROR,
                message=f"Price {price} outside expected bounds {bounds}",
                source=source,
                symbol=symbol,
                details={'price': price, 'bounds': bounds}
            ))
            return 0.0
        
        # Check against historical prices for outlier detection
        key = f"{source}:{symbol}"
        if key in self.price_history and len(self.price_history[key]) > 10:
            history = list(self.price_history[key])
            median_price = statistics.median(history)
            
            # Check for extreme deviation (>50% from median)
            deviation = abs(price - median_price) / median_price
            if deviation > 0.5:
                issues.append(ValidationIssue(
                    metric=QualityMetric.ACCURACY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Price deviation {deviation:.1%} from historical median",
                    source=source,
                    symbol=symbol,
                    details={'current_price': price, 'median_price': median_price, 'deviation': deviation}
                ))
                return 0.7
        
        return 1.0
    
    def _validate_timeliness(
        self, 
        timestamp: Optional[Any], 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate data timeliness"""
        if timestamp is None:
            return 1.0  # No timestamp to validate
        
        try:
            # Convert to datetime if needed
            if isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp)
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = timestamp
            
            # Check if data is too old (>5 minutes for real-time data)
            age_seconds = (datetime.now() - dt).total_seconds()
            if age_seconds > 300:  # 5 minutes
                issues.append(ValidationIssue(
                    metric=QualityMetric.TIMELINESS,
                    severity=ValidationSeverity.WARNING,
                    message=f"Data is {age_seconds:.0f} seconds old",
                    source=source,
                    details={'age_seconds': age_seconds}
                ))
                return max(0.0, 1.0 - (age_seconds - 300) / 1800)  # Linear decay over 30 min
            
            return 1.0
            
        except Exception as e:
            issues.append(ValidationIssue(
                metric=QualityMetric.TIMELINESS,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid timestamp format: {timestamp}",
                source=source,
                details={'timestamp': str(timestamp), 'error': str(e)}
            ))
            return 0.0
    
    def _validate_ticker_consistency(
        self, 
        data: Dict, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate ticker data consistency"""
        consistency_score = 1.0
        
        # Bid/Ask spread validation
        bid = data.get('bid')
        ask = data.get('ask')
        last = data.get('last')
        
        if bid and ask:
            if bid >= ask:
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Bid {bid} >= Ask {ask}",
                    source=source,
                    symbol=data.get('symbol')
                ))
                consistency_score -= 0.5
            
            # Last price should be between bid and ask
            if last and not (bid <= last <= ask):
                spread_ratio = abs(last - (bid + ask) / 2) / ((ask - bid) / 2)
                if spread_ratio > 2.0:  # More than 2x the spread away from mid
                    issues.append(ValidationIssue(
                        metric=QualityMetric.CONSISTENCY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Last price {last} outside bid-ask range [{bid}, {ask}]",
                        source=source,
                        symbol=data.get('symbol')
                    ))
                    consistency_score -= 0.2
        
        # High/Low validation
        high = data.get('high')
        low = data.get('low')
        
        if high and low:
            if high < low:
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"High {high} < Low {low}",
                    source=source,
                    symbol=data.get('symbol')
                ))
                consistency_score -= 0.3
            
            # Last price should be between high and low
            if last and not (low <= last <= high):
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Last price {last} outside daily range [{low}, {high}]",
                    source=source,
                    symbol=data.get('symbol')
                ))
                consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _validate_orderbook_validity(
        self, 
        data: Dict, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate order book data validity"""
        validity_score = 1.0
        
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        # Validate bid structure
        for i, bid in enumerate(bids):
            if not isinstance(bid, list) or len(bid) < 2:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid bid format at index {i}: {bid}",
                    source=source
                ))
                validity_score -= 0.1
            elif not (isinstance(bid[0], (int, float)) and isinstance(bid[1], (int, float))):
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid bid data types at index {i}: {bid}",
                    source=source
                ))
                validity_score -= 0.1
        
        # Validate ask structure
        for i, ask in enumerate(asks):
            if not isinstance(ask, list) or len(ask) < 2:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid ask format at index {i}: {ask}",
                    source=source
                ))
                validity_score -= 0.1
            elif not (isinstance(ask[0], (int, float)) and isinstance(ask[1], (int, float))):
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid ask data types at index {i}: {ask}",
                    source=source
                ))
                validity_score -= 0.1
        
        return max(0.0, validity_score)
    
    def _validate_orderbook_consistency(
        self, 
        data: Dict, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate order book consistency"""
        consistency_score = 1.0
        
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        # Check bid ordering (should be descending)
        for i in range(1, len(bids)):
            if bids[i][0] > bids[i-1][0]:
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Bids not in descending order at index {i}",
                    source=source
                ))
                consistency_score -= 0.2
                break
        
        # Check ask ordering (should be ascending)
        for i in range(1, len(asks)):
            if asks[i][0] < asks[i-1][0]:
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Asks not in ascending order at index {i}",
                    source=source
                ))
                consistency_score -= 0.2
                break
        
        # Check that best bid < best ask
        if bids and asks and bids[0][0] >= asks[0][0]:
            issues.append(ValidationIssue(
                metric=QualityMetric.CONSISTENCY,
                severity=ValidationSeverity.CRITICAL,
                message=f"Best bid {bids[0][0]} >= best ask {asks[0][0]}",
                source=source
            ))
            consistency_score -= 0.5
        
        return max(0.0, consistency_score)
    
    def _validate_orderbook_accuracy(
        self, 
        data: Dict, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate order book price accuracy"""
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if not bids or not asks:
            return 1.0
        
        # Check for reasonable spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread_pct = (best_ask - best_bid) / best_bid * 100
        
        if spread_pct > 10:  # >10% spread is suspicious
            issues.append(ValidationIssue(
                metric=QualityMetric.ACCURACY,
                severity=ValidationSeverity.WARNING,
                message=f"Unusually wide spread: {spread_pct:.2f}%",
                source=source,
                details={'spread_pct': spread_pct, 'best_bid': best_bid, 'best_ask': best_ask}
            ))
            return 0.7
        
        return 1.0
    
    def _validate_ohlcv_completeness(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV data completeness"""
        if not data:
            return 0.0
        
        complete_candles = 0
        total_candles = len(data)
        
        for i, candle in enumerate(data):
            if len(candle) >= 6:  # timestamp, open, high, low, close, volume
                complete_candles += 1
            else:
                issues.append(ValidationIssue(
                    metric=QualityMetric.COMPLETENESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Incomplete candle at index {i}: {len(candle)} fields",
                    source=source,
                    details={'candle_index': i, 'fields_count': len(candle)}
                ))
        
        return complete_candles / total_candles if total_candles > 0 else 0.0
    
    def _validate_ohlcv_validity(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV data validity"""
        validity_score = 1.0
        
        for i, candle in enumerate(data):
            if len(candle) < 6:
                continue
            
            timestamp, open_price, high, low, close, volume = candle[:6]
            
            # Validate data types
            if not all(isinstance(x, (int, float)) for x in [open_price, high, low, close, volume]):
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid data types in candle {i}",
                    source=source
                ))
                validity_score -= 0.1
            
            # Validate positive prices
            if any(x <= 0 for x in [open_price, high, low, close]):
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Non-positive prices in candle {i}",
                    source=source
                ))
                validity_score -= 0.1
            
            # Validate non-negative volume
            if volume < 0:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Negative volume in candle {i}: {volume}",
                    source=source
                ))
                validity_score -= 0.05
        
        return max(0.0, validity_score)
    
    def _validate_ohlcv_consistency(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV data consistency"""
        consistency_score = 1.0
        
        for i, candle in enumerate(data):
            if len(candle) < 6:
                continue
            
            timestamp, open_price, high, low, close, volume = candle[:6]
            
            # High should be >= open, close, low
            if not (high >= max(open_price, close) and high >= low):
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid high price in candle {i}: {high}",
                    source=source,
                    details={'candle': candle[:6]}
                ))
                consistency_score -= 0.1
            
            # Low should be <= open, close, high
            if not (low <= min(open_price, close) and low <= high):
                issues.append(ValidationIssue(
                    metric=QualityMetric.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid low price in candle {i}: {low}",
                    source=source,
                    details={'candle': candle[:6]}
                ))
                consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _validate_ohlcv_timeliness(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV timestamp ordering"""
        if len(data) < 2:
            return 1.0
        
        for i in range(1, len(data)):
            if data[i][0] <= data[i-1][0]:
                issues.append(ValidationIssue(
                    metric=QualityMetric.TIMELINESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Timestamps not in ascending order at index {i}",
                    source=source,
                    details={'prev_ts': data[i-1][0], 'curr_ts': data[i][0]}
                ))
                return 0.0
        
        return 1.0
    
    def _validate_ohlcv_uniqueness(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV timestamp uniqueness"""
        timestamps = [candle[0] for candle in data if len(candle) > 0]
        unique_timestamps = set(timestamps)
        
        if len(unique_timestamps) < len(timestamps):
            duplicates = len(timestamps) - len(unique_timestamps)
            issues.append(ValidationIssue(
                metric=QualityMetric.UNIQUENESS,
                severity=ValidationSeverity.WARNING,
                message=f"{duplicates} duplicate timestamps found",
                source=source,
                details={'duplicates': duplicates, 'total': len(timestamps)}
            ))
            return len(unique_timestamps) / len(timestamps)
        
        return 1.0
    
    def _validate_ohlcv_accuracy(
        self, 
        data: List, 
        issues: List[ValidationIssue], 
        source: str
    ) -> float:
        """Validate OHLCV price accuracy"""
        if not data:
            return 1.0
        
        # Check for extreme price movements between candles
        for i in range(1, len(data)):
            if len(data[i]) < 6 or len(data[i-1]) < 6:
                continue
            
            prev_close = data[i-1][4]
            curr_open = data[i][1]
            
            # Check for gaps > 50%
            if prev_close > 0:
                gap_pct = abs(curr_open - prev_close) / prev_close
                if gap_pct > 0.5:
                    issues.append(ValidationIssue(
                        metric=QualityMetric.ACCURACY,
                        severity=ValidationSeverity.WARNING,
                        message=f"Large price gap {gap_pct:.1%} between candles {i-1} and {i}",
                        source=source,
                        details={'gap_pct': gap_pct, 'prev_close': prev_close, 'curr_open': curr_open}
                    ))
                    return 0.8
        
        return 1.0
    
    async def _validate_multiple_tickers(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate multiple ticker data"""
        issues = []
        data = response.data
        
        if not isinstance(data, dict):
            return [ValidationIssue(
                metric=QualityMetric.VALIDITY,
                severity=ValidationSeverity.ERROR,
                message="Multiple tickers data is not a dictionary",
                source=response.source
            )], {metric: 0.0 for metric in QualityMetric}
        
        total_symbols = len(data)
        valid_symbols = 0
        
        for symbol, ticker_data in data.items():
            if isinstance(ticker_data, dict) and ticker_data.get('last') is not None:
                valid_symbols += 1
            else:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Invalid ticker data for {symbol}",
                    source=response.source,
                    symbol=symbol
                ))
        
        # Calculate metrics
        completeness = valid_symbols / total_symbols if total_symbols > 0 else 0.0
        validity = completeness  # Similar for multiple tickers
        
        return issues, {
            QualityMetric.COMPLETENESS: completeness,
            QualityMetric.VALIDITY: validity,
            QualityMetric.ACCURACY: 1.0,
            QualityMetric.CONSISTENCY: 1.0,
            QualityMetric.TIMELINESS: 1.0,
            QualityMetric.UNIQUENESS: 1.0
        }
    
    async def _validate_trades_data(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate trades data"""
        issues = []
        data = response.data
        
        if not data:
            return [ValidationIssue(
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                message="Empty trades data",
                source=response.source
            )], {metric: 0.5 for metric in QualityMetric}
        
        valid_trades = 0
        for i, trade in enumerate(data):
            if isinstance(trade, dict) and 'price' in trade and 'amount' in trade:
                valid_trades += 1
            else:
                issues.append(ValidationIssue(
                    metric=QualityMetric.VALIDITY,
                    severity=ValidationSeverity.WARNING,
                    message=f"Invalid trade data at index {i}",
                    source=response.source
                ))
        
        completeness = valid_trades / len(data)
        
        return issues, {
            QualityMetric.COMPLETENESS: completeness,
            QualityMetric.VALIDITY: completeness,
            QualityMetric.ACCURACY: 1.0,
            QualityMetric.CONSISTENCY: 1.0,
            QualityMetric.TIMELINESS: 1.0,
            QualityMetric.UNIQUENESS: 1.0
        }
    
    async def _validate_generic_dict(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate generic dictionary data"""
        data = response.data
        
        if not data:
            return [ValidationIssue(
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                message="Empty data dictionary",
                source=response.source
            )], {metric: 0.5 for metric in QualityMetric}
        
        # Basic validation - just check it's not empty
        return [], {
            QualityMetric.COMPLETENESS: 1.0,
            QualityMetric.VALIDITY: 1.0,
            QualityMetric.ACCURACY: 1.0,
            QualityMetric.CONSISTENCY: 1.0,
            QualityMetric.TIMELINESS: 1.0,
            QualityMetric.UNIQUENESS: 1.0
        }
    
    async def _validate_generic_list(self, response: DataResponse) -> Tuple[List[ValidationIssue], Dict[QualityMetric, float]]:
        """Validate generic list data"""
        data = response.data
        
        if not data:
            return [ValidationIssue(
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                message="Empty data list",
                source=response.source
            )], {metric: 0.5 for metric in QualityMetric}
        
        # Basic validation - just check it's not empty
        return [], {
            QualityMetric.COMPLETENESS: 1.0,
            QualityMetric.VALIDITY: 1.0,
            QualityMetric.ACCURACY: 1.0,
            QualityMetric.CONSISTENCY: 1.0,
            QualityMetric.TIMELINESS: 1.0,
            QualityMetric.UNIQUENESS: 1.0
        }
    
    def _calculate_overall_score(self, metric_scores: Dict[QualityMetric, float]) -> float:
        """Calculate overall quality score"""
        if not metric_scores:
            return 0.0
        
        # Weighted average of metrics
        weights = {
            QualityMetric.COMPLETENESS: 0.25,
            QualityMetric.VALIDITY: 0.25,
            QualityMetric.ACCURACY: 0.20,
            QualityMetric.CONSISTENCY: 0.15,
            QualityMetric.TIMELINESS: 0.10,
            QualityMetric.UNIQUENESS: 0.05
        }
        
        weighted_sum = sum(
            metric_scores.get(metric, 0.0) * weight 
            for metric, weight in weights.items()
        )
        
        return weighted_sum
    
    def _update_historical_data(self, response: DataResponse):
        """Update historical data for future validations"""
        if response.status != "success" or not response.data:
            return
        
        # Extract key data points
        symbol = self._extract_symbol(response.data)
        if not symbol:
            return
        
        key = f"{response.source}:{symbol}"
        
        # Update price history
        price = self._extract_price(response.data)
        if price:
            self.price_history[key].append(price)
        
        # Update volume history
        volume = self._extract_volume(response.data)
        if volume:
            self.volume_history[key].append(volume)
        
        # Update timestamp history
        self.timestamp_history[key].append(response.timestamp)
    
    def _update_validation_stats(self, overall_score: float, issues: List[ValidationIssue]):
        """Update validation statistics"""
        self.validation_stats['total_validations'] += 1
        
        if overall_score < 0.8:
            self.validation_stats['failed_validations'] += 1
        
        critical_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        self.validation_stats['critical_issues'] += critical_issues
        
        # Update running average
        total = self.validation_stats['total_validations']
        current_avg = self.validation_stats['avg_quality_score']
        self.validation_stats['avg_quality_score'] = (
            (current_avg * (total - 1) + overall_score) / total
        )
    
    def _create_failed_report(self, response: DataResponse, reason: str) -> QualityReport:
        """Create a failed quality report"""
        return QualityReport(
            source=response.source,
            symbol=None,
            timestamp=response.timestamp,
            overall_score=0.0,
            metric_scores={metric: 0.0 for metric in QualityMetric},
            issues=[ValidationIssue(
                metric=QualityMetric.VALIDITY,
                severity=ValidationSeverity.CRITICAL,
                message=reason,
                source=response.source
            )],
            data_points=0,
            assessment_duration_ms=0.0
        )
    
    def _extract_symbol(self, data: Any) -> Optional[str]:
        """Extract symbol from data"""
        if isinstance(data, dict):
            return data.get('symbol')
        return None
    
    def _extract_price(self, data: Any) -> Optional[float]:
        """Extract price from data"""
        if isinstance(data, dict):
            return data.get('last') or data.get('price')
        return None
    
    def _extract_volume(self, data: Any) -> Optional[float]:
        """Extract volume from data"""
        if isinstance(data, dict):
            return data.get('volume') or data.get('baseVolume')
        return None
    
    def _count_data_points(self, data: Any) -> int:
        """Count data points in response"""
        if isinstance(data, dict):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        return 1
    
    def _is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol format is valid"""
        if symbol in self.known_symbols:
            return True
        
        # Check against patterns
        for pattern_name, pattern in self.symbol_patterns.items():
            import re
            if re.match(pattern, symbol):
                return True
        
        # Basic format check (e.g., BTC/USD)
        if '/' in symbol and len(symbol.split('/')) == 2:
            base, quote = symbol.split('/')
            if len(base) >= 2 and len(quote) >= 3:
                return True
        
        return False
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation statistics summary"""
        return {
            'total_validations': self.validation_stats['total_validations'],
            'failed_validations': self.validation_stats['failed_validations'],
            'success_rate': (
                (self.validation_stats['total_validations'] - self.validation_stats['failed_validations']) /
                max(1, self.validation_stats['total_validations'])
            ),
            'critical_issues': self.validation_stats['critical_issues'],
            'avg_quality_score': self.validation_stats['avg_quality_score'],
            'thresholds': self.thresholds
        }


# Factory function
def create_data_quality_validator(config: Dict[str, Any]) -> DataQualityValidator:
    """Create configured data quality validator"""
    return DataQualityValidator(config)