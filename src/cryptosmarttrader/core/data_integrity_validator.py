#!/usr/bin/env python3
"""
Advanced Data Integrity Validator
Implements sophisticated detection of synthetic, forward-filled, and corrupted data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import re
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')

from ..core.logging_manager import get_logger

@dataclass
class IntegrityViolation:
    """Data integrity violation record"""
    violation_type: str
    field: str
    value: Any
    confidence: float  # 0.0 to 1.0
    description: str
    timestamp: datetime

class DataIntegrityValidator:
    """Advanced validator for detecting synthetic and corrupted data"""

    def __init__(self):
        self.logger = get_logger()

        # Synthetic data patterns
        self.synthetic_patterns = {
            'obvious_placeholders': [0, -1, 999999, 1000000, -999999],
            'common_test_values': [123.45, 100.00, 0.01, 99.99],
            'repeated_decimals': r'\.(\d)\1{4,}',  # .00000, .11111, etc
            'sequential_numbers': r'\d{6,}',  # Long sequences
        }

        # Forward-fill detection thresholds
        self.ff_detection = {
            'max_identical_consecutive': 3,  # Max identical values in sequence
            'variance_threshold': 0.001,     # Min variance for realistic data
            'correlation_threshold': 0.99    # Max correlation for realistic data
        }

        # Data validation ranges
        self.validation_ranges = {
            'price': {'min': 0.000001, 'max': 10000000},
            'volume': {'min': 0, 'max': 1e12},
            'market_cap': {'min': 1000, 'max': 1e15},
            'bid': {'min': 0.000001, 'max': 10000000},
            'ask': {'min': 0.000001, 'max': 10000000},
            'high': {'min': 0.000001, 'max': 10000000},
            'low': {'min': 0.000001, 'max': 10000000},
            'open': {'min': 0.000001, 'max': 10000000},
            'close': {'min': 0.000001, 'max': 10000000}
        }

    def validate_data_integrity(
        self,
        data: Dict[str, Any],
        symbol: str,
        historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, List[IntegrityViolation]]:
        """
        Comprehensive data integrity validation
        Returns (is_valid, violations_list)
        """
        violations = []

        # 1. Check for obvious synthetic data
        synthetic_violations = self._detect_synthetic_data(data, symbol)
        violations.extend(synthetic_violations)

        # 2. Check value ranges and logical consistency
        range_violations = self._validate_value_ranges(data, symbol)
        violations.extend(range_violations)

        # 3. Check for forward-filled data patterns
        if historical_data:
            ff_violations = self._detect_forward_fill(data, historical_data, symbol)
            violations.extend(ff_violations)

        # 4. Check for data corruption patterns
        corruption_violations = self._detect_data_corruption(data, symbol)
        violations.extend(corruption_violations)

        # 5. Statistical anomaly detection
        stat_violations = self._detect_statistical_anomalies(data, symbol)
        violations.extend(stat_violations)

        # Determine overall validity
        critical_violations = [v for v in violations if v.confidence >= 0.8]
        is_valid = len(critical_violations) == 0

        if violations:
            self.logger.warning(
                f"Data integrity violations detected for {symbol}",
                extra={
                    'symbol': symbol,
                    'total_violations': len(violations),
                    'critical_violations': len(critical_violations),
                    'is_valid': is_valid,
                    'violation_types': [v.violation_type for v in violations]
                }
            )

        return is_valid, violations

    def _detect_synthetic_data(self, data: Dict[str, Any], symbol: str) -> List[IntegrityViolation]:
        """Detect synthetic/placeholder data patterns"""
        violations = []

        for field, value in data.items():
            if value is None:
                continue

            # Check for obvious placeholder values
            if isinstance(value, (int, float)):
                if value in self.synthetic_patterns['obvious_placeholders']:
                    violations.append(IntegrityViolation(
                        violation_type='synthetic_placeholder',
                        field=field,
                        value=value,
                        confidence=1.0,
                        description=f"Obvious placeholder value: {value}",
                        timestamp=datetime.now()
                    ))

                # Check for common test values
                if value in self.synthetic_patterns['common_test_values']:
                    violations.append(IntegrityViolation(
                        violation_type='synthetic_test_value',
                        field=field,
                        value=value,
                        confidence=0.7,
                        description=f"Common test value: {value}",
                        timestamp=datetime.now()
                    ))

                # Check for repeated decimal patterns
                str_value = str(value)
                if '.' in str_value:
                    decimal_part = str_value.split('.')[1]
                    if len(decimal_part) >= 5:
                        if re.search(self.synthetic_patterns['repeated_decimals'], str_value):
                            violations.append(IntegrityViolation(
                                violation_type='synthetic_repeated_decimals',
                                field=field,
                                value=value,
                                confidence=0.9,
                                description=f"Repeated decimal pattern in {value}",
                                timestamp=datetime.now()
                            ))

            elif isinstance(value, str):
                # Check for synthetic string indicators
                synthetic_indicators = [
                    'null', 'none', 'n/a', 'placeholder', 'synthetic',
                    'forward_fill', 'interpolated', 'estimated', 'mock', 'test'
                ]
                if value.lower().strip() in synthetic_indicators:
                    violations.append(IntegrityViolation(
                        violation_type='synthetic_string',
                        field=field,
                        value=value,
                        confidence=1.0,
                        description=f"Synthetic string indicator: {value}",
                        timestamp=datetime.now()
                    ))

        return violations

    def _validate_value_ranges(self, data: Dict[str, Any], symbol: str) -> List[IntegrityViolation]:
        """Validate that values are within realistic ranges"""
        violations = []

        for field, value in data.items():
            if field in self.validation_ranges and isinstance(value, (int, float)):
                range_config = self.validation_ranges[field]

                if value < range_config['min'] or value > range_config['max']:
                    violations.append(IntegrityViolation(
                        violation_type='value_out_of_range',
                        field=field,
                        value=value,
                        confidence=0.9,
                        description=f"{field} value {value} outside valid range "
                                  f"[{range_config['min']}, {range_config['max']}]",
                        timestamp=datetime.now()
                    ))

        # Check logical relationships
        try:
            if all(k in data for k in ['high', 'low', 'open', 'close']):
                high, low, open_price, close = data['high'], data['low'], data['open'], data['close']

                if high < low:
                    violations.append(IntegrityViolation(
                        violation_type='logical_inconsistency',
                        field='high_low',
                        value={'high': high, 'low': low},
                        confidence=1.0,
                        description=f"High ({high}) less than low ({low})",
                        timestamp=datetime.now()
                    ))

                if open_price < low or open_price > high:
                    violations.append(IntegrityViolation(
                        violation_type='logical_inconsistency',
                        field='open_range',
                        value={'open': open_price, 'high': high, 'low': low},
                        confidence=1.0,
                        description=f"Open ({open_price}) outside high-low range",
                        timestamp=datetime.now()
                    ))

                if close < low or close > high:
                    violations.append(IntegrityViolation(
                        violation_type='logical_inconsistency',
                        field='close_range',
                        value={'close': close, 'high': high, 'low': low},
                        confidence=1.0,
                        description=f"Close ({close}) outside high-low range",
                        timestamp=datetime.now()
                    ))

            if all(k in data for k in ['bid', 'ask']):
                bid, ask = data['bid'], data['ask']
                if bid >= ask:
                    violations.append(IntegrityViolation(
                        violation_type='logical_inconsistency',
                        field='bid_ask_spread',
                        value={'bid': bid, 'ask': ask},
                        confidence=1.0,
                        description=f"Bid ({bid}) >= ask ({ask}) - invalid spread",
                        timestamp=datetime.now()
                    ))

        except (KeyError, TypeError, ValueError):
            # Skip logical checks if data is incomplete or malformed
            pass

        return violations

    def _detect_forward_fill(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        symbol: str
    ) -> List[IntegrityViolation]:
        """Detect forward-filled or interpolated data"""
        violations = []

        if len(historical_data) < 3:
            return violations  # Need sufficient history

        # Combine current with recent historical data
        all_data = historical_data + [current_data]

        # Check each numeric field for forward-fill patterns
        for field in ['price', 'volume', 'high', 'low', 'open', 'close', 'bid', 'ask']:
            if field not in current_data:
                continue

            # Extract values for this field
            values = []
            for record in all_data:
                if field in record and isinstance(record[field], (int, float)):
                    values.append(record[field])

            if len(values) < 3:
                continue

            # Check for identical consecutive values (forward-fill indicator)
            consecutive_identical = 1
            max_consecutive = 1

            for i in range(1, len(values)):
                if abs(values[i] - values[i-1]) < 1e-10:  # Essentially identical
                    consecutive_identical += 1
                    max_consecutive = max(max_consecutive, consecutive_identical)
                else:
                    consecutive_identical = 1

            if max_consecutive > self.ff_detection['max_identical_consecutive']:
                violations.append(IntegrityViolation(
                    violation_type='forward_fill_pattern',
                    field=field,
                    value=values[-1],
                    confidence=0.8,
                    description=f"{max_consecutive} consecutive identical values in {field}",
                    timestamp=datetime.now()
                ))

            # Check for unrealistic low variance (interpolation indicator)
            if len(values) >= 5:
                variance = np.var(values[-5:])  # Last 5 values
                mean_value = np.mean(values[-5:])

                if mean_value > 0:
                    relative_variance = variance / (mean_value ** 2)
                    if relative_variance < self.ff_detection['variance_threshold']:
                        violations.append(IntegrityViolation(
                            violation_type='low_variance_interpolation',
                            field=field,
                            value=current_data[field],
                            confidence=0.6,
                            description=f"Unrealistically low variance in {field}: {relative_variance:.6f}",
                            timestamp=datetime.now()
                        ))

        return violations

    def _detect_data_corruption(self, data: Dict[str, Any], symbol: str) -> List[IntegrityViolation]:
        """Detect data corruption patterns"""
        violations = []

        for field, value in data.items():
            if not isinstance(value, (int, float, str)):
                continue

            # Check for NaN or infinite values
            if isinstance(value, float):
                if math.isnan(value):
                    violations.append(IntegrityViolation(
                        violation_type='data_corruption',
                        field=field,
                        value=value,
                        confidence=1.0,
                        description=f"NaN value in {field}",
                        timestamp=datetime.now()
                    ))

                if math.isinf(value):
                    violations.append(IntegrityViolation(
                        violation_type='data_corruption',
                        field=field,
                        value=value,
                        confidence=1.0,
                        description=f"Infinite value in {field}",
                        timestamp=datetime.now()
                    ))

            # Check for malformed string data
            if isinstance(value, str):
                # Check for obviously corrupted strings
                if len(value) > 100:  # Unreasonably long for financial data
                    violations.append(IntegrityViolation(
                        violation_type='data_corruption',
                        field=field,
                        value=value[:50] + "...",  # Truncate for logging
                        confidence=0.8,
                        description=f"Unreasonably long string in {field} ({len(value)} chars)",
                        timestamp=datetime.now()
                    ))

                # Check for control characters or binary data
                if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
                    violations.append(IntegrityViolation(
                        violation_type='data_corruption',
                        field=field,
                        value=repr(value),
                        confidence=0.9,
                        description=f"Control characters detected in {field}",
                        timestamp=datetime.now()
                    ))

        return violations

    def _detect_statistical_anomalies(self, data: Dict[str, Any], symbol: str) -> List[IntegrityViolation]:
        """Detect statistical anomalies that suggest synthetic data"""
        violations = []

        # Check for suspiciously round numbers
        for field in ['price', 'volume', 'market_cap']:
            if field not in data or not isinstance(data[field], (int, float)):
                continue

            value = data[field]

            # Check if value is suspiciously round
            if value > 1:
                # For values > 1, check if it's a round number
                str_value = str(value)
                if '.' not in str_value and str_value.endswith('000'):
                    trailing_zeros = len(str_value) - len(str_value.rstrip('0'))
                    if trailing_zeros >= 3:
                        violations.append(IntegrityViolation(
                            violation_type='suspicious_round_number',
                            field=field,
                            value=value,
                            confidence=0.4,
                            description=f"Suspiciously round number in {field}: {value}",
                            timestamp=datetime.now()
                        ))

            # Check for values that are exact powers of 10
            if value > 0:
                log_value = math.log10(value)
                if abs(log_value - round(log_value)) < 0.001:  # Very close to power of 10
                    violations.append(IntegrityViolation(
                        violation_type='power_of_ten',
                        field=field,
                        value=value,
                        confidence=0.5,
                        description=f"Value is exact power of 10 in {field}: {value}",
                        timestamp=datetime.now()
                    ))

        return violations

    def validate_time_series_integrity(
        self,
        time_series_data: List[Dict[str, Any]],
        symbol: str
    ) -> Tuple[bool, List[IntegrityViolation]]:
        """Validate integrity of time series data"""
        violations = []

        if len(time_series_data) < 2:
            return True, violations

        # Convert to DataFrame for easier analysis
        try:
            df = pd.DataFrame(time_series_data)

            # Check for missing timestamps
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')

                # Check for duplicate timestamps
                duplicates = df[df.duplicated('timestamp', keep=False)]
                if not duplicates.empty:
                    violations.append(IntegrityViolation(
                        violation_type='duplicate_timestamps',
                        field='timestamp',
                        value=len(duplicates),
                        confidence=0.9,
                        description=f"{len(duplicates)} duplicate timestamps found",
                        timestamp=datetime.now()
                    ))

                # Check for irregular time intervals
                time_diffs = df['timestamp'].diff().dropna()
                if len(time_diffs) > 1:
                    median_diff = time_diffs.median()
                    irregular_diffs = time_diffs[abs(time_diffs - median_diff) > median_diff * 0.5]

                    if len(irregular_diffs) > len(time_diffs) * 0.2:  # More than 20% irregular
                        violations.append(IntegrityViolation(
                            violation_type='irregular_time_intervals',
                            field='timestamp',
                            value=len(irregular_diffs),
                            confidence=0.6,
                            description=f"Irregular time intervals detected: {len(irregular_diffs)} anomalies",
                            timestamp=datetime.now()
                        ))

            # Check for data gaps or suspicious patterns in price data
            for price_field in ['price', 'close', 'open']:
                if price_field in df.columns:
                    prices = df[price_field].dropna()

                    if len(prices) > 5:
                        # Check for flat lines (no price movement)
                        flat_sequences = 0
                        current_flat = 1

                        for i in range(1, len(prices)):
                            if abs(prices.iloc[i] - prices.iloc[i-1]) < 1e-10:
                                current_flat += 1
                            else:
                                if current_flat >= 5:
                                    flat_sequences += 1
                                current_flat = 1

                        if flat_sequences > 0:
                            violations.append(IntegrityViolation(
                                violation_type='flat_price_sequences',
                                field=price_field,
                                value=flat_sequences,
                                confidence=0.7,
                                description=f"{flat_sequences} flat price sequences in {price_field}",
                                timestamp=datetime.now()
                            ))

        except Exception as e:
            violations.append(IntegrityViolation(
                violation_type='time_series_analysis_error',
                field='general',
                value=str(e),
                confidence=0.5,
                description=f"Error analyzing time series: {e}",
                timestamp=datetime.now()
            ))

        # Time series is valid if no high-confidence violations
        critical_violations = [v for v in violations if v.confidence >= 0.8]
        is_valid = len(critical_violations) == 0

        return is_valid, violations

# Global instance
_data_integrity_validator = None

def get_data_integrity_validator() -> DataIntegrityValidator:
    """Get global data integrity validator instance"""
    global _data_integrity_validator
    if _data_integrity_validator is None:
        _data_integrity_validator = DataIntegrityValidator()
    return _data_integrity_validator
