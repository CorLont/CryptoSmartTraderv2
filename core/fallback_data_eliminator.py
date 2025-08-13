#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Fallback Data Eliminator
Complete elimination of fallback/synthetic data with strict validation
"""

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

class DataValidationLevel(Enum):
    STRICT = "strict"          # Zero tolerance for fallback/synthetic data
    MODERATE = "moderate"      # Warning for fallback, error for synthetic
    PERMISSIVE = "permissive"  # Log only, allow fallback in emergencies

class DataSource(Enum):
    AUTHENTIC = "authentic"    # Real data from exchanges/APIs
    FALLBACK = "fallback"     # Cached/historical fallback data
    SYNTHETIC = "synthetic"   # Generated/interpolated data
    UNKNOWN = "unknown"       # Source cannot be determined

@dataclass
class DataValidationRule:
    """Data validation rule configuration"""
    name: str
    check_function: Callable[[Any], bool]
    error_message: str
    validation_level: DataValidationLevel = DataValidationLevel.STRICT
    allow_empty: bool = False
    require_recent: bool = True
    max_age_seconds: int = 300  # 5 minutes default

@dataclass
class ValidationResult:
    """Data validation result"""
    is_valid: bool
    data_source: DataSource
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    data_age_seconds: Optional[float] = None
    data_quality_score: float = 1.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class FallbackDataEliminator:
    """Enterprise-grade fallback and synthetic data eliminator"""
    
    def __init__(self, validation_level: DataValidationLevel = DataValidationLevel.STRICT):
        self.validation_level = validation_level
        self.logger = logging.getLogger(f"{__name__}.FallbackDataEliminator")
        
        # Tracking and statistics
        self.validation_stats = {
            "total_validations": 0,
            "authentic_data_count": 0,
            "fallback_data_rejected": 0,
            "synthetic_data_rejected": 0,
            "validation_failures": 0,
            "last_validation_time": None
        }
        
        # Validation rules registry
        self.validation_rules: Dict[str, List[DataValidationRule]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize standard validation rules
        self._initialize_standard_rules()
        
        self.logger.info(f"Fallback Data Eliminator initialized with {validation_level.value} validation")
    
    def _initialize_standard_rules(self):
        """Initialize standard data validation rules"""
        # Price data validation rules
        price_rules = [
            DataValidationRule(
                name="no_zero_prices",
                check_function=lambda data: self._check_no_zero_values(data, 'price'),
                error_message="Zero prices detected - indicates synthetic/fallback data"
            ),
            DataValidationRule(
                name="realistic_price_range",
                check_function=lambda data: self._check_realistic_prices(data),
                error_message="Unrealistic price values detected"
            ),
            DataValidationRule(
                name="no_constant_prices",
                check_function=lambda data: self._check_no_constant_values(data, 'price'),
                error_message="Constant price values indicate synthetic data"
            ),
            DataValidationRule(
                name="price_volatility_check",
                check_function=lambda data: self._check_price_volatility(data),
                error_message="Artificial price volatility patterns detected"
            )
        ]
        
        # Volume data validation rules
        volume_rules = [
            DataValidationRule(
                name="no_zero_volume",
                check_function=lambda data: self._check_no_zero_values(data, 'volume'),
                error_message="Zero volume detected - indicates synthetic data"
            ),
            DataValidationRule(
                name="realistic_volume_range", 
                check_function=lambda data: self._check_realistic_volumes(data),
                error_message="Unrealistic volume values detected"
            ),
            DataValidationRule(
                name="volume_distribution_check",
                check_function=lambda data: self._check_volume_distribution(data),
                error_message="Artificial volume distribution detected"
            )
        ]
        
        # Timestamp validation rules
        timestamp_rules = [
            DataValidationRule(
                name="recent_timestamps",
                check_function=lambda data: self._check_timestamp_recency(data),
                error_message="Stale data detected - may be fallback data"
            ),
            DataValidationRule(
                name="sequential_timestamps",
                check_function=lambda data: self._check_timestamp_sequence(data),
                error_message="Non-sequential timestamps indicate synthetic data"
            ),
            DataValidationRule(
                name="realistic_intervals",
                check_function=lambda data: self._check_realistic_intervals(data),
                error_message="Unrealistic timestamp intervals detected"
            )
        ]
        
        # General data validation rules
        general_rules = [
            DataValidationRule(
                name="no_missing_values",
                check_function=lambda data: self._check_no_missing_values(data),
                error_message="Missing values detected - incomplete authentic data"
            ),
            DataValidationRule(
                name="data_completeness",
                check_function=lambda data: self._check_data_completeness(data),
                error_message="Incomplete data suggests fallback/synthetic source"
            ),
            DataValidationRule(
                name="authentic_data_patterns",
                check_function=lambda data: self._check_authentic_patterns(data),
                error_message="Data patterns suggest synthetic generation"
            )
        ]
        
        # Register rule sets
        self.validation_rules["price_data"] = price_rules
        self.validation_rules["volume_data"] = volume_rules
        self.validation_rules["timestamp_data"] = timestamp_rules
        self.validation_rules["general"] = general_rules
        
        self.logger.info(f"Initialized {sum(len(rules) for rules in self.validation_rules.values())} validation rules")
    
    def validate_data(
        self, 
        data: Any, 
        data_type: str = "general",
        require_authentic: bool = True,
        custom_rules: Optional[List[DataValidationRule]] = None
    ) -> ValidationResult:
        """
        Validate data for authenticity and quality
        
        Args:
            data: Data to validate (DataFrame, dict, array, etc.)
            data_type: Type of data for specific validation rules
            require_authentic: Whether to require authentic data source
            custom_rules: Additional custom validation rules
            
        Returns:
            ValidationResult with validation outcome
        """
        with self._lock:
            self.validation_stats["total_validations"] += 1
            self.validation_stats["last_validation_time"] = datetime.now()
            
            result = ValidationResult(
                is_valid=True,
                data_source=DataSource.UNKNOWN,
                data_quality_score=1.0
            )
            
            try:
                # Check if data exists and is not None
                if data is None:
                    result.is_valid = False
                    result.validation_errors.append("Data is None - no authentic data available")
                    result.data_source = DataSource.UNKNOWN
                    self.validation_stats["validation_failures"] += 1
                    return result
                
                # Determine data source
                result.data_source = self._determine_data_source(data)
                
                # Apply strict validation based on level and source
                if self.validation_level == DataValidationLevel.STRICT:
                    if result.data_source in (DataSource.FALLBACK, DataSource.SYNTHETIC):
                        result.is_valid = False
                        result.validation_errors.append(
                            f"STRICT MODE: {result.data_source.value.title()} data rejected"
                        )
                        self.validation_stats["fallback_data_rejected" if result.data_source == DataSource.FALLBACK else "synthetic_data_rejected"] += 1
                        return result
                
                # Get applicable validation rules
                applicable_rules = self.validation_rules.get(data_type, self.validation_rules["general"])
                if custom_rules:
                    applicable_rules.extend(custom_rules)
                
                # Apply validation rules
                for rule in applicable_rules:
                    try:
                        if not rule.check_function(data):
                            if rule.validation_level == DataValidationLevel.STRICT or self.validation_level == DataValidationLevel.STRICT:
                                result.is_valid = False
                                result.validation_errors.append(rule.error_message)
                            else:
                                result.validation_warnings.append(rule.error_message)
                                result.data_quality_score *= 0.9  # Reduce quality score
                    
                    except Exception as e:
                        self.logger.warning(f"Validation rule '{rule.name}' failed: {e}")
                        result.validation_warnings.append(f"Validation rule error: {rule.name}")
                        result.data_quality_score *= 0.95
                
                # Calculate data age
                result.data_age_seconds = self._calculate_data_age(data)
                
                # Check data recency requirements
                if require_authentic and result.data_age_seconds and result.data_age_seconds > 300:  # 5 minutes
                    if self.validation_level == DataValidationLevel.STRICT:
                        result.is_valid = False
                        result.validation_errors.append(f"Data too old ({result.data_age_seconds:.1f}s) - may be fallback")
                    else:
                        result.validation_warnings.append(f"Data age warning: {result.data_age_seconds:.1f}s")
                        result.data_quality_score *= 0.8
                
                # Final validation outcome
                if result.is_valid and result.data_source == DataSource.AUTHENTIC:
                    self.validation_stats["authentic_data_count"] += 1
                elif not result.is_valid:
                    self.validation_stats["validation_failures"] += 1
                
                # Log validation result
                self._log_validation_result(result, data_type)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Data validation error: {e}")
                result.is_valid = False
                result.validation_errors.append(f"Validation process error: {str(e)}")
                self.validation_stats["validation_failures"] += 1
                return result
    
    def _determine_data_source(self, data: Any) -> DataSource:
        """Determine the source of the data"""
        try:
            # Check for explicit source markers
            if hasattr(data, 'attrs') and 'data_source' in data.attrs:
                source_str = data.attrs['data_source'].lower()
                if 'synthetic' in source_str or 'generated' in source_str:
                    return DataSource.SYNTHETIC
                elif 'fallback' in source_str or 'cached' in source_str:
                    return DataSource.FALLBACK
                elif 'authentic' in source_str or 'live' in source_str:
                    return DataSource.AUTHENTIC
            
            # Check DataFrame metadata
            if isinstance(data, pd.DataFrame):
                # Look for source indicators in column names or metadata
                if any('synthetic' in str(col).lower() for col in data.columns):
                    return DataSource.SYNTHETIC
                elif any('fallback' in str(col).lower() or 'cached' in str(col).lower() for col in data.columns):
                    return DataSource.FALLBACK
                
                # Check for patterns that suggest synthetic data
                if self._has_synthetic_patterns(data):
                    return DataSource.SYNTHETIC
                
                # Check for fresh timestamps (likely authentic)
                if 'timestamp' in data.columns or 'time' in data.columns:
                    time_col = 'timestamp' if 'timestamp' in data.columns else 'time'
                    if not data.empty:
                        latest_time = pd.to_datetime(data[time_col]).max()
                        if isinstance(latest_time, pd.Timestamp):
                            age_minutes = (datetime.now() - latest_time.to_pydatetime()).total_seconds() / 60
                            if age_minutes < 10:  # Fresh data, likely authentic
                                return DataSource.AUTHENTIC
            
            # Default to unknown if cannot determine
            return DataSource.UNKNOWN
            
        except Exception as e:
            self.logger.debug(f"Error determining data source: {e}")
            return DataSource.UNKNOWN
    
    def _has_synthetic_patterns(self, data: pd.DataFrame) -> bool:
        """Check for patterns that suggest synthetic data generation"""
        try:
            # Check for perfectly uniform distributions (synthetic indicator)
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if len(data[col].unique()) == len(data) and len(data) > 10:
                    # All unique values in a large dataset - suspicious
                    continue
                
                # Check for repeated patterns
                if len(data) > 20:
                    # Look for repeated sequences
                    values = data[col].values
                    for window_size in [5, 10]:
                        if len(values) >= window_size * 3:
                            first_window = values[:window_size]
                            second_window = values[window_size:window_size*2]
                            if np.array_equal(first_window, second_window):
                                return True
            
            # Check for artificial precision patterns
            for col in numeric_columns:
                if data[col].dtype == float:
                    # Count decimal places
                    decimal_places = []
                    for val in data[col].head(20):
                        if pd.notna(val):
                            str_val = f"{val:.10f}".rstrip('0')
                            if '.' in str_val:
                                decimal_places.append(len(str_val.split('.')[1]))
                    
                    # If all values have same decimal places, might be synthetic
                    if len(set(decimal_places)) == 1 and decimal_places[0] > 6:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _check_no_zero_values(self, data: Any, column: str) -> bool:
        """Check for zero values that indicate synthetic/fallback data"""
        try:
            if isinstance(data, pd.DataFrame) and column in data.columns:
                return (data[column] > 0).all()
            elif isinstance(data, dict) and column in data:
                values = data[column]
                if isinstance(values, (list, np.ndarray)):
                    return all(v > 0 for v in values if v is not None)
                else:
                    return values > 0
            return True
        except Exception:
            return False
    
    def _check_realistic_prices(self, data: Any) -> bool:
        """Check for realistic price ranges"""
        try:
            if isinstance(data, pd.DataFrame):
                price_cols = [col for col in data.columns if 'price' in col.lower()]
                for col in price_cols:
                    prices = data[col].dropna()
                    if len(prices) == 0:
                        continue
                    
                    # Check for unrealistic ranges
                    if prices.min() < 0.000001:  # Too small for realistic crypto
                        return False
                    if prices.max() > 1000000:  # Suspiciously high
                        return False
                    
                    # Check for unrealistic precision
                    if all(len(f"{p:.10f}".rstrip('0').split('.')[-1]) > 8 for p in prices.head(10)):
                        return False
            
            return True
        except Exception:
            return False
    
    def _check_no_constant_values(self, data: Any, column: str) -> bool:
        """Check for constant values that indicate synthetic data"""
        try:
            if isinstance(data, pd.DataFrame) and column in data.columns:
                values = data[column].dropna()
                if len(values) < 2:
                    return True
                return len(values.unique()) > 1
            return True
        except Exception:
            return False
    
    def _check_price_volatility(self, data: Any) -> bool:
        """Check for artificial price volatility patterns"""
        try:
            if isinstance(data, pd.DataFrame):
                price_cols = [col for col in data.columns if 'price' in col.lower()]
                for col in price_cols:
                    prices = data[col].dropna()
                    if len(prices) < 10:
                        continue
                    
                    # Calculate returns
                    returns = prices.pct_change().dropna()
                    if len(returns) == 0:
                        continue
                    
                    # Check for suspiciously uniform volatility
                    volatility = returns.std()
                    if volatility == 0:  # No volatility - synthetic
                        return False
                    
                    # Check for artificial patterns in returns
                    if len(set(np.round(returns.values, 6))) == 1:  # All same return
                        return False
            
            return True
        except Exception:
            return False
    
    def _check_realistic_volumes(self, data: Any) -> bool:
        """Check for realistic volume ranges"""
        try:
            if isinstance(data, pd.DataFrame):
                volume_cols = [col for col in data.columns if 'volume' in col.lower()]
                for col in volume_cols:
                    volumes = data[col].dropna()
                    if len(volumes) == 0:
                        continue
                    
                    # Check for zero volumes (suspicious)
                    if (volumes == 0).any():
                        return False
                    
                    # Check for negative volumes
                    if (volumes < 0).any():
                        return False
            
            return True
        except Exception:
            return False
    
    def _check_volume_distribution(self, data: Any) -> bool:
        """Check for artificial volume distributions"""
        try:
            if isinstance(data, pd.DataFrame):
                volume_cols = [col for col in data.columns if 'volume' in col.lower()]
                for col in volume_cols:
                    volumes = data[col].dropna()
                    if len(volumes) < 10:
                        continue
                    
                    # Check for suspiciously uniform distribution
                    if len(volumes.unique()) == len(volumes) and len(volumes) > 50:
                        # Too many unique values might indicate synthetic data
                        continue
                    
                    # Check for artificial patterns
                    volume_diff = volumes.diff().dropna()
                    if len(volume_diff.unique()) == 1 and len(volume_diff) > 5:
                        return False  # Constant differences indicate synthetic
            
            return True
        except Exception:
            return False
    
    def _check_timestamp_recency(self, data: Any) -> bool:
        """Check timestamp recency"""
        try:
            age_seconds = self._calculate_data_age(data)
            if age_seconds is None:
                return True  # Cannot determine age
            
            # Data older than 10 minutes is suspicious for real-time trading
            return age_seconds < 600
            
        except Exception:
            return True
    
    def _check_timestamp_sequence(self, data: Any) -> bool:
        """Check for sequential timestamps"""
        try:
            if isinstance(data, pd.DataFrame):
                time_cols = [col for col in data.columns 
                           if any(word in col.lower() for word in ['time', 'timestamp', 'date'])]
                
                for col in time_cols:
                    timestamps = pd.to_datetime(data[col], errors='coerce').dropna()
                    if len(timestamps) < 2:
                        continue
                    
                    # Check if timestamps are sorted
                    if not timestamps.is_monotonic_increasing:
                        return False
                    
                    # Check for duplicate timestamps (suspicious)
                    if len(timestamps) != len(timestamps.unique()):
                        return False
            
            return True
        except Exception:
            return True
    
    def _check_realistic_intervals(self, data: Any) -> bool:
        """Check for realistic timestamp intervals"""
        try:
            if isinstance(data, pd.DataFrame):
                time_cols = [col for col in data.columns 
                           if any(word in col.lower() for word in ['time', 'timestamp', 'date'])]
                
                for col in time_cols:
                    timestamps = pd.to_datetime(data[col], errors='coerce').dropna()
                    if len(timestamps) < 2:
                        continue
                    
                    # Calculate intervals
                    intervals = timestamps.diff().dropna()
                    interval_seconds = intervals.dt.total_seconds()
                    
                    # Check for unrealistic intervals
                    if (interval_seconds <= 0).any():  # Non-positive intervals
                        return False
                    
                    # Check for suspiciously uniform intervals
                    unique_intervals = interval_seconds.unique()
                    if len(unique_intervals) == 1 and len(interval_seconds) > 10:
                        # Perfectly uniform intervals might be synthetic
                        interval_value = unique_intervals[0]
                        if interval_value in [1, 5, 10, 30, 60, 300, 600]:  # Common synthetic intervals
                            return False
            
            return True
        except Exception:
            return True
    
    def _check_no_missing_values(self, data: Any) -> bool:
        """Check for missing values"""
        try:
            if isinstance(data, pd.DataFrame):
                return not data.isnull().any().any()
            elif isinstance(data, dict):
                return all(v is not None for v in data.values())
            elif isinstance(data, (list, np.ndarray)):
                return all(v is not None for v in data)
            return True
        except Exception:
            return False
    
    def _check_data_completeness(self, data: Any) -> bool:
        """Check for data completeness"""
        try:
            if isinstance(data, pd.DataFrame):
                # Check if we have minimum required columns
                required_cols = ['price', 'volume', 'timestamp']
                available_cols = [col for col in data.columns 
                                if any(req in col.lower() for req in required_cols)]
                
                # Should have at least price and timestamp data
                has_price = any('price' in col.lower() for col in data.columns)
                has_time = any(word in col.lower() for col in data.columns 
                             for word in ['time', 'timestamp', 'date'])
                
                return has_price and has_time
            
            return True
        except Exception:
            return False
    
    def _check_authentic_patterns(self, data: Any) -> bool:
        """Check for patterns that indicate authentic market data"""
        try:
            if isinstance(data, pd.DataFrame):
                # Real market data should have some irregularity and noise
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if len(data[col].dropna()) < 10:
                        continue
                    
                    values = data[col].dropna().values
                    
                    # Check for some randomness/noise (authentic market data)
                    # Calculate coefficient of variation
                    if np.std(values) == 0:
                        return False  # No variation - suspicious
                    
                    cv = np.std(values) / np.abs(np.mean(values))
                    if cv < 0.001:  # Too low variation for real market data
                        return False
            
            return True
        except Exception:
            return True
    
    def _calculate_data_age(self, data: Any) -> Optional[float]:
        """Calculate age of data in seconds"""
        try:
            latest_timestamp = None
            
            if isinstance(data, pd.DataFrame):
                time_cols = [col for col in data.columns 
                           if any(word in col.lower() for word in ['time', 'timestamp', 'date'])]
                
                for col in time_cols:
                    timestamps = pd.to_datetime(data[col], errors='coerce').dropna()
                    if not timestamps.empty:
                        col_latest = timestamps.max()
                        if latest_timestamp is None or col_latest > latest_timestamp:
                            latest_timestamp = col_latest
            
            elif isinstance(data, dict):
                for key, value in data.items():
                    if 'time' in key.lower():
                        try:
                            timestamp = pd.to_datetime(value)
                            if latest_timestamp is None or timestamp > latest_timestamp:
                                latest_timestamp = timestamp
                        except Exception:
                            continue
            
            if latest_timestamp is not None:
                if isinstance(latest_timestamp, pd.Timestamp):
                    latest_timestamp = latest_timestamp.to_pydatetime()
                return (datetime.now() - latest_timestamp).total_seconds()
            
            return None
            
        except Exception:
            return None
    
    def _log_validation_result(self, result: ValidationResult, data_type: str):
        """Log validation result"""
        if not result.is_valid:
            self.logger.error(f"Data validation FAILED for {data_type}: {'; '.join(result.validation_errors)}")
        elif result.validation_warnings:
            self.logger.warning(f"Data validation warnings for {data_type}: {'; '.join(result.validation_warnings)}")
        else:
            self.logger.debug(f"Data validation PASSED for {data_type} (quality: {result.data_quality_score:.2f})")
    
    def add_custom_rule(self, data_type: str, rule: DataValidationRule):
        """Add a custom validation rule"""
        with self._lock:
            if data_type not in self.validation_rules:
                self.validation_rules[data_type] = []
            self.validation_rules[data_type].append(rule)
            self.logger.info(f"Added custom validation rule '{rule.name}' for {data_type}")
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        with self._lock:
            stats = self.validation_stats.copy()
            
            # Calculate success rates
            if stats["total_validations"] > 0:
                stats["authentic_data_rate"] = stats["authentic_data_count"] / stats["total_validations"]
                stats["validation_success_rate"] = (stats["total_validations"] - stats["validation_failures"]) / stats["total_validations"]
            else:
                stats["authentic_data_rate"] = 0.0
                stats["validation_success_rate"] = 0.0
            
            return stats
    
    def get_health_score(self) -> float:
        """Calculate health score based on data validation performance"""
        with self._lock:
            stats = self.get_validation_statistics()
            
            # Base score from authentic data rate
            base_score = stats.get("authentic_data_rate", 0.0)
            
            # Penalty for fallback/synthetic data
            total_rejected = stats.get("fallback_data_rejected", 0) + stats.get("synthetic_data_rejected", 0)
            if stats.get("total_validations", 0) > 0:
                rejection_penalty = total_rejected / stats["total_validations"]
                base_score = max(0.0, base_score - rejection_penalty)
            
            # Bonus for consistent validation success
            success_rate = stats.get("validation_success_rate", 0.0)
            health_score = (base_score * 0.7) + (success_rate * 0.3)
            
            return min(1.0, max(0.0, health_score))


# Singleton eliminator
_fallback_eliminator = None
_eliminator_lock = threading.Lock()

def get_fallback_eliminator(validation_level: DataValidationLevel = DataValidationLevel.STRICT) -> FallbackDataEliminator:
    """Get the singleton fallback data eliminator"""
    global _fallback_eliminator
    
    with _eliminator_lock:
        if _fallback_eliminator is None:
            _fallback_eliminator = FallbackDataEliminator(validation_level)
        return _fallback_eliminator

def validate_authentic_data(data: Any, data_type: str = "general") -> ValidationResult:
    """Convenient function to validate authentic data"""
    eliminator = get_fallback_eliminator()
    return eliminator.validate_data(data, data_type, require_authentic=True)