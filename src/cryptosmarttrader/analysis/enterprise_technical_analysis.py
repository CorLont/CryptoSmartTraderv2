#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise Technical Analysis Framework
Unified, robust TA indicators met consistent error handling en validation
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def array(data, dtype=None):
            return list(data)
        @staticmethod
        def nan():
            return float('nan')
        @staticmethod
        def isnan(x):
            return x != x if isinstance(x, (int, float)) else [v != v for v in x]
        @staticmethod
        def mean(x):
            return sum(x) / len(x) if x else 0
        @staticmethod
        def var(x):
            if not x: return 0
            mean_val = sum(x) / len(x)
            return sum((v - mean_val) ** 2 for v in x) / len(x)
    np = MockNumpy()
import pandas as pd
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
import threading
from collections import defaultdict

from core.structured_logger import get_structured_logger


class IndicatorType(Enum):
    """Types of technical indicators"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OSCILLATOR = "oscillator"


@dataclass
class IndicatorResult:
    """Standardized result for technical indicators"""
    indicator_name: str
    indicator_type: IndicatorType
    values: Union[List[float], Dict[str, List[float]]]
    parameters: Dict[str, Any]
    valid_from_index: int  # Index from which values are meaningful
    calculation_time: float
    data_quality_score: float  # 0-1 score based on data completeness
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TAConfig:
    """Configuration for technical analysis"""
    min_data_points: int = 50
    max_lookback_periods: int = 500
    outlier_threshold: float = 5.0  # Standard deviations
    fill_method: str = "forward"  # forward, backward, interpolate, none
    validation_enabled: bool = True
    cache_enabled: bool = True
    error_on_insufficient_data: bool = False


class DataValidator:
    """Validates and preprocesses price data"""
    
    def __init__(self, config: TAConfig):
        self.config = config
        self.logger = get_structured_logger("DataValidator")
    
    def validate_and_clean(self, 
                          prices: Union[List[float], pd.Series, np.ndarray],
                          price_type: str = "close") -> Tuple[np.ndarray, List[str], float]:
        """Validate and clean price data"""
        warnings = []
        
        # Convert to numpy array
        if isinstance(prices, (list, pd.Series)):
            prices = np.array(prices, dtype=float)
        elif not isinstance(prices, np.ndarray):
            raise ValueError(f"Unsupported price data type: {type(prices)}")
        
        original_length = len(prices)
        
        # Check minimum data requirements
        if len(prices) < self.config.min_data_points:
            if self.config.error_on_insufficient_data:
                raise ValueError(f"Insufficient data: {len(prices)} < {self.config.min_data_points}")
            warnings.append(f"Warning: Only {len(prices)} data points, minimum recommended: {self.config.min_data_points}")
        
        # Handle missing values
        nan_count = np.isnan(prices).sum()
        if nan_count > 0:
            warnings.append(f"Found {nan_count} NaN values in {price_type} prices")
            
            if self.config.fill_method == "forward":
                prices = pd.Series(prices).fillna(method='ffill').values
            elif self.config.fill_method == "backward":
                prices = pd.Series(prices).fillna(method='bfill').values
            elif self.config.fill_method == "interpolate":
                prices = pd.Series(prices).interpolate().values
            elif self.config.fill_method == "none":
                pass  # Keep NaN values
            
            remaining_nans = np.isnan(prices).sum()
            if remaining_nans > 0:
                warnings.append(f"After cleaning: {remaining_nans} NaN values remain")
        
        # Detect outliers
        if len(prices) > 10:  # Need sufficient data for outlier detection
            median = np.nanmedian(prices)
            mad = np.nanmedian(np.abs(prices - median))
            if mad > 0:
                outlier_threshold = median + (self.config.outlier_threshold * mad)
                outliers = np.sum(prices > outlier_threshold)
                if outliers > 0:
                    warnings.append(f"Detected {outliers} potential outliers in {price_type} prices")
        
        # Calculate data quality score
        valid_data_ratio = (len(prices) - np.isnan(prices).sum()) / len(prices)
        length_score = min(1.0, len(prices) / self.config.min_data_points)
        data_quality_score = (valid_data_ratio * 0.7) + (length_score * 0.3)
        
        return prices, warnings, data_quality_score


class BaseIndicator(ABC):
    """Base class for all technical indicators"""
    
    def __init__(self, config: TAConfig = None):
        self.config = config or TAConfig()
        self.validator = DataValidator(self.config)
        self.logger = get_structured_logger(f"TA_{self.__class__.__name__}")
        self._cache = {} if self.config.cache_enabled else None
    
    @abstractmethod
    def calculate(self, prices: np.ndarray, **kwargs) -> IndicatorResult:
        """Calculate indicator values"""
        pass
    
    @property
    @abstractmethod
    def indicator_type(self) -> IndicatorType:
        """Return indicator type"""
        pass
    
    @property
    @abstractmethod
    def required_periods(self) -> int:
        """Minimum periods required for calculation"""
        pass
    
    def _get_cache_key(self, prices: np.ndarray, **kwargs) -> str:
        """Generate cache key for results"""
        if self._cache is None:
            return None
        
        # Simple hash of prices and parameters
        import hashlib
        price_hash = hashlib.md5(str(prices.tobytes()).encode()).hexdigest()[:8]
        param_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()[:8]
        return f"{self.__class__.__name__}_{price_hash}_{param_hash}"
    
    def _validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and clean parameters"""
        # Override in subclasses for specific validation
        return kwargs


class RSIIndicator(BaseIndicator):
    """Relative Strength Index"""
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.MOMENTUM
    
    @property
    def required_periods(self) -> int:
        return 15  # period + 1 for delta calculation
    
    def calculate(self, prices: np.ndarray, period: int = 14) -> IndicatorResult:
        """Calculate RSI"""
        start_time = datetime.now()
        
        # Validate inputs
        prices, warnings, quality_score = self.validator.validate_and_clean(prices, "RSI")
        period = max(1, int(period))
        
        # Check cache
        cache_key = self._get_cache_key(prices, period=period)
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        errors = []
        
        try:
            if len(prices) < self.required_periods:
                if self.config.error_on_insufficient_data:
                    raise ValueError(f"RSI requires at least {self.required_periods} data points")
                warnings.append(f"Insufficient data for RSI calculation")
                rsi_values = [np.nan] * len(prices)
                valid_from_index = len(prices)
            else:
                rsi_values = self._get_technical_analyzer().calculate_indicator("RSI", prices, period).values
                valid_from_index = period
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            result = IndicatorResult(
                indicator_name="RSI",
                indicator_type=self.indicator_type,
                values=rsi_values,
                parameters={"period": period},
                valid_from_index=valid_from_index,
                calculation_time=calculation_time,
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            # Cache result
            if cache_key:
                self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            errors.append(f"RSI calculation failed: {str(e)}")
            self.logger.error(f"RSI calculation error: {e}")
            
            return IndicatorResult(
                indicator_name="RSI",
                indicator_type=self.indicator_type,
                values=[np.nan] * len(prices),
                parameters={"period": period},
                valid_from_index=len(prices),
                calculation_time=(datetime.now() - start_time).total_seconds(),
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
    
    def _get_technical_analyzer().calculate_indicator("RSI", self, prices: np.ndarray, period: int).values -> List[float]:
        """Core RSI calculation with robust error handling"""
        try:
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Initialize result array
            rsi_values = [np.nan] * len(prices)
            
            if len(gains) < period:
                return rsi_values
            
            # Calculate initial averages
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RSI for each period
            for i in range(period, len(deltas)):
                # Wilder's smoothing
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
                
                # Calculate RSI
                if avg_loss == 0:
                    rsi = 100.0
                elif avg_gain == 0:
                    rsi = 0.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                rsi_values[i + 1] = rsi
            
            return rsi_values
            
        except Exception as e:
            self.logger.error(f"RSI core calculation failed: {e}")
            return [np.nan] * len(prices)


class MACDIndicator(BaseIndicator):
    """Moving Average Convergence Divergence"""
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.TREND
    
    @property
    def required_periods(self) -> int:
        return 35  # max(fast, slow) + signal periods
    
    def calculate(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> IndicatorResult:
        """Calculate MACD"""
        start_time = datetime.now()
        
        # Validate inputs
        prices, warnings, quality_score = self.validator.validate_and_clean(prices, "MACD")
        fast, slow, signal = max(1, int(fast)), max(1, int(slow)), max(1, int(signal))
        
        if fast >= slow:
            warnings.append("MACD fast period should be < slow period")
            fast, slow = min(fast, slow - 1), max(fast + 1, slow)
        
        errors = []
        
        try:
            if len(prices) < self.required_periods:
                if self.config.error_on_insufficient_data:
                    raise ValueError(f"MACD requires at least {self.required_periods} data points")
                warnings.append("Insufficient data for MACD calculation")
                macd_values = {"macd": [np.nan] * len(prices), 
                             "signal": [np.nan] * len(prices),
                             "histogram": [np.nan] * len(prices)}
                valid_from_index = len(prices)
            else:
                macd_values = self._get_technical_analyzer().calculate_indicator("MACD", prices, fast, slow, signal).values
                valid_from_index = slow + signal - 1
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            result = IndicatorResult(
                indicator_name="MACD",
                indicator_type=self.indicator_type,
                values=macd_values,
                parameters={"fast": fast, "slow": slow, "signal": signal},
                valid_from_index=valid_from_index,
                calculation_time=calculation_time,
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            return result
            
        except Exception as e:
            errors.append(f"MACD calculation failed: {str(e)}")
            self.logger.error(f"MACD calculation error: {e}")
            
            return IndicatorResult(
                indicator_name="MACD",
                indicator_type=self.indicator_type,
                values={"macd": [np.nan] * len(prices), 
                       "signal": [np.nan] * len(prices),
                       "histogram": [np.nan] * len(prices)},
                parameters={"fast": fast, "slow": slow, "signal": signal},
                valid_from_index=len(prices),
                calculation_time=(datetime.now() - start_time).total_seconds(),
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            alpha = 2.0 / (period + 1.0)
            ema = np.empty_like(prices)
            ema[0] = prices[0]
            
            for i in range(1, len(prices)):
                if np.isnan(prices[i]):
                    ema[i] = ema[i-1]
                else:
                    ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
            return ema
            
        except Exception as e:
            self.logger.error(f"EMA calculation failed: {e}")
            return np.full_like(prices, np.nan)
    
    def _get_technical_analyzer().calculate_indicator("MACD", self, prices: np.ndarray, fast: int, slow: int, signal: int).values -> Dict[str, List[float]]:
        """Core MACD calculation"""
        try:
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line (EMA of MACD line)
            signal_line = self._calculate_ema(macd_line, signal)
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                "macd": macd_line.tolist(),
                "signal": signal_line.tolist(), 
                "histogram": histogram.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"MACD core calculation failed: {e}")
            length = len(prices)
            return {
                "macd": [np.nan] * length,
                "signal": [np.nan] * length,
                "histogram": [np.nan] * length
            }


class BollingerBandsIndicator(BaseIndicator):
    """Bollinger Bands"""
    
    @property
    def indicator_type(self) -> IndicatorType:
        return IndicatorType.VOLATILITY
    
    @property
    def required_periods(self) -> int:
        return 20  # Default period
    
    def calculate(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> IndicatorResult:
        """Calculate Bollinger Bands"""
        start_time = datetime.now()
        
        # Validate inputs
        prices, warnings, quality_score = self.validator.validate_and_clean(prices, "Bollinger")
        period = max(1, int(period))
        std_dev = max(0.1, float(std_dev))
        
        errors = []
        
        try:
            if len(prices) < period:
                if self.config.error_on_insufficient_data:
                    raise ValueError(f"Bollinger Bands require at least {period} data points")
                warnings.append("Insufficient data for Bollinger Bands calculation")
                bb_values = {"upper": [np.nan] * len(prices),
                           "middle": [np.nan] * len(prices), 
                           "lower": [np.nan] * len(prices)}
                valid_from_index = len(prices)
            else:
                bb_values = self._get_technical_analyzer().calculate_indicator("BollingerBands", prices, period, std_dev).values
                valid_from_index = period - 1
            
            calculation_time = (datetime.now() - start_time).total_seconds()
            
            result = IndicatorResult(
                indicator_name="BollingerBands",
                indicator_type=self.indicator_type,
                values=bb_values,
                parameters={"period": period, "std_dev": std_dev},
                valid_from_index=valid_from_index,
                calculation_time=calculation_time,
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
            
            return result
            
        except Exception as e:
            errors.append(f"Bollinger Bands calculation failed: {str(e)}")
            self.logger.error(f"Bollinger calculation error: {e}")
            
            return IndicatorResult(
                indicator_name="BollingerBands",
                indicator_type=self.indicator_type,
                values={"upper": [np.nan] * len(prices),
                       "middle": [np.nan] * len(prices),
                       "lower": [np.nan] * len(prices)},
                parameters={"period": period, "std_dev": std_dev},
                valid_from_index=len(prices),
                calculation_time=(datetime.now() - start_time).total_seconds(),
                data_quality_score=quality_score,
                warnings=warnings,
                errors=errors
            )
    
    def _get_technical_analyzer().calculate_indicator("BollingerBands", self, prices: np.ndarray, period: int, std_dev: float).values -> Dict[str, List[float]]:
        """Core Bollinger Bands calculation"""
        try:
            length = len(prices)
            upper = np.full(length, np.nan)
            middle = np.full(length, np.nan)
            lower = np.full(length, np.nan)
            
            for i in range(period - 1, length):
                window = prices[i - period + 1:i + 1]
                
                # Handle NaN values in window
                valid_window = window[~np.isnan(window)]
                if len(valid_window) >= period // 2:  # At least half the window should be valid
                    sma = np.mean(valid_window)
                    std = np.std(valid_window, ddof=1)
                    
                    middle[i] = sma
                    upper[i] = sma + (std_dev * std)
                    lower[i] = sma - (std_dev * std)
            
            return {
                "upper": upper.tolist(),
                "middle": middle.tolist(),
                "lower": lower.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Bollinger core calculation failed: {e}")
            length = len(prices)
            return {
                "upper": [np.nan] * length,
                "middle": [np.nan] * length,
                "lower": [np.nan] * length
            }


class EnterpriseTechnicalAnalyzer:
    """Unified technical analysis coordinator"""
    
    def __init__(self, config: TAConfig = None):
        self.config = config or TAConfig()
        self.logger = get_structured_logger("EnterpriseTechnicalAnalyzer")
        
        # Initialize indicators
        self.indicators = {
            "RSI": RSIIndicator(self.config),
            "MACD": MACDIndicator(self.config),
            "BollingerBands": BollingerBandsIndicator(self.config)
        }
        
        # Metrics tracking
        self.calculation_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
        self.logger.info("Enterprise Technical Analyzer initialized")
    
    def calculate_indicator(self, 
                          indicator_name: str,
                          prices: Union[List[float], pd.Series, np.ndarray],
                          **kwargs) -> IndicatorResult:
        """Calculate specific indicator"""
        
        if indicator_name not in self.indicators:
            raise ValueError(f"Unknown indicator: {indicator_name}. Available: {list(self.indicators.keys())}")
        
        start_time = datetime.now()
        
        try:
            # Convert prices to numpy array
            if isinstance(prices, (list, pd.Series)):
                prices = np.array(prices, dtype=float)
            
            indicator = self.indicators[indicator_name]
            result = indicator.calculate(prices, **kwargs)
            
            # Record metrics
            with self.lock:
                self.calculation_metrics[indicator_name].append({
                    "timestamp": start_time,
                    "calculation_time": result.calculation_time,
                    "data_points": len(prices),
                    "quality_score": result.data_quality_score,
                    "warning_count": len(result.warnings),
                    "error_count": len(result.errors)
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed for {indicator_name}: {e}")
            
            return IndicatorResult(
                indicator_name=indicator_name,
                indicator_type=IndicatorType.TREND,  # Default
                values=[np.nan] * len(prices),
                parameters=kwargs,
                valid_from_index=len(prices),
                calculation_time=(datetime.now() - start_time).total_seconds(),
                data_quality_score=0.0,
                warnings=[],
                errors=[f"Calculation failed: {str(e)}"]
            )
    
    def calculate_multiple(self, 
                          prices: Union[List[float], pd.Series, np.ndarray],
                          indicators: Dict[str, Dict[str, Any]]) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators efficiently"""
        
        results = {}
        
        for indicator_name, params in indicators.items():
            try:
                result = self.calculate_indicator(indicator_name, prices, **params)
                results[indicator_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to calculate {indicator_name}: {e}")
                results[indicator_name] = IndicatorResult(
                    indicator_name=indicator_name,
                    indicator_type=IndicatorType.TREND,
                    values=[np.nan] * len(prices),
                    parameters=params,
                    valid_from_index=len(prices),
                    calculation_time=0.0,
                    data_quality_score=0.0,
                    warnings=[],
                    errors=[f"Multiple calculation failed: {str(e)}"]
                )
        
        return results
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get comprehensive analysis summary"""
        with self.lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "available_indicators": list(self.indicators.keys()),
                "calculation_metrics": {}
            }
            
            for indicator_name, metrics in self.calculation_metrics.items():
                if metrics:
                    recent_metrics = metrics[-100:]  # Last 100 calculations
                    
                    summary["calculation_metrics"][indicator_name] = {
                        "total_calculations": len(metrics),
                        "avg_calculation_time": np.mean([m["calculation_time"] for m in recent_metrics]),
                        "avg_quality_score": np.mean([m["quality_score"] for m in recent_metrics]),
                        "error_rate": np.mean([m["error_count"] > 0 for m in recent_metrics]),
                        "last_calculation": recent_metrics[-1]["timestamp"].isoformat() if recent_metrics else None
                    }
            
            return summary


# Global singleton
_ta_analyzer_instance = None

def get_technical_analyzer(config: TAConfig = None) -> EnterpriseTechnicalAnalyzer:
    """Get singleton technical analyzer"""
    global _ta_analyzer_instance
    if _ta_analyzer_instance is None:
        _ta_analyzer_instance = EnterpriseTechnicalAnalyzer(config)
    return _ta_analyzer_instance


if __name__ == "__main__":
    # Basic validation
    analyzer = get_technical_analyzer()
    
    # Test with sample data
    sample_prices = np.random.randn(100).cumsum() + 100
    
    # Calculate single indicator
    rsi_result = analyzer.calculate_indicator("RSI", sample_prices, period=14)
    print(f"RSI calculation: {len(rsi_result.warnings)} warnings, {len(rsi_result.errors)} errors")
    
    # Calculate multiple indicators
    indicators = {
        "RSI": {"period": 14},
        "MACD": {"fast": 12, "slow": 26, "signal": 9},
        "BollingerBands": {"period": 20, "std_dev": 2.0}
    }
    
    results = analyzer.calculate_multiple(sample_prices, indicators)
    print(f"Multiple calculation results: {list(results.keys())}")
    
    # Show summary
    summary = analyzer.get_analysis_summary()
    print(f"Analysis summary: {summary}")