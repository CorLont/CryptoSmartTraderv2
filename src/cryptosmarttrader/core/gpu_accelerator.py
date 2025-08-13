"""
CryptoSmartTrader V2 - GPU Accelerator
GPU-accelerated computing voor maximum performance
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import sys
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# GPU libraries
try:
    import cupy as cp
    import cudf
    import cuml
    from cuml.ensemble import RandomForestRegressor as CuRandomForestRegressor
    from cuml.linear_model import LinearRegression as CuLinearRegression
    from cuml.preprocessing import StandardScaler as CuStandardScaler
    GPU_AVAILABLE = True
    CUPY_AVAILABLE = True
    RAPIDS_AVAILABLE = True
except ImportError as e:
    logging.info(f"GPU libraries not available: {e}")
    GPU_AVAILABLE = False
    CUPY_AVAILABLE = False
    RAPIDS_AVAILABLE = False
    # Fallback imports
    import numpy as cp
    cudf = pd

try:
    import numba
    from numba import cuda, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.info("Numba not available - JIT compilation disabled")

class GPUAccelerator:
    """GPU-accelerated computing engine"""

    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        self.cache_manager = container.cache_manager()

        # GPU status
        self.gpu_available = self._check_gpu_availability()
        self.device_info = self._get_device_info()

        # Performance tracking
        self.performance_stats = {
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'total_speedup': 0.0,
            'average_speedup': 1.0
        }

        self.logger.info(f"GPU Accelerator initialized. GPU available: {self.gpu_available}")
        if self.gpu_available:
            self.logger.info(f"Device info: {self.device_info}")

    def _check_gpu_availability(self) -> bool:
        """Check GPU availability"""
        try:
            if CUPY_AVAILABLE:
                # Test CuPy
                test_array = cp.array([1, 2, 3])
                result = cp.sum(test_array)
                cp.cuda.Stream.null.synchronize()
                return True
            return False
        except Exception as e:
            self.logger.warning(f"GPU availability check failed: {e}")
            return False

    def _get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information"""
        try:
            if not self.gpu_available:
                return {'device': 'CPU', 'memory': 'N/A'}

            device = cp.cuda.Device()
            meminfo = cp.cuda.MemoryInfo()

            return {
                'device': f"GPU {device.id}",
                'name': device.attributes.get('name', 'Unknown'),
                'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                'total_memory_gb': meminfo.total / (1024**3),
                'free_memory_gb': meminfo.free / (1024**3)
            }
        except Exception as e:
            self.logger.error(f"Failed to get device info: {e}")
            return {'device': 'GPU (info unavailable)'}

    def gpu_accelerated_operation(self, func, *args, fallback_func=None, **kwargs):
        """Execute operation with GPU acceleration and CPU fallback"""
        start_time = time.time()

        try:
            if self.gpu_available:
                # Try GPU operation
                result = func(*args, **kwargs)

                # Synchronize GPU
                if CUPY_AVAILABLE:
                    cp.cuda.Stream.null.synchronize()

                execution_time = time.time() - start_time
                self.performance_stats['gpu_operations'] += 1

                return result
            else:
                raise Exception("GPU not available")

        except Exception as e:
            # Fallback to CPU
            self.logger.debug(f"GPU operation failed, falling back to CPU: {e}")
            self.performance_stats['cpu_fallbacks'] += 1

            if fallback_func:
                return fallback_func(*args, **kwargs)
            else:
                # Convert CuPy arrays to NumPy if needed
                cpu_args = []
                for arg in args:
                    if hasattr(arg, 'get'):  # CuPy array
                        cpu_args.append(arg.get())
                    else:
                        cpu_args.append(arg)

                return func(*cpu_args, **kwargs)

    def accelerated_technical_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated technical indicators calculation"""
        try:
            if self.gpu_available and CUPY_AVAILABLE:
                return self._gpu_technical_indicators(price_data)
            else:
                return self._cpu_technical_indicators(price_data)
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed: {e}")
            return self._cpu_technical_indicators(price_data)

    def _gpu_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """GPU-accelerated technical indicators"""
        try:
            # Convert to CuPy arrays
            closes = cp.array(df['close'].values)
            highs = cp.array(df['high'].values)
            lows = cp.array(df['low'].values)
            volumes = cp.array(df['volume'].values)

            result_df = df.copy()

            # Moving averages
            result_df['sma_20'] = self._gpu_sma(closes, 20).get()
            result_df['sma_50'] = self._gpu_sma(closes, 50).get()
            result_df['ema_20'] = self._gpu_ema(closes, 20).get()

            # RSI
            result_df['rsi'] = self._gpu_rsi(closes, 14).get()

            # MACD
            macd, signal = self._gpu_macd(closes)
            result_df['macd'] = macd.get()
            result_df['macd_signal'] = signal.get()

            # Bollinger Bands
            bb_upper, bb_lower = self._gpu_bollinger_bands(closes, 20, 2)
            result_df['bb_upper'] = bb_upper.get()
            result_df['bb_lower'] = bb_lower.get()

            # Volume indicators
            result_df['volume_sma'] = self._gpu_sma(volumes, 20).get()
            result_df['volume_ratio'] = (volumes / self._gpu_sma(volumes, 20)).get()

            # Price ratios
            result_df['high_low_ratio'] = (highs / lows).get()

            return result_df

        except Exception as e:
            self.logger.error(f"GPU technical indicators failed: {e}")
            return self._cpu_technical_indicators(df)

    @staticmethod
    def _gpu_sma(data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated Simple Moving Average"""
        if NUMBA_AVAILABLE and CUPY_AVAILABLE:
            return GPUAccelerator._gpu_sma_kernel(data, window)
        else:
            # Fallback using CuPy
            return cp.convolve(data, cp.ones(window)/window, mode='same')

    @staticmethod
    def _gpu_sma_kernel(data, window):
        """CUDA kernel for SMA calculation (if NUMBA available)"""
        if NUMBA_AVAILABLE:
            @cuda.jit
            def sma_kernel_impl(data, window):
                idx = cuda.grid(1)
                if idx < len(data):
                    if idx < window - 1:
                        data[idx] = cp.nan
                    else:
                        total = 0.0
                        for i in range(window):
                            total += data[idx - i]
                        data[idx] = total / window
            return sma_kernel_impl(data, window)
        else:
            # Fallback to simple convolution
            return cp.convolve(data, cp.ones(window)/window, mode='same')

    @staticmethod
    def _gpu_ema(data: cp.ndarray, window: int) -> cp.ndarray:
        """GPU-accelerated Exponential Moving Average"""
        alpha = 2.0 / (window + 1)
        ema = cp.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]

        return ema

    @staticmethod
    def _gpu_rsi(closes: cp.ndarray, window: int = 14) -> cp.ndarray:
        """GPU-accelerated RSI calculation"""
        deltas = cp.diff(closes)
        gains = cp.where(deltas > 0, deltas, 0)
        losses = cp.where(deltas < 0, -deltas, 0)

        avg_gains = GPUAccelerator._gpu_sma(gains, window)
        avg_losses = GPUAccelerator._gpu_sma(losses, window)

        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return cp.concatenate([cp.array([cp.nan]), rsi])

    @staticmethod
    def _gpu_macd(closes: cp.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[cp.ndarray, cp.ndarray]:
        """GPU-accelerated MACD calculation"""
        ema_fast = GPUAccelerator._gpu_ema(closes, fast)
        ema_slow = GPUAccelerator._gpu_ema(closes, slow)

        macd = ema_fast - ema_slow
        signal = GPUAccelerator._gpu_ema(macd, signal_period)

        return macd, signal

    @staticmethod
    def _gpu_bollinger_bands(closes: cp.ndarray, window: int = 20, std_dev: float = 2) -> Tuple[cp.ndarray, cp.ndarray]:
        """GPU-accelerated Bollinger Bands calculation"""
        sma = GPUAccelerator._gpu_sma(closes, window)

        # Rolling standard deviation
        rolling_std = cp.zeros_like(closes)
        for i in range(window-1, len(closes)):
            window_data = closes[i-window+1:i+1]
            rolling_std[i] = cp.std(window_data)

        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)

        return upper_band, lower_band

    def _cpu_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU fallback for technical indicators"""
        try:
            result_df = df.copy()

            # Moving averages
            result_df['sma_20'] = df['close'].rolling(window=20).mean()
            result_df['sma_50'] = df['close'].rolling(window=50).mean()
            result_df['ema_20'] = df['close'].ewm(span=20).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            result_df['macd'] = ema_12 - ema_26
            result_df['macd_signal'] = result_df['macd'].ewm(span=9).mean()

            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            result_df['bb_upper'] = sma_20 + (std_20 * 2)
            result_df['bb_lower'] = sma_20 - (std_20 * 2)

            # Volume indicators
            result_df['volume_sma'] = df['volume'].rolling(window=20).mean()
            result_df['volume_ratio'] = df['volume'] / result_df['volume_sma']

            # Price ratios
            result_df['high_low_ratio'] = df['high'] / df['low']

            return result_df

        except Exception as e:
            self.logger.error(f"CPU technical indicators failed: {e}")
            return df

    def accelerated_ml_training(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'random_forest') -> Any:
        """GPU-accelerated ML training"""
        try:
            if self.gpu_available and RAPIDS_AVAILABLE:
                return self._gpu_ml_training(X, y, model_type)
            else:
                return self._cpu_ml_training(X, y, model_type)
        except Exception as e:
            self.logger.error(f"ML training failed: {e}")
            return self._cpu_ml_training(X, y, model_type)

    def _gpu_ml_training(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Any:
        """GPU-accelerated ML training using RAPIDS cuML"""
        try:
            # Convert to cuDF
            X_gpu = cudf.DataFrame(X)
            y_gpu = cudf.Series(y)

            # Scale features
            scaler = CuStandardScaler()
            X_scaled = scaler.fit_transform(X_gpu)

            # Train model
            if model_type == 'random_forest':
                model = CuRandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            elif model_type == 'linear_regression':
                model = CuLinearRegression()
            else:
                # Fallback to CPU
                return self._cpu_ml_training(X, y, model_type)

            model.fit(X_scaled, y_gpu)

            return {
                'model': model,
                'scaler': scaler,
                'training_method': 'GPU',
                'model_type': model_type
            }

        except Exception as e:
            self.logger.error(f"GPU ML training failed: {e}")
            return self._cpu_ml_training(X, y, model_type)

    def _cpu_ml_training(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Any:
        """CPU fallback for ML training"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            elif model_type == 'linear_regression':
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            model.fit(X_scaled, y)

            return {
                'model': model,
                'scaler': scaler,
                'training_method': 'CPU',
                'model_type': model_type
            }

        except Exception as e:
            self.logger.error(f"CPU ML training failed: {e}")
            return None

    def accelerated_data_processing(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """GPU-accelerated data processing operations"""
        try:
            if self.gpu_available and RAPIDS_AVAILABLE:
                return self._gpu_data_processing(data, operations)
            else:
                return self._cpu_data_processing(data, operations)
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            return self._cpu_data_processing(data, operations)

    def _gpu_data_processing(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """GPU-accelerated data processing using cuDF"""
        try:
            # Convert to cuDF
            df_gpu = cudf.DataFrame(data)

            for operation in operations:
                if operation == 'normalize':
                    numeric_cols = df_gpu.select_dtypes(include=['float64', 'int64']).columns
                    df_gpu[numeric_cols] = (df_gpu[numeric_cols] - df_gpu[numeric_cols].mean()) / df_gpu[numeric_cols].std()

                elif operation == 'log_transform':
                    numeric_cols = df_gpu.select_dtypes(include=['float64', 'int64']).columns
                    df_gpu[numeric_cols] = cudf.log(df_gpu[numeric_cols] + 1)

                elif operation == 'rolling_stats':
                    if 'close' in df_gpu.columns:
                        df_gpu['rolling_mean_10'] = df_gpu['close'].rolling(window=10).mean()
                        df_gpu['rolling_std_10'] = df_gpu['close'].rolling(window=10).std()

                elif operation == 'outlier_removal':
                    numeric_cols = df_gpu.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        Q1 = df_gpu[col].quantile(0.25)
                        Q3 = df_gpu[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df_gpu = df_gpu[(df_gpu[col] >= lower_bound) & (df_gpu[col] <= upper_bound)]

            # Convert back to pandas
            return df_gpu.to_pandas()

        except Exception as e:
            self.logger.error(f"GPU data processing failed: {e}")
            return self._cpu_data_processing(data, operations)

    def _cpu_data_processing(self, data: pd.DataFrame, operations: List[str]) -> pd.DataFrame:
        """CPU fallback for data processing"""
        try:
            df = data.copy()

            for operation in operations:
                if operation == 'normalize':
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

                elif operation == 'log_transform':
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    df[numeric_cols] = np.log(df[numeric_cols] + 1)

                elif operation == 'rolling_stats':
                    if 'close' in df.columns:
                        df['rolling_mean_10'] = df['close'].rolling(window=10).mean()
                        df['rolling_std_10'] = df['close'].rolling(window=10).std()

                elif operation == 'outlier_removal':
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    for col in numeric_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

            return df

        except Exception as e:
            self.logger.error(f"CPU data processing failed: {e}")
            return data

    def benchmark_performance(self, data_size: int = 10000) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        try:
            # Generate test data
            np.random.seed(42)
            test_data = pd.DataFrame({
                'close': np.random.randn(data_size).cumsum() + 100,
                'high': np.random.randn(data_size).cumsum() + 105,
                'low': np.random.randn(data_size).cumsum() + 95,
                'volume': np.random.normal(0, 1)
            })

            results = {
                'data_size': data_size,
                'gpu_available': self.gpu_available,
                'tests': {}
            }

            # Test technical indicators
            start_time = time.time()
            cpu_result = self._cpu_technical_indicators(test_data)
            cpu_time = time.time() - start_time

            start_time = time.time()
            gpu_result = self._gpu_technical_indicators(test_data) if self.gpu_available else cpu_result
            gpu_time = time.time() - start_time

            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            results['tests']['technical_indicators'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'gpu_faster': speedup > 1.0
            }

            # Test data processing
            start_time = time.time()
            cpu_processed = self._cpu_data_processing(test_data, ['normalize', 'rolling_stats'])
            cpu_time = time.time() - start_time

            start_time = time.time()
            gpu_processed = self._gpu_data_processing(test_data, ['normalize', 'rolling_stats']) if self.gpu_available else cpu_processed
            gpu_time = time.time() - start_time

            speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0

            results['tests']['data_processing'] = {
                'cpu_time': cpu_time,
                'gpu_time': gpu_time,
                'speedup': speedup,
                'gpu_faster': speedup > 1.0
            }

            # Overall stats
            total_speedup = sum(test['speedup'] for test in results['tests'].values())
            results['average_speedup'] = total_speedup / len(results['tests'])
            results['overall_faster'] = results['average_speedup'] > 1.0

            return results

        except Exception as e:
            self.logger.error(f"Performance benchmark failed: {e}")
            return {'error': str(e)}

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get GPU accelerator status"""
        return {
            'gpu_available': self.gpu_available,
            'cupy_available': CUPY_AVAILABLE,
            'rapids_available': RAPIDS_AVAILABLE,
            'numba_available': NUMBA_AVAILABLE,
            'device_info': self.device_info,
            'performance_stats': self.performance_stats,
            'timestamp': datetime.now()
        }

    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        try:
            if self.gpu_available and CUPY_AVAILABLE:
                # Clear GPU memory
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

                self.logger.info("GPU memory optimized")

        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")


# Global GPU accelerator instance
gpu_accelerator = None

def get_gpu_accelerator(container=None):
    """Get global GPU accelerator instance"""
    global gpu_accelerator
    if gpu_accelerator is None and container is not None:
        gpu_accelerator = GPUAccelerator(container)
    return gpu_accelerator
