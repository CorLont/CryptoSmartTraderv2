"""
CryptoSmartTrader V2 - GPU Acceleration Manager
Automatic GPU detection and acceleration for computation-intensive operations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class GPUAccelerator:
    """GPU acceleration manager with automatic fallback to CPU"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gpu_available = False
        self.cupy_available = False
        self.numba_available = False
        
        # Initialize GPU libraries
        self._initialize_gpu_libraries()
        
        # Performance tracking
        self.performance_stats = {
            'gpu_operations': 0,
            'cpu_operations': 0,
            'gpu_speedup_ratio': 0.0,
            'memory_usage_gb': 0.0
        }
        
        self.logger.info(f"GPU Accelerator initialized - GPU: {self.gpu_available}, CuPy: {self.cupy_available}, Numba: {self.numba_available}")
    
    def _initialize_gpu_libraries(self):
        """Initialize GPU acceleration libraries"""
        # Try to import CuPy for GPU arrays
        try:
            import cupy as cp
            self.cp = cp
            self.cupy_available = True
            
            # Test GPU availability
            try:
                test_array = cp.array([1, 2, 3])
                _ = cp.sum(test_array)
                self.gpu_available = True
                self.logger.info(f"CuPy GPU acceleration available - Device: {cp.cuda.get_device_name()}")
            except Exception as e:
                self.cupy_available = False
                self.logger.warning(f"CuPy available but GPU not accessible: {e}")
                
        except ImportError:
            self.logger.info("CuPy not available - using CPU numpy")
            self.cp = np  # Fallback to numpy
        
        # Try to import Numba for JIT compilation
        try:
            import numba
            from numba import jit, cuda
            self.numba = numba
            self.jit = jit
            self.cuda = cuda
            self.numba_available = True
            
            # Test CUDA availability for Numba
            try:
                if cuda.is_available():
                    self.logger.info(f"Numba CUDA available - {len(cuda.gpus)} GPU(s) detected")
                else:
                    self.logger.info("Numba available but CUDA not detected")
            except Exception as e:
                self.logger.warning(f"Numba CUDA check failed: {e}")
                
        except ImportError:
            self.logger.info("Numba not available - JIT compilation disabled")
            self.numba_available = False
    
    def to_gpu(self, data: Union[np.ndarray, pd.DataFrame, list]) -> Union[np.ndarray, Any]:
        """Convert data to GPU array if available"""
        try:
            if not self.cupy_available:
                return np.array(data) if not isinstance(data, np.ndarray) else data
            
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to GPU arrays for numerical columns
                return self.cp.array(data.select_dtypes(include=[np.number]).values)
            elif isinstance(data, (list, tuple)):
                return self.cp.array(data)
            elif isinstance(data, np.ndarray):
                return self.cp.array(data)
            else:
                return self.cp.array(data)
                
        except Exception as e:
            self.logger.warning(f"GPU conversion failed, using CPU: {e}")
            return np.array(data) if not isinstance(data, np.ndarray) else data
    
    def to_cpu(self, data: Any) -> np.ndarray:
        """Convert GPU array back to CPU numpy array"""
        try:
            if self.cupy_available and hasattr(data, 'get'):
                return data.get()  # CuPy to numpy
            else:
                return np.array(data) if not isinstance(data, np.ndarray) else data
        except Exception as e:
            self.logger.warning(f"CPU conversion failed: {e}")
            return np.array(data)
    
    def accelerated_mean(self, data: Union[np.ndarray, list]) -> float:
        """GPU-accelerated mean calculation"""
        try:
            if self.cupy_available:
                gpu_data = self.to_gpu(data)
                result = self.cp.mean(gpu_data)
                self.performance_stats['gpu_operations'] += 1
                return float(self.to_cpu(result))
            else:
                self.performance_stats['cpu_operations'] += 1
                return float(np.mean(data))
        except Exception as e:
            self.logger.error(f"Accelerated mean failed: {e}")
            return float(np.mean(data))
    
    def accelerated_std(self, data: Union[np.ndarray, list]) -> float:
        """GPU-accelerated standard deviation calculation"""
        try:
            if self.cupy_available:
                gpu_data = self.to_gpu(data)
                result = self.cp.std(gpu_data)
                self.performance_stats['gpu_operations'] += 1
                return float(self.to_cpu(result))
            else:
                self.performance_stats['cpu_operations'] += 1
                return float(np.std(data))
        except Exception as e:
            self.logger.error(f"Accelerated std failed: {e}")
            return float(np.std(data))
    
    def accelerated_correlation(self, x: Union[np.ndarray, list], y: Union[np.ndarray, list]) -> float:
        """GPU-accelerated correlation calculation"""
        try:
            if self.cupy_available:
                gpu_x = self.to_gpu(x)
                gpu_y = self.to_gpu(y)
                
                # Calculate correlation coefficient
                corr_matrix = self.cp.corrcoef(gpu_x, gpu_y)
                result = corr_matrix[0, 1]
                
                self.performance_stats['gpu_operations'] += 1
                return float(self.to_cpu(result))
            else:
                self.performance_stats['cpu_operations'] += 1
                return float(np.corrcoef(x, y)[0, 1])
        except Exception as e:
            self.logger.error(f"Accelerated correlation failed: {e}")
            return float(np.corrcoef(x, y)[0, 1])
    
    def accelerated_rolling_mean(self, data: Union[np.ndarray, list], window: int) -> np.ndarray:
        """GPU-accelerated rolling mean calculation"""
        try:
            if self.cupy_available and len(data) > 1000:  # Use GPU for larger datasets
                gpu_data = self.to_gpu(data)
                
                # Manual rolling mean implementation for GPU
                result = self.cp.zeros(len(gpu_data))
                
                for i in range(len(gpu_data)):
                    start_idx = max(0, i - window + 1)
                    result[i] = self.cp.mean(gpu_data[start_idx:i+1])
                
                self.performance_stats['gpu_operations'] += 1
                return self.to_cpu(result)
            else:
                # Use pandas rolling for CPU (more efficient for small datasets)
                self.performance_stats['cpu_operations'] += 1
                return pd.Series(data).rolling(window, min_periods=1).mean().values
                
        except Exception as e:
            self.logger.error(f"Accelerated rolling mean failed: {e}")
            return pd.Series(data).rolling(window, min_periods=1).mean().values
    
    @property
    def jit_compile(self):
        """JIT compilation decorator for performance-critical functions"""
        if self.numba_available:
            return self.jit(nopython=True, cache=True)
        else:
            # Return identity decorator if Numba not available
            return lambda func: func
    
    @property
    def cuda_jit(self):
        """CUDA JIT compilation for GPU kernels"""
        if self.numba_available and hasattr(self.cuda, 'jit'):
            return self.cuda.jit
        else:
            # Return identity decorator if CUDA not available
            return lambda func: func
    
    def accelerated_technical_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """GPU-accelerated technical indicator calculations"""
        try:
            results = {}
            
            # Extract price data
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            volume = df['volume'].values
            
            if self.cupy_available and len(close_prices) > 500:
                # GPU acceleration for larger datasets
                gpu_close = self.to_gpu(close_prices)
                gpu_high = self.to_gpu(high_prices)
                gpu_low = self.to_gpu(low_prices)
                gpu_volume = self.to_gpu(volume)
                
                # SMA calculations
                results['sma_20'] = self.to_cpu(self._gpu_sma(gpu_close, 20))
                results['sma_50'] = self.to_cpu(self._gpu_sma(gpu_close, 50))
                
                # EMA calculations
                results['ema_12'] = self.to_cpu(self._gpu_ema(gpu_close, 12))
                results['ema_26'] = self.to_cpu(self._gpu_ema(gpu_close, 26))
                
                # RSI calculation
                results['rsi'] = self.to_cpu(self._gpu_rsi(gpu_close, 14))
                
                # Bollinger Bands
                bb_results = self._gpu_bollinger_bands(gpu_close, 20, 2)
                results['bb_upper'] = self.to_cpu(bb_results[0])
                results['bb_middle'] = self.to_cpu(bb_results[1])
                results['bb_lower'] = self.to_cpu(bb_results[2])
                
                # Volume indicators
                results['volume_sma'] = self.to_cpu(self._gpu_sma(gpu_volume, 20))
                
                self.performance_stats['gpu_operations'] += 1
                
            else:
                # CPU calculations for smaller datasets
                results['sma_20'] = self._cpu_sma(close_prices, 20)
                results['sma_50'] = self._cpu_sma(close_prices, 50)
                results['ema_12'] = self._cpu_ema(close_prices, 12)
                results['ema_26'] = self._cpu_ema(close_prices, 26)
                results['rsi'] = self._cpu_rsi(close_prices, 14)
                
                bb_results = self._cpu_bollinger_bands(close_prices, 20, 2)
                results['bb_upper'] = bb_results[0]
                results['bb_middle'] = bb_results[1]
                results['bb_lower'] = bb_results[2]
                
                results['volume_sma'] = self._cpu_sma(volume, 20)
                
                self.performance_stats['cpu_operations'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed: {e}")
            return {}
    
    def _gpu_sma(self, data, window):
        """GPU Simple Moving Average"""
        if not self.cupy_available:
            return self._cpu_sma(self.to_cpu(data), window)
        
        result = self.cp.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            result[i] = self.cp.mean(data[start_idx:i+1])
        return result
    
    def _gpu_ema(self, data, window):
        """GPU Exponential Moving Average"""
        if not self.cupy_available:
            return self._cpu_ema(self.to_cpu(data), window)
        
        alpha = 2.0 / (window + 1)
        result = self.cp.zeros_like(data)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    def _gpu_rsi(self, data, window):
        """GPU Relative Strength Index"""
        if not self.cupy_available:
            return self._cpu_rsi(self.to_cpu(data), window)
        
        delta = self.cp.diff(data)
        gain = self.cp.where(delta > 0, delta, 0)
        loss = self.cp.where(delta < 0, -delta, 0)
        
        avg_gain = self.cp.zeros_like(data)
        avg_loss = self.cp.zeros_like(data)
        
        # Initial averages
        avg_gain[window] = self.cp.mean(gain[1:window+1])
        avg_loss[window] = self.cp.mean(loss[1:window+1])
        
        # Smoothed averages
        for i in range(window+1, len(data)):
            avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
        
        rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _gpu_bollinger_bands(self, data, window, num_std):
        """GPU Bollinger Bands"""
        if not self.cupy_available:
            return self._cpu_bollinger_bands(self.to_cpu(data), window, num_std)
        
        sma = self._gpu_sma(data, window)
        
        # Calculate rolling standard deviation
        std = self.cp.zeros_like(data)
        for i in range(len(data)):
            start_idx = max(0, i - window + 1)
            std[i] = self.cp.std(data[start_idx:i+1])
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, sma, lower
    
    def _cpu_sma(self, data, window):
        """CPU Simple Moving Average"""
        return pd.Series(data).rolling(window, min_periods=1).mean().values
    
    def _cpu_ema(self, data, window):
        """CPU Exponential Moving Average"""
        return pd.Series(data).ewm(span=window).mean().values
    
    def _cpu_rsi(self, data, window):
        """CPU Relative Strength Index"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    def _cpu_bollinger_bands(self, data, window, num_std):
        """CPU Bollinger Bands"""
        series = pd.Series(data)
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper.values, sma.values, lower.values
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage statistics"""
        try:
            if self.cupy_available:
                pool = self.cp.get_default_memory_pool()
                return {
                    'used_bytes': pool.used_bytes(),
                    'total_bytes': pool.total_bytes(),
                    'used_gb': pool.used_bytes() / (1024**3),
                    'total_gb': pool.total_bytes() / (1024**3)
                }
            else:
                return {'used_gb': 0.0, 'total_gb': 0.0}
        except Exception as e:
            self.logger.error(f"Memory usage check failed: {e}")
            return {'used_gb': 0.0, 'total_gb': 0.0}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_ops = self.performance_stats['gpu_operations'] + self.performance_stats['cpu_operations']
        
        if total_ops > 0:
            gpu_ratio = self.performance_stats['gpu_operations'] / total_ops
        else:
            gpu_ratio = 0.0
        
        memory_usage = self.get_memory_usage()
        
        return {
            **self.performance_stats,
            'gpu_usage_ratio': gpu_ratio,
            'gpu_available': self.gpu_available,
            'cupy_available': self.cupy_available,
            'numba_available': self.numba_available,
            'memory_usage': memory_usage
        }
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if self.cupy_available:
                self.cp.get_default_memory_pool().free_all_blocks()
                self.logger.info("GPU memory cleaned up")
        except Exception as e:
            self.logger.error(f"GPU memory cleanup failed: {e}")

# Global GPU accelerator instance
gpu_accelerator = GPUAccelerator()