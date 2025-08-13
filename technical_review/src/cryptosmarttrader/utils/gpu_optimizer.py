import threading
import time
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import cuda
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False

class GPUOptimizer:
    """GPU optimization and performance monitoring"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # GPU availability and status
        self.gpu_available = False
        self.gpu_info = {}
        self.gpu_memory_info = {}
        self.performance_metrics = {}
        self._lock = threading.Lock()

        # Performance monitoring
        self.monitoring_active = False
        self.monitor_thread = None

        # Initialize GPU detection
        self._detect_gpu_capabilities()

        # Start monitoring if GPU is available
        if self.gpu_available and self.config_manager.get("enable_gpu", False):
            self.start_monitoring()

    def _detect_gpu_capabilities(self):
        """Detect available GPU capabilities"""
        try:
            # Check CuPy availability
            if CUPY_AVAILABLE:
                try:
                    # Test basic CuPy functionality
                    test_array = cp.array([1, 2, 3])
                    result = cp.sum(test_array)
                    cp.cuda.Device().synchronize()

                    self.gpu_available = True
                    self.gpu_info['cupy_available'] = True
                    self.gpu_info['cupy_version'] = cp.__version__

                    # Get GPU device info
                    device = cp.cuda.Device()
                    self.gpu_info['device_id'] = device.id
                    self.gpu_info['device_name'] = device.attributes['Name'].decode()
                    self.gpu_info['compute_capability'] = device.compute_capability

                    self.logger.info(f"GPU detected: {self.gpu_info['device_name']}")

                except Exception as e:
                    self.logger.warning(f"CuPy test failed: {str(e)}")
                    self.gpu_info['cupy_available'] = False
            else:
                self.gpu_info['cupy_available'] = False

            # Check Numba CUDA availability
            if NUMBA_CUDA_AVAILABLE:
                try:
                    # Test basic CUDA functionality
                    if cuda.is_available():
                        self.gpu_info['numba_cuda_available'] = True
                        self.gpu_info['cuda_devices'] = len(cuda.gpus)
                        self.gpu_available = True

                        # Get device info
                        gpu = cuda.gpus[0]
                        self.gpu_info['numba_device_name'] = gpu.name.decode()
                        self.gpu_info['numba_compute_capability'] = gpu.compute_capability

                        self.logger.info(f"Numba CUDA available with {len(cuda.gpus)} devices")
                    else:
                        self.gpu_info['numba_cuda_available'] = False
                except Exception as e:
                    self.logger.warning(f"Numba CUDA test failed: {str(e)}")
                    self.gpu_info['numba_cuda_available'] = False
            else:
                self.gpu_info['numba_cuda_available'] = False

            # Overall GPU status
            if not self.gpu_available:
                self.logger.info("No GPU acceleration available, using CPU only")

        except Exception as e:
            self.logger.error(f"GPU detection error: {str(e)}")
            self.gpu_available = False

    def start_monitoring(self):
        """Start GPU performance monitoring"""
        if not self.monitoring_active and self.gpu_available:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("GPU monitoring started")

    def stop_monitoring(self):
        """Stop GPU performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("GPU monitoring stopped")

    def _monitoring_loop(self):
        """GPU monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_gpu_metrics()
                time.sleep(30)  # Update every 30 seconds

            except Exception as e:
                self.logger.error(f"GPU monitoring error: {str(e)}")
                time.sleep(60)

    def _update_gpu_metrics(self):
        """Update GPU performance metrics"""
        try:
            with self._lock:
                metrics = {
                    'timestamp': time.time(),
                    'gpu_available': self.gpu_available
                }

                if self.gpu_available and CUPY_AVAILABLE:
                    # Get GPU memory info
                    mempool = cp.get_default_memory_pool()
                    metrics['memory_used_bytes'] = mempool.used_bytes()
                    metrics['memory_total_bytes'] = mempool.total_bytes()

                    # Get device utilization if available
                    try:
                        with cp.cuda.Device() as device:
                            metrics['device_id'] = device.id
                            # Note: GPU utilization requires nvidia-ml-py or similar
                            # For now, we'll track memory utilization as proxy
                            if metrics['memory_total_bytes'] > 0:
                                metrics['memory_utilization'] = metrics['memory_used_bytes'] / metrics['memory_total_bytes']
                            else:
                                metrics['memory_utilization'] = 0
                    except Exception as e:
                        self.logger.debug(f"GPU device metrics error: {str(e)}")

                self.performance_metrics = metrics

        except Exception as e:
            self.logger.error(f"GPU metrics update error: {str(e)}")

    def optimize_array_computation(self, data: np.ndarray, operation: str = 'sum') -> Tuple[Any, float]:
        """Optimize array computation using GPU if available"""
        start_time = time.time()

        try:
            if self.gpu_available and CUPY_AVAILABLE and self.config_manager.get("enable_gpu", False):
                # Use GPU computation
                gpu_data = cp.asarray(data)

                if operation == 'sum':
                    result = cp.sum(gpu_data)
                elif operation == 'mean':
                    result = cp.mean(gpu_data)
                elif operation == 'std':
                    result = cp.std(gpu_data)
                elif operation == 'max':
                    result = cp.max(gpu_data)
                elif operation == 'min':
                    result = cp.min(gpu_data)
                elif operation == 'sort':
                    result = cp.sort(gpu_data)
                else:
                    result = gpu_data

                # Convert back to CPU if needed
                if hasattr(result, 'get'):
                    result = result.get()

                computation_time = time.time() - start_time
                self.logger.debug(f"GPU computation ({operation}) completed in {computation_time:.4f}s")

                return result, computation_time
            else:
                # Fall back to CPU computation
                if operation == 'sum':
                    result = np.sum(data)
                elif operation == 'mean':
                    result = np.mean(data)
                elif operation == 'std':
                    result = np.std(data)
                elif operation == 'max':
                    result = np.max(data)
                elif operation == 'min':
                    result = np.min(data)
                elif operation == 'sort':
                    result = np.sort(data)
                else:
                    result = data

                computation_time = time.time() - start_time
                self.logger.debug(f"CPU computation ({operation}) completed in {computation_time:.4f}s")

                return result, computation_time

        except Exception as e:
            self.logger.error(f"Array computation error: {str(e)}")
            # Fall back to CPU
            if operation == 'sum':
                result = np.sum(data)
            else:
                result = data

            return result, time.time() - start_time

    def optimize_matrix_operations(self, matrix_a: np.ndarray, matrix_b: np.ndarray = None,
                                  operation: str = 'multiply') -> Tuple[Any, float]:
        """Optimize matrix operations using GPU if available"""
        start_time = time.time()

        try:
            if self.gpu_available and CUPY_AVAILABLE and self.config_manager.get("enable_gpu", False):
                # Use GPU computation
                gpu_a = cp.asarray(matrix_a)

                if matrix_b is not None:
                    gpu_b = cp.asarray(matrix_b)

                    if operation == 'multiply' or operation == 'matmul':
                        result = cp.matmul(gpu_a, gpu_b)
                    elif operation == 'add':
                        result = gpu_a + gpu_b
                    elif operation == 'subtract':
                        result = gpu_a - gpu_b
                    elif operation == 'divide':
                        result = gpu_a / gpu_b
                    else:
                        result = gpu_a
                else:
                    if operation == 'transpose':
                        result = cp.transpose(gpu_a)
                    elif operation == 'inverse':
                        result = cp.linalg.inv(gpu_a)
                    elif operation == 'eigenvalues':
                        result = cp.linalg.eigvals(gpu_a)
                    else:
                        result = gpu_a

                # Convert back to CPU
                if hasattr(result, 'get'):
                    result = result.get()

                computation_time = time.time() - start_time
                self.logger.debug(f"GPU matrix operation ({operation}) completed in {computation_time:.4f}s")

                return result, computation_time
            else:
                # Fall back to CPU computation
                if matrix_b is not None:
                    if operation == 'multiply' or operation == 'matmul':
                        result = np.matmul(matrix_a, matrix_b)
                    elif operation == 'add':
                        result = matrix_a + matrix_b
                    elif operation == 'subtract':
                        result = matrix_a - matrix_b
                    elif operation == 'divide':
                        result = matrix_a / matrix_b
                    else:
                        result = matrix_a
                else:
                    if operation == 'transpose':
                        result = np.transpose(matrix_a)
                    elif operation == 'inverse':
                        result = np.linalg.inv(matrix_a)
                    elif operation == 'eigenvalues':
                        result = np.linalg.eigvals(matrix_a)
                    else:
                        result = matrix_a

                computation_time = time.time() - start_time
                self.logger.debug(f"CPU matrix operation ({operation}) completed in {computation_time:.4f}s")

                return result, computation_time

        except Exception as e:
            self.logger.error(f"Matrix operation error: {str(e)}")
            # Fall back to simple operation
            result = matrix_a
            return result, time.time() - start_time

    def benchmark_performance(self, test_size: int = 10000) -> Dict[str, Any]:
        """Benchmark GPU vs CPU performance"""
        results = {
            'test_size': test_size,
            'gpu_available': self.gpu_available,
            'timestamp': time.time()
        }

        try:
            # Create test data
            test_data = np.random.random((test_size, test_size)).astype(np.float32)
            test_vector = np.random.random(test_size).astype(np.float32)

            # CPU benchmarks
            cpu_times = {}

            start = time.time()
            cpu_result = np.sum(test_data)
            cpu_times['sum'] = time.time() - start

            start = time.time()
            cpu_result = np.mean(test_data)
            cpu_times['mean'] = time.time() - start

            start = time.time()
            cpu_result = np.matmul(test_data, test_vector)
            cpu_times['matmul'] = time.time() - start

            results['cpu_times'] = cpu_times

            # GPU benchmarks if available
            if self.gpu_available and CUPY_AVAILABLE:
                gpu_times = {}

                # Transfer to GPU
                start = time.time()
                gpu_data = cp.asarray(test_data)
                gpu_vector = cp.asarray(test_vector)
                gpu_times['transfer_to_gpu'] = time.time() - start

                # GPU operations
                start = time.time()
                gpu_result = cp.sum(gpu_data)
                cp.cuda.Device().synchronize()
                gpu_times['sum'] = time.time() - start

                start = time.time()
                gpu_result = cp.mean(gpu_data)
                cp.cuda.Device().synchronize()
                gpu_times['mean'] = time.time() - start

                start = time.time()
                gpu_result = cp.matmul(gpu_data, gpu_vector)
                cp.cuda.Device().synchronize()
                gpu_times['matmul'] = time.time() - start

                # Transfer back to CPU
                start = time.time()
                cpu_result = gpu_result.get()
                gpu_times['transfer_to_cpu'] = time.time() - start

                results['gpu_times'] = gpu_times

                # Calculate speedup
                speedup = {}
                for op in ['sum', 'mean', 'matmul']:
                    if gpu_times[op] > 0:
                        speedup[op] = cpu_times[op] / gpu_times[op]
                    else:
                        speedup[op] = 0

                results['speedup'] = speedup
                results['avg_speedup'] = np.mean(list(speedup.values()))

                self.logger.info(f"GPU benchmark completed. Average speedup: {results['avg_speedup']:.2f}x")

            return results

        except Exception as e:
            self.logger.error(f"Performance benchmark error: {str(e)}")
            results['error'] = str(e)
            return results

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status and information"""
        with self._lock:
            return {
                'gpu_available': self.gpu_available,
                'gpu_enabled': self.config_manager.get("enable_gpu", False),
                'gpu_info': self.gpu_info.copy(),
                'performance_metrics': self.performance_metrics.copy(),
                'libraries': {
                    'cupy_available': CUPY_AVAILABLE,
                    'numba_cuda_available': NUMBA_CUDA_AVAILABLE
                }
            }

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current system"""
        recommendations = []

        if not self.gpu_available:
            recommendations.append("Consider installing GPU-compatible libraries (CuPy, Numba) for acceleration")
            recommendations.append("GPU acceleration could significantly improve ML model training performance")

        if self.gpu_available and not self.config_manager.get("enable_gpu", False):
            recommendations.append("GPU is available but not enabled in configuration")
            recommendations.append("Enable GPU acceleration in settings for better performance")

        # Check memory usage
        metrics = self.performance_metrics
        if metrics.get('memory_utilization', 0) > 0.8:
            recommendations.append("High GPU memory usage detected - consider reducing batch sizes")

        # Check if monitoring is active
        if self.gpu_available and not self.monitoring_active:
            recommendations.append("Enable GPU monitoring for performance tracking")

        return recommendations

    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        try:
            if self.gpu_available and CUPY_AVAILABLE:
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
                self.logger.info("GPU memory cache cleared")
                return True
        except Exception as e:
            self.logger.error(f"GPU memory clear error: {str(e)}")

        return False
