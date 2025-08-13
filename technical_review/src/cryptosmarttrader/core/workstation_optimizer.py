#!/usr/bin/env python3
"""
Workstation Optimizer
Optimize CryptoSmartTrader for i9-32GB-RTX2000 workstation
"""

import os
import sys
import json
import psutil
import platform
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try imports for hardware detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class WorkstationOptimizer:
    """
    Optimize system configuration for specific workstation hardware
    """

    def __init__(self, target_specs: Optional[Dict[str, Any]] = None):
        self.target_specs = target_specs or {
            'cpu': 'i9',
            'ram_gb': 32,
            'gpu': 'RTX2000',
            'gpu_vram_gb': 8,
            'storage': 'SSD'
        }

        self.detected_specs = self._detect_hardware()
        self.optimization_config = self._generate_optimization_config()

    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect actual hardware specifications"""

        specs = {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            'ram_gb': round(psutil.virtual_memory().total / (1024**3)),
            'platform': platform.system(),
            'python_version': sys.version,
            'gpu_available': False,
            'gpu_count': 0,
            'gpu_memory_gb': 0
        }

        # Detect GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            specs['gpu_available'] = True
            specs['gpu_count'] = torch.cuda.device_count()
            specs['gpu_name'] = torch.cuda.get_device_name(0)
            specs['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))

        elif GPUTIL_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                specs['gpu_available'] = True
                specs['gpu_count'] = len(gpus)
                specs['gpu_name'] = gpus[0].name
                specs['gpu_memory_gb'] = round(gpus[0].memoryTotal / 1024)

        return specs

    def _generate_optimization_config(self) -> Dict[str, Any]:
        """Generate optimized configuration based on detected hardware"""

        config = {
            'cpu_optimization': self._optimize_cpu_config(),
            'memory_optimization': self._optimize_memory_config(),
            'gpu_optimization': self._optimize_gpu_config(),
            'storage_optimization': self._optimize_storage_config(),
            'process_optimization': self._optimize_process_config()
        }

        return config

    def _optimize_cpu_config(self) -> Dict[str, Any]:
        """Optimize CPU configuration"""

        cpu_count = self.detected_specs['cpu_count']

        # Conservative allocation for stability
        worker_processes = max(1, int(cpu_count * 0.75))
        async_workers = max(1, int(cpu_count * 0.5))

        return {
            'worker_processes': worker_processes,
            'async_workers': async_workers,
            'parallel_inference': cpu_count >= 8,
            'numa_optimization': cpu_count >= 16,
            'cpu_affinity_enabled': True,
            'process_priority': 'above_normal'
        }

    def _optimize_memory_config(self) -> Dict[str, Any]:
        """Optimize memory configuration"""

        total_ram_gb = self.detected_specs['ram_gb']

        # Allocate memory conservatively (75% max usage)
        available_gb = int(total_ram_gb * 0.75)

        return {
            'max_cache_size_gb': min(8, available_gb // 4),
            'feature_cache_gb': min(4, available_gb // 8),
            'model_cache_gb': min(2, available_gb // 16),
            'data_buffer_gb': min(4, available_gb // 8),
            'memory_mapping_enabled': total_ram_gb >= 16,
            'swap_management': 'aggressive' if total_ram_gb >= 32 else 'conservative'
        }

    def _optimize_gpu_config(self) -> Dict[str, Any]:
        """Optimize GPU configuration"""

        if not self.detected_specs['gpu_available']:
            return {
                'enabled': False,
                'fallback_to_cpu': True,
                'max_batch_size': 64,
                'mixed_precision': False,
                'memory_fraction': 0.0,
                'allow_memory_growth': False,
                'cache_size_mb': 0,
                'cuda_optimization': False,
                'tensor_cores_enabled': False,
                'memory_pool_size_mb': 0
            }

        gpu_memory_gb = self.detected_specs['gpu_memory_gb']

        # RTX 2000 specific optimizations (8GB VRAM)
        if gpu_memory_gb <= 8:
            batch_size = 512
            memory_fraction = 0.8
            cache_size_mb = 1024
        elif gpu_memory_gb <= 16:
            batch_size = 1024
            memory_fraction = 0.85
            cache_size_mb = 2048
        else:
            batch_size = 2048
            memory_fraction = 0.9
            cache_size_mb = 4096

        return {
            'enabled': True,
            'max_batch_size': batch_size,
            'mixed_precision': True,
            'memory_fraction': memory_fraction,
            'allow_memory_growth': True,
            'cache_size_mb': cache_size_mb,
            'cuda_optimization': True,
            'tensor_cores_enabled': True,
            'memory_pool_size_mb': int(gpu_memory_gb * 1024 * 0.1)
        }

    def _optimize_storage_config(self) -> Dict[str, Any]:
        """Optimize storage configuration"""

        # Check if running on SSD (heuristic based on disk performance)
        try:
            disk_usage = psutil.disk_usage('.')
            disk_io = psutil.disk_io_counters()

            # Simple heuristic: assume SSD if high free space ratio
            is_ssd = (disk_usage.free / disk_usage.total) > 0.1
        except:
            is_ssd = True  # Default assumption

        return {
            'cache_to_disk': True,
            'async_io': is_ssd,
            'prefetch_enabled': is_ssd,
            'compression_enabled': not is_ssd,  # Compress for HDD, not SSD
            'temp_dir': './cache/temp',
            'model_storage_dir': './models/optimized'
        }

    def _optimize_process_config(self) -> Dict[str, Any]:
        """Optimize process management"""

        return {
            'agent_isolation': True,
            'process_restart_threshold': 1000,  # MB
            'health_check_interval': 30,  # seconds
            'circuit_breaker_threshold': 5,
            'exponential_backoff': True,
            'graceful_shutdown_timeout': 30
        }

    def apply_optimizations(self) -> Dict[str, Any]:
        """Apply all optimizations and return summary"""

        summary = {
            'timestamp': psutil.boot_time(),
            'detected_hardware': self.detected_specs,
            'optimization_applied': self.optimization_config,
            'compatibility_check': self._check_compatibility(),
            'recommendations': self._generate_recommendations()
        }

        # Write optimization config to file
        config_path = Path('./config/workstation_optimization.json')
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def _check_compatibility(self) -> Dict[str, Any]:
        """Check compatibility with target specifications"""

        compatibility = {
            'cpu_compatible': self.detected_specs['cpu_count'] >= 8,
            'ram_compatible': self.detected_specs['ram_gb'] >= 16,
            'gpu_compatible': self.detected_specs['gpu_available'],
            'platform_compatible': self.detected_specs['platform'] in ['Windows', 'Linux'],
            'python_compatible': sys.version_info >= (3, 9),
            'overall_compatible': True
        }

        compatibility['overall_compatible'] = all([
            compatibility['cpu_compatible'],
            compatibility['ram_compatible'],
            compatibility['platform_compatible'],
            compatibility['python_compatible']
        ])

        return compatibility

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""

        recommendations = []

        # CPU recommendations
        if self.detected_specs['cpu_count'] < 8:
            recommendations.append("Consider upgrading to a multi-core CPU for better performance")

        # RAM recommendations
        if self.detected_specs['ram_gb'] < 16:
            recommendations.append("Increase RAM to at least 16GB for optimal performance")
        elif self.detected_specs['ram_gb'] >= 32:
            recommendations.append("Excellent RAM capacity - enable aggressive caching")

        # GPU recommendations
        if not self.detected_specs['gpu_available']:
            recommendations.append("GPU not detected - install CUDA-compatible GPU for acceleration")
        elif self.detected_specs['gpu_memory_gb'] < 4:
            recommendations.append("GPU has limited VRAM - consider batch size optimization")
        elif self.detected_specs['gpu_memory_gb'] >= 8:
            recommendations.append("Excellent GPU memory - enable large batch processing")

        # Platform recommendations
        if self.detected_specs['platform'] == 'Windows':
            recommendations.append("Configure Windows Defender exceptions for optimal performance")
            recommendations.append("Enable high-performance power plan")

        return recommendations

    def get_workstation_report(self) -> str:
        """Generate human-readable workstation report"""

        report = f"""
CryptoSmartTrader V2 - Workstation Analysis Report
=================================================

Hardware Detected:
- CPU: {self.detected_specs['cpu_count']} cores @ {self.detected_specs.get('cpu_freq', 0):.0f} MHz
- RAM: {self.detected_specs['ram_gb']} GB
- GPU: {self.detected_specs.get('gpu_name', 'Not Available')}
- GPU Memory: {self.detected_specs['gpu_memory_gb']} GB
- Platform: {self.detected_specs['platform']}

Optimization Settings:
- Worker Processes: {self.optimization_config['cpu_optimization']['worker_processes']}
- Max Cache Size: {self.optimization_config['memory_optimization']['max_cache_size_gb']} GB
- GPU Batch Size: {self.optimization_config['gpu_optimization']['max_batch_size']}
- Mixed Precision: {self.optimization_config['gpu_optimization']['mixed_precision']}

Compatibility Status:
- Overall Compatible: {self._check_compatibility()['overall_compatible']}

Recommendations:
"""

        for rec in self._generate_recommendations():
            report += f"- {rec}\n"

        return report

class WorkstationHealthMonitor:
    """
    Monitor workstation health and performance
    """

    def __init__(self):
        self.metrics_history = []

    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""

        metrics = {
            'timestamp': psutil.boot_time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('.').percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'process_count': len(psutil.pids()),
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }

        # GPU metrics if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            metrics['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                if torch.cuda.max_memory_allocated() > 0 else 0
            )
            metrics['gpu_utilization'] = self._get_gpu_utilization()

        self.metrics_history.append(metrics)

        # Keep only last 1000 measurements
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        return metrics

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage"""

        try:
            if GPUTIL_AVAILABLE:
                gpus = GPUtil.getGPUs()
                return gpus[0].load * 100 if gpus else 0
            elif NVML_AVAILABLE:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return utilization.gpu
        except:
            pass

        return 0.0

    def get_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""

        if not self.metrics_history:
            return 50.0

        latest = self.metrics_history[-1]

        # Calculate health components
        cpu_health = max(0, 100 - latest['cpu_percent'])
        memory_health = max(0, 100 - latest['memory_percent'])
        disk_health = max(0, 100 - latest['disk_usage_percent'])

        # Weight the components
        health_score = (
            cpu_health * 0.3 +
            memory_health * 0.4 +
            disk_health * 0.3
        )

        return round(health_score, 2)

def optimize_workstation() -> Dict[str, Any]:
    """Main function to optimize workstation"""

    print("🔧 OPTIMIZING WORKSTATION FOR CRYPTOSMARTTRADER V2")
    print("=" * 60)

    optimizer = WorkstationOptimizer()
    summary = optimizer.apply_optimizations()

    print("📊 Hardware Detection:")
    print(f"   CPU: {summary['detected_hardware']['cpu_count']} cores")
    print(f"   RAM: {summary['detected_hardware']['ram_gb']} GB")
    print(f"   GPU: {summary['detected_hardware'].get('gpu_name', 'Not Available')}")

    print(f"\n⚙️ Optimization Applied:")
    print(f"   Worker Processes: {summary['optimization_applied']['cpu_optimization']['worker_processes']}")
    print(f"   Max Cache: {summary['optimization_applied']['memory_optimization']['max_cache_size_gb']} GB")
    print(f"   GPU Batch Size: {summary['optimization_applied']['gpu_optimization']['max_batch_size']}")

    print(f"\n✅ Compatibility: {summary['compatibility_check']['overall_compatible']}")

    if summary['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations'][:3]:
            print(f"   - {rec}")

    return summary

if __name__ == "__main__":
    result = optimize_workstation()

    # Save detailed report
    report_path = Path('./logs/workstation_optimization_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = WorkstationOptimizer()
    with open(report_path, 'w') as f:
        f.write(optimizer.get_workstation_report())

    print(f"\n📄 Detailed report saved to: {report_path}")
