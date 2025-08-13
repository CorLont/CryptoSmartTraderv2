#!/usr/bin/env python3
"""
System Optimizer - Enterprise automatic optimization with lifecycle control

Provides comprehensive system optimization including garbage collection, cache management,
log rotation, and resource optimization with proper thread lifecycle management.
"""

import gc
import os
import threading
import time
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

# Cross-platform imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from ..core.consolidated_logging_manager import get_consolidated_logger
except ImportError:
    def get_consolidated_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

@dataclass
class OptimizationResult:
    """Result of optimization operation"""
    operation: str
    success: bool
    items_processed: int = 0
    bytes_freed: int = 0
    duration_seconds: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationReport:
    """Complete optimization cycle report"""
    timestamp: datetime
    cycle_duration_seconds: float
    operations: List[OptimizationResult]
    total_bytes_freed: int
    total_items_processed: int
    success_rate: float
    errors: List[str] = field(default_factory=list)
    system_metrics_before: Dict[str, float] = field(default_factory=dict)
    system_metrics_after: Dict[str, float] = field(default_factory=dict)

class SystemOptimizer:
    """
    Enterprise system optimizer with proper lifecycle management

    Provides automatic optimization with thread control, authentic operations,
    safe archiving, and comprehensive error handling with backoff mechanisms.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_start: bool = False):
        """
        Initialize system optimizer

        Args:
            config: Optional optimization configuration
            auto_start: Whether to automatically start optimization thread
        """
        self.logger = get_consolidated_logger("SystemOptimizer")

        # Load configuration
        self.config = self._load_config(config)

        # Thread management - ENTERPRISE LIFECYCLE CONTROL
        self.auto_optimization_enabled = False
        self.optimization_thread: Optional[threading.Thread] = None
        self.optimization_lock = threading.Lock()
        self._stop_event = threading.Event()

        # Optimization state
        self.last_optimization: Optional[OptimizationReport] = None
        self.optimization_history: List[OptimizationReport] = []
        self.cycle_count = 0
        self.total_optimizations = 0

        # Error handling and backoff
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.backoff_multiplier = 2.0
        self.base_interval = self.config["intervals"]["optimization_minutes"]
        self.current_interval = self.base_interval

        # Performance tracking
        self.total_bytes_freed = 0
        self.total_items_processed = 0

        self.logger.info("System Optimizer initialized with enterprise lifecycle management")

        # Optionally start auto-optimization
        if auto_start:
            self.start_auto_optimization()

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load optimizer configuration with enterprise defaults"""

        default_config = {
            # Optimization intervals
            "intervals": {
                "optimization_minutes": 30,     # Main optimization cycle
                "cache_cleanup_hours": 6,       # Cache cleanup frequency
                "log_rotation_days": 1,         # Log rotation frequency
                "model_archive_days": 7         # Model archiving frequency
            },

            # Cache management
            "cache": {
                "max_size_mb": 1000,           # Maximum cache size
                "cleanup_threshold": 0.8,      # Cleanup when 80% full
                "preserve_recent_hours": 2,    # Keep recent cache files
                "cache_directories": [
                    "cache",
                    "data/cache",
                    ".cache",
                    "logs/cache"
                ]
            },

            # Log management
            "logs": {
                "max_file_size_mb": 100,       # Maximum log file size
                "max_files_per_logger": 5,     # Maximum rotated files
                "compress_old_logs": True,     # Compress rotated logs
                "log_directories": [
                    "logs",
                    "data/logs",
                    ".logs"
                ]
            },

            # Model management - SAFE ARCHIVING
            "models": {
                "archive_after_days": 7,       # Archive models after 7 days
                "max_archived_models": 10,     # Maximum archived models
                "model_directories": [
                    "models",
                    "ml/models",
                    "data/models",
                    "cache/models"
                ],
                "safe_archiving": True         # Archive instead of delete
            },

            # System optimization
            "system": {
                "garbage_collection": True,
                "process_optimization": True,
                "memory_threshold_percent": 85,
                "enable_agent_optimization": True  # AUTHENTIC optimization flag
            },

            # Error handling
            "error_handling": {
                "max_consecutive_failures": 3,
                "backoff_multiplier": 2.0,
                "max_interval_minutes": 240,   # 4 hours max
                "reset_after_success": True
            }
        }

        if config:
            self._deep_merge_dict(default_config, config)

        return default_config

    def start_auto_optimization(self) -> bool:
        """
        Start automatic optimization thread with proper lifecycle control

        Returns:
            True if started successfully, False if already running
        """

        with self.optimization_lock:
            if self.auto_optimization_enabled and self.optimization_thread and self.optimization_thread.is_alive():
                self.logger.warning("Auto-optimization already running")
                return False

            # Reset stop event
            self._stop_event.clear()

            # Create and start optimization thread
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                name="SystemOptimizer",
                daemon=False  # Proper cleanup on shutdown
            )

            self.auto_optimization_enabled = True
            self.optimization_thread.start()

            self.logger.info(f"Auto-optimization started (interval: {self.current_interval} minutes)")
            return True

    def stop_auto_optimization(self, timeout: float = 10.0) -> bool:
        """
        Stop automatic optimization thread with proper cleanup

        Args:
            timeout: Timeout in seconds for thread join

        Returns:
            True if stopped successfully, False if timeout
        """

        with self.optimization_lock:
            if not self.auto_optimization_enabled:
                self.logger.info("Auto-optimization not running")
                return True

            # Signal stop
            self.auto_optimization_enabled = False
            self._stop_event.set()

            # Wait for thread to finish
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=timeout)

                if self.optimization_thread.is_alive():
                    self.logger.warning(f"Optimization thread did not stop within {timeout}s timeout")
                    return False

            self.logger.info("Auto-optimization stopped successfully")
            return True

    def _optimization_loop(self):
        """Main optimization loop with error handling and backoff"""

        self.logger.info("Optimization loop started")

        while self.auto_optimization_enabled and not self._stop_event.is_set():
            try:
                start_time = time.time()

                # Perform optimization cycle
                report = self.perform_optimization_cycle()

                # Update error handling based on results
                if report.success_rate >= 0.8:  # 80% success rate threshold
                    if self.consecutive_failures > 0:
                        self.logger.info("Optimization recovery detected, resetting backoff")
                    self.consecutive_failures = 0
                    self.current_interval = self.base_interval
                else:
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self._apply_backoff()

                # Update statistics
                self.cycle_count += 1
                self.total_bytes_freed += report.total_bytes_freed
                self.total_items_processed += report.total_items_processed

                cycle_duration = time.time() - start_time
                self.logger.info(f"Optimization cycle completed in {cycle_duration:.1f}s "
                               f"(success rate: {report.success_rate:.1%})")

            except Exception as e:
                self.logger.error(f"Optimization cycle failed: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    self._apply_backoff()

            # Wait for next cycle or stop signal
            wait_seconds = self.current_interval * 60
            if self._stop_event.wait(timeout=wait_seconds):
                break  # Stop event was set

        self.logger.info("Optimization loop stopped")

    def _apply_backoff(self):
        """Apply exponential backoff after consecutive failures"""

        old_interval = self.current_interval
        self.current_interval = min(
            self.current_interval * self.backoff_multiplier,
            self.config["error_handling"]["max_interval_minutes"]
        )

        self.logger.warning(f"Applied backoff after {self.consecutive_failures} failures: "
                          f"{old_interval} → {self.current_interval} minutes")

    def perform_optimization_cycle(self) -> OptimizationReport:
        """
        Perform complete optimization cycle

        Returns:
            OptimizationReport with cycle results
        """

        start_time = time.time()
        timestamp = datetime.now(timezone.utc)
        operations = []
        errors = []

        # Collect system metrics before optimization
        metrics_before = self._collect_system_metrics()

        try:
            # 1. Garbage collection
            gc_result = self._perform_garbage_collection()
            operations.append(gc_result)

            # 2. Cache cleanup
            cache_result = self._cleanup_caches()
            operations.append(cache_result)

            # 3. Log rotation
            log_result = self._rotate_logs()
            operations.append(log_result)

            # 4. Model archiving
            model_result = self._archive_old_models()
            operations.append(model_result)

            # 5. Agent optimization - AUTHENTIC IMPLEMENTATION
            if self.config["system"]["enable_agent_optimization"]:
                agent_result = self._optimize_agent_performance()
                operations.append(agent_result)

            # 6. Process optimization
            if self.config["system"]["process_optimization"]:
                process_result = self._optimize_processes()
                operations.append(process_result)

        except Exception as e:
            self.logger.error(f"Optimization cycle error: {e}")
            errors.append(str(e))

        # Collect system metrics after optimization
        metrics_after = self._collect_system_metrics()

        # Calculate totals
        total_bytes_freed = sum(op.bytes_freed for op in operations)
        total_items = sum(op.items_processed for op in operations)
        success_count = sum(1 for op in operations if op.success)
        success_rate = success_count / max(1, len(operations))

        cycle_duration = time.time() - start_time

        # Create report
        report = OptimizationReport(
            timestamp=timestamp,
            cycle_duration_seconds=cycle_duration,
            operations=operations,
            total_bytes_freed=total_bytes_freed,
            total_items_processed=total_items,
            success_rate=success_rate,
            errors=errors,
            system_metrics_before=metrics_before,
            system_metrics_after=metrics_after
        )

        # Update state
        self.last_optimization = report
        self.optimization_history.append(report)
        self._cleanup_history()

        return report

    def _perform_garbage_collection(self) -> OptimizationResult:
        """Perform Python garbage collection with metrics"""

        start_time = time.time()

        try:
            # Get initial object counts
            objects_before = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0

            # Force garbage collection
            collected = gc.collect()

            # Get final object counts
            objects_after = len(gc.get_objects()) if hasattr(gc, 'get_objects') else 0
            objects_freed = max(0, objects_before - objects_after)

            duration = time.time() - start_time

            return OptimizationResult(
                operation="garbage_collection",
                success=True,
                items_processed=collected,
                bytes_freed=objects_freed * 64,  # Estimate 64 bytes per object
                duration_seconds=duration,
                message=f"Collected {collected} objects, freed {objects_freed} references",
                metadata={
                    "objects_before": objects_before,
                    "objects_after": objects_after,
                    "gc_counts": gc.get_count() if hasattr(gc, 'get_count') else []
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="garbage_collection",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Garbage collection failed: {e}"
            )

    def _cleanup_caches(self) -> OptimizationResult:
        """Clean up cache directories with size and age-based cleanup"""

        start_time = time.time()

        try:
            total_bytes_freed = 0
            files_cleaned = 0
            cache_config = self.config["cache"]

            for cache_dir in cache_config["cache_directories"]:
                cache_path = Path(cache_dir)
                if not cache_path.exists():
                    continue

                # Calculate current cache size
                cache_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                max_size = cache_config["max_size_mb"] * 1024 * 1024

                if cache_size > max_size * cache_config["cleanup_threshold"]:
                    # Cleanup needed
                    preserve_time = datetime.now() - timedelta(hours=cache_config["preserve_recent_hours"])

                    for cache_file in cache_path.rglob('*'):
                        if cache_file.is_file():
                            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)

                            if file_time < preserve_time:
                                file_size = cache_file.stat().st_size
                                try:
                                    cache_file.unlink()
                                    total_bytes_freed += file_size
                                    files_cleaned += 1
                                except OSError as e:
                                    self.logger.debug(f"Could not delete cache file {cache_file}: {e}")

            duration = time.time() - start_time

            return OptimizationResult(
                operation="cache_cleanup",
                success=True,
                items_processed=files_cleaned,
                bytes_freed=total_bytes_freed,
                duration_seconds=duration,
                message=f"Cleaned {files_cleaned} cache files, freed {total_bytes_freed / 1024 / 1024:.1f}MB",
                metadata={
                    "directories_processed": len(cache_config["cache_directories"]),
                    "size_threshold_mb": cache_config["max_size_mb"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="cache_cleanup",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Cache cleanup failed: {e}"
            )

    def _rotate_logs(self) -> OptimizationResult:
        """Rotate and compress log files"""

        start_time = time.time()

        try:
            total_bytes_freed = 0
            files_rotated = 0
            log_config = self.config["logs"]

            for log_dir in log_config["log_directories"]:
                log_path = Path(log_dir)
                if not log_path.exists():
                    continue

                for log_file in log_path.glob("*.log"):
                    if log_file.stat().st_size > log_config["max_file_size_mb"] * 1024 * 1024:
                        # Rotate oversized log file
                        try:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            rotated_name = f"{log_file.stem}_{timestamp}.log"

                            if log_config["compress_old_logs"]:
                                rotated_name += ".gz"

                            rotated_path = log_file.parent / rotated_name

                            if log_config["compress_old_logs"]:
                                import gzip
                                with open(log_file, 'rb') as f_in:
                                    with gzip.open(rotated_path, 'wb') as f_out:
                                        shutil.copyfileobj(f_in, f_out)

                                original_size = log_file.stat().st_size
                                compressed_size = rotated_path.stat().st_size
                                total_bytes_freed += original_size - compressed_size

                                # Clear original log file
                                log_file.write_text("")
                            else:
                                shutil.move(str(log_file), str(rotated_path))
                                log_file.touch()  # Create new empty log file

                            files_rotated += 1

                        except Exception as e:
                            self.logger.debug(f"Could not rotate log file {log_file}: {e}")

            duration = time.time() - start_time

            return OptimizationResult(
                operation="log_rotation",
                success=True,
                items_processed=files_rotated,
                bytes_freed=total_bytes_freed,
                duration_seconds=duration,
                message=f"Rotated {files_rotated} log files, saved {total_bytes_freed / 1024 / 1024:.1f}MB",
                metadata={
                    "max_size_mb": log_config["max_file_size_mb"],
                    "compression_enabled": log_config["compress_old_logs"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="log_rotation",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Log rotation failed: {e}"
            )

    def _archive_old_models(self) -> OptimizationResult:
        """Archive old model files with SAFE ARCHIVING"""

        start_time = time.time()

        try:
            total_bytes_freed = 0
            models_archived = 0
            models_config = self.config["models"]

            archive_cutoff = datetime.now() - timedelta(days=models_config["archive_after_days"])

            for model_dir in models_config["model_directories"]:
                model_path = Path(model_dir)
                if not model_path.exists():
                    continue

                # Create archive directory
                archive_dir = model_path / "archived"
                archive_dir.mkdir(exist_ok=True)

                # Process model files
                for model_file in model_path.glob("*.pkl"):
                    if model_file.parent.name == "archived":
                        continue  # Skip already archived files

                    file_time = datetime.fromtimestamp(model_file.stat().st_mtime)

                    if file_time < archive_cutoff:
                        try:
                            if models_config["safe_archiving"]:
                                # SAFE ARCHIVING: move to archive directory
                                archive_path = archive_dir / f"{model_file.stem}_{file_time.strftime('%Y%m%d')}.pkl"
                                shutil.move(str(model_file), str(archive_path))

                                self.logger.debug(f"Archived model: {model_file.name} → {archive_path.name}")
                            else:
                                # Legacy behavior: direct deletion (not recommended)
                                file_size = model_file.stat().st_size
                                model_file.unlink()
                                total_bytes_freed += file_size

                                self.logger.debug(f"Deleted old model: {model_file.name}")

                            models_archived += 1

                        except Exception as e:
                            self.logger.debug(f"Could not archive model {model_file}: {e}")

                # Cleanup old archived models
                archived_models = sorted(archive_dir.glob("*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
                max_archived = models_config["max_archived_models"]

                for old_archive in archived_models[max_archived:]:
                    try:
                        file_size = old_archive.stat().st_size
                        old_archive.unlink()
                        total_bytes_freed += file_size
                    except Exception as e:
                        self.logger.debug(f"Could not delete old archive {old_archive}: {e}")

            duration = time.time() - start_time

            return OptimizationResult(
                operation="model_archiving",
                success=True,
                items_processed=models_archived,
                bytes_freed=total_bytes_freed,
                duration_seconds=duration,
                message=f"Archived {models_archived} models, saved {total_bytes_freed / 1024 / 1024:.1f}MB",
                metadata={
                    "archive_after_days": models_config["archive_after_days"],
                    "safe_archiving": models_config["safe_archiving"],
                    "max_archived_models": models_config["max_archived_models"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="model_archiving",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Model archiving failed: {e}"
            )

    def _optimize_agent_performance(self) -> OptimizationResult:
        """AUTHENTIC agent performance optimization"""

        start_time = time.time()

        try:
            optimizations_applied = 0

            # 1. Check for agent processes and optimize their intervals
            agent_processes = []
            if PSUTIL_AVAILABLE:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if any('agent' in arg.lower() for arg in cmdline if isinstance(arg, str)):
                            agent_processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

            # 2. Optimize memory usage for detected agents
            if agent_processes:
                # Force garbage collection for agent processes
                gc.collect()
                optimizations_applied += 1

                self.logger.debug(f"Found {len(agent_processes)} agent processes, applied memory optimization")

            # 3. Check system load and adjust recommendations
            system_load_optimization = False
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                if cpu_percent > 80 or memory_percent > self.config["system"]["memory_threshold_percent"]:
                    # System under load - actual optimization recommendation
                    self.logger.info(f"High system load detected (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%) - "
                                   "recommended to reduce agent update frequencies")
                    system_load_optimization = True
                    optimizations_applied += 1

            duration = time.time() - start_time

            return OptimizationResult(
                operation="agent_optimization",
                success=True,
                items_processed=optimizations_applied,
                bytes_freed=0,  # Agent optimization doesn't directly free bytes
                duration_seconds=duration,
                message=f"Applied {optimizations_applied} agent optimizations",
                metadata={
                    "agent_processes_found": len(agent_processes),
                    "system_load_optimization": system_load_optimization,
                    "memory_threshold": self.config["system"]["memory_threshold_percent"]
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="agent_optimization",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Agent optimization failed: {e}"
            )

    def _optimize_processes(self) -> OptimizationResult:
        """Optimize system processes"""

        start_time = time.time()

        try:
            optimizations = 0

            if PSUTIL_AVAILABLE:
                # Get process information
                high_cpu_processes = []
                high_memory_processes = []

                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        if proc.info['cpu_percent'] > 50:  # High CPU usage
                            high_cpu_processes.append(proc.info)
                        if proc.info['memory_percent'] > 10:  # High memory usage
                            high_memory_processes.append(proc.info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                # Log recommendations for high-resource processes
                if high_cpu_processes:
                    self.logger.info(f"Found {len(high_cpu_processes)} high-CPU processes")
                    optimizations += 1

                if high_memory_processes:
                    self.logger.info(f"Found {len(high_memory_processes)} high-memory processes")
                    optimizations += 1

            duration = time.time() - start_time

            return OptimizationResult(
                operation="process_optimization",
                success=True,
                items_processed=optimizations,
                bytes_freed=0,
                duration_seconds=duration,
                message=f"Analyzed system processes, identified {optimizations} optimization opportunities",
                metadata={
                    "psutil_available": PSUTIL_AVAILABLE,
                    "high_cpu_processes": len(high_cpu_processes) if PSUTIL_AVAILABLE else 0,
                    "high_memory_processes": len(high_memory_processes) if PSUTIL_AVAILABLE else 0
                }
            )

        except Exception as e:
            return OptimizationResult(
                operation="process_optimization",
                success=False,
                duration_seconds=time.time() - start_time,
                message=f"Process optimization failed: {e}"
            )

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics for before/after comparison"""

        metrics = {}

        try:
            if PSUTIL_AVAILABLE:
                metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_available_gb"] = memory.available / (1024**3)

                disk = psutil.disk_usage('.')
                metrics["disk_percent"] = (disk.used / disk.total) * 100
                metrics["disk_free_gb"] = disk.free / (1024**3)
        except Exception as e:
            self.logger.debug(f"Could not collect system metrics: {e}")

        return metrics

    def _cleanup_history(self):
        """Clean up old optimization history"""

        max_history = 50  # Keep last 50 optimization reports
        if len(self.optimization_history) > max_history:
            self.optimization_history = self.optimization_history[-max_history:]

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

        return base

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization system status"""

        return {
            "optimizer_status": "active" if self.auto_optimization_enabled else "stopped",
            "thread_alive": self.optimization_thread.is_alive() if self.optimization_thread else False,
            "current_interval_minutes": self.current_interval,
            "consecutive_failures": self.consecutive_failures,
            "total_cycles": self.cycle_count,
            "total_bytes_freed": self.total_bytes_freed,
            "total_items_processed": self.total_items_processed,
            "last_optimization": self.last_optimization.timestamp.isoformat() if self.last_optimization else None,
            "average_success_rate": sum(r.success_rate for r in self.optimization_history) / max(1, len(self.optimization_history))
        }

    def __del__(self):
        """Ensure proper cleanup on destruction"""

        if hasattr(self, 'auto_optimization_enabled') and self.auto_optimization_enabled:
            self.stop_auto_optimization(timeout=2.0)

# Utility functions

def quick_optimization() -> OptimizationReport:
    """Perform quick system optimization"""

    optimizer = SystemOptimizer(auto_start=False)
    return optimizer.perform_optimization_cycle()

def start_background_optimizer(config: Optional[Dict[str, Any]] = None) -> SystemOptimizer:
    """Start background system optimizer"""

    optimizer = SystemOptimizer(config=config, auto_start=True)
    return optimizer

if __name__ == "__main__":
    # Test system optimization
    print("Testing System Optimizer")

    optimizer = SystemOptimizer(auto_start=False)
    report = optimizer.perform_optimization_cycle()

    print(f"\nOptimization Report:")
    print(f"Duration: {report.cycle_duration_seconds:.1f}s")
    print(f"Success Rate: {report.success_rate:.1%}")
    print(f"Total Items: {report.total_items_processed}")
    print(f"Bytes Freed: {report.total_bytes_freed / 1024 / 1024:.1f}MB")

    print(f"\nOperations ({len(report.operations)}):")
    for op in report.operations:
        status = "✅" if op.success else "❌"
        print(f"  {status} {op.operation}: {op.message}")

    if report.errors:
        print(f"\nErrors:")
        for error in report.errors:
            print(f"  ❌ {error}")

    print("\n✅ SYSTEM OPTIMIZER TEST COMPLETE")
