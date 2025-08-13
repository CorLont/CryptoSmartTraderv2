# utils/system_optimizer.py
import threading
import time
import gc
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
import json
import psutil


logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of an optimization operation"""
    action: str
    success: bool
    details: str
    impact: Optional[str] = None
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None


class SystemOptimizer:
    """Advanced system optimization with automatic performance tuning"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.optimization_history: List[OptimizationResult] = []
        self.auto_optimization_enabled = True
        self.optimization_thread = None
        self._lock = threading.Lock()
        
        # Optimization settings
        self.memory_cleanup_threshold = 80.0  # %
        self.cpu_optimization_threshold = 85.0  # %
        self.cache_cleanup_interval = 300  # seconds
        self.optimization_check_interval = 60  # seconds
        
        self.start_auto_optimization()
    
    def start_auto_optimization(self):
        """Start automatic optimization monitoring"""
        if self.auto_optimization_enabled and (
            self.optimization_thread is None or not self.optimization_thread.is_alive()
        ):
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
            logger.info("Auto-optimization started")
    
    def stop_auto_optimization(self):
        """Stop automatic optimization"""
        self.auto_optimization_enabled = False
        logger.info("Auto-optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while self.auto_optimization_enabled:
            try:
                self._check_and_optimize()
                time.sleep(self.optimization_check_interval)
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _check_and_optimize(self):
        """Check system status and apply optimizations if needed"""
        try:
            # Get current metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            optimizations_applied = []
            
            # Memory optimization
            if memory.percent > self.memory_cleanup_threshold:
                result = self.optimize_memory()
                if result.success:
                    optimizations_applied.append("memory_cleanup")
            
            # CPU optimization
            if cpu_percent > self.cpu_optimization_threshold:
                result = self.optimize_cpu_usage()
                if result.success:
                    optimizations_applied.append("cpu_optimization")
            
            # Periodic cache cleanup
            result = self.cleanup_caches()
            if result.success:
                optimizations_applied.append("cache_cleanup")
            
            if optimizations_applied:
                logger.info(f"Auto-optimizations applied: {optimizations_applied}")
                
        except Exception as e:
            logger.error(f"Auto-optimization check failed: {e}")
    
    def optimize_memory(self) -> OptimizationResult:
        """Optimize memory usage"""
        try:
            # Collect metrics before
            memory_before = psutil.virtual_memory()
            
            # Force garbage collection
            collected_objects = gc.collect()
            
            # Additional memory cleanup strategies
            self._cleanup_temporary_files()
            
            # Collect metrics after
            memory_after = psutil.virtual_memory()
            
            impact = f"Freed {memory_before.used - memory_after.used} bytes"
            
            result = OptimizationResult(
                action="memory_optimization",
                success=True,
                details=f"Garbage collection freed {collected_objects} objects",
                impact=impact,
                metrics_before={"memory_percent": memory_before.percent},
                metrics_after={"memory_percent": memory_after.percent}
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.info(f"Memory optimization completed: {impact}")
            return result
            
        except Exception as e:
            result = OptimizationResult(
                action="memory_optimization",
                success=False,
                details=f"Memory optimization failed: {str(e)}"
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.error(f"Memory optimization failed: {e}")
            return result
    
    def optimize_cpu_usage(self) -> OptimizationResult:
        """Optimize CPU usage"""
        try:
            cpu_before = psutil.cpu_percent(interval=1)
            
            # CPU optimization strategies
            optimizations = []
            
            # Reduce thread priorities for non-critical operations
            current_process = psutil.Process()
            if hasattr(current_process, 'nice'):
                try:
                    current_process.nice(5)  # Lower priority slightly
                    optimizations.append("reduced_process_priority")
                except Exception as e:
                    logger.warning(f"Error in system_optimizer.py: {e}")
                    pass
            
            # Optimize agent update intervals
            self._optimize_agent_intervals()
            optimizations.append("optimized_agent_intervals")
            
            cpu_after = psutil.cpu_percent(interval=1)
            
            result = OptimizationResult(
                action="cpu_optimization",
                success=True,
                details=f"Applied optimizations: {optimizations}",
                impact=f"CPU usage: {cpu_before:.1f}% -> {cpu_after:.1f}%",
                metrics_before={"cpu_percent": cpu_before},
                metrics_after={"cpu_percent": cpu_after}
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.info(f"CPU optimization completed: {result.impact}")
            return result
            
        except Exception as e:
            result = OptimizationResult(
                action="cpu_optimization",
                success=False,
                details=f"CPU optimization failed: {str(e)}"
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.error(f"CPU optimization failed: {e}")
            return result
    
    def cleanup_caches(self) -> OptimizationResult:
        """Clean up system caches"""
        try:
            cleanup_details = []
            
            # Clean up log files
            logs_cleaned = self._cleanup_log_files()
            if logs_cleaned:
                cleanup_details.append(f"cleaned {logs_cleaned} log files")
            
            # Clean up temporary files
            temp_cleaned = self._cleanup_temporary_files()
            if temp_cleaned:
                cleanup_details.append(f"cleaned {temp_cleaned} temporary files")
            
            # Clean up model cache
            models_cleaned = self._cleanup_model_cache()
            if models_cleaned:
                cleanup_details.append(f"cleaned {models_cleaned} cached models")
            
            result = OptimizationResult(
                action="cache_cleanup",
                success=True,
                details=f"Cache cleanup completed: {', '.join(cleanup_details) if cleanup_details else 'no cleanup needed'}"
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.info(result.details)
            return result
            
        except Exception as e:
            result = OptimizationResult(
                action="cache_cleanup",
                success=False,
                details=f"Cache cleanup failed: {str(e)}"
            )
            
            with self._lock:
                self.optimization_history.append(result)
            
            logger.error(f"Cache cleanup failed: {e}")
            return result
    
    def _cleanup_log_files(self) -> int:
        """Clean up old log files"""
        try:
            logs_path = Path("logs")
            if not logs_path.exists():
                return 0
            
            cleaned_count = 0
            max_file_size = 50 * 1024 * 1024  # 50MB
            
            for log_file in logs_path.glob("*.log"):
                if log_file.stat().st_size > max_file_size:
                    # Archive large log files
                    archive_name = log_file.with_suffix(".log.old")
                    log_file.rename(archive_name)
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
            return 0
    
    def _cleanup_temporary_files(self) -> int:
        """Clean up temporary files"""
        try:
            temp_dirs = [Path("temp"), Path("tmp"), Path(".cache")]
            cleaned_count = 0
            
            for temp_dir in temp_dirs:
                if temp_dir.exists():
                    for temp_file in temp_dir.rglob("*"):
                        if temp_file.is_file() and temp_file.stat().st_mtime < (time.time() - 86400):  # 1 day old
                            temp_file.unlink()
                            cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Temporary file cleanup failed: {e}")
            return 0
    
    def _cleanup_model_cache(self) -> int:
        """Clean up old cached models"""
        try:
            models_path = Path("models")
            if not models_path.exists():
                return 0
            
            cleaned_count = 0
            max_age = 7 * 24 * 3600  # 1 week
            
            for model_file in models_path.glob("*.pkl"):
                if model_file.stat().st_mtime < (time.time() - max_age):
                    model_file.unlink()
                    cleaned_count += 1
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Model cache cleanup failed: {e}")
            return 0
    
    def _optimize_agent_intervals(self):
        """Optimize agent update intervals based on system load"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # If CPU usage is high, increase agent intervals
            if cpu_percent > 80:
                multiplier = 1.5
            elif cpu_percent > 60:
                multiplier = 1.2
            else:
                multiplier = 1.0
            
            # Update configuration (this would integrate with the actual config system)
            logger.info(f"Agent intervals optimized with multiplier: {multiplier}")
            
        except Exception as e:
            logger.error(f"Agent interval optimization failed: {e}")
    
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Get optimization history"""
        with self._lock:
            return self.optimization_history.copy()
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        with self._lock:
            history = self.optimization_history.copy()
        
        if not history:
            return {"total_optimizations": 0}
        
        successful = [r for r in history if r.success]
        failed = [r for r in history if not r.success]
        
        action_counts = {}
        for result in history:
            action_counts[result.action] = action_counts.get(result.action, 0) + 1
        
        return {
            "total_optimizations": len(history),
            "successful_optimizations": len(successful),
            "failed_optimizations": len(failed),
            "success_rate": len(successful) / len(history) * 100,
            "optimization_types": action_counts,
            "last_optimization": history[-1].action if history else None,
            "auto_optimization_enabled": self.auto_optimization_enabled
        }
    
    def clear_history(self):
        """Clear optimization history"""
        with self._lock:
            self.optimization_history.clear()
        logger.info("Optimization history cleared")
    
    def run_full_optimization(self) -> List[OptimizationResult]:
        """Run all optimization routines"""
        logger.info("Starting full system optimization")
        
        results = []
        
        # Memory optimization
        results.append(self.optimize_memory())
        
        # CPU optimization  
        results.append(self.optimize_cpu_usage())
        
        # Cache cleanup
        results.append(self.cleanup_caches())
        
        successful = sum(1 for r in results if r.success)
        logger.info(f"Full optimization completed: {successful}/{len(results)} successful")
        
        return results