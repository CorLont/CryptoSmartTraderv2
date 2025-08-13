#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Async Coordination Manager
True async processing with 100+ concurrent tasks and zero blocking operations
"""

import asyncio
import aiohttp
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Coroutine, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import weakref
import traceback


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class AsyncTask:
    """Async task with comprehensive tracking"""

    task_id: str
    name: str
    coroutine: Optional[Coroutine] = None
    function: Optional[Callable] = None
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 5  # 1=highest, 10=lowest
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[Exception] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class CoordinatorConfig:
    """Configuration for async coordinator"""

    max_concurrent_tasks: int = 100
    max_thread_pool_workers: int = 20
    default_timeout: float = 30.0
    task_cleanup_interval: int = 300  # 5 minutes
    max_task_history: int = 10000
    enable_performance_monitoring: bool = True
    rate_limit_requests_per_second: float = 50.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60


class AsyncCoordinator:
    """Enterprise async coordination manager for true non-blocking operations"""

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()
        self.logger = logging.getLogger(f"{__name__}.AsyncCoordinator")

        # Task management
        self.tasks: Dict[str, AsyncTask] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_history: List[AsyncTask] = []
        self._task_counter = 0
        self._lock = threading.RLock()

        # Async infrastructure
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_thread_pool_workers)

        # Performance monitoring
        self.performance_metrics = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "average_task_duration": 0.0,
            "current_concurrent_tasks": 0,
            "peak_concurrent_tasks": 0,
            "last_cleanup": datetime.now(),
        }

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(int(self.config.rate_limit_requests_per_second))
        self.request_timestamps: List[float] = []

        # Circuit breaker pattern
        self.circuit_breakers: Dict[str, Dict] = {}

        # Event loop management
        self._ensure_event_loop()

        # Start background tasks
        self._start_background_tasks()

        self.logger.info(
            f"Async Coordinator initialized with {self.config.max_concurrent_tasks} max concurrent tasks"
        )

    def _ensure_event_loop(self):
        """Ensure we have a running event loop"""
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            # Create new event loop if none exists
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Start event loop in separate thread
            self.loop_thread = threading.Thread(
                target=self._run_event_loop, daemon=True, name="AsyncCoordinatorEventLoop"
            )
            self.loop_thread.start()
            time.sleep(0.1)  # Allow loop to start

    def _run_event_loop(self):
        """Run the event loop in a separate thread"""
        try:
            self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Event loop crashed: {e}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.loop and not self.loop.is_closed():
            # Schedule periodic cleanup
            asyncio.run_coroutine_threadsafe(self._periodic_cleanup(), self.loop)

            # Schedule performance monitoring
            if self.config.enable_performance_monitoring:
                asyncio.run_coroutine_threadsafe(self._performance_monitor(), self.loop)

    async def _periodic_cleanup(self):
        """Periodic cleanup of completed tasks and metrics"""
        while True:
            try:
                await asyncio.sleep(self.config.task_cleanup_interval)
                await self._cleanup_completed_tasks()
                await self._cleanup_circuit_breakers()
                await self._update_performance_metrics()

            except Exception as e:
                self.logger.error(f"Periodic cleanup error: {e}")

    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Log every minute
                metrics = self.get_performance_metrics()
                self.logger.info(f"Performance metrics: {metrics}")

            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")

    async def get_http_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper configuration"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.default_timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.max_concurrent_tasks,
                limit_per_host=20,
                enable_cleanup_closed=True,
            )

            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": "CryptoSmartTrader/2.0 (Async)"},
            )

        return self.session

    def submit_task(
        self,
        task_name: str,
        coroutine: Optional[Coroutine] = None,
        function: Optional[Callable] = None,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 5,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        dependencies: List[str] = None,
    ) -> str:
        """
        Submit a task for async execution

        Args:
            task_name: Human-readable task name
            coroutine: Async coroutine to execute
            function: Sync function to execute in thread pool
            args: Function arguments
            kwargs: Function keyword arguments
            priority: Task priority (1=highest, 10=lowest)
            timeout: Task timeout in seconds
            max_retries: Maximum retry attempts
            dependencies: List of task IDs that must complete first

        Returns:
            Task ID for tracking
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}_{int(time.time())}"

            task = AsyncTask(
                task_id=task_id,
                name=task_name,
                coroutine=coroutine,
                function=function,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                timeout=timeout or self.config.default_timeout,
                max_retries=max_retries,
                dependencies=dependencies or [],
            )

            self.tasks[task_id] = task
            self.performance_metrics["total_tasks_submitted"] += 1

            # Schedule task execution
            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self._execute_task(task), self.loop)

            self.logger.debug(f"Submitted task {task_id}: {task_name}")
            return task_id

    async def _execute_task(self, task: AsyncTask):
        """Execute a single task with full error handling and retry logic"""
        task_id = task.task_id

        try:
            # Check dependencies
            if not await self._check_dependencies(task):
                task.status = TaskStatus.FAILED
                task.error = Exception("Dependencies not satisfied")
                return

            # Apply rate limiting
            async with self.rate_limiter:
                await self._apply_rate_limiting()

                # Check circuit breaker
                if self._is_circuit_breaker_open(task.name):
                    task.status = TaskStatus.FAILED
                    task.error = Exception(f"Circuit breaker open for {task.name}")
                    return

                # Execute task with timeout
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                try:
                    if task.coroutine:
                        result = await asyncio.wait_for(task.coroutine, timeout=task.timeout)
                    elif task.function:
                        result = await asyncio.wait_for(
                            self._run_in_thread_pool(task), timeout=task.timeout
                        )
                    else:
                        raise ValueError("No coroutine or function provided")

                    task.result = result
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()

                    self.performance_metrics["total_tasks_completed"] += 1
                    self._record_circuit_breaker_success(task.name)

                except asyncio.TimeoutError:
                    task.status = TaskStatus.TIMEOUT
                    task.error = Exception(f"Task timed out after {task.timeout}s")
                    self._record_circuit_breaker_failure(task.name)

                except Exception as e:
                    task.error = e
                    task.status = TaskStatus.FAILED
                    self.performance_metrics["total_tasks_failed"] += 1
                    self._record_circuit_breaker_failure(task.name)

                    # Retry logic
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING

                        # Exponential backoff
                        delay = task.retry_delay * (2 ** (task.retry_count - 1))
                        await asyncio.sleep(delay)

                        self.logger.warning(f"Retrying task {task_id} (attempt {task.retry_count})")
                        await self._execute_task(task)
                        return

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            self.logger.error(f"Critical error executing task {task_id}: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        finally:
            # Cleanup
            if task_id in self.running_tasks:
                self.running_tasks.pop(task_id, None)

            # Move to history
            self.task_history.append(task)
            if len(self.task_history) > self.config.max_task_history:
                self.task_history = self.task_history[-self.config.max_task_history :]

    async def _run_in_thread_pool(self, task: AsyncTask):
        """Run sync function in thread pool"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.thread_pool, lambda: task.function(*task.args, **task.kwargs)

    async def _check_dependencies(self, task: AsyncTask) -> bool:
        """Check if all task dependencies are satisfied"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False

        return True

    async def _apply_rate_limiting(self):
        """Apply rate limiting to prevent overwhelming external services"""
        current_time = time.time()

        # Clean old timestamps
        self.request_timestamps = [ts for ts in self.request_timestamps if current_time - ts < 1.0]

        # Check if we need to wait
        if len(self.request_timestamps) >= self.config.rate_limit_requests_per_second:
            sleep_time = 1.0 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(current_time)

    def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open for a service"""
        cb = self.circuit_breakers.get(service_name)
        if not cb:
            return False

        if cb["state"] == "open":
            if time.time() - cb["last_failure"] > self.config.circuit_breaker_recovery_timeout:
                cb["state"] = "half_open"
                cb["failure_count"] = 0
                return False
            return True

        return False

    def _record_circuit_breaker_failure(self, service_name: str):
        """Record a failure for circuit breaker"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = {
                "failure_count": 0,
                "last_failure": 0,
                "state": "closed",
            }

        cb = self.circuit_breakers[service_name]
        cb["failure_count"] += 1
        cb["last_failure"] = time.time()

        if cb["failure_count"] >= self.config.circuit_breaker_failure_threshold:
            cb["state"] = "open"
            self.logger.warning(f"Circuit breaker opened for {service_name}")

    def _record_circuit_breaker_success(self, service_name: str):
        """Record a success for circuit breaker"""
        if service_name in self.circuit_breakers:
            cb = self.circuit_breakers[service_name]
            if cb["state"] == "half_open":
                cb["state"] = "closed"
                cb["failure_count"] = 0

    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks to prevent memory leaks"""
        with self._lock:
            completed_tasks = [
                task_id
                for task_id, task in self.tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
                and task.completed_at
                and (datetime.now() - task.completed_at).total_seconds() > 300  # 5 minutes
            ]

            for task_id in completed_tasks:
                self.tasks.pop(task_id, None)

            if completed_tasks:
                self.logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks")

    async def _cleanup_circuit_breakers(self):
        """Clean up old circuit breaker states"""
        current_time = time.time()

        for service_name, cb in list(self.circuit_breakers.items()):
            if (current_time - cb["last_failure"]) > (
                self.config.circuit_breaker_recovery_timeout * 2
            ):
                if cb["state"] == "closed" and cb["failure_count"] == 0:
                    del self.circuit_breakers[service_name]

    async def _update_performance_metrics(self):
        """Update performance metrics"""
        with self._lock:
            # Update current concurrent tasks
            self.performance_metrics["current_concurrent_tasks"] = len(
                [task for task in self.tasks.values() if task.status == TaskStatus.RUNNING]
            )

            # Update peak concurrent tasks
            current_concurrent = self.performance_metrics["current_concurrent_tasks"]
            if current_concurrent > self.performance_metrics["peak_concurrent_tasks"]:
                self.performance_metrics["peak_concurrent_tasks"] = current_concurrent

            # Calculate average task duration
            completed_tasks = [
                task
                for task in self.task_history
                if task.status == TaskStatus.COMPLETED and task.started_at and task.completed_at
            ]

            if completed_tasks:
                total_duration = sum(
                    (task.completed_at - task.started_at).total_seconds()
                    for task in completed_tasks
                )
                self.performance_metrics["average_task_duration"] = total_duration / len(
                    completed_tasks
                )

            self.performance_metrics["last_cleanup"] = datetime.now()

    def get_task_status(self, task_id: str) -> Optional[AsyncTask]:
        """Get status of a specific task"""
        return self.tasks.get(task_id)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self._lock:
            return self.performance_metrics.copy()

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            return False

        task.status = TaskStatus.CANCELLED

        # Cancel the actual asyncio task if running
        if task_id in self.running_tasks:
            asyncio_task = self.running_tasks[task_id]
            asyncio_task.cancel()

        return True

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        with self._lock:
            health = {
                "event_loop_running": self.loop is not None and not self.loop.is_closed(),
                "thread_pool_active": not self.thread_pool._shutdown,
                "http_session_active": self.session is not None and not self.session.closed,
                "total_tasks": len(self.tasks),
                "running_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]
                ),
                "failed_tasks": len(
                    [t for t in self.tasks.values() if t.status == TaskStatus.FAILED]
                ),
                "circuit_breakers": len(self.circuit_breakers),
                "open_circuit_breakers": len(
                    [cb for cb in self.circuit_breakers.values() if cb["state"] == "open"]
                ),
                "performance_metrics": self.performance_metrics,
                "timestamp": datetime.now().isoformat(),
            }

            return health

    async def shutdown(self):
        """Graceful shutdown of the coordinator"""
        self.logger.info("Starting graceful shutdown...")

        # Cancel all pending tasks
        for task_id in list(self.tasks.keys()):
            self.cancel_task(task_id)

        # Wait for running tasks to complete (with timeout)
        try:
            await asyncio.wait_for(self._wait_for_tasks_completion(), timeout=30.0)
        except asyncio.TimeoutError:
            self.logger.warning("Shutdown timeout, forcing termination")

        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Stop event loop
        if self.loop and not self.loop.is_closed():
            self.loop.stop()

        self.logger.info("Async coordinator shutdown completed")

    async def _wait_for_tasks_completion(self):
        """Wait for all running tasks to complete"""
        while any(task.status == TaskStatus.RUNNING for task in self.tasks.values()):
            await asyncio.sleep(0.1)


# Singleton coordinator
_async_coordinator = None
_coordinator_lock = threading.Lock()


def get_async_coordinator(config: Optional[CoordinatorConfig] = None) -> AsyncCoordinator:
    """Get the singleton async coordinator instance"""
    global _async_coordinator

    with _coordinator_lock:
        if _async_coordinator is None:
            _async_coordinator = AsyncCoordinator(config)
        return _async_coordinator


async def submit_async_task(
    task_name: str, coroutine: Coroutine, priority: int = 5, timeout: Optional[float] = None
) -> str:
    """Convenient function to submit async task"""
    coordinator = get_async_coordinator()
    return coordinator.submit_task(
        task_name=task_name, coroutine=coroutine, priority=priority, timeout=timeout
    )


def submit_sync_task(
    task_name: str,
    function: Callable,
    args: tuple = (),
    kwargs: dict = None,
    priority: int = 5,
    timeout: Optional[float] = None,
) -> str:
    """Convenient function to submit sync task"""
    coordinator = get_async_coordinator()
    return coordinator.submit_task(
        task_name=task_name,
        function=function,
        args=args,
        kwargs=kwargs or {},
        priority=priority,
        timeout=timeout,
    )
