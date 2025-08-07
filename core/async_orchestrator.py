"""
Async Orchestrator - True Asynchronous Processing
Eliminates blocking operations for maximum performance
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Callable, Optional, Awaitable
from datetime import datetime, timedelta
import json
import concurrent.futures
from dataclasses import dataclass
import time
import threading
from collections import deque
import signal
import sys

@dataclass
class AsyncTask:
    """Represents an asynchronous task"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 1
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class AsyncOrchestrator:
    """
    High-performance async orchestrator for truly non-blocking operations
    Handles ML, scraping, whale detection, and all background tasks
    """
    
    def __init__(self, container):
        self.container = container
        self.logger = logging.getLogger(__name__)
        
        # Async configuration
        self.config = {
            'max_workers': 50,
            'max_concurrent_tasks': 100,
            'default_timeout': 300,
            'batch_size': 20,
            'retry_delay': 1.0,
            'health_check_interval': 30
        }
        
        # Task management
        self.task_queue = asyncio.Queue(maxsize=1000)
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        self.failed_tasks = deque(maxlen=500)
        
        # Performance monitoring
        self.performance_stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_execution_time': 0.0,
            'max_execution_time': 0.0,
            'current_load': 0
        }
        
        # Async components
        self.event_loop = None
        self.session = None
        self.executor = None
        
        # Graceful shutdown
        self.shutdown_event = asyncio.Event()
        self.running = False
        
        self.logger.critical("ASYNC ORCHESTRATOR INITIALIZED - Zero blocking operations mode")
    
    async def initialize(self):
        """Initialize async components"""
        
        try:
            # Create event loop if needed
            try:
                self.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
            
            # Initialize HTTP session with optimized settings
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config['default_timeout'],
                connect=10,
                sock_read=30
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'CryptoSmartTrader/2.0'}
            )
            
            # Initialize thread pool executor for CPU-intensive tasks
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config['max_workers'],
                thread_name_prefix='AsyncOrchestrator'
            )
            
            # Start background workers
            self.running = True
            
            # Start task processor
            asyncio.create_task(self._task_processor())
            
            # Start performance monitor
            asyncio.create_task(self._performance_monitor())
            
            # Start health checker
            asyncio.create_task(self._health_checker())
            
            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.critical("ASYNC ORCHESTRATOR FULLY INITIALIZED - All systems async")
            
        except Exception as e:
            self.logger.error(f"Async orchestrator initialization failed: {e}")
            raise
    
    async def submit_task(self, 
                         task_function: Callable,
                         *args,
                         task_id: str = None,
                         priority: int = 1,
                         timeout: float = None,
                         max_retries: int = 3,
                         **kwargs) -> str:
        """Submit task for async execution"""
        
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        if timeout is None:
            timeout = self.config['default_timeout']
        
        task = AsyncTask(
            task_id=task_id,
            function=task_function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        try:
            await self.task_queue.put(task)
            self.performance_stats['tasks_submitted'] += 1
            
            self.logger.debug(f"Task submitted: {task_id}")
            return task_id
            
        except asyncio.QueueFull:
            self.logger.error(f"Task queue full, rejecting task: {task_id}")
            raise RuntimeError("Task queue is full")
    
    async def submit_batch_tasks(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """Submit multiple tasks as a batch"""
        
        task_ids = []
        
        for task_config in tasks:
            task_id = await self.submit_task(
                task_function=task_config['function'],
                *task_config.get('args', []),
                task_id=task_config.get('task_id'),
                priority=task_config.get('priority', 1),
                timeout=task_config.get('timeout'),
                max_retries=task_config.get('max_retries', 3),
                **task_config.get('kwargs', {})
            )
            task_ids.append(task_id)
        
        self.logger.info(f"Batch submitted: {len(task_ids)} tasks")
        return task_ids
    
    async def _task_processor(self):
        """Main task processing loop"""
        
        while self.running:
            try:
                # Process tasks with proper concurrency control
                current_load = len(self.active_tasks)
                
                if current_load < self.config['max_concurrent_tasks']:
                    try:
                        # Get task with timeout to avoid blocking
                        task = await asyncio.wait_for(
                            self.task_queue.get(),
                            timeout=1.0
                        )
                        
                        # Execute task asynchronously
                        asyncio.create_task(self._execute_task(task))
                        
                    except asyncio.TimeoutError:
                        # No tasks available, continue
                        pass
                else:
                    # Too many active tasks, wait briefly
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Task processor error: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_task(self, task: AsyncTask):
        """Execute individual task asynchronously"""
        
        start_time = time.time()
        self.active_tasks[task.task_id] = task
        self.performance_stats['current_load'] = len(self.active_tasks)
        
        try:
            # Determine execution method based on function type
            if asyncio.iscoroutinefunction(task.function):
                # Async function - execute directly
                result = await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.timeout
                )
            else:
                # Sync function - run in executor to avoid blocking
                result = await asyncio.wait_for(
                    self.event_loop.run_in_executor(
                        self.executor,
                        lambda: task.function(*task.args, **task.kwargs)
                    ),
                    timeout=task.timeout
                )
            
            # Task completed successfully
            execution_time = time.time() - start_time
            
            completed_task = {
                'task_id': task.task_id,
                'result': result,
                'execution_time': execution_time,
                'completed_at': datetime.now(),
                'retry_count': task.retry_count
            }
            
            self.completed_tasks.append(completed_task)
            self.performance_stats['tasks_completed'] += 1
            
            # Update performance stats
            self._update_performance_stats(execution_time)
            
            self.logger.debug(f"Task completed: {task.task_id} ({execution_time:.2f}s)")
            
        except asyncio.TimeoutError:
            await self._handle_task_timeout(task)
            
        except Exception as e:
            await self._handle_task_error(task, e)
            
        finally:
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.performance_stats['current_load'] = len(self.active_tasks)
    
    async def _handle_task_timeout(self, task: AsyncTask):
        """Handle task timeout"""
        
        self.logger.warning(f"Task timeout: {task.task_id} (timeout: {task.timeout}s)")
        
        if task.retry_count < task.max_retries:
            # Retry task with longer timeout
            task.retry_count += 1
            task.timeout *= 1.5  # Increase timeout for retry
            
            await self.task_queue.put(task)
            self.logger.info(f"Task retry scheduled: {task.task_id} (attempt {task.retry_count})")
        else:
            # Max retries exceeded
            failed_task = {
                'task_id': task.task_id,
                'error': f'Timeout after {task.timeout}s (max retries exceeded)',
                'failed_at': datetime.now(),
                'retry_count': task.retry_count
            }
            
            self.failed_tasks.append(failed_task)
            self.performance_stats['tasks_failed'] += 1
    
    async def _handle_task_error(self, task: AsyncTask, error: Exception):
        """Handle task execution error"""
        
        self.logger.warning(f"Task error: {task.task_id} - {error}")
        
        if task.retry_count < task.max_retries:
            # Retry task with exponential backoff
            task.retry_count += 1
            
            # Wait before retry
            retry_delay = self.config['retry_delay'] * (2 ** task.retry_count)
            await asyncio.sleep(retry_delay)
            
            await self.task_queue.put(task)
            self.logger.info(f"Task retry scheduled: {task.task_id} (attempt {task.retry_count})")
        else:
            # Max retries exceeded
            failed_task = {
                'task_id': task.task_id,
                'error': str(error),
                'failed_at': datetime.now(),
                'retry_count': task.retry_count
            }
            
            self.failed_tasks.append(failed_task)
            self.performance_stats['tasks_failed'] += 1
    
    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics"""
        
        # Update average execution time (exponential moving average)
        alpha = 0.1
        current_avg = self.performance_stats['average_execution_time']
        self.performance_stats['average_execution_time'] = (
            alpha * execution_time + (1 - alpha) * current_avg
        )
        
        # Update max execution time
        if execution_time > self.performance_stats['max_execution_time']:
            self.performance_stats['max_execution_time'] = execution_time
    
    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        
        while self.running:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                stats = self.get_performance_stats()
                
                self.logger.info(
                    f"ASYNC PERFORMANCE: "
                    f"Load: {stats['current_load']}/{self.config['max_concurrent_tasks']}, "
                    f"Completed: {stats['tasks_completed']}, "
                    f"Failed: {stats['tasks_failed']}, "
                    f"Avg Time: {stats['average_execution_time']:.2f}s"
                )
                
                # Check for performance issues
                if stats['current_load'] > self.config['max_concurrent_tasks'] * 0.9:
                    self.logger.warning("HIGH ASYNC LOAD - Consider increasing max_concurrent_tasks")
                
                if stats['average_execution_time'] > 30.0:
                    self.logger.warning("HIGH EXECUTION TIME - Tasks taking too long")
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {e}")
    
    async def _health_checker(self):
        """Check system health regularly"""
        
        while self.running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])
                
                # Check queue size
                queue_size = self.task_queue.qsize()
                if queue_size > 500:
                    self.logger.warning(f"HIGH QUEUE SIZE: {queue_size} tasks pending")
                
                # Check for stuck tasks
                current_time = datetime.now()
                stuck_tasks = []
                
                for task_id, task in self.active_tasks.items():
                    if (current_time - task.created_at).total_seconds() > task.timeout * 2:
                        stuck_tasks.append(task_id)
                
                if stuck_tasks:
                    self.logger.error(f"STUCK TASKS DETECTED: {stuck_tasks}")
                
                # Check HTTP session health
                if self.session and self.session.closed:
                    self.logger.error("HTTP SESSION CLOSED - Reinitializing")
                    await self._reinitialize_session()
                
            except Exception as e:
                self.logger.error(f"Health checker error: {e}")
    
    async def _reinitialize_session(self):
        """Reinitialize HTTP session if needed"""
        
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config['default_timeout'],
                connect=10,
                sock_read=30
            )
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'CryptoSmartTrader/2.0'}
            )
            
            self.logger.info("HTTP session reinitialized")
            
        except Exception as e:
            self.logger.error(f"Session reinitialization failed: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        
        if self.event_loop and not self.event_loop.is_closed():
            self.event_loop.create_task(self.shutdown())
    
    async def shutdown(self):
        """Graceful shutdown of async orchestrator"""
        
        self.logger.critical("INITIATING ASYNC ORCHESTRATOR SHUTDOWN")
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete")
            
            shutdown_timeout = 30.0
            start_time = time.time()
            
            while self.active_tasks and (time.time() - start_time) < shutdown_timeout:
                await asyncio.sleep(0.5)
            
            if self.active_tasks:
                self.logger.warning(f"Forcing shutdown with {len(self.active_tasks)} active tasks")
        
        # Close HTTP session
        if self.session and not self.session.closed:
            await self.session.close()
            self.logger.info("HTTP session closed")
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, timeout=10.0)
            self.logger.info("Thread pool executor shutdown")
        
        self.logger.critical("ASYNC ORCHESTRATOR SHUTDOWN COMPLETE")
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of specific task"""
        
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'status': 'running',
                'task_id': task_id,
                'created_at': task.created_at.isoformat(),
                'retry_count': task.retry_count,
                'timeout': task.timeout
            }
        
        # Check completed tasks
        for completed in self.completed_tasks:
            if completed['task_id'] == task_id:
                return {
                    'status': 'completed',
                    'task_id': task_id,
                    'completed_at': completed['completed_at'].isoformat(),
                    'execution_time': completed['execution_time'],
                    'retry_count': completed['retry_count']
                }
        
        # Check failed tasks
        for failed in self.failed_tasks:
            if failed['task_id'] == task_id:
                return {
                    'status': 'failed',
                    'task_id': task_id,
                    'failed_at': failed['failed_at'].isoformat(),
                    'error': failed['error'],
                    'retry_count': failed['retry_count']
                }
        
        return {'status': 'not_found', 'task_id': task_id}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        
        total_tasks = (
            self.performance_stats['tasks_completed'] + 
            self.performance_stats['tasks_failed']
        )
        
        success_rate = 0.0
        if total_tasks > 0:
            success_rate = self.performance_stats['tasks_completed'] / total_tasks * 100
        
        return {
            **self.performance_stats,
            'queue_size': self.task_queue.qsize(),
            'active_tasks_count': len(self.active_tasks),
            'success_rate_percent': round(success_rate, 2),
            'total_tasks_processed': total_tasks,
            'async_mode': 'FULLY_ASYNC',
            'last_updated': datetime.now().isoformat()
        }
    
    async def execute_ml_batch_async(self, ml_function: Callable, data_batches: List[Any]) -> List[Any]:
        """Execute ML batch processing asynchronously"""
        
        batch_tasks = []
        
        for i, batch in enumerate(data_batches):
            task_config = {
                'function': ml_function,
                'args': (batch,),
                'task_id': f'ml_batch_{i}',
                'priority': 2,  # High priority for ML
                'timeout': 600.0  # 10 minutes for ML tasks
            }
            batch_tasks.append(task_config)
        
        task_ids = await self.submit_batch_tasks(batch_tasks)
        
        # Wait for all batches to complete
        results = []
        
        for task_id in task_ids:
            while True:
                status = await self.get_task_status(task_id)
                
                if status['status'] == 'completed':
                    # Get result from completed tasks
                    for completed in self.completed_tasks:
                        if completed['task_id'] == task_id:
                            results.append(completed['result'])
                            break
                    break
                elif status['status'] == 'failed':
                    self.logger.error(f"ML batch task failed: {task_id}")
                    results.append(None)
                    break
                else:
                    # Still running, wait
                    await asyncio.sleep(1.0)
        
        self.logger.info(f"ML batch processing completed: {len(results)} results")
        return results
    
    async def execute_scraping_async(self, urls: List[str], scraping_function: Callable) -> List[Any]:
        """Execute web scraping asynchronously"""
        
        scraping_tasks = []
        
        for i, url in enumerate(urls):
            task_config = {
                'function': scraping_function,
                'args': (url,),
                'kwargs': {'session': self.session},
                'task_id': f'scraping_{i}',
                'priority': 1,
                'timeout': 60.0  # 1 minute for scraping
            }
            scraping_tasks.append(task_config)
        
        task_ids = await self.submit_batch_tasks(scraping_tasks)
        
        self.logger.info(f"Scraping batch submitted: {len(task_ids)} URLs")
        return task_ids

# Global async orchestrator instance
async_orchestrator = None

async def get_async_orchestrator(container) -> AsyncOrchestrator:
    """Get or create async orchestrator instance"""
    
    global async_orchestrator
    
    if async_orchestrator is None:
        async_orchestrator = AsyncOrchestrator(container)
        await async_orchestrator.initialize()
    
    return async_orchestrator