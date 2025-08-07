# utils/orchestrator.py
import asyncio
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json


logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Task definition for the orchestrator"""
    id: str
    name: str
    agent_type: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    max_retries: int = 3
    retry_delay: int = 60
    timeout: int = 300
    depends_on: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0


class SystemOrchestrator:
    """Enterprise-grade system orchestrator with workflow management"""
    
    def __init__(self, config_manager, health_monitor):
        self.config_manager = config_manager
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(__name__)
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
        # Worker management
        self.max_workers = self.config_manager.get("parallel_workers", 4)
        self.workers: List[asyncio.Task] = []
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        
        # Orchestrator state
        self.running = False
        self.paused = False
        self.shutdown_event = asyncio.Event()
        
        # Task dependencies
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Metrics
        self.total_tasks_executed = 0
        self.total_tasks_failed = 0
        self.average_execution_time = 0.0
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        
    async def start(self):
        """Start the orchestrator"""
        if self.running:
            self.logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker coroutines
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
            self.worker_stats[i] = {
                "tasks_processed": 0,
                "last_activity": datetime.now(),
                "current_task": None
            }
        
        # Start monitoring task
        monitoring_task = asyncio.create_task(self._monitor_system())
        self.workers.append(monitoring_task)
        
        self.logger.info(f"System orchestrator started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the orchestrator gracefully"""
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all running tasks
        for task_id, task in self.running_tasks.items():
            task.cancel()
            self.logger.info(f"Cancelled running task: {task_id}")
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.logger.info("System orchestrator stopped")
    
    def register_agent_task(self, task: Task) -> str:
        """Register a new agent task"""
        if task.id in self.tasks:
            raise ValueError(f"Task {task.id} already registered")
        
        self.tasks[task.id] = task
        
        # Build dependency graph
        if task.depends_on:
            self.dependency_graph[task.id] = task.depends_on
        
        self.logger.info(f"Registered task {task.id} for agent {task.agent_type}")
        return task.id
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        task_id = self.register_agent_task(task)
        
        # Check dependencies before queuing
        if self._dependencies_satisfied(task_id):
            await self.task_queue.put(task_id)
            self.logger.debug(f"Task {task_id} queued for execution")
        else:
            self.logger.debug(f"Task {task_id} waiting for dependencies: {task.depends_on}")
        
        return task_id
    
    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        task = self.tasks[task_id]
        
        for dep_id in task.depends_on:
            if dep_id not in self.completed_tasks:
                return False
        
        return True
    
    async def _worker(self, worker_id: int):
        """Worker coroutine to process tasks"""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Wait for task or shutdown
                try:
                    task_id = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                if self.paused:
                    # Re-queue the task if paused
                    await self.task_queue.put(task_id)
                    await asyncio.sleep(1)
                    continue
                
                # Process the task
                await self._execute_task(worker_id, task_id)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def _execute_task(self, worker_id: int, task_id: str):
        """Execute a specific task"""
        task = self.tasks[task_id]
        
        # Update worker stats
        self.worker_stats[worker_id]["current_task"] = task_id
        self.worker_stats[worker_id]["last_activity"] = datetime.now()
        
        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # Add to running tasks
        execution_task = asyncio.create_task(self._run_task_with_timeout(task))
        self.running_tasks[task_id] = execution_task
        
        try:
            # Execute the task
            result = await execution_task
            
            # Task completed successfully
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            self.completed_tasks.append(task_id)
            self.total_tasks_executed += 1
            
            # Update average execution time
            execution_time = (task.completed_at - task.started_at).total_seconds()
            self._update_average_execution_time(execution_time)
            
            # Check for dependent tasks
            await self._check_dependent_tasks(task_id)
            
            self.logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self.logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            # Task failed
            task.error = str(e)
            
            if task.retry_count < task.max_retries:
                # Schedule retry
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Schedule retry after delay
                await asyncio.sleep(task.retry_delay)
                await self.task_queue.put(task_id)
                
                self.logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries}): {e}")
            else:
                # Max retries reached
                task.status = TaskStatus.FAILED
                self.failed_tasks.append(task_id)
                self.total_tasks_failed += 1
                
                # Try recovery strategy
                await self._try_recovery(task)
                
                self.logger.error(f"Task {task_id} failed permanently: {e}")
        
        finally:
            # Cleanup
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]
            
            self.worker_stats[worker_id]["current_task"] = None
            self.worker_stats[worker_id]["tasks_processed"] += 1
    
    async def _run_task_with_timeout(self, task: Task) -> Any:
        """Run task with timeout protection"""
        try:
            return await asyncio.wait_for(
                task.function(*task.args, **task.kwargs),
                timeout=task.timeout
            )
        except asyncio.TimeoutError:
            raise Exception(f"Task {task.id} timed out after {task.timeout} seconds")
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check if any waiting tasks can now be executed"""
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and 
                completed_task_id in task.depends_on and
                self._dependencies_satisfied(task_id)):
                
                await self.task_queue.put(task_id)
                self.logger.debug(f"Task {task_id} dependencies satisfied, queued for execution")
    
    async def _try_recovery(self, task: Task):
        """Try to recover from task failure using registered strategies"""
        if task.agent_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[task.agent_type]
                await recovery_func(task)
                self.logger.info(f"Recovery strategy applied for {task.agent_type}")
            except Exception as e:
                self.logger.error(f"Recovery strategy failed for {task.agent_type}: {e}")
    
    def register_recovery_strategy(self, agent_type: str, recovery_func: Callable):
        """Register a recovery strategy for an agent type"""
        self.recovery_strategies[agent_type] = recovery_func
        self.logger.info(f"Recovery strategy registered for {agent_type}")
    
    async def _monitor_system(self):
        """Monitor system health and task execution"""
        while self.running:
            try:
                # Check system health
                health_status = self.health_monitor.get_system_health()
                
                # Pause execution if system health is critical
                if health_status.get('grade') in ['E', 'F']:
                    if not self.paused:
                        self.paused = True
                        self.logger.warning("System paused due to critical health status")
                elif self.paused:
                    self.paused = False
                    self.logger.info("System resumed - health status improved")
                
                # Check for stuck tasks
                current_time = datetime.now()
                for task_id, task in self.running_tasks.items():
                    if task.id in self.tasks:
                        task_obj = self.tasks[task.id]
                        if (current_time - task_obj.started_at).total_seconds() > task_obj.timeout * 1.5:
                            task.cancel()
                            self.logger.warning(f"Cancelled stuck task: {task_id}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _update_average_execution_time(self, execution_time: float):
        """Update average execution time metric"""
        if self.total_tasks_executed == 1:
            self.average_execution_time = execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_execution_time = (
                alpha * execution_time + 
                (1 - alpha) * self.average_execution_time
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "running": self.running,
            "paused": self.paused,
            "total_tasks": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "total_executed": self.total_tasks_executed,
            "total_failed": self.total_tasks_failed,
            "success_rate": (self.total_tasks_executed - self.total_tasks_failed) / max(1, self.total_tasks_executed) * 100,
            "average_execution_time": self.average_execution_time,
            "worker_count": len(self.workers),
            "worker_stats": self.worker_stats,
            "queue_size": self.task_queue.qsize()
        }
    
    def pause(self):
        """Pause task execution"""
        self.paused = True
        self.logger.info("Orchestrator paused")
    
    def resume(self):
        """Resume task execution"""
        self.paused = False
        self.logger.info("Orchestrator resumed")