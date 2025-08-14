#!/usr/bin/env python3
"""
Centrale Data Scheduler
Coördineert alle data ingestion met prioriteiten, scheduling en load balancing
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import croniter
import threading

from .enterprise_data_ingestion import (
    EnterpriseDataIngestion, 
    DataRequest, 
    DataPriority, 
    DataSourceStatus,
    create_market_data_request,
    create_orderbook_request,
    create_ohlcv_request
)


class ScheduleFrequency(Enum):
    """Data collection frequencies"""
    REALTIME = "*/5 * * * * *"      # Every 5 seconds
    HIGH_FREQ = "*/30 * * * * *"    # Every 30 seconds  
    MEDIUM_FREQ = "*/5 * * * *"     # Every 5 minutes
    LOW_FREQ = "0 */15 * * * *"     # Every 15 minutes
    HOURLY = "0 0 * * * *"          # Every hour
    DAILY = "0 0 0 * * *"           # Daily at midnight


@dataclass
class ScheduledTask:
    """Scheduled data collection task"""
    task_id: str
    name: str
    frequency: ScheduleFrequency
    request_generator: Callable[[], List[DataRequest]]
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    avg_duration_ms: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    max_failures: int = 5
    timeout_seconds: float = 300.0


class DataScheduler:
    """Centrale scheduler voor alle data collection"""
    
    def __init__(self, data_ingestion: EnterpriseDataIngestion, config: Dict[str, Any]):
        self.data_ingestion = data_ingestion
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Task management
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.task_execution_history = deque(maxlen=1000)
        self.running_tasks: Set[str] = set()
        
        # Scheduling state
        self.scheduler_active = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        # Load balancing
        self.exchange_loads = defaultdict(int)
        self.priority_queues = {
            DataPriority.CRITICAL: deque(),
            DataPriority.HIGH: deque(),
            DataPriority.MEDIUM: deque(),
            DataPriority.LOW: deque()
        }
        
        # Metrics
        self.metrics = {
            'total_scheduled': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'start_time': datetime.now()
        }
        
        # Initialize default tasks
        self._init_default_tasks()
    
    def _init_default_tasks(self):
        """Initialize default scheduled tasks"""
        
        # Critical real-time data
        self.add_task(ScheduledTask(
            task_id="critical_tickers",
            name="Critical Market Tickers",
            frequency=ScheduleFrequency.REALTIME,
            request_generator=self._generate_critical_ticker_requests
        ))
        
        self.add_task(ScheduledTask(
            task_id="critical_orderbooks", 
            name="Critical Order Books",
            frequency=ScheduleFrequency.REALTIME,
            request_generator=self._generate_critical_orderbook_requests
        ))
        
        # High frequency price feeds
        self.add_task(ScheduledTask(
            task_id="price_feeds",
            name="Price Feeds Collection",
            frequency=ScheduleFrequency.HIGH_FREQ,
            request_generator=self._generate_price_feed_requests
        ))
        
        # Medium frequency OHLCV data
        self.add_task(ScheduledTask(
            task_id="ohlcv_data",
            name="OHLCV Data Collection",
            frequency=ScheduleFrequency.MEDIUM_FREQ,
            request_generator=self._generate_ohlcv_requests
        ))
        
        # Low frequency market analytics
        self.add_task(ScheduledTask(
            task_id="market_analytics",
            name="Market Analytics Data",
            frequency=ScheduleFrequency.LOW_FREQ,
            request_generator=self._generate_analytics_requests
        ))
        
        # Daily comprehensive updates
        self.add_task(ScheduledTask(
            task_id="daily_comprehensive",
            name="Daily Comprehensive Update",
            frequency=ScheduleFrequency.DAILY,
            request_generator=self._generate_daily_requests,
            timeout_seconds=1800.0  # 30 minutes for daily updates
        ))
    
    def add_task(self, task: ScheduledTask):
        """Add scheduled task"""
        self.scheduled_tasks[task.task_id] = task
        self._calculate_next_run(task)
        self.logger.info(f"Added scheduled task: {task.name}")
    
    def remove_task(self, task_id: str):
        """Remove scheduled task"""
        if task_id in self.scheduled_tasks:
            del self.scheduled_tasks[task_id]
            self.logger.info(f"Removed scheduled task: {task_id}")
    
    def enable_task(self, task_id: str):
        """Enable scheduled task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = True
            self._calculate_next_run(self.scheduled_tasks[task_id])
    
    def disable_task(self, task_id: str):
        """Disable scheduled task"""
        if task_id in self.scheduled_tasks:
            self.scheduled_tasks[task_id].enabled = False
    
    async def start_scheduler(self):
        """Start the data scheduler"""
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        self.logger.info("Started data scheduler")
    
    async def stop_scheduler(self):
        """Stop the data scheduler"""
        if not self.scheduler_active:
            return
        
        self.scheduler_active = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped data scheduler")
    
    async def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.scheduler_active:
            try:
                current_time = datetime.now()
                
                # Find tasks ready to run
                ready_tasks = [
                    task for task in self.scheduled_tasks.values()
                    if (task.enabled and 
                        task.next_run and 
                        task.next_run <= current_time and
                        task.task_id not in self.running_tasks)
                ]
                
                # Sort by priority (critical tasks first)
                ready_tasks.sort(key=lambda t: (
                    t.failure_count,  # Failed tasks get lower priority
                    t.next_run        # Earlier scheduled tasks first
                ))
                
                # Execute ready tasks
                for task in ready_tasks:
                    if self._can_execute_task(task):
                        asyncio.create_task(self._execute_task(task))
                
                # Wait before next iteration
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5.0)
    
    def _can_execute_task(self, task: ScheduledTask) -> bool:
        """Check if task can be executed"""
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.running_tasks:
                return False
        
        # Check failure threshold
        if task.failure_count >= task.max_failures:
            return False
        
        # Check exchange load balancing
        if hasattr(task, 'primary_exchange'):
            if self.exchange_loads[task.primary_exchange] > 10:
                return False
        
        return True
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute scheduled task"""
        start_time = datetime.now()
        self.running_tasks.add(task.task_id)
        
        try:
            self.logger.info(f"Executing task: {task.name}")
            
            # Generate requests
            requests = task.request_generator()
            if not requests:
                self.logger.warning(f"No requests generated for task: {task.name}")
                return
            
            # Execute requests with timeout
            responses = []
            async with asyncio.timeout(task.timeout_seconds):
                for request in requests:
                    response = await self.data_ingestion.request_data(request)
                    responses.append(response)
            
            # Analyze results
            successful_responses = [r for r in responses if r.status == "success"]
            success_rate = len(successful_responses) / len(responses) if responses else 0
            
            # Update task metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            if success_rate >= 0.8:  # 80% success threshold
                task.success_count += 1
                task.failure_count = max(0, task.failure_count - 1)  # Reduce failure count on success
                self.metrics['successful_executions'] += 1
            else:
                task.failure_count += 1
                self.metrics['failed_executions'] += 1
                self.logger.warning(f"Task {task.name} failed with {success_rate:.1%} success rate")
            
            # Update average duration
            task.avg_duration_ms = (
                (task.avg_duration_ms * task.success_count + execution_time) / 
                (task.success_count + 1)
            )
            
            # Record execution
            self.task_execution_history.append({
                'task_id': task.task_id,
                'timestamp': start_time,
                'duration_ms': execution_time,
                'success_rate': success_rate,
                'requests_count': len(requests)
            })
            
            self.logger.info(f"Task {task.name} completed: {success_rate:.1%} success, {execution_time:.0f}ms")
            
        except asyncio.TimeoutError:
            task.failure_count += 1
            self.metrics['failed_executions'] += 1
            self.logger.error(f"Task {task.name} timed out after {task.timeout_seconds}s")
            
        except Exception as e:
            task.failure_count += 1
            self.metrics['failed_executions'] += 1
            self.logger.error(f"Task {task.name} failed: {e}")
            
        finally:
            # Update timing
            task.last_run = start_time
            self._calculate_next_run(task)
            
            # Remove from running tasks
            self.running_tasks.discard(task.task_id)
            
            # Update metrics
            self.metrics['total_scheduled'] += 1
    
    def _calculate_next_run(self, task: ScheduledTask):
        """Calculate next run time for task"""
        if not task.enabled:
            task.next_run = None
            return
        
        cron = croniter.croniter(task.frequency.value, datetime.now())
        task.next_run = cron.get_next(datetime)
    
    def _generate_critical_ticker_requests(self) -> List[DataRequest]:
        """Generate critical ticker requests"""
        critical_symbols = self.config.get('critical_symbols', [
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD'
        ])
        
        requests = []
        for symbol in critical_symbols:
            for exchange in ['kraken', 'binance']:
                if exchange in self.data_ingestion.async_exchanges:
                    request = create_market_data_request(
                        exchange=exchange,
                        symbol=symbol,
                        priority=DataPriority.CRITICAL,
                        cache_ttl=5
                    )
                    requests.append(request)
        
        return requests
    
    def _generate_critical_orderbook_requests(self) -> List[DataRequest]:
        """Generate critical orderbook requests"""
        critical_symbols = self.config.get('critical_symbols', [
            'BTC/USD', 'ETH/USD'
        ])
        
        requests = []
        for symbol in critical_symbols:
            for exchange in ['kraken', 'binance']:
                if exchange in self.data_ingestion.async_exchanges:
                    request = create_orderbook_request(
                        exchange=exchange,
                        symbol=symbol,
                        limit=50,
                        priority=DataPriority.CRITICAL
                    )
                    requests.append(request)
        
        return requests
    
    def _generate_price_feed_requests(self) -> List[DataRequest]:
        """Generate price feed requests"""
        tracked_symbols = self.config.get('tracked_symbols', [
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD',
            'AVAX/USD', 'DOT/USD', 'MATIC/USD', 'LINK/USD', 'UNI/USD'
        ])
        
        requests = []
        for symbol in tracked_symbols:
            for exchange in self.data_ingestion.async_exchanges.keys():
                request = create_market_data_request(
                    exchange=exchange,
                    symbol=symbol,
                    priority=DataPriority.HIGH,
                    cache_ttl=30
                )
                requests.append(request)
        
        return requests
    
    def _generate_ohlcv_requests(self) -> List[DataRequest]:
        """Generate OHLCV requests"""
        symbols = self.config.get('tracked_symbols', [
            'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD'
        ])
        
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        requests = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                for exchange in self.data_ingestion.async_exchanges.keys():
                    # Adjust cache TTL based on timeframe
                    cache_ttl = {
                        '1m': 60, '5m': 300, '15m': 900,
                        '1h': 3600, '4h': 14400, '1d': 86400
                    }.get(timeframe, 300)
                    
                    request = create_ohlcv_request(
                        exchange=exchange,
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=100,
                        priority=DataPriority.MEDIUM,
                        cache_ttl=cache_ttl
                    )
                    requests.append(request)
        
        return requests
    
    def _generate_analytics_requests(self) -> List[DataRequest]:
        """Generate analytics data requests"""
        # Lower priority analytics data
        symbols = self.config.get('analytics_symbols', [
            'BTC/USD', 'ETH/USD', 'BNB/USD'
        ])
        
        requests = []
        for symbol in symbols:
            for exchange in self.data_ingestion.async_exchanges.keys():
                # Historical trades for analytics
                request = DataRequest(
                    source=exchange,
                    endpoint="fetch_trades",
                    params={'symbol': symbol, 'limit': 100},
                    priority=DataPriority.LOW,
                    cache_ttl=900  # 15 minutes
                )
                requests.append(request)
        
        return requests
    
    def _generate_daily_requests(self) -> List[DataRequest]:
        """Generate daily comprehensive requests"""
        # Daily full market scan
        requests = []
        
        for exchange in self.data_ingestion.async_exchanges.keys():
            # All tickers for daily analysis
            request = DataRequest(
                source=exchange,
                endpoint="fetch_tickers",
                params={'symbols': None},  # All symbols
                priority=DataPriority.LOW,
                cache_ttl=3600,  # 1 hour cache
                timeout=120.0
            )
            requests.append(request)
        
        return requests
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status"""
        uptime = datetime.now() - self.metrics['start_time']
        
        return {
            'active': self.scheduler_active,
            'uptime_seconds': uptime.total_seconds(),
            'metrics': self.metrics.copy(),
            'tasks': {
                task_id: {
                    'name': task.name,
                    'enabled': task.enabled,
                    'last_run': task.last_run.isoformat() if task.last_run else None,
                    'next_run': task.next_run.isoformat() if task.next_run else None,
                    'success_count': task.success_count,
                    'failure_count': task.failure_count,
                    'avg_duration_ms': task.avg_duration_ms,
                    'max_failures': task.max_failures
                }
                for task_id, task in self.scheduled_tasks.items()
            },
            'running_tasks': list(self.running_tasks),
            'queue_sizes': {
                priority.name: len(queue)
                for priority, queue in self.priority_queues.items()
            },
            'recent_executions': list(self.task_execution_history)[-10:]  # Last 10 executions
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        recent_executions = list(self.task_execution_history)[-100:]  # Last 100
        
        if not recent_executions:
            return {'status': 'no_data'}
        
        # Calculate performance metrics
        avg_duration = sum(e['duration_ms'] for e in recent_executions) / len(recent_executions)
        avg_success_rate = sum(e['success_rate'] for e in recent_executions) / len(recent_executions)
        
        # Task performance breakdown
        task_performance = defaultdict(list)
        for execution in recent_executions:
            task_performance[execution['task_id']].append(execution)
        
        task_stats = {}
        for task_id, executions in task_performance.items():
            task_stats[task_id] = {
                'avg_duration_ms': sum(e['duration_ms'] for e in executions) / len(executions),
                'avg_success_rate': sum(e['success_rate'] for e in executions) / len(executions),
                'execution_count': len(executions)
            }
        
        return {
            'avg_duration_ms': avg_duration,
            'avg_success_rate': avg_success_rate,
            'total_executions': len(recent_executions),
            'task_breakdown': task_stats,
            'system_health': 'healthy' if avg_success_rate > 0.8 else 'degraded'
        }


# Factory function
def create_data_scheduler(
    data_ingestion: EnterpriseDataIngestion, 
    config: Dict[str, Any]
) -> DataScheduler:
    """Create configured data scheduler"""
    return DataScheduler(data_ingestion, config)