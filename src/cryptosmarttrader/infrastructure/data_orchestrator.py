#!/usr/bin/env python3
"""
Data Orchestrator
Centrale orchestrator voor alle data ingestion met APScheduler, per-source rate limits en health monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from pathlib import Path

# APScheduler imports
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.jobstores.memory import MemoryJobStore
    from apscheduler.executors.asyncio import AsyncIOExecutor
except ImportError:
    # Fallback for development
    AsyncIOScheduler = None
    CronTrigger = None
    IntervalTrigger = None
    MemoryJobStore = None
    AsyncIOExecutor = None

from .hardened_http_client import HardenedHTTPClient, HTTPConfig, RequestMetrics

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


class DataSourceHealth(Enum):
    HEALTHY = "healthy"       # All metrics green
    DEGRADED = "degraded"     # Some issues but operational
    CRITICAL = "critical"     # Major issues, may fail
    OFFLINE = "offline"       # Completely unavailable


@dataclass
class DataJob:
    """Data collection job definition"""
    job_id: str
    name: str
    source: str
    endpoint: str
    schedule: str  # Cron expression or interval
    enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 3
    rate_limit_per_minute: int = 60
    priority: int = 1  # 1=highest, 5=lowest
    dependencies: Set[str] = field(default_factory=set)
    
    # Job execution state
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    avg_duration_ms: float = 0.0
    status: JobStatus = JobStatus.PENDING
    last_error: Optional[str] = None


@dataclass
class SourceHealth:
    """Health status per data source"""
    source: str
    status: DataSourceHealth = DataSourceHealth.HEALTHY
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    circuit_breaker_open: bool = False
    total_jobs: int = 0
    active_jobs: int = 0
    failed_jobs: int = 0


class DataOrchestrator:
    """Centrale orchestrator voor data ingestion"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.http_client: Optional[HardenedHTTPClient] = None
        self.scheduler: Optional[AsyncIOScheduler] = None
        
        # Job management
        self.jobs: Dict[str, DataJob] = {}
        self.job_execution_history: List[Dict[str, Any]] = []
        self.source_health: Dict[str, SourceHealth] = {}
        
        # State management
        self.running = False
        self.startup_time: Optional[datetime] = None
        
        # Metrics
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        
    async def start(self):
        """Start orchestrator"""
        if self.running:
            return
        
        self.logger.info("🚀 Starting Data Orchestrator...")
        
        # Initialize HTTP client
        http_config = HTTPConfig(
            base_timeout=10.0,
            max_timeout=30.0,
            max_retries=3,
            circuit_breaker_threshold=5,
            rate_limit_per_minute=60
        )
        self.http_client = HardenedHTTPClient(http_config)
        await self.http_client.start()
        
        # Initialize scheduler
        if AsyncIOScheduler is None:
            raise ImportError("APScheduler not available. Install with: pip install apscheduler")
        
        jobstores = {'default': MemoryJobStore()}
        executors = {'default': AsyncIOExecutor()}
        job_defaults = {
            'coalesce': True,
            'max_instances': 3,
            'misfire_grace_time': 30
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults
        )
        
        # Start scheduler
        self.scheduler.start()
        
        # Load predefined jobs
        await self.load_default_jobs()
        
        # Start health monitoring
        await self.start_health_monitoring()
        
        self.running = True
        self.startup_time = datetime.now()
        self.logger.info("✅ Data Orchestrator started successfully")
    
    async def stop(self):
        """Stop orchestrator"""
        if not self.running:
            return
        
        self.logger.info("🛑 Stopping Data Orchestrator...")
        
        # Stop scheduler
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.close()
        
        self.running = False
        self.logger.info("✅ Data Orchestrator stopped")
    
    async def load_default_jobs(self):
        """Load default data collection jobs"""
        
        # High-frequency market data
        market_jobs = [
            DataJob(
                job_id="kraken_ticker_btc",
                name="Kraken BTC Ticker",
                source="kraken",
                endpoint="https://api.kraken.com/0/public/Ticker?pair=BTCUSD",
                schedule="*/10 * * * * *",  # Every 10 seconds
                timeout_seconds=10.0,
                rate_limit_per_minute=120,
                priority=1
            ),
            DataJob(
                job_id="kraken_ticker_eth", 
                name="Kraken ETH Ticker",
                source="kraken",
                endpoint="https://api.kraken.com/0/public/Ticker?pair=ETHUSD",
                schedule="*/10 * * * * *",
                timeout_seconds=10.0,
                rate_limit_per_minute=120,
                priority=1
            ),
            DataJob(
                job_id="kraken_orderbook_btc",
                name="Kraken BTC Orderbook",
                source="kraken", 
                endpoint="https://api.kraken.com/0/public/Depth?pair=BTCUSD&count=100",
                schedule="*/30 * * * * *",  # Every 30 seconds
                timeout_seconds=15.0,
                rate_limit_per_minute=60,
                priority=2
            )
        ]
        
        # Medium-frequency OHLCV data
        ohlcv_jobs = [
            DataJob(
                job_id="kraken_ohlcv_btc_1m",
                name="Kraken BTC 1m OHLCV",
                source="kraken",
                endpoint="https://api.kraken.com/0/public/OHLC?pair=BTCUSD&interval=1",
                schedule="0 */1 * * * *",  # Every minute
                timeout_seconds=20.0,
                rate_limit_per_minute=30,
                priority=3
            ),
            DataJob(
                job_id="kraken_ohlcv_btc_5m",
                name="Kraken BTC 5m OHLCV", 
                source="kraken",
                endpoint="https://api.kraken.com/0/public/OHLC?pair=BTCUSD&interval=5",
                schedule="0 */5 * * * *",  # Every 5 minutes
                timeout_seconds=20.0,
                rate_limit_per_minute=20,
                priority=4
            )
        ]
        
        # Add all jobs
        all_jobs = market_jobs + ohlcv_jobs
        
        for job in all_jobs:
            await self.add_job(job)
        
        self.logger.info(f"📋 Loaded {len(all_jobs)} default data collection jobs")
    
    async def add_job(self, job: DataJob):
        """Add new data collection job"""
        self.jobs[job.job_id] = job
        
        # Initialize source health if not exists
        if job.source not in self.source_health:
            self.source_health[job.source] = SourceHealth(source=job.source)
        
        # Update source health job counts
        source_health = self.source_health[job.source]
        source_health.total_jobs += 1
        if job.enabled:
            source_health.active_jobs += 1
        
        # Schedule job if enabled
        if job.enabled and self.scheduler:
            await self.schedule_job(job)
        
        self.logger.info(f"➕ Added job: {job.name} ({job.job_id})")
    
    async def schedule_job(self, job: DataJob):
        """Schedule job with APScheduler"""
        try:
            # Parse schedule (support both cron and interval)
            if CronTrigger is None or IntervalTrigger is None:
                raise ImportError("APScheduler triggers not available")
                
            if job.schedule.count(' ') >= 5:  # Cron with seconds
                trigger = CronTrigger.from_crontab(job.schedule)
            elif job.schedule.startswith('*/'):  # Interval format
                seconds = int(job.schedule.split('*/')[1].split(' ')[0])
                trigger = IntervalTrigger(seconds=seconds)
            else:
                # Fallback to cron without seconds  
                trigger = CronTrigger.from_crontab(job.schedule)
            
            # Add job to scheduler
            self.scheduler.add_job(
                func=self.execute_job,
                trigger=trigger,
                args=[job.job_id],
                id=job.job_id,
                name=job.name,
                replace_existing=True,
                max_instances=1
            )
            
            job.status = JobStatus.PENDING
            self.logger.debug(f"📅 Scheduled job: {job.name}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to schedule job {job.name}: {e}")
            job.status = JobStatus.FAILED
            job.last_error = str(e)
    
    async def execute_job(self, job_id: str):
        """Execute data collection job"""
        job = self.jobs.get(job_id)
        if not job:
            self.logger.error(f"❌ Job not found: {job_id}")
            return
        
        start_time = datetime.now()
        job.status = JobStatus.RUNNING
        self.total_executions += 1
        
        try:
            self.logger.debug(f"🚀 Executing job: {job.name}")
            
            # Execute HTTP request
            response = await self.http_client.get(
                url=job.endpoint,
                source=job.source,
                timeout=job.timeout_seconds
            )
            
            # Calculate duration
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update job metrics
            job.last_run = start_time
            job.success_count += 1
            job.consecutive_failures = 0
            job.status = JobStatus.SUCCESS
            job.last_error = None
            
            # Update average duration
            if job.avg_duration_ms == 0:
                job.avg_duration_ms = duration_ms
            else:
                job.avg_duration_ms = (job.avg_duration_ms * 0.9) + (duration_ms * 0.1)
            
            # Update source health
            await self.update_source_health(job.source, success=True, latency_ms=duration_ms)
            
            # Record execution
            self.successful_executions += 1
            await self.record_execution(job_id, True, duration_ms, response)
            
            self.logger.debug(f"✅ Job completed: {job.name} ({duration_ms:.0f}ms)")
            
        except Exception as e:
            # Calculate duration even for failures
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Update job metrics
            job.last_run = start_time
            job.failure_count += 1
            job.consecutive_failures += 1
            job.status = JobStatus.FAILED
            job.last_error = str(e)
            
            # Update source health
            await self.update_source_health(job.source, success=False, error=str(e))
            
            # Record execution
            self.failed_executions += 1
            await self.record_execution(job_id, False, duration_ms, None, str(e))
            
            self.logger.error(f"❌ Job failed: {job.name} - {e}")
    
    async def update_source_health(self, source: str, success: bool, latency_ms: float = 0, error: str = None):
        """Update health status for data source"""
        health = self.source_health.get(source)
        if not health:
            return
        
        if success:
            health.last_success = datetime.now()
            
            # Update average latency
            if health.avg_latency_ms == 0:
                health.avg_latency_ms = latency_ms
            else:
                health.avg_latency_ms = (health.avg_latency_ms * 0.9) + (latency_ms * 0.1)
        else:
            health.last_failure = datetime.now()
            health.failed_jobs += 1
        
        # Calculate error rate (over last 100 requests)
        recent_executions = [e for e in self.job_execution_history[-100:] if e.get('source') == source]
        if recent_executions:
            failures = sum(1 for e in recent_executions if not e.get('success', False))
            health.error_rate = failures / len(recent_executions)
        
        # Determine health status
        if self.http_client:
            circuit_status = self.http_client.get_health_status().get(source, {})
            health.circuit_breaker_open = circuit_status.get('circuit_breaker_state') == 'open'
        
        # Set overall health status
        if health.circuit_breaker_open or health.error_rate > 0.5:
            health.status = DataSourceHealth.OFFLINE
        elif health.error_rate > 0.2 or health.avg_latency_ms > 5000:
            health.status = DataSourceHealth.CRITICAL
        elif health.error_rate > 0.1 or health.avg_latency_ms > 2000:
            health.status = DataSourceHealth.DEGRADED
        else:
            health.status = DataSourceHealth.HEALTHY
    
    async def record_execution(self, job_id: str, success: bool, duration_ms: float, response: Any = None, error: str = None):
        """Record job execution for history tracking"""
        job = self.jobs.get(job_id)
        
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'job_id': job_id,
            'job_name': job.name if job else 'Unknown',
            'source': job.source if job else 'Unknown',
            'success': success,
            'duration_ms': duration_ms,
            'error': error,
            'data_size': len(str(response)) if response else 0
        }
        
        self.job_execution_history.append(execution_record)
        
        # Keep only last 1000 executions
        if len(self.job_execution_history) > 1000:
            self.job_execution_history = self.job_execution_history[-1000:]
    
    async def start_health_monitoring(self):
        """Start periodic health monitoring"""
        
        async def health_check():
            """Periodic health check of all sources"""
            try:
                for source, health in self.source_health.items():
                    # Check if source has been failing too long
                    if health.last_failure and health.last_success:
                        time_since_success = datetime.now() - health.last_success
                        if time_since_success > timedelta(minutes=10):
                            self.logger.warning(f"⚠️  Source {source} has not succeeded in {time_since_success}")
                    
                    # Log health status changes
                    if health.status in [DataSourceHealth.CRITICAL, DataSourceHealth.OFFLINE]:
                        self.logger.error(f"🔴 Source {source} is {health.status.value}")
            except Exception as e:
                self.logger.error(f"❌ Health check failed: {e}")
        
        # Schedule health check every minute
        if self.scheduler:
            self.scheduler.add_job(
                func=health_check,
                trigger=IntervalTrigger(minutes=1),
                id="health_monitor",
                name="Health Monitor"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        uptime = (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0
        
        return {
            'orchestrator': {
                'running': self.running,
                'uptime_seconds': uptime,
                'total_jobs': len(self.jobs),
                'active_jobs': sum(1 for job in self.jobs.values() if job.enabled),
                'total_executions': self.total_executions,
                'successful_executions': self.successful_executions,
                'failed_executions': self.failed_executions,
                'success_rate': self.successful_executions / max(self.total_executions, 1)
            },
            'sources': {
                source: {
                    'status': health.status.value,
                    'error_rate': health.error_rate,
                    'avg_latency_ms': health.avg_latency_ms,
                    'circuit_breaker_open': health.circuit_breaker_open,
                    'total_jobs': health.total_jobs,
                    'active_jobs': health.active_jobs,
                    'failed_jobs': health.failed_jobs,
                    'last_success': health.last_success.isoformat() if health.last_success else None,
                    'last_failure': health.last_failure.isoformat() if health.last_failure else None
                }
                for source, health in self.source_health.items()
            },
            'jobs': {
                job_id: {
                    'name': job.name,
                    'source': job.source,
                    'enabled': job.enabled,
                    'status': job.status.value,
                    'success_count': job.success_count,
                    'failure_count': job.failure_count,
                    'consecutive_failures': job.consecutive_failures,
                    'avg_duration_ms': job.avg_duration_ms,
                    'last_run': job.last_run.isoformat() if job.last_run else None,
                    'last_error': job.last_error
                }
                for job_id, job in self.jobs.items()
            }
        }
    
    async def enable_job(self, job_id: str):
        """Enable specific job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = True
            await self.schedule_job(job)
            self.logger.info(f"✅ Enabled job: {job.name}")
    
    async def disable_job(self, job_id: str):
        """Disable specific job"""
        job = self.jobs.get(job_id)
        if job:
            job.enabled = False
            job.status = JobStatus.DISABLED
            if self.scheduler:
                try:
                    self.scheduler.remove_job(job_id)
                except:
                    pass  # Job might not be scheduled
            self.logger.info(f"🛑 Disabled job: {job.name}")