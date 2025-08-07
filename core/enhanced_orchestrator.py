"""
Enhanced Orchestrator - Addresses failover, self-healing, resource management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import time
import psutil
import threading
from pathlib import Path
from enum import Enum
import signal
import sys

from utils.daily_logger import get_daily_logger

class AgentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    FAILED = "failed"
    RESTARTING = "restarting"
    STOPPED = "stopped"

@dataclass
class AgentHealth:
    """Agent health monitoring"""
    agent_name: str
    status: AgentStatus
    last_heartbeat: datetime
    error_count: int = 0
    restart_count: int = 0
    response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_error: Optional[str] = None
    uptime: timedelta = field(default_factory=lambda: timedelta(0))

@dataclass
class SystemResources:
    """System resource monitoring"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    active_agents: int
    failed_agents: int
    timestamp: datetime

class ResourceMonitor:
    """Monitor system resources and prevent overload"""
    
    def __init__(self, max_cpu_percent: float = 80, max_memory_percent: float = 85):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.logger = get_daily_logger().get_logger('performance')
        self.monitoring = False
        
    async def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        while self.monitoring:
            try:
                resources = self._get_current_resources()
                
                # Log resource usage
                self.logger.info(f"Resources: CPU={resources.cpu_percent:.1f}%, "
                               f"Memory={resources.memory_percent:.1f}%, "
                               f"Agents={resources.active_agents}/{resources.active_agents + resources.failed_agents}")
                
                # Check for resource constraints
                if resources.cpu_percent > self.max_cpu_percent:
                    self.logger.warning(f"High CPU usage: {resources.cpu_percent:.1f}%")
                
                if resources.memory_percent > self.max_memory_percent:
                    self.logger.warning(f"High memory usage: {resources.memory_percent:.1f}%")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
    
    def _get_current_resources(self) -> SystemResources:
        """Get current system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net_io = psutil.net_io_counters()._asdict()
            
            return SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk.percent,
                network_io=net_io,
                active_agents=0,  # Will be updated by orchestrator
                failed_agents=0,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"Error getting system resources: {e}")
            return SystemResources(0, 0, 0, {}, 0, 0, datetime.now())
    
    def is_resource_available(self, required_cpu: float = 10, required_memory: float = 10) -> bool:
        """Check if resources are available for new operations"""
        resources = self._get_current_resources()
        
        cpu_available = resources.cpu_percent + required_cpu <= self.max_cpu_percent
        memory_available = resources.memory_percent + required_memory <= self.max_memory_percent
        
        return cpu_available and memory_available

class AgentManager:
    """Manage individual agents with health monitoring and auto-restart"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.logger = get_daily_logger().get_logger('health')
        self.agents: Dict[str, AgentHealth] = {}
        self.agent_tasks: Dict[str, asyncio.Task] = {}
        self.heartbeat_interval = 30  # seconds
        self.max_restart_attempts = 3
        self.restart_cooldown = 60  # seconds
        
    async def register_agent(self, 
                           agent_name: str,
                           agent_instance: Any,
                           health_check_func: Optional[Callable] = None):
        """Register an agent for monitoring"""
        
        self.agents[agent_name] = AgentHealth(
            agent_name=agent_name,
            status=AgentStatus.HEALTHY,
            last_heartbeat=datetime.now()
        )
        
        # Start health monitoring task
        task = asyncio.create_task(
            self._monitor_agent_health(agent_name, agent_instance, health_check_func)
        )
        self.agent_tasks[agent_name] = task
        
        self.logger.info(f"Registered agent: {agent_name}")
    
    async def _monitor_agent_health(self, 
                                  agent_name: str,
                                  agent_instance: Any,
                                  health_check_func: Optional[Callable]):
        """Monitor individual agent health"""
        
        start_time = datetime.now()
        
        while True:
            try:
                agent_health = self.agents[agent_name]
                
                # Perform health check
                health_ok = await self._perform_health_check(
                    agent_name, agent_instance, health_check_func
                )
                
                if health_ok:
                    agent_health.status = AgentStatus.HEALTHY
                    agent_health.last_heartbeat = datetime.now()
                    agent_health.uptime = datetime.now() - start_time
                    agent_health.error_count = max(0, agent_health.error_count - 1)  # Decay errors
                else:
                    await self._handle_agent_failure(agent_name, agent_instance)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error for {agent_name}: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _perform_health_check(self, 
                                  agent_name: str,
                                  agent_instance: Any,
                                  health_check_func: Optional[Callable]) -> bool:
        """Perform health check on agent"""
        
        try:
            start_time = time.time()
            
            if health_check_func:
                # Custom health check
                result = await asyncio.wait_for(
                    health_check_func(agent_instance),
                    timeout=10
                )
                health_ok = bool(result)
            else:
                # Default health check - check if agent has get_status method
                if hasattr(agent_instance, 'get_status'):
                    status = agent_instance.get_status()
                    health_ok = status.get('status') == 'operational'
                else:
                    health_ok = True  # Assume healthy if no check available
            
            response_time = time.time() - start_time
            self.agents[agent_name].response_time = response_time
            
            return health_ok
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Health check timeout for {agent_name}")
            return False
        except Exception as e:
            self.logger.error(f"Health check failed for {agent_name}: {e}")
            return False
    
    async def _handle_agent_failure(self, agent_name: str, agent_instance: Any):
        """Handle agent failure with restart logic"""
        
        agent_health = self.agents[agent_name]
        agent_health.error_count += 1
        agent_health.status = AgentStatus.FAILED
        
        self.logger.error(f"Agent {agent_name} failed (errors: {agent_health.error_count})")
        
        # Attempt restart if within limits
        if (agent_health.restart_count < self.max_restart_attempts and
            (datetime.now() - agent_health.last_heartbeat).seconds > self.restart_cooldown):
            
            await self._restart_agent(agent_name, agent_instance)
        else:
            self.logger.critical(f"Agent {agent_name} permanently failed - max restarts exceeded")
            agent_health.status = AgentStatus.STOPPED
    
    async def _restart_agent(self, agent_name: str, agent_instance: Any):
        """Restart failed agent"""
        
        agent_health = self.agents[agent_name]
        agent_health.status = AgentStatus.RESTARTING
        agent_health.restart_count += 1
        
        self.logger.info(f"Restarting agent {agent_name} (attempt {agent_health.restart_count})")
        
        try:
            # Call restart method if available
            if hasattr(agent_instance, 'restart'):
                await agent_instance.restart()
            elif hasattr(agent_instance, '__init__'):
                # Reinitialize agent
                agent_instance.__init__()
            
            # Wait for restart
            await asyncio.sleep(5)
            
            # Verify restart
            health_ok = await self._perform_health_check(agent_name, agent_instance, None)
            
            if health_ok:
                agent_health.status = AgentStatus.HEALTHY
                agent_health.last_heartbeat = datetime.now()
                self.logger.info(f"Agent {agent_name} restarted successfully")
            else:
                agent_health.status = AgentStatus.FAILED
                self.logger.error(f"Agent {agent_name} restart failed")
                
        except Exception as e:
            agent_health.status = AgentStatus.FAILED
            agent_health.last_error = str(e)
            self.logger.error(f"Error restarting agent {agent_name}: {e}")
    
    def get_agent_status(self, agent_name: str) -> Optional[AgentHealth]:
        """Get status of specific agent"""
        return self.agents.get(agent_name)
    
    def get_all_agents_status(self) -> Dict[str, AgentHealth]:
        """Get status of all agents"""
        return self.agents.copy()
    
    async def shutdown_all_agents(self):
        """Gracefully shutdown all agents"""
        self.logger.info("Shutting down all agents")
        
        # Cancel monitoring tasks
        for agent_name, task in self.agent_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Update agent status
        for agent_health in self.agents.values():
            agent_health.status = AgentStatus.STOPPED

class EnhancedOrchestrator:
    """Production-grade orchestrator with failover and self-healing"""
    
    def __init__(self):
        self.logger = get_daily_logger().get_logger('main')
        self.agent_manager = AgentManager(self)
        self.resource_monitor = ResourceMonitor()
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0,
            'uptime_start': datetime.now()
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """Start the orchestrator"""
        
        if self.running:
            self.logger.warning("Orchestrator already running")
            return
        
        self.running = True
        self.logger.info("Starting Enhanced Orchestrator")
        
        try:
            # Start resource monitoring
            resource_task = asyncio.create_task(self.resource_monitor.start_monitoring())
            
            # Initialize and register agents
            await self._initialize_agents()
            
            # Start main orchestration loop
            orchestration_task = asyncio.create_task(self._orchestration_loop())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.critical(f"Orchestrator startup failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _initialize_agents(self):
        """Initialize and register all agents"""
        
        self.logger.info("Initializing agents")
        
        try:
            # Import and register sentiment agent
            from agents.enhanced_sentiment_agent import sentiment_agent
            await self.agent_manager.register_agent(
                'sentiment_agent',
                sentiment_agent,
                self._sentiment_health_check
            )
            
            # Import and register technical agent
            from agents.enhanced_technical_agent import technical_agent
            await self.agent_manager.register_agent(
                'technical_agent',
                technical_agent,
                self._technical_health_check
            )
            
            # Import and register whale agent
            from agents.enhanced_whale_agent import whale_agent
            await self.agent_manager.register_agent(
                'whale_agent',
                whale_agent,
                self._whale_health_check
            )
            
            # Import and register ML agent
            from agents.enhanced_ml_agent import ml_agent
            await self.agent_manager.register_agent(
                'ml_agent',
                ml_agent,
                self._ml_health_check
            )
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def _sentiment_health_check(self, agent) -> bool:
        """Health check for sentiment agent"""
        try:
            status = agent.get_status()
            return status.get('status') == 'operational'
        except:
            return False
    
    async def _technical_health_check(self, agent) -> bool:
        """Health check for technical agent"""
        try:
            status = agent.get_status()
            return status.get('status') == 'operational'
        except:
            return False
    
    async def _whale_health_check(self, agent) -> bool:
        """Health check for whale agent"""
        try:
            status = agent.get_status()
            return status.get('status') == 'operational'
        except:
            return False
    
    async def _ml_health_check(self, agent) -> bool:
        """Health check for ML agent"""
        try:
            status = agent.get_status()
            return status.get('status') == 'operational'
        except:
            return False
    
    async def _orchestration_loop(self):
        """Main orchestration loop"""
        
        self.logger.info("Starting orchestration loop")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                loop_start = time.time()
                
                # Check system resources
                if not self.resource_monitor.is_resource_available():
                    self.logger.warning("System resources constrained, throttling operations")
                    await asyncio.sleep(60)
                    continue
                
                # Perform orchestration tasks
                await self._coordinate_agents()
                
                # Update performance metrics
                loop_time = time.time() - loop_start
                self._update_performance_metrics(loop_time, success=True)
                
                # Sleep before next iteration
                await asyncio.sleep(30)  # 30 second cycle
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                self._update_performance_metrics(0, success=False)
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _coordinate_agents(self):
        """Coordinate agent activities"""
        
        # Get list of coins to analyze
        coins = ['BTC/USD', 'ETH/USD', 'SOL/USD']  # Example coins
        
        # Check which agents are healthy
        healthy_agents = []
        for agent_name, agent_health in self.agent_manager.get_all_agents_status().items():
            if agent_health.status == AgentStatus.HEALTHY:
                healthy_agents.append(agent_name)
        
        if not healthy_agents:
            self.logger.warning("No healthy agents available")
            return
        
        self.logger.info(f"Coordinating {len(healthy_agents)} healthy agents for {len(coins)} coins")
        
        # Log coordination activity
        get_daily_logger().log_performance_metric(
            'agent_coordination_cycle',
            len(healthy_agents),
            'agents',
            {'coins': len(coins), 'healthy_agents': healthy_agents}
        )
    
    def _update_performance_metrics(self, response_time: float, success: bool):
        """Update performance metrics"""
        
        self.performance_metrics['total_requests'] += 1
        
        if success:
            self.performance_metrics['successful_requests'] += 1
        else:
            self.performance_metrics['failed_requests'] += 1
        
        # Update average response time
        total_requests = self.performance_metrics['total_requests']
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        agent_statuses = self.agent_manager.get_all_agents_status()
        
        healthy_count = sum(1 for h in agent_statuses.values() if h.status == AgentStatus.HEALTHY)
        failed_count = sum(1 for h in agent_statuses.values() if h.status == AgentStatus.FAILED)
        
        uptime = datetime.now() - self.performance_metrics['uptime_start']
        
        return {
            'orchestrator_status': 'running' if self.running else 'stopped',
            'uptime_seconds': uptime.total_seconds(),
            'agents': {
                'healthy': healthy_count,
                'failed': failed_count,
                'total': len(agent_statuses)
            },
            'performance': self.performance_metrics,
            'agent_details': {name: {
                'status': health.status.value,
                'uptime': health.uptime.total_seconds(),
                'error_count': health.error_count,
                'restart_count': health.restart_count,
                'response_time': health.response_time
            } for name, health in agent_statuses.items()}
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        
        self.logger.info("Initiating graceful shutdown")
        self.running = False
        
        # Stop resource monitoring
        self.resource_monitor.stop_monitoring()
        
        # Shutdown all agents
        await self.agent_manager.shutdown_all_agents()
        
        # Signal shutdown complete
        self.shutdown_event.set()
        
        self.logger.info("Orchestrator shutdown complete")
    
    async def _cleanup(self):
        """Cleanup resources"""
        
        self.logger.info("Cleaning up resources")
        
        # Additional cleanup tasks can be added here
        
        self.logger.info("Cleanup complete")

# Global instance
orchestrator = EnhancedOrchestrator()

# CLI for running orchestrator
async def main():
    """Main entry point"""
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        await orchestrator.shutdown()
    except Exception as e:
        logging.critical(f"Orchestrator failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())