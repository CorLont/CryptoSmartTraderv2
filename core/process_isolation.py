#!/usr/bin/env python3
"""
Process Isolation System
Multi-process agent architecture with autorestart + exponential backoff
"""

import multiprocessing as mp
import asyncio
import time
import json
import signal
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

class AgentState(Enum):
    """Agent process states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"

@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    target_function: str
    module_path: str
    restart_policy: str = "always"  # always, on-failure, never
    max_restarts: int = 10
    restart_delay_base: float = 1.0  # Base delay for exponential backoff
    restart_delay_max: float = 60.0  # Max delay
    health_check_interval: float = 30.0
    memory_limit_mb: int = 1024
    cpu_limit_percent: float = 50.0

@dataclass
class AgentStatus:
    """Agent runtime status"""
    name: str
    state: AgentState
    pid: Optional[int]
    start_time: Optional[datetime]
    restart_count: int
    last_restart: Optional[datetime]
    last_health_check: Optional[datetime]
    memory_usage_mb: float
    cpu_usage_percent: float
    error_message: Optional[str]

class ExponentialBackoff:
    """Exponential backoff calculator"""
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0):
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

class AgentProcess:
    """Individual agent process wrapper"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_structured_logger(f"Agent_{config.name}")
        self.process: Optional[mp.Process] = None
        self.status = AgentStatus(
            name=config.name,
            state=AgentState.STOPPED,
            pid=None,
            start_time=None,
            restart_count=0,
            last_restart=None,
            last_health_check=None,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            error_message=None
        )
        self.backoff = ExponentialBackoff(
            config.restart_delay_base,
            config.restart_delay_max
        )
        self.stop_event = mp.Event()
    
    def start(self) -> bool:
        """Start the agent process"""
        
        if self.status.state in [AgentState.RUNNING, AgentState.STARTING]:
            self.logger.warning(f"Agent {self.config.name} already running")
            return False
        
        try:
            self.status.state = AgentState.STARTING
            self.logger.info(f"Starting agent {self.config.name}")
            
            # Create and start process
            self.process = mp.Process(
                target=self._run_agent,
                name=f"Agent_{self.config.name}",
                daemon=False
            )
            
            self.process.start()
            self.status.pid = self.process.pid
            self.status.start_time = datetime.now()
            self.status.state = AgentState.RUNNING
            
            self.logger.info(f"Agent {self.config.name} started with PID {self.status.pid}")
            
            return True
            
        except Exception as e:
            self.status.state = AgentState.FAILED
            self.status.error_message = str(e)
            self.logger.error(f"Failed to start agent {self.config.name}: {e}")
            return False
    
    def stop(self, timeout: float = 10.0) -> bool:
        """Stop the agent process gracefully"""
        
        if self.status.state not in [AgentState.RUNNING, AgentState.STARTING]:
            return True
        
        try:
            self.status.state = AgentState.STOPPING
            self.stop_event.set()
            
            if self.process and self.process.is_alive():
                self.logger.info(f"Stopping agent {self.config.name}")
                
                # Try graceful shutdown first
                self.process.terminate()
                self.process.join(timeout=timeout)
                
                # Force kill if still alive
                if self.process.is_alive():
                    self.logger.warning(f"Force killing agent {self.config.name}")
                    self.process.kill()
                    self.process.join(timeout=5.0)
            
            self.status.state = AgentState.STOPPED
            self.status.pid = None
            
            self.logger.info(f"Agent {self.config.name} stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop agent {self.config.name}: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart the agent process"""
        
        self.logger.info(f"Restarting agent {self.config.name}")
        
        # Calculate restart delay with exponential backoff
        delay = self.backoff.calculate_delay(self.status.restart_count)
        
        if delay > 0:
            self.logger.info(f"Waiting {delay:.1f}s before restart (attempt {self.status.restart_count + 1})")
            time.sleep(delay)
        
        # Stop and start
        self.stop()
        time.sleep(1.0)  # Brief pause
        
        success = self.start()
        
        if success:
            self.status.restart_count += 1
            self.status.last_restart = datetime.now()
            self.logger.info(f"Agent {self.config.name} restarted successfully")
        else:
            self.logger.error(f"Failed to restart agent {self.config.name}")
        
        return success
    
    def is_healthy(self) -> bool:
        """Check if agent process is healthy"""
        
        try:
            if not self.process or not self.process.is_alive():
                return False
            
            # Basic process health check
            if self.status.state != AgentState.RUNNING:
                return False
            
            # Check if process is responsive (simplified)
            # In production, this would include more sophisticated health checks
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed for agent {self.config.name}: {e}")
            return False
    
    def update_status(self) -> None:
        """Update agent status metrics"""
        
        try:
            if self.process and self.process.is_alive():
                # In production, would use psutil for accurate metrics
                self.status.memory_usage_mb = 100.0  # Mock value
                self.status.cpu_usage_percent = 15.0  # Mock value
            else:
                self.status.memory_usage_mb = 0.0
                self.status.cpu_usage_percent = 0.0
            
            self.status.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update status for agent {self.config.name}: {e}")
    
    def _run_agent(self) -> None:
        """Main agent process function"""
        
        try:
            # Set up signal handlers
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)
            
            self.logger.info(f"Agent {self.config.name} process started")
            
            # Import and run the target function
            module_parts = self.config.module_path.split('.')
            module_name = '.'.join(module_parts[:-1])
            function_name = module_parts[-1]
            
            module = __import__(module_name, fromlist=[function_name])
            target_func = getattr(module, function_name)
            
            # Run the agent function
            if asyncio.iscoroutinefunction(target_func):
                asyncio.run(target_func(self.stop_event))
            else:
                target_func(self.stop_event)
            
        except Exception as e:
            self.logger.error(f"Agent {self.config.name} process failed: {e}")
            self.logger.error(traceback.format_exc())
            sys.exit(1)
        
        self.logger.info(f"Agent {self.config.name} process finished")
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Agent {self.config.name} received signal {signum}")
        self.stop_event.set()

class ProcessIsolationManager:
    """Manager for isolated agent processes"""
    
    def __init__(self):
        self.logger = get_structured_logger("ProcessIsolationManager")
        self.agents: Dict[str, AgentProcess] = {}
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def register_agent(self, config: AgentConfig) -> bool:
        """Register a new agent"""
        
        try:
            if config.name in self.agents:
                self.logger.warning(f"Agent {config.name} already registered")
                return False
            
            agent = AgentProcess(config)
            self.agents[config.name] = agent
            
            self.logger.info(f"Registered agent {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {config.name}: {e}")
            return False
    
    def start_agent(self, name: str) -> bool:
        """Start a specific agent"""
        
        if name not in self.agents:
            self.logger.error(f"Agent {name} not registered")
            return False
        
        return self.agents[name].start()
    
    def stop_agent(self, name: str) -> bool:
        """Stop a specific agent"""
        
        if name not in self.agents:
            self.logger.error(f"Agent {name} not registered")
            return False
        
        return self.agents[name].stop()
    
    def restart_agent(self, name: str) -> bool:
        """Restart a specific agent"""
        
        if name not in self.agents:
            self.logger.error(f"Agent {name} not registered")
            return False
        
        return self.agents[name].restart()
    
    def start_all_agents(self) -> bool:
        """Start all registered agents"""
        
        self.logger.info("Starting all agents")
        
        success_count = 0
        for name, agent in self.agents.items():
            if agent.start():
                success_count += 1
        
        self.logger.info(f"Started {success_count}/{len(self.agents)} agents")
        return success_count == len(self.agents)
    
    def stop_all_agents(self) -> bool:
        """Stop all agents"""
        
        self.logger.info("Stopping all agents")
        
        success_count = 0
        for name, agent in self.agents.items():
            if agent.stop():
                success_count += 1
        
        self.logger.info(f"Stopped {success_count}/{len(self.agents)} agents")
        return success_count == len(self.agents)
    
    async def start_monitoring(self) -> None:
        """Start the monitoring loop"""
        
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting process isolation monitoring")
        
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring loop"""
        
        self.running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Process isolation monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.running:
            try:
                await self._monitor_agents()
                await asyncio.sleep(5.0)  # Monitor every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10.0)
    
    async def _monitor_agents(self) -> None:
        """Monitor all agents and handle failures"""
        
        for name, agent in self.agents.items():
            try:
                # Update agent status
                agent.update_status()
                
                # Check if agent needs restart
                should_restart = False
                
                if not agent.is_healthy():
                    if agent.status.state == AgentState.RUNNING:
                        self.logger.warning(f"Agent {name} unhealthy, marking for restart")
                        agent.status.state = AgentState.FAILED
                        should_restart = True
                
                # Check if process died unexpectedly
                if agent.process and not agent.process.is_alive() and agent.status.state == AgentState.RUNNING:
                    self.logger.error(f"Agent {name} process died unexpectedly")
                    agent.status.state = AgentState.FAILED
                    should_restart = True
                
                # Handle restart if needed
                if should_restart and agent.config.restart_policy == "always":
                    if agent.status.restart_count < agent.config.max_restarts:
                        self.logger.info(f"Auto-restarting agent {name}")
                        agent.restart()
                    else:
                        self.logger.error(f"Agent {name} exceeded max restarts ({agent.config.max_restarts})")
                        agent.status.state = AgentState.FAILED
                
            except Exception as e:
                self.logger.error(f"Error monitoring agent {name}: {e}")
    
    def get_agent_status(self, name: str) -> Optional[AgentStatus]:
        """Get status of specific agent"""
        
        if name not in self.agents:
            return None
        
        return self.agents[name].status
    
    def get_all_status(self) -> Dict[str, AgentStatus]:
        """Get status of all agents"""
        
        return {name: agent.status for name, agent in self.agents.items()}
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary"""
        
        total_agents = len(self.agents)
        running_agents = sum(1 for agent in self.agents.values() 
                           if agent.status.state == AgentState.RUNNING)
        failed_agents = sum(1 for agent in self.agents.values() 
                          if agent.status.state == AgentState.FAILED)
        
        return {
            'total_agents': total_agents,
            'running_agents': running_agents,
            'failed_agents': failed_agents,
            'health_percentage': (running_agents / max(total_agents, 1)) * 100,
            'monitoring_active': self.running
        }

# Mock agent functions for testing
async def mock_data_collector_agent(stop_event: mp.Event) -> None:
    """Mock data collector agent"""
    logger = get_structured_logger("DataCollectorAgent")
    logger.info("Data collector agent started")
    
    while not stop_event.is_set():
        try:
            # Simulate data collection work
            logger.info("Collecting market data...")
            await asyncio.sleep(10.0)
            
        except Exception as e:
            logger.error(f"Data collector error: {e}")
            await asyncio.sleep(5.0)
    
    logger.info("Data collector agent stopped")

async def mock_ml_predictor_agent(stop_event: mp.Event) -> None:
    """Mock ML predictor agent"""
    logger = get_structured_logger("MLPredictorAgent")
    logger.info("ML predictor agent started")
    
    while not stop_event.is_set():
        try:
            # Simulate ML prediction work
            logger.info("Running ML predictions...")
            await asyncio.sleep(15.0)
            
        except Exception as e:
            logger.error(f"ML predictor error: {e}")
            await asyncio.sleep(5.0)
    
    logger.info("ML predictor agent stopped")

def mock_failing_agent(stop_event: mp.Event) -> None:
    """Mock agent that fails for testing restart logic"""
    logger = get_structured_logger("FailingAgent")
    logger.info("Failing agent started")
    
    time.sleep(2.0)  # Run briefly then fail
    raise Exception("Simulated agent failure")

if __name__ == "__main__":
    async def test_process_isolation():
        """Test process isolation system"""
        
        print("🔍 TESTING PROCESS ISOLATION SYSTEM")
        print("=" * 60)
        
        # Create manager
        manager = ProcessIsolationManager()
        
        # Register agents
        agents_config = [
            AgentConfig(
                name="DataCollector",
                target_function="mock_data_collector_agent",
                module_path="core.process_isolation.mock_data_collector_agent",
                restart_delay_base=1.0,
                max_restarts=3
            ),
            AgentConfig(
                name="MLPredictor", 
                target_function="mock_ml_predictor_agent",
                module_path="core.process_isolation.mock_ml_predictor_agent",
                restart_delay_base=2.0,
                max_restarts=3
            ),
            AgentConfig(
                name="FailingAgent",
                target_function="mock_failing_agent", 
                module_path="core.process_isolation.mock_failing_agent",
                restart_delay_base=1.0,
                max_restarts=2
            )
        ]
        
        print("📝 Registering agents...")
        for config in agents_config:
            success = manager.register_agent(config)
            print(f"   {'✅' if success else '❌'} {config.name}")
        
        print("\n🚀 Starting all agents...")
        await manager.start_monitoring()
        success = manager.start_all_agents()
        print(f"   Start result: {'✅' if success else '❌'}")
        
        print("\n⏱️  Monitoring for 30 seconds...")
        
        # Monitor for 30 seconds
        for i in range(6):
            await asyncio.sleep(5.0)
            
            health = manager.get_health_summary()
            print(f"   Health check {i+1}: {health['running_agents']}/{health['total_agents']} running "
                  f"({health['health_percentage']:.1f}%)")
            
            # Show agent status
            for name, status in manager.get_all_status().items():
                state_emoji = {"running": "🟢", "failed": "🔴", "restarting": "🟡"}.get(status.state.value, "⚪")
                print(f"     {state_emoji} {name}: {status.state.value} (restarts: {status.restart_count})")
        
        print("\n🛑 Stopping all agents...")
        await manager.stop_monitoring()
        success = manager.stop_all_agents()
        print(f"   Stop result: {'✅' if success else '❌'}")
        
        print("\n✅ PROCESS ISOLATION TEST COMPLETED")
    
    # Run test
    asyncio.run(test_process_isolation())