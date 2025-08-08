#!/usr/bin/env python3
"""
Distributed Multi-Process Orchestrator
Replaces the monolithic orchestrator with isolated agent processes
"""

import asyncio
import multiprocessing
import signal
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import psutil
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    name: str
    module_path: str
    restart_limit: int = 3
    timeout: int = 300  # 5 minutes
    circuit_breaker_threshold: int = 5
    backoff_multiplier: float = 2.0
    max_backoff: int = 60

@dataclass
class AgentStatus:
    process: Optional[multiprocessing.Process]
    last_heartbeat: datetime
    restart_count: int
    failures: int
    circuit_open: bool
    next_retry: Optional[datetime]

class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if datetime.now() - self.last_failure > timedelta(seconds=self.timeout):
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = datetime.now()
            
            if self.failures >= self.threshold:
                self.state = 'OPEN'
            
            raise e

class DistributedOrchestrator:
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.message_queue = asyncio.Queue()
        self.running = True
        
        # Define isolated agents
        self.setup_agents()
    
    def setup_agents(self):
        """Configure all independent agent processes"""
        agent_configs = [
            AgentConfig("async_data_collector", "agents.data_collector", timeout=120),  # Faster with async
            AgentConfig("sentiment_analyzer", "agents.sentiment_analyzer", timeout=120),
            AgentConfig("technical_analyzer", "agents.technical_analyzer", timeout=90),
            AgentConfig("ml_predictor", "agents.ml_predictor", timeout=300),
            AgentConfig("whale_detector", "agents.whale_detector", timeout=150),
            AgentConfig("risk_manager", "agents.risk_manager", timeout=60),
            AgentConfig("portfolio_optimizer", "agents.portfolio_optimizer", timeout=120),
            AgentConfig("health_monitor", "agents.health_monitor", timeout=30),
        ]
        
        for config in agent_configs:
            self.agents[config.name] = config
            self.agent_status[config.name] = AgentStatus(
                process=None,
                last_heartbeat=datetime.now(),
                restart_count=0,
                failures=0,
                circuit_open=False,
                next_retry=None
            )
            self.circuit_breakers[config.name] = CircuitBreaker()
    
    def start_agent(self, agent_name: str) -> bool:
        """Start an individual agent process with isolation"""
        try:
            config = self.agents[agent_name]
            status = self.agent_status[agent_name]
            
            # Check if agent is already running
            if status.process and status.process.is_alive():
                return True
            
            # Create isolated process
            process = multiprocessing.Process(
                target=self._run_agent,
                args=(agent_name, config),
                name=f"agent_{agent_name}"
            )
            
            process.start()
            status.process = process
            status.last_heartbeat = datetime.now()
            
            logger.info(f"Started agent {agent_name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent {agent_name}: {e}")
            self.agent_status[agent_name].failures += 1
            return False
    
    def _run_agent(self, agent_name: str, config: AgentConfig):
        """Run agent in isolated process"""
        try:
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._agent_shutdown_handler)
            signal.signal(signal.SIGINT, self._agent_shutdown_handler)
            
            # Import and run agent module
            module = __import__(config.module_path, fromlist=[''])
            
            if hasattr(module, 'run'):
                logger.info(f"Running agent {agent_name}")
                module.run()
            else:
                logger.error(f"Agent {agent_name} has no run() function")
                
        except Exception as e:
            logger.error(f"Agent {agent_name} failed: {e}")
            sys.exit(1)
    
    def _agent_shutdown_handler(self, signum, frame):
        """Handle graceful shutdown of agent"""
        logger.info(f"Agent received shutdown signal {signum}")
        sys.exit(0)
    
    def monitor_agents(self):
        """Monitor agent health and restart failed agents"""
        while self.running:
            try:
                for agent_name, status in self.agent_status.items():
                    config = self.agents[agent_name]
                    
                    # Check if process is alive
                    if not status.process or not status.process.is_alive():
                        self._handle_agent_failure(agent_name)
                        continue
                    
                    # Check for timeout
                    if datetime.now() - status.last_heartbeat > timedelta(seconds=config.timeout):
                        logger.warning(f"Agent {agent_name} timeout detected")
                        self._handle_agent_failure(agent_name)
                        continue
                    
                    # Check resource usage
                    try:
                        process = psutil.Process(status.process.pid)
                        cpu_percent = process.cpu_percent()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        
                        # Kill runaway processes
                        if cpu_percent > 90 or memory_mb > 1024:  # 1GB limit
                            logger.warning(f"Agent {agent_name} consuming too many resources")
                            self._handle_agent_failure(agent_name)
                    
                    except psutil.NoSuchProcess:
                        self._handle_agent_failure(agent_name)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(5)
    
    def _handle_agent_failure(self, agent_name: str):
        """Handle agent failure with exponential backoff"""
        status = self.agent_status[agent_name]
        config = self.agents[agent_name]
        
        # Terminate existing process if still running
        if status.process and status.process.is_alive():
            try:
                status.process.terminate()
                status.process.join(timeout=5)
                if status.process.is_alive():
                    status.process.kill()
            except Exception as e:
                logger.error(f"Error terminating agent {agent_name}: {e}")
        
        status.failures += 1
        status.restart_count += 1
        
        # Check restart limits
        if status.restart_count > config.restart_limit:
            logger.error(f"Agent {agent_name} exceeded restart limit, opening circuit breaker")
            status.circuit_open = True
            status.next_retry = datetime.now() + timedelta(minutes=30)
            return
        
        # Exponential backoff
        backoff_time = min(
            config.backoff_multiplier ** status.failures,
            config.max_backoff
        )
        
        status.next_retry = datetime.now() + timedelta(seconds=backoff_time)
        logger.info(f"Agent {agent_name} will restart in {backoff_time} seconds")
        
        # Schedule restart
        asyncio.create_task(self._delayed_restart(agent_name, backoff_time))
    
    async def _delayed_restart(self, agent_name: str, delay: float):
        """Restart agent after backoff delay"""
        await asyncio.sleep(delay)
        
        status = self.agent_status[agent_name]
        if not status.circuit_open and datetime.now() >= status.next_retry:
            logger.info(f"Attempting to restart agent {agent_name}")
            if self.start_agent(agent_name):
                status.failures = 0  # Reset on successful restart
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "orchestrator": {
                "running": self.running,
                "timestamp": datetime.now().isoformat()
            },
            "agents": {}
        }
        
        for agent_name, agent_status in self.agent_status.items():
            status["agents"][agent_name] = {
                "running": agent_status.process and agent_status.process.is_alive(),
                "pid": agent_status.process.pid if agent_status.process else None,
                "last_heartbeat": agent_status.last_heartbeat.isoformat(),
                "restart_count": agent_status.restart_count,
                "failures": agent_status.failures,
                "circuit_open": agent_status.circuit_open,
                "next_retry": agent_status.next_retry.isoformat() if agent_status.next_retry else None
            }
        
        return status
    
    def start_all_agents(self):
        """Start all configured agents"""
        logger.info("Starting distributed agent system...")
        
        for agent_name in self.agents:
            if not self.start_agent(agent_name):
                logger.error(f"Failed to start agent {agent_name}")
        
        # Start monitoring
        monitor_thread = multiprocessing.Process(target=self.monitor_agents)
        monitor_thread.start()
        
        logger.info("Distributed orchestrator running")
        return monitor_thread
    
    def shutdown(self):
        """Graceful shutdown of all agents"""
        logger.info("Shutting down distributed orchestrator...")
        self.running = False
        
        for agent_name, status in self.agent_status.items():
            if status.process and status.process.is_alive():
                try:
                    status.process.terminate()
                    status.process.join(timeout=10)
                    if status.process.is_alive():
                        status.process.kill()
                    logger.info(f"Stopped agent {agent_name}")
                except Exception as e:
                    logger.error(f"Error stopping agent {agent_name}: {e}")

def main():
    """Main entry point for distributed orchestrator"""
    orchestrator = DistributedOrchestrator()
    
    try:
        monitor_process = orchestrator.start_all_agents()
        
        # Keep main process alive
        while True:
            time.sleep(60)
            
            # Log system status
            status = orchestrator.get_system_status()
            logger.info(f"System status: {json.dumps(status, indent=2)}")
    
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()