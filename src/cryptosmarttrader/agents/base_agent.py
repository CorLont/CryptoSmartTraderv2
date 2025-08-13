#!/usr/bin/env python3
"""
Base Agent Class for Distributed Architecture
All agents inherit from this for consistent behavior
"""

import asyncio
import signal
import sys
import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
from pathlib import Path

class BaseAgent(ABC):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.running = True
        self.last_heartbeat = datetime.now()
        self.logger = self.setup_logging()

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown_handler)
        signal.signal(signal.SIGINT, self.shutdown_handler)

        # Health check settings
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.heartbeat_file = Path(f"logs/{self.name}_heartbeat.json")

    def setup_logging(self) -> logging.Logger:
        """Setup isolated logging for this agent"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

        # File handler
        handler = logging.FileHandler(f"logs/{self.name}.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def shutdown_handler(self, signum, frame):
        """Handle graceful shutdown"""
        self.logger.info(f"Agent {self.name} received shutdown signal {signum}")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def update_heartbeat(self):
        """Update heartbeat for health monitoring"""
        import os

        self.last_heartbeat = datetime.now()

        heartbeat_data = {
            "name": self.name,
            "timestamp": self.last_heartbeat.isoformat(),
            "status": "healthy",
            "pid": os.getpid()
        }

        try:
            with open(self.heartbeat_file, 'w') as f:
                json.dump(heartbeat_data, f)
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat: {e}")

    async def health_check(self):
        """Periodic health check"""
        while self.running:
            try:
                self.update_heartbeat()
                await self.perform_health_check()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                await asyncio.sleep(5)

    @abstractmethod
    async def perform_health_check(self):
        """Agent-specific health check implementation"""
        pass

    @abstractmethod
    async def main_loop(self):
        """Main agent logic loop"""
        pass

    def cleanup(self):
        """Cleanup resources before shutdown"""
        self.logger.info(f"Agent {self.name} cleaning up...")

        # Remove heartbeat file
        try:
            if self.heartbeat_file.exists():
                self.heartbeat_file.unlink()
        except Exception as e:
            self.logger.error(f"Failed to cleanup heartbeat file: {e}")

    async def run(self):
        """Run the agent with health monitoring"""
        self.logger.info(f"Starting agent {self.name}")

        try:
            # Start health check task
            health_task = asyncio.create_task(self.health_check())

            # Start main loop task
            main_task = asyncio.create_task(self.main_loop())

            # Wait for either to complete
            await asyncio.gather(health_task, main_task)

        except Exception as e:
            self.logger.error(f"Agent {self.name} failed: {e}")
            raise
        finally:
            self.cleanup()

def run_agent(agent_class, config: Dict[str, Any] = None):
    """Utility function to run an agent"""

    agent = agent_class(config)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.logger.info("Agent interrupted by user")
    except Exception as e:
        agent.logger.error(f"Agent failed: {e}")
        sys.exit(1)
