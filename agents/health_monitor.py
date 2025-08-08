#!/usr/bin/env python3
"""
Health Monitor Agent - System Health Monitoring
Monitors all other agents and system resources
"""

import asyncio
import psutil
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from .base_agent import BaseAgent

class HealthMonitorAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("health_monitor", config)
        
        self.monitor_interval = self.config.get('monitor_interval', 30)
        self.status_file = Path("health_status.json")
        self.logs_dir = Path("logs")
        
        # Alert thresholds
        self.cpu_threshold = 80
        self.memory_threshold = 80
        self.disk_threshold = 90
    
    async def perform_health_check(self):
        """Monitor system health"""
        system_status = await self.get_system_status()
        
        # Check for critical issues
        if system_status['cpu_percent'] > self.cpu_threshold:
            self.logger.warning(f"High CPU usage: {system_status['cpu_percent']}%")
        
        if system_status['memory_percent'] > self.memory_threshold:
            self.logger.warning(f"High memory usage: {system_status['memory_percent']}%")
        
        if system_status['disk_percent'] > self.disk_threshold:
            self.logger.warning(f"High disk usage: {system_status['disk_percent']}%")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_status = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
        
        return system_status
    
    async def check_agent_health(self) -> Dict[str, Any]:
        """Check health of all agents via heartbeat files"""
        agent_status = {}
        
        for heartbeat_file in self.logs_dir.glob("*_heartbeat.json"):
            try:
                agent_name = heartbeat_file.stem.replace('_heartbeat', '')
                
                with open(heartbeat_file, 'r') as f:
                    heartbeat_data = json.load(f)
                
                last_heartbeat = datetime.fromisoformat(heartbeat_data['timestamp'])
                age_seconds = (datetime.now() - last_heartbeat).total_seconds()
                
                agent_status[agent_name] = {
                    "status": "healthy" if age_seconds < 120 else "unhealthy",
                    "last_heartbeat": heartbeat_data['timestamp'],
                    "age_seconds": age_seconds,
                    "pid": heartbeat_data.get('pid')
                }
                
            except Exception as e:
                agent_status[agent_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return agent_status
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        system_status = await self.get_system_status()
        agent_status = await self.check_agent_health()
        
        # Count healthy vs unhealthy agents
        healthy_agents = sum(1 for status in agent_status.values() 
                           if status.get("status") == "healthy")
        total_agents = len(agent_status)
        
        overall_health = "healthy"
        if healthy_agents < total_agents * 0.5:
            overall_health = "critical"
        elif healthy_agents < total_agents * 0.8:
            overall_health = "degraded"
        
        health_report = {
            "overall_health": overall_health,
            "system": system_status,
            "agents": {
                "healthy_count": healthy_agents,
                "total_count": total_agents,
                "details": agent_status
            },
            "alerts": self.generate_alerts(system_status, agent_status)
        }
        
        return health_report
    
    def generate_alerts(self, system_status: Dict, agent_status: Dict) -> List[str]:
        """Generate alerts based on health data"""
        alerts = []
        
        # System alerts
        if system_status['cpu_percent'] > self.cpu_threshold:
            alerts.append(f"High CPU usage: {system_status['cpu_percent']}%")
        
        if system_status['memory_percent'] > self.memory_threshold:
            alerts.append(f"High memory usage: {system_status['memory_percent']}%")
        
        if system_status['disk_percent'] > self.disk_threshold:
            alerts.append(f"Low disk space: {system_status['disk_percent']}% used")
        
        # Agent alerts
        for agent_name, status in agent_status.items():
            if status.get("status") != "healthy":
                alerts.append(f"Agent {agent_name} is {status.get('status', 'unknown')}")
        
        return alerts
    
    async def store_health_status(self, health_report: Dict[str, Any]):
        """Store health status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(health_report, f, indent=2)
            
            self.logger.info(f"Health status updated: {health_report['overall_health']}")
            
            # Log alerts
            for alert in health_report['alerts']:
                self.logger.warning(f"ALERT: {alert}")
                
        except Exception as e:
            self.logger.error(f"Failed to store health status: {e}")
    
    async def main_loop(self):
        """Main health monitoring loop"""
        self.logger.info("Starting health monitoring")
        
        while self.running:
            try:
                # Generate health report
                health_report = await self.generate_health_report()
                
                # Store status
                await self.store_health_status(health_report)
                
                # Wait for next check
                await asyncio.sleep(self.monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

def run():
    """Entry point for the health monitor agent"""
    from .base_agent import run_agent
    
    config = {
        'monitor_interval': 30,
        'health_check_interval': 60
    }
    
    run_agent(HealthMonitorAgent, config)

if __name__ == "__main__":
    run()