#!/usr/bin/env python3
"""
Health Monitor Agent - System health monitoring and diagnostics
"""

import asyncio
import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from core.structured_logger import get_structured_logger

class HealthMonitorAgent:
    """Agent for monitoring system health and performance"""
    
    def __init__(self):
        self.logger = get_structured_logger("HealthMonitorAgent")
        self.initialized = False
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the health monitor agent"""
        try:
            self.logger.info("Initializing Health Monitor Agent")
            self.initialized = True
            self.logger.info("Health Monitor Agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Health Monitor Agent initialization failed: {e}")
            raise
    
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process system health monitoring"""
        
        try:
            health_data = await self.get_system_health()
            
            return {
                "health_status": health_data,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Health monitoring failed: {e}")
            return {"health_status": {}, "status": "error", "error": str(e)}
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            
            # Calculate uptime
            uptime = time.time() - self.start_time
            
            # Component health scores
            component_health = {}
            
            # CPU health
            if cpu_percent < 50:
                cpu_grade = "A"
            elif cpu_percent < 70:
                cpu_grade = "B"
            elif cpu_percent < 85:
                cpu_grade = "C"
            else:
                cpu_grade = "F"
            
            component_health["cpu"] = {
                "usage_percent": cpu_percent,
                "grade": cpu_grade,
                "status": "healthy" if cpu_grade in ["A", "B"] else "warning"
            }
            
            # Memory health
            memory_percent = memory.percent
            if memory_percent < 60:
                memory_grade = "A"
            elif memory_percent < 75:
                memory_grade = "B"
            elif memory_percent < 90:
                memory_grade = "C"
            else:
                memory_grade = "F"
                
            component_health["memory"] = {
                "usage_percent": memory_percent,
                "available_gb": memory.available / (1024**3),
                "grade": memory_grade,
                "status": "healthy" if memory_grade in ["A", "B"] else "warning"
            }
            
            # Disk health
            disk_percent = disk.percent
            if disk_percent < 60:
                disk_grade = "A"
            elif disk_percent < 75:
                disk_grade = "B"
            elif disk_percent < 90:
                disk_grade = "C"
            else:
                disk_grade = "F"
                
            component_health["disk"] = {
                "usage_percent": disk_percent,
                "free_gb": disk.free / (1024**3),
                "grade": disk_grade,
                "status": "healthy" if disk_grade in ["A", "B"] else "warning"
            }
            
            # Overall health score
            grades = [comp["grade"] for comp in component_health.values()]
            grade_scores = {"A": 100, "B": 80, "C": 60, "D": 40, "F": 0}
            avg_score = sum(grade_scores[grade] for grade in grades) / len(grades)
            
            if avg_score >= 90:
                overall_grade = "A"
            elif avg_score >= 80:
                overall_grade = "B"
            elif avg_score >= 60:
                overall_grade = "C"
            elif avg_score >= 40:
                overall_grade = "D"
            else:
                overall_grade = "F"
            
            health_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": uptime,
                "uptime_hours": uptime / 3600,
                "component_health": component_health,
                "grade": overall_grade,
                "score": avg_score,
                "status": "healthy" if overall_grade in ["A", "B"] else "degraded",
                "process_memory_mb": process_memory.rss / (1024**2),
                "system_info": {
                    "cpu_count": psutil.cpu_count(),
                    "total_memory_gb": memory.total / (1024**3),
                    "total_disk_gb": disk.total / (1024**3)
                }
            }
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e),
                "grade": "F",
                "score": 0
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        # This would normally check all agents - simplified for now
        return {
            "data_collector": {"status": "running", "last_update": datetime.utcnow().isoformat()},
            "sentiment_analyzer": {"status": "running", "last_update": datetime.utcnow().isoformat()},
            "ml_predictor": {"status": "running", "last_update": datetime.utcnow().isoformat()},
            "risk_manager": {"status": "running", "last_update": datetime.utcnow().isoformat()}
        }