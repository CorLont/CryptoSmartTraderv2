import psutil
import time
import threading
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from pathlib import Path

class HealthMonitor:
    """Automated health and completeness monitoring system"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.health_data = {}
        self.alerts = []
        self.monitoring_active = False
        self._lock = threading.Lock()
        self.health_file = Path("health_status.json")
        
        # Initialize monitoring thread
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_health_metrics()
                self._check_alerts()
                
                # Sleep based on config interval
                interval = self.config_manager.get("health_check_interval", 5) * 60
                time.sleep(interval)
                
            except Exception as e:
                self._add_alert("error", f"Health monitoring error: {str(e)}")
                time.sleep(60)  # Fallback sleep
    
    def _update_health_metrics(self):
        """Update all health metrics"""
        with self._lock:
            timestamp = datetime.now()
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate system health score
            system_score = self._calculate_system_score(cpu_percent, memory.percent, disk.percent)
            
            # Agent health (mock for now - would be populated by actual agents)
            agent_health = self._get_agent_health()
            
            # Data coverage (mock - would be calculated from actual data)
            data_coverage = self._calculate_data_coverage()
            
            self.health_data = {
                "timestamp": timestamp.isoformat(),
                "grade": self._score_to_grade(system_score),
                "score": system_score,
                "resources": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "agent_health": agent_health,
                "data_coverage": data_coverage,
                "uptime": self._get_uptime()
            }
            
            # Save to file
            self._save_health_data()
    
    def _calculate_system_score(self, cpu: float, memory: float, disk: float) -> float:
        """Calculate overall system health score"""
        # Weight different metrics
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory) 
        disk_score = max(0, 100 - disk)
        
        # Weighted average
        total_score = (cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2)
        return min(100, max(0, total_score))
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        elif score >= 50:
            return 'E'
        else:
            return 'F'
    
    def _get_agent_health(self) -> Dict[str, Dict]:
        """Get health status of all agents"""
        agents = self.config_manager.get("agents", {})
        agent_health = {}
        
        for agent_name, config in agents.items():
            if config.get("enabled", True):
                # Mock health data - in real implementation, agents would report their status
                agent_health[agent_name] = {
                    "status": "healthy",
                    "last_update": datetime.now().isoformat(),
                    "uptime": 24.5,  # hours
                    "error_rate": 0.01,
                    "processed_items": 1000
                }
        
        return agent_health
    
    def _calculate_data_coverage(self) -> float:
        """Calculate data coverage percentage"""
        # Mock calculation - in real implementation, would check actual data
        max_coins = self.config_manager.get("max_coins", 453)
        covered_coins = 450  # Mock value
        return (covered_coins / max_coins) * 100
    
    def _get_uptime(self) -> float:
        """Get system uptime in hours"""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            uptime = datetime.now() - boot_time
            return uptime.total_seconds() / 3600
        except:
            return 0.0
    
    def _check_alerts(self):
        """Check for alert conditions"""
        threshold = self.config_manager.get("alert_threshold", 80)
        
        # Check system resources
        if self.health_data.get("resources", {}).get("cpu_percent", 0) > 90:
            self._add_alert("warning", "High CPU usage detected")
        
        if self.health_data.get("resources", {}).get("memory_percent", 0) > 90:
            self._add_alert("warning", "High memory usage detected")
        
        if self.health_data.get("score", 100) < threshold:
            self._add_alert("error", f"System health below threshold: {self.health_data.get('score', 0):.1f}%")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add alert to the system"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message
        }
        
        with self._lock:
            self.alerts.append(alert)
            # Keep only last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
    
    def _save_health_data(self):
        """Save health data to file"""
        try:
            combined_data = {
                "health": self.health_data,
                "alerts": self.alerts[-10:]  # Save last 10 alerts
            }
            
            with open(self.health_file, 'w') as f:
                json.dump(combined_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving health data: {e}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health summary"""
        with self._lock:
            if not self.health_data:
                return {"grade": "F", "score": 0, "message": "Health data not available"}
            
            return {
                "grade": self.health_data.get("grade", "F"),
                "score": self.health_data.get("score", 0),
                "timestamp": self.health_data.get("timestamp"),
                "data_coverage": self.health_data.get("data_coverage", 0)
            }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        with self._lock:
            return {
                **self.health_data,
                "recent_alerts": self.alerts[-10:],
                "alert_count": len(self.alerts),
                "grade_change": 0,  # Mock - would track changes
                "score_change": 0,  # Mock - would track changes
                "active_agents": len([a for a in self.health_data.get("agent_health", {}).values() 
                                    if a.get("status") == "healthy"]),
                "coverage_change": 0  # Mock - would track changes
            }
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get alerts from the last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_alerts = []
            for alert in self.alerts:
                try:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time > cutoff:
                        recent_alerts.append(alert)
                except:
                    continue
            
            return recent_alerts
