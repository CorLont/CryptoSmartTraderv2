#!/usr/bin/env python3
"""
Daily Health Check Script for CryptoSmartTrader V2
Comprehensive system health validation and reporting
"""

import json
import time
import requests
import subprocess
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any


class DailyHealthChecker:
    def __init__(self):
        self.timestamp = datetime.now()
        self.results = {
            "timestamp": self.timestamp.isoformat(),
            "overall_health": "unknown",
            "services": {},
            "system": {},
            "application": {},
            "warnings": [],
            "errors": []
        }
    
    def check_service_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        services = {
            "dashboard": {"port": 5000, "path": "/_stcore/health"},
            "api": {"port": 8001, "path": "/health"},
            "metrics": {"port": 8000, "path": "/health"}
        }
        
        service_results = {}
        
        for service_name, config in services.items():
            url = f"http://localhost:{config['port']}{config['path']}"
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    service_results[service_name] = {
                        "status": "healthy",
                        "response_time": response.elapsed.total_seconds(),
                        "port": config["port"]
                    }
                else:
                    service_results[service_name] = {
                        "status": "unhealthy",
                        "status_code": response.status_code,
                        "port": config["port"]
                    }
                    self.results["errors"].append(f"{service_name} returned {response.status_code}")
            except Exception as e:
                service_results[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "port": config["port"]
                }
                self.results["errors"].append(f"{service_name} connection failed: {e}")
        
        return service_results
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Process count
            process_count = len(psutil.pids())
            
            system_results = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": round(memory_available_gb, 2),
                "disk_percent": round(disk_percent, 1),
                "disk_free_gb": round(disk_free_gb, 2),
                "process_count": process_count
            }
            
            # Warnings
            if cpu_percent > 80:
                self.results["warnings"].append(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 85:
                self.results["warnings"].append(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                self.results["warnings"].append(f"High disk usage: {disk_percent}%")
            if disk_free_gb < 1:
                self.results["errors"].append(f"Low disk space: {disk_free_gb}GB free")
            
            return system_results
            
        except Exception as e:
            self.results["errors"].append(f"System resource check failed: {e}")
            return {"error": str(e)}
    
    def check_application_health(self) -> Dict[str, Any]:
        """Check application-specific health metrics"""
        app_results = {}
        
        try:
            # Get detailed health from API
            response = requests.get("http://localhost:8001/health/detailed", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                app_results["health_score"] = health_data.get("application_health", {}).get("health_score", 0)
                app_results["trading_status"] = health_data.get("application_health", {}).get("trading_status", "unknown")
                app_results["uptime_seconds"] = health_data.get("uptime_seconds", 0)
                
                # Check health score
                health_score = app_results.get("health_score", 0)
                if health_score < 60:
                    self.results["errors"].append(f"Critical health score: {health_score}")
                elif health_score < 80:
                    self.results["warnings"].append(f"Low health score: {health_score}")
            else:
                self.results["errors"].append("Unable to get detailed health status")
                
        except Exception as e:
            self.results["errors"].append(f"Application health check failed: {e}")
        
        # Check log files for recent errors
        try:
            log_file = Path("logs/app.log")
            if log_file.exists():
                # Count errors in last 24 hours
                with open(log_file, 'r') as f:
                    recent_errors = 0
                    for line in f.readlines()[-1000:]:  # Check last 1000 lines
                        if "ERROR" in line:
                            recent_errors += 1
                
                app_results["recent_errors"] = recent_errors
                if recent_errors > 10:
                    self.results["warnings"].append(f"High error count: {recent_errors} errors in logs")
            else:
                self.results["warnings"].append("Log file not found")
                
        except Exception as e:
            self.results["warnings"].append(f"Log analysis failed: {e}")
        
        return app_results
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check if data is fresh and up-to-date"""
        data_results = {}
        
        try:
            # Check cache directories
            cache_dirs = ["cache", "data", "logs"]
            for dir_name in cache_dirs:
                dir_path = Path(dir_name)
                if dir_path.exists():
                    # Get newest file modification time
                    newest_file = max(dir_path.rglob("*"), key=lambda x: x.stat().st_mtime, default=None)
                    if newest_file:
                        mod_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
                        age_hours = (self.timestamp - mod_time).total_seconds() / 3600
                        data_results[f"{dir_name}_freshness_hours"] = round(age_hours, 1)
                        
                        if age_hours > 24:
                            self.results["warnings"].append(f"{dir_name} data is {age_hours:.1f} hours old")
            
            # Check if services are getting real-time data
            try:
                response = requests.get("http://localhost:8000/metrics", timeout=5)
                if response.status_code == 200:
                    metrics_text = response.text
                    # Look for trading metrics
                    if "cryptotrader_portfolio_value" in metrics_text:
                        data_results["metrics_available"] = True
                    else:
                        self.results["warnings"].append("Trading metrics not available")
                else:
                    self.results["warnings"].append("Metrics endpoint not accessible")
            except:
                self.results["warnings"].append("Unable to check metrics")
                
        except Exception as e:
            self.results["errors"].append(f"Data freshness check failed: {e}")
        
        return data_results
    
    def check_process_health(self) -> Dict[str, Any]:
        """Check if all required processes are running"""
        process_results = {}
        
        required_processes = ["streamlit", "uvicorn", "python"]
        running_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                if any(req_proc in cmdline.lower() for req_proc in required_processes):
                    running_processes.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "cmdline": cmdline[:100] + "..." if len(cmdline) > 100 else cmdline
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        process_results["running_processes"] = len(running_processes)
        process_results["process_details"] = running_processes
        
        if len(running_processes) < 3:
            self.results["warnings"].append(f"Expected 3+ processes, found {len(running_processes)}")
        
        return process_results
    
    def determine_overall_health(self) -> str:
        """Determine overall system health based on checks"""
        if self.results["errors"]:
            return "critical"
        elif len(self.results["warnings"]) > 5:
            return "degraded"
        elif self.results["warnings"]:
            return "warning"
        else:
            return "healthy"
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run all health checks and return results"""
        print("🏥 CryptoSmartTrader V2 - Daily Health Check")
        print("=" * 50)
        print(f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Run all checks
        print("🔍 Checking services...")
        self.results["services"] = self.check_service_health()
        
        print("🖥️  Checking system resources...")
        self.results["system"] = self.check_system_resources()
        
        print("📊 Checking application health...")
        self.results["application"] = self.check_application_health()
        
        print("🔄 Checking data freshness...")
        self.results["data"] = self.check_data_freshness()
        
        print("⚙️  Checking processes...")
        self.results["processes"] = self.check_process_health()
        
        # Determine overall health
        self.results["overall_health"] = self.determine_overall_health()
        
        return self.results
    
    def print_summary(self):
        """Print health check summary"""
        print("\n📋 Health Check Summary")
        print("-" * 30)
        
        # Overall status
        health_status = self.results["overall_health"]
        status_emoji = {
            "healthy": "✅",
            "warning": "⚠️",
            "degraded": "🔶",
            "critical": "❌"
        }
        print(f"Overall Health: {status_emoji.get(health_status, '❓')} {health_status.upper()}")
        
        # Service status
        print("\nService Status:")
        for service, details in self.results.get("services", {}).items():
            status = details.get("status", "unknown")
            emoji = "✅" if status == "healthy" else "❌"
            print(f"  {emoji} {service}: {status}")
        
        # System resources
        system = self.results.get("system", {})
        if system:
            print(f"\nSystem Resources:")
            print(f"  CPU: {system.get('cpu_percent', 0)}%")
            print(f"  Memory: {system.get('memory_percent', 0)}%")
            print(f"  Disk: {system.get('disk_percent', 0)}%")
            print(f"  Free Space: {system.get('disk_free_gb', 0)}GB")
        
        # Application health
        app = self.results.get("application", {})
        if app:
            print(f"\nApplication Health:")
            print(f"  Health Score: {app.get('health_score', 'unknown')}")
            print(f"  Trading Status: {app.get('trading_status', 'unknown')}")
            if app.get('uptime_seconds'):
                uptime_hours = app['uptime_seconds'] / 3600
                print(f"  Uptime: {uptime_hours:.1f} hours")
        
        # Warnings and errors
        if self.results["warnings"]:
            print(f"\n⚠️  Warnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"][:5]:  # Show first 5
                print(f"  - {warning}")
            if len(self.results["warnings"]) > 5:
                print(f"  ... and {len(self.results['warnings']) - 5} more")
        
        if self.results["errors"]:
            print(f"\n❌ Errors ({len(self.results['errors'])}):")
            for error in self.results["errors"][:5]:  # Show first 5
                print(f"  - {error}")
            if len(self.results["errors"]) > 5:
                print(f"  ... and {len(self.results['errors']) - 5} more")
        
        print()
    
    def save_results(self):
        """Save results to log file"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        log_file = log_dir / f"daily_health_{self.timestamp.strftime('%Y%m%d')}.json"
        with open(log_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Append summary to health log
        summary_file = log_dir / "health_summary.log"
        with open(summary_file, 'a') as f:
            f.write(f"{self.timestamp.isoformat()},{self.results['overall_health']},{len(self.results['warnings'])},{len(self.results['errors'])}\n")
        
        print(f"📁 Results saved to {log_file}")


def main():
    """Main execution function"""
    checker = DailyHealthChecker()
    
    try:
        # Run health check
        checker.run_health_check()
        
        # Print summary
        checker.print_summary()
        
        # Save results
        checker.save_results()
        
        # Exit with appropriate code
        if checker.results["overall_health"] in ["critical"]:
            return 2  # Critical issues
        elif checker.results["overall_health"] in ["degraded", "warning"]:
            return 1  # Warnings
        else:
            return 0  # Healthy
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return 3  # Script failure


if __name__ == "__main__":
    exit(main())