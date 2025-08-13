import logging

logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
Multi-Service Starter for Replit - Start all services with proper coordination
"""

import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path


class MultiServiceManager:
    """Manage multiple services for CryptoSmartTrader V2"""

    def __init__(self):
        self.processes = {}
        self.running = True
        self.repo_root = Path.cwd()

        # Service configurations
        self.services = {
            "api": {
                "command": ["python", "api/health_endpoint.py"],
                "port": 8001,
                "health_path": "/health",
                "description": "Health & Status API",
            },
            "metrics": {
                "command": ["python", "metrics/metrics_server.py"],
                "port": 8000,
                "health_path": "/health",
                "description": "Prometheus Metrics Server",
            },
            "dashboard": {
                "command": [
                    "streamlit",
                    "run",
                    "app_fixed_all_issues.py",
                    "--server.port",
                    "5000",
                    "--server.address",
                    "0.0.0.0",
                ],
                "port": 5000,
                "health_path": "/_stcore/health",
                "description": "Main Trading Dashboard",
            },
        }

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""

        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, shutting down services...")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start_service(self, name: str, config: dict):
        """Start a single service"""
        try:
            print(f"🚀 Starting {config['description']} on port {config['port']}...")

            process = subprocess.Popen(
                config["command"],
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            self.processes[name] = {"process": process, "config": config, "start_time": time.time()}

            # Start log monitoring thread
            threading.Thread(
                target=self.monitor_service_logs, args=(name, process), daemon=True
            ).start()

            print(f"✅ {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start {name}: {e}")
            return False

    def monitor_service_logs(self, name: str, process: subprocess.Popen):
        """Monitor service logs and print with prefix"""
        while self.running and process.poll() is None:
            try:
                # Read stdout
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        print(f"[{name.upper()}] {line.strip()}")

                # Read stderr
                if process.stderr:
                    line = process.stderr.readline()
                    if line:
                        print(f"[{name.upper()}:ERR] {line.strip()}")

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

    def check_service_health(self, name: str) -> bool:
        """Check if service is healthy"""
        if name not in self.processes:
            return False

        process_info = self.processes[name]
        process = process_info["process"]
        config = process_info["config"]

        # Check if process is still running
        if process.poll() is not None:
            return False

        # Wait a bit for service to start up
        if time.time() - process_info["start_time"] < 10:
            return True  # Give it time to start

        # Check HTTP health endpoint
        try:
            import requests

            url = f"http://localhost:{config['port']}{config['health_path']}"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except Exception as e:
            return False

    def wait_for_services(self, timeout: int = 60):
        """Wait for all services to become healthy"""
        print("\n⏳ Waiting for services to become ready...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True

            for name in self.services.keys():
                if not self.check_service_health(name):
                    all_healthy = False
                    break

            if all_healthy:
                print("✅ All services are healthy!")
                return True

            time.sleep(2)

        print("⚠️  Timeout waiting for services to become ready")
        return False

    def show_service_status(self):
        """Show status of all services"""
        print("\n📊 Service Status:")
        print("-" * 60)

        for name, config in self.services.items():
            if name in self.processes:
                process = self.processes[name]["process"]
                is_running = process.poll() is None
                is_healthy = self.check_service_health(name)

                status = (
                    "🟢 HEALTHY" if is_healthy else ("🟡 STARTING" if is_running else "🔴 STOPPED")
                )
                print(f"{config['description']:25} | Port {config['port']} | {status}")
            else:
                print(f"{config['description']:25} | Port {config['port']} | 🔴 NOT STARTED")

        print("-" * 60)

    def show_urls(self):
        """Show service URLs"""
        print("\n🌐 Service URLs:")
        print("-" * 40)
        print(f"Dashboard:  http://localhost:5000")
        print(f"API:        http://localhost:8001")
        print(f"Metrics:    http://localhost:8000")
        print(f"API Docs:   http://localhost:8001/api/docs")
        print("-" * 40)

    def start_all_services(self):
        """Start all services"""
        print("🚀 CryptoSmartTrader V2 Multi-Service Startup")
        print("=" * 50)

        # Start services in order
        success_count = 0
        for name, config in self.services.items():
            if self.start_service(name, config):
                success_count += 1
                time.sleep(2)  # Small delay between starts

        if success_count == 0:
            print("❌ No services started successfully")
            return False

        # Wait for services to become ready
        self.wait_for_services()

        # Show status
        self.show_service_status()
        self.show_urls()

        return True

    def shutdown(self):
        """Shutdown all services gracefully"""
        self.running = False

        print("\n🛑 Shutting down services...")

        for name, process_info in self.processes.items():
            process = process_info["process"]

            if process.poll() is None:
                print(f"Stopping {name}...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"Force killing {name}...")
                    process.kill()
                except Exception:
                    pass

        print("✅ All services stopped")

    def run(self):
        """Main run loop"""
        self.setup_signal_handlers()

        if not self.start_all_services():
            return 1

        try:
            print("\n✅ All services running! Press Ctrl+C to stop")
            print("💡 Monitor logs above, check service URLs, or view Replit Ports panel")

            # Keep main thread alive and monitor services
            while self.running:
                time.sleep(10)

                # Check if any service died
                for name, process_info in list(self.processes.items()):
                    process = process_info["process"]
                    if process.poll() is not None:
                        print(f"⚠️  Service {name} stopped unexpectedly")
                        # Could implement restart logic here

        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")

        finally:
            self.shutdown()

        return 0


if __name__ == "__main__":
    manager = MultiServiceManager()
    sys.exit(manager.run())
