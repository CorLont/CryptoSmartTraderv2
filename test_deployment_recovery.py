"""
Deployment & Recovery System Test

Test process management, auto-restart, health probes and
recovery time objectives (0-to-healthy < 30s).
"""

import sys
import os
import time
import signal
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.deployment.process_manager import (
    ProcessManager,
    ProcessConfig,
    ProcessState,
    HealthStatus,
)
from cryptosmarttrader.deployment.health_checker import (
    HealthChecker,
    DependencyCheck,
    DependencyType,
    HealthLevel,
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DeploymentRecoveryTester:
    """
    Comprehensive test suite for deployment and recovery systems
    """

    def __init__(self):
        self.test_results = {}
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()

    def test_process_auto_restart(self) -> bool:
        """Test automatic process restart functionality"""

        logger.info("Testing process auto-restart...")

        try:
            # Create a simple test process that we can kill
            test_script = """
import time
import sys
print("Test process started", flush=True)
try:
    while True:
        print("Test process running...", flush=True)
        time.sleep(2)
except KeyboardInterrupt:
    print("Test process stopping", flush=True)
    sys.exit(0)
"""

            script_path = Path("test_process.py")
            script_path.write_text(test_script)

            # Configure test process
            config = ProcessConfig(
                name="test_process",
                command=sys.executable,
                args=[str(script_path)],
                auto_restart=True,
                max_restarts=5,
                restart_delay=2.0,
                exponential_backoff=False,
            )

            # Register and start process
            self.process_manager.register_process(config)
            start_success = self.process_manager.start_process("test_process")

            if not start_success:
                logger.error("Failed to start test process")
                return False

            # Wait for process to be running
            time.sleep(3)

            status = self.process_manager.get_process_status("test_process")
            if not status or status.state != ProcessState.RUNNING:
                logger.error("Process not running after start")
                return False

            initial_pid = status.pid
            logger.info(f"Test process started with PID {initial_pid}")

            # Kill the process to trigger auto-restart
            if initial_pid:
                logger.info("Killing test process to trigger restart...")
                os.kill(initial_pid, signal.SIGTERM)

            # Wait for restart
            restart_detected = False
            max_wait = 15  # 15 seconds max wait
            start_wait = time.time()

            while time.time() - start_wait < max_wait:
                status = self.process_manager.get_process_status("test_process")
                if status and status.state == ProcessState.RUNNING and status.pid != initial_pid:
                    restart_detected = True
                    logger.info(f"Process restarted with new PID {status.pid}")
                    break
                time.sleep(0.5)

            # Stop the process
            self.process_manager.stop_process("test_process")

            # Cleanup
            if script_path.exists():
                script_path.unlink()

            self.test_results["auto_restart"] = {
                "success": restart_detected,
                "initial_pid": initial_pid,
                "restart_detected": restart_detected,
            }

            return restart_detected

        except Exception as e:
            logger.error(f"Auto-restart test failed: {e}")
            return False

    def test_health_probe_real_dependencies(self) -> bool:
        """Test health probes with real dependency checks"""

        logger.info("Testing health probes with real dependencies...")

        try:
            # Add various dependency checks

            # 1. File system check
            self.health_checker.add_dependency_check(
                DependencyCheck(
                    name="test_write_dir",
                    type=DependencyType.FILE_SYSTEM,
                    description="Test write directory",
                    critical=True,
                    path="./test_data",
                )
            )

            # 2. Network port check (use a common port that should be closed)
            self.health_checker.add_dependency_check(
                DependencyCheck(
                    name="test_closed_port",
                    type=DependencyType.NETWORK_PORT,
                    description="Test closed port",
                    critical=False,
                    host="localhost",
                    port=9999,  # Should be closed
                    timeout_seconds=2.0,
                )
            )

            # 3. System resource check
            self.health_checker.add_dependency_check(
                DependencyCheck(
                    name="test_system_resources",
                    type=DependencyType.SYSTEM_RESOURCE,
                    description="System resource usage",
                    critical=True,
                    max_memory_percent=95.0,
                    max_cpu_percent=90.0,
                    max_disk_percent=95.0,
                )
            )

            # Run health checks
            results = self.health_checker.check_all_dependencies()

            logger.info("Health check results:")
            healthy_count = 0
            total_count = len(results)

            for name, result in results.items():
                logger.info(f"  {name}: {result.status.value} - {result.message}")
                if result.is_healthy:
                    healthy_count += 1

            # Get overall health report
            health_report = self.health_checker.get_health_report()
            overall_status = health_report["overall_status"]

            logger.info(f"Overall health status: {overall_status}")
            logger.info(f"Healthy dependencies: {healthy_count}/{total_count}")

            # Test should pass if:
            # 1. File system check passes (we can create the directory)
            # 2. System resource check passes (normal system)
            # 3. Port check fails as expected (port should be closed)

            file_system_healthy = results.get("test_write_dir", {}).is_healthy
            system_resources_healthy = results.get("test_system_resources", {}).is_healthy
            port_check_failed = not results.get("test_closed_port", {}).is_healthy

            success = file_system_healthy and system_resources_healthy and port_check_failed

            self.test_results["health_probes"] = {
                "success": success,
                "total_checks": total_count,
                "healthy_count": healthy_count,
                "overall_status": overall_status,
                "file_system_healthy": file_system_healthy,
                "system_resources_healthy": system_resources_healthy,
                "port_check_failed_as_expected": port_check_failed,
            }

            return success

        except Exception as e:
            logger.error(f"Health probe test failed: {e}")
            return False

    def test_exponential_backoff(self) -> bool:
        """Test exponential back-off for upstream failures"""

        logger.info("Testing exponential back-off...")

        try:
            # Create a process that will fail repeatedly
            failing_script = """
import sys
print("Failing process started", flush=True)
sys.exit(1)  # Always fail
"""

            script_path = Path("failing_process.py")
            script_path.write_text(failing_script)

            # Configure with exponential backoff
            config = ProcessConfig(
                name="failing_process",
                command=sys.executable,
                args=[str(script_path)],
                auto_restart=True,
                max_restarts=3,
                restart_delay=1.0,
                exponential_backoff=True,
                max_backoff_delay=10.0,
            )

            self.process_manager.register_process(config)

            # Track restart timing
            restart_times = []

            def monitor_restarts():
                last_restart_count = 0
                while len(restart_times) < 3:
                    status = self.process_manager.get_process_status("failing_process")
                    if status and status.restart_count > last_restart_count:
                        restart_times.append(time.time())
                        last_restart_count = status.restart_count
                        logger.info(f"Restart {status.restart_count} detected")
                    time.sleep(0.1)

            # Start monitoring in background
            monitor_thread = threading.Thread(target=monitor_restarts, daemon=True)
            monitor_thread.start()

            # Start the failing process
            start_time = time.time()
            self.process_manager.start_process("failing_process")

            # Wait for restarts to complete
            monitor_thread.join(timeout=30)

            # Calculate delays between restarts
            delays = []
            if len(restart_times) >= 2:
                for i in range(1, len(restart_times)):
                    delay = restart_times[i] - restart_times[i - 1]
                    delays.append(delay)
                    logger.info(f"Delay between restart {i} and {i + 1}: {delay:.1f}s")

            # Cleanup
            if script_path.exists():
                script_path.unlink()

            # Verify exponential backoff (each delay should be roughly double the previous)
            backoff_working = True
            if len(delays) >= 2:
                for i in range(1, len(delays)):
                    # Allow some tolerance (1.5x to 3x increase)
                    if delays[i] < delays[i - 1] * 1.5:
                        backoff_working = False
                        break
            else:
                backoff_working = False

            self.test_results["exponential_backoff"] = {
                "success": backoff_working,
                "restart_count": len(restart_times),
                "delays": delays,
                "backoff_working": backoff_working,
            }

            return backoff_working

        except Exception as e:
            logger.error(f"Exponential backoff test failed: {e}")
            return False

    def test_recovery_time_objective(self) -> bool:
        """Test RTO: 0-to-healthy < 30s after kill process"""

        logger.info("Testing Recovery Time Objective (0-to-healthy < 30s)...")

        try:
            # Create a simple HTTP server process for realistic health check
            server_script = """
import http.server
import socketserver
import threading
import time
import sys

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

PORT = 8899
print(f"Starting health server on port {PORT}", flush=True)

try:
    with socketserver.TCPServer(("", PORT), HealthHandler) as httpd:
        print(f"Health server running on port {PORT}", flush=True)
        httpd.serve_forever()
except KeyboardInterrupt:
    print("Health server stopping", flush=True)
    sys.exit(0)
"""

            script_path = Path("health_server.py")
            script_path.write_text(server_script)

            # Configure process with health check
            config = ProcessConfig(
                name="health_server",
                command=sys.executable,
                args=[str(script_path)],
                auto_restart=True,
                max_restarts=2,
                restart_delay=1.0,
                health_check_url="http://localhost:8899/health",
                health_check_interval=1.0,
                health_check_timeout=5.0,
            )

            # Add health check for the server
            self.health_checker.add_api_check(
                name="test_health_server",
                url="http://localhost:8899/health",
                critical=True,
                timeout=5.0,
            )

            self.process_manager.register_process(config)

            # Start process and wait for it to be healthy
            logger.info("Starting health server...")
            start_success = self.process_manager.start_process("health_server")

            if not start_success:
                logger.error("Failed to start health server")
                return False

            # Wait for server to be ready
            server_ready = False
            for _ in range(20):  # Wait up to 20 seconds
                try:
                    result = self.health_checker.check_dependency("test_health_server")
                    if result and result.is_healthy:
                        server_ready = True
                        logger.info("Health server is ready")
                        break
                except Exception:
                    pass
                time.sleep(1)

            if not server_ready:
                logger.error("Health server did not become ready")
                self.process_manager.stop_process("health_server")
                if script_path.exists():
                    script_path.unlink()
                return False

            # Record kill time and kill the process
            status = self.process_manager.get_process_status("health_server")
            if not status or not status.pid:
                logger.error("Cannot get server PID")
                return False

            logger.info(f"Killing health server (PID {status.pid}) to test recovery...")
            kill_time = time.time()

            try:
                os.kill(status.pid, signal.SIGKILL)  # Force kill for immediate death
            except ProcessLookupError:
                logger.warning("Process already dead")

            # Monitor for recovery to healthy state
            recovery_time = None
            max_wait = 45  # Allow 45 seconds total

            while time.time() - kill_time < max_wait:
                try:
                    # Check if process is healthy again
                    result = self.health_checker.check_dependency("test_health_server")
                    if result and result.is_healthy:
                        recovery_time = time.time() - kill_time
                        logger.info(f"Service recovered to healthy in {recovery_time:.1f} seconds")
                        break
                except Exception:
                    pass
                time.sleep(0.5)

            # Stop the process
            self.process_manager.stop_process("health_server")

            # Cleanup
            if script_path.exists():
                script_path.unlink()

            # RTO target: < 30 seconds
            rto_target = 30.0
            rto_met = recovery_time is not None and recovery_time < rto_target

            logger.info(f"RTO Test Result:")
            logger.info(
                f"  Recovery time: {recovery_time:.1f}s" if recovery_time else "  Recovery: FAILED"
            )
            logger.info(f"  RTO target: {rto_target}s")
            logger.info(f"  RTO met: {rto_met}")

            self.test_results["rto_test"] = {
                "success": rto_met,
                "recovery_time_seconds": recovery_time,
                "rto_target_seconds": rto_target,
                "rto_met": rto_met,
            }

            return rto_met

        except Exception as e:
            logger.error(f"RTO test failed: {e}")
            return False

    def test_write_directory_rotation(self) -> bool:
        """Test write directory with log rotation capabilities"""

        logger.info("Testing write directory with rotation...")

        try:
            # Create test write directory
            write_dir = Path("./test_write_data")
            write_dir.mkdir(exist_ok=True)

            # Test basic write capability
            test_file = write_dir / "test_log.txt"

            # Write some test data
            with open(test_file, "w") as f:
                f.write("Test log entry 1\n")
                f.write("Test log entry 2\n")

            # Verify read capability
            with open(test_file, "r") as f:
                content = f.read()

            write_success = "Test log entry 1" in content

            # Test rotation (rename current file, create new one)
            if test_file.exists():
                rotated_file = write_dir / f"test_log_{int(time.time())}.txt"
                test_file.rename(rotated_file)

                # Create new file
                with open(test_file, "w") as f:
                    f.write("New log entry after rotation\n")

                rotation_success = rotated_file.exists() and test_file.exists()
            else:
                rotation_success = False

            # Test directory space and permissions
            import shutil

            total, used, free = shutil.disk_usage(write_dir)
            free_space_gb = free / (1024**3)

            space_adequate = free_space_gb > 0.1  # At least 100MB free

            # Cleanup
            import shutil

            shutil.rmtree(write_dir, ignore_errors=True)

            overall_success = write_success and rotation_success and space_adequate

            logger.info(f"Write directory test results:")
            logger.info(f"  Write capability: {write_success}")
            logger.info(f"  Rotation capability: {rotation_success}")
            logger.info(f"  Free space: {free_space_gb:.2f} GB")
            logger.info(f"  Space adequate: {space_adequate}")

            self.test_results["write_directory"] = {
                "success": overall_success,
                "write_success": write_success,
                "rotation_success": rotation_success,
                "free_space_gb": free_space_gb,
                "space_adequate": space_adequate,
            }

            return overall_success

        except Exception as e:
            logger.error(f"Write directory test failed: {e}")
            return False

    def test_rpo_data_protection(self) -> bool:
        """Test RPO (Recovery Point Objective) data protection"""

        logger.info("Testing RPO data protection...")

        try:
            # RPO test: ensure data loss is minimized
            rpo_metrics = self.health_checker.get_rto_rpo_metrics()

            logger.info("RPO Metrics:")
            logger.info(f"  RPO target: {rpo_metrics['rpo_target_seconds']}s")
            logger.info(f"  Backup frequency: {rpo_metrics['backup_frequency_seconds']}s")
            logger.info(f"  Checkpoint frequency: {rpo_metrics['checkpoint_frequency_seconds']}s")

            # Verify backup frequency meets RPO target
            backup_meets_rpo = (
                rpo_metrics["backup_frequency_seconds"] <= rpo_metrics["rpo_target_seconds"]
            )
            checkpoint_meets_rpo = (
                rpo_metrics["checkpoint_frequency_seconds"] <= rpo_metrics["rpo_target_seconds"]
            )

            # Test data persistence capability
            persistence_dir = Path("./test_persistence")
            persistence_dir.mkdir(exist_ok=True)

            # Simulate critical data write
            critical_data_file = persistence_dir / "critical_data.json"
            test_data = {
                "timestamp": datetime.now().isoformat(),
                "critical_value": 12345,
                "status": "active",
            }

            import json

            with open(critical_data_file, "w") as f:
                json.dump(test_data, f)

            # Verify data can be recovered
            with open(critical_data_file, "r") as f:
                recovered_data = json.load(f)

            data_integrity = recovered_data["critical_value"] == test_data["critical_value"]

            # Cleanup
            import shutil

            shutil.rmtree(persistence_dir, ignore_errors=True)

            rpo_success = backup_meets_rpo and data_integrity

            logger.info(f"RPO test results:")
            logger.info(f"  Backup frequency meets RPO: {backup_meets_rpo}")
            logger.info(f"  Data integrity verified: {data_integrity}")
            logger.info(f"  RPO success: {rpo_success}")

            self.test_results["rpo_test"] = {
                "success": rpo_success,
                "backup_meets_rpo": backup_meets_rpo,
                "checkpoint_meets_rpo": checkpoint_meets_rpo,
                "data_integrity": data_integrity,
                "rpo_target_seconds": rpo_metrics["rpo_target_seconds"],
            }

            return rpo_success

        except Exception as e:
            logger.error(f"RPO test failed: {e}")
            return False

    def run_comprehensive_tests(self):
        """Run all deployment and recovery tests"""

        logger.info("=" * 60)
        logger.info("🧪 DEPLOYMENT & RECOVERY SYSTEM TESTS")
        logger.info("=" * 60)

        tests = [
            ("Process Auto-Restart", self.test_process_auto_restart),
            ("Health Probe Dependencies", self.test_health_probe_real_dependencies),
            ("Exponential Back-off", self.test_exponential_backoff),
            ("Recovery Time Objective", self.test_recovery_time_objective),
            ("Write Directory Rotation", self.test_write_directory_rotation),
            ("RPO Data Protection", self.test_rpo_data_protection),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n📋 {test_name}")
            try:
                success = test_func()
                if success:
                    logger.info(f"✅ {test_name} - PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} failed with exception: {e}")

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("🏁 DEPLOYMENT & RECOVERY TEST RESULTS")
        logger.info("=" * 60)

        for test_name, _ in tests:
            test_key = test_name.lower().replace(" ", "_").replace("-", "_")
            result = (
                "✅ PASSED"
                if self.test_results.get(test_key, {}).get("success", False)
                else "❌ FAILED"
            )
            logger.info(f"{test_name:<35} {result}")

        logger.info("=" * 60)
        logger.info(
            f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests / total_tests * 100:.1f}%)"
        )

        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - DEPLOYMENT SYSTEM READY")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")

        # Key metrics summary
        logger.info("\n📊 KEY DEPLOYMENT METRICS:")

        if "rto_test" in self.test_results:
            rto = self.test_results["rto_test"]
            recovery_time = rto.get("recovery_time_seconds")
            if recovery_time:
                logger.info(f"• Recovery Time: {recovery_time:.1f}s (target: 30s)")
            logger.info(f"• RTO Met: {rto.get('rto_met', False)}")

        if "auto_restart" in self.test_results:
            restart = self.test_results["auto_restart"]
            logger.info(f"• Auto-restart: {restart.get('restart_detected', False)}")

        if "health_probes" in self.test_results:
            health = self.test_results["health_probes"]
            logger.info(
                f"• Health checks: {health.get('healthy_count', 0)}/{health.get('total_checks', 0)}"
            )

        if "exponential_backoff" in self.test_results:
            backoff = self.test_results["exponential_backoff"]
            logger.info(f"• Exponential backoff: {backoff.get('backoff_working', False)}")

        return passed_tests == total_tests


def main():
    """Run deployment and recovery tests"""

    tester = DeploymentRecoveryTester()
    success = tester.run_comprehensive_tests()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
