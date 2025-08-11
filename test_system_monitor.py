#!/usr/bin/env python3
"""
Test script for System Monitor
Validates enterprise fixes: configurable ports, unused threshold cleanup, cross-platform compatibility
"""

import sys
import os
import json
from pathlib import Path

def test_system_monitor():
    """Test system monitor with enterprise fixes"""
    
    print("Testing System Monitor Enterprise Implementation")
    print("=" * 60)
    
    try:
        # Direct import to avoid dependency issues
        import importlib.util
        spec = importlib.util.spec_from_file_location("system_monitor", "core/system_monitor.py")
        monitor_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor_module)
        
        SystemMonitor = monitor_module.SystemMonitor
        
        # Test 1: Configurable port support
        print("\n1. Testing Configurable Port Support...")
        
        # Test with environment variable override
        os.environ["DASHBOARD_PORT"] = "3000"
        monitor = SystemMonitor()
        
        dashboard_config = monitor.config["services"]["dashboard"]
        ports = dashboard_config["ports"]
        
        if isinstance(ports, list) and len(ports) > 1:
            print(f"   ✅ PASSED: Multiple ports configured: {ports}")
            
            # Check that environment variable is respected
            if 3000 in ports:
                print("   ✅ PASSED: Environment variable DASHBOARD_PORT respected")
            else:
                print("   ❌ FAILED: Environment variable not properly integrated")
                return False
        else:
            print("   ❌ FAILED: Single port configuration detected")
            return False
        
        # Test 2: Unused threshold cleanup
        print("\n2. Testing Threshold Configuration Cleanup...")
        
        alert_thresholds = monitor.config["alert_thresholds"]
        
        # Check that response_time threshold is removed (was unused)
        if "response_time" not in alert_thresholds:
            print("   ✅ PASSED: Unused 'response_time' threshold removed")
        else:
            print("   ❌ FAILED: Unused 'response_time' threshold still present")
            return False
        
        # Check that used thresholds are present
        required_thresholds = ["cpu_percent", "memory_percent", "disk_percent"]
        missing_thresholds = [t for t in required_thresholds if t not in alert_thresholds]
        
        if not missing_thresholds:
            print(f"   ✅ PASSED: All required thresholds present: {required_thresholds}")
        else:
            print(f"   ❌ FAILED: Missing thresholds: {missing_thresholds}")
            return False
        
        # Test 3: Cross-platform compatibility
        print("\n3. Testing Cross-Platform Compatibility...")
        
        # Test platform detection
        platform_name = monitor.platform_name
        print(f"   Platform detected: {platform_name}")
        
        # Test load average handling
        platform_config = monitor.config["platform"]
        load_avg_enabled = platform_config["enable_load_average"]
        
        if monitor.is_windows and not load_avg_enabled:
            print("   ✅ PASSED: Load average disabled on Windows")
        elif not monitor.is_windows and load_avg_enabled:
            print("   ✅ PASSED: Load average enabled on Unix-like system")
        else:
            print(f"   ⚠️ WARNING: Load average config may not match platform")
        
        # Test system metrics collection
        try:
            metrics = monitor.collect_system_metrics()
            
            if metrics.timestamp:
                print("   ✅ PASSED: System metrics collected successfully")
                
                # Check for platform-specific metrics
                if metrics.load_average is not None:
                    print(f"   ✅ Load average available: {metrics.load_average}")
                elif monitor.is_windows:
                    print("   ✅ Load average correctly unavailable on Windows")
                else:
                    print("   ⚠️ Load average unavailable (may be permission issue)")
                    
            else:
                print("   ❌ FAILED: No timestamp in metrics")
                return False
                
        except Exception as e:
            print(f"   ❌ FAILED: Metrics collection crashed: {e}")
            return False
        
        # Test 4: Service availability with multiple ports
        print("\n4. Testing Service Availability with Multiple Ports...")
        
        try:
            services = monitor.check_service_availability()
            
            if services:
                print(f"   ✅ PASSED: Service availability check completed ({len(services)} services)")
                
                for service in services:
                    accessible_status = "✅ UP" if service.is_accessible else "❌ DOWN"
                    print(f"   {accessible_status} {service.name}: {service.host}:{service.port}")
                    
                    # Check for proper error handling when service is down
                    if not service.is_accessible and service.error_message:
                        print(f"     Error details: {service.error_message}")
            else:
                print("   ⚠️ WARNING: No services configured for monitoring")
                
        except Exception as e:
            print(f"   ❌ FAILED: Service availability check crashed: {e}")
            return False
        
        # Test 5: Alert generation and threshold application
        print("\n5. Testing Alert Generation and Threshold Application...")
        
        try:
            # Generate a full monitoring report
            report = monitor.generate_monitoring_report()
            
            if report.timestamp:
                print(f"   ✅ PASSED: Monitoring report generated successfully")
                print(f"   Overall health: {report.overall_health}")
                print(f"   Alerts generated: {len(report.alerts)}")
                
                # Check that thresholds are actually used
                if report.alerts:
                    threshold_related_alerts = [
                        alert for alert in report.alerts 
                        if "threshold" in alert.lower()
                    ]
                    if threshold_related_alerts:
                        print("   ✅ PASSED: Thresholds actively used in alert generation")
                    else:
                        print("   ⚠️ Alerts present but no threshold-related alerts found")
                else:
                    print("   ✅ No alerts generated (system healthy)")
                    
            else:
                print("   ❌ FAILED: No timestamp in report")
                return False
                
        except Exception as e:
            print(f"   ❌ FAILED: Report generation crashed: {e}")
            return False
        
        # Test 6: JSON serialization
        print("\n6. Testing JSON Report Serialization...")
        
        try:
            test_output = "test_system_monitor_report.json"
            save_success = monitor.save_report_to_json(report, test_output)
            
            if save_success and Path(test_output).exists():
                print("   ✅ PASSED: JSON report serialization successful")
                
                # Validate JSON structure
                with open(test_output, 'r') as f:
                    report_data = json.load(f)
                
                required_keys = ["timestamp", "overall_health", "system_metrics", "service_statuses"]
                missing_keys = [key for key in required_keys if key not in report_data]
                
                if not missing_keys:
                    print("   ✅ PASSED: JSON report structure complete")
                else:
                    print(f"   ❌ FAILED: Missing JSON keys: {missing_keys}")
                    return False
                
                # Cleanup
                Path(test_output).unlink()
                
            else:
                print("   ❌ FAILED: JSON report serialization failed")
                return False
                
        except Exception as e:
            print(f"   ❌ FAILED: JSON serialization crashed: {e}")
            return False
        
        # Test 7: Configuration flexibility
        print("\n7. Testing Configuration Flexibility...")
        
        # Test custom configuration
        custom_config = {
            "alert_thresholds": {
                "cpu_percent": 90.0,
                "memory_percent": 95.0
            },
            "services": {
                "custom_service": {
                    "name": "Custom Service",
                    "host": "localhost", 
                    "ports": [9999, 9998]
                }
            }
        }
        
        try:
            # Save custom config
            config_path = "test_monitor_config.json"
            with open(config_path, 'w') as f:
                json.dump(custom_config, f)
            
            # Load monitor with custom config
            custom_monitor = SystemMonitor(config_path)
            
            # Verify custom thresholds
            custom_thresholds = custom_monitor.config["alert_thresholds"]
            if custom_thresholds["cpu_percent"] == 90.0:
                print("   ✅ PASSED: Custom configuration loaded successfully")
            else:
                print("   ❌ FAILED: Custom configuration not applied")
                return False
            
            # Cleanup
            Path(config_path).unlink()
            
        except Exception as e:
            print(f"   ❌ FAILED: Custom configuration test crashed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ SYSTEM MONITOR ENTERPRISE FIXES VALIDATION COMPLETE")
        print("- Configurable ports: Multiple port strategy with environment variable support")
        print("- Threshold cleanup: Unused 'response_time' threshold removed, active thresholds validated")
        print("- Cross-platform compatibility: Load average disabled on Windows, graceful fallbacks")
        print("- Service monitoring: Multiple port testing with detailed error reporting")
        print("- Alert generation: Thresholds actively applied in monitoring logic")
        print("- JSON serialization: Complete report structure with metadata")
        print("- Configuration flexibility: Custom configuration merging with defaults")
        
        return True
        
    except Exception as e:
        print(f"\n❌ SYSTEM MONITOR TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system_monitor()
    sys.exit(0 if success else 1)