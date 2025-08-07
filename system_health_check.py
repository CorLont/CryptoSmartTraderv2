#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Complete System Health Check
Pre-installation verification script for hardware, dependencies, and system requirements
"""

import sys
import os
import platform
import subprocess
import socket
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

class SystemHealthChecker:
    """Comprehensive system health and dependency checker"""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'checks': []
        }
        
    def check(self, name: str, condition: bool, requirement: str, warning: bool = False) -> bool:
        """Add a check result"""
        status = "✅ PASS" if condition else ("⚠️ WARNING" if warning else "❌ FAIL")
        
        self.results['checks'].append({
            'name': name,
            'status': status,
            'requirement': requirement,
            'passed': condition,
            'warning': warning
        })
        
        if condition:
            self.results['passed'] += 1
        elif warning:
            self.results['warnings'] += 1
        else:
            self.results['failed'] += 1
            
        print(f"{status} {name}: {requirement}")
        return condition
    
    def run_all_checks(self) -> Dict:
        """Run comprehensive system health checks"""
        print("🔍 CryptoSmartTrader V2 - System Health Check")
        print("=" * 70)
        
        # A. Hardware & System Checks
        print("\n📊 HARDWARE & SYSTEM CHECKS")
        print("-" * 40)
        self._check_python_version()
        self._check_operating_system()
        self._check_ram()
        self._check_cpu()
        self._check_gpu_cuda()
        self._check_disk_space()
        
        # B. System Tools & Dependencies
        print("\n🔧 SYSTEM TOOLS & DEPENDENCIES")
        print("-" * 40)
        self._check_pip()
        self._check_build_tools()
        self._check_talib()
        self._check_redis()
        self._check_cuda_toolkit()
        
        # C. Network & API Connectivity
        print("\n🌐 NETWORK & API CONNECTIVITY")
        print("-" * 40)
        self._check_internet()
        self._check_api_endpoints()
        
        # D. Optional Advanced Features
        print("\n⚡ OPTIONAL ADVANCED FEATURES")
        print("-" * 40)
        self._check_advanced_ml_libraries()
        self._check_monitoring_tools()
        
        # Summary
        self._print_summary()
        return self.results
    
    def _check_python_version(self):
        """Check Python version requirement"""
        version = sys.version_info
        required = version >= (3, 9)
        self.check(
            "Python Version",
            required,
            f"Python 3.9+ required, found {version.major}.{version.minor}.{version.micro}"
        )
        
    def _check_operating_system(self):
        """Check operating system compatibility"""
        system = platform.system()
        is_windows = system == "Windows"
        self.check(
            "Operating System",
            True,  # Accept all OS but warn for non-Windows
            f"Found {system} {platform.release()}",
            warning=not is_windows
        )
        
    def _check_ram(self):
        """Check RAM requirements"""
        try:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
            sufficient = total_ram_gb >= 16
            self.check(
                "RAM Memory",
                sufficient,
                f"{total_ram_gb:.1f}GB total (16GB+ recommended for optimal performance)"
            )
        except ImportError:
            self.check("RAM Memory", False, "psutil not available for memory check")
            
    def _check_cpu(self):
        """Check CPU requirements"""
        cores = os.cpu_count()
        sufficient = cores >= 8
        self.check(
            "CPU Cores",
            sufficient,
            f"{cores} logical cores (8+ recommended for distributed processing)"
        )
        
    def _check_gpu_cuda(self):
        """Check GPU and CUDA availability"""
        # Check PyTorch CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                self.check(
                    "GPU CUDA Support",
                    True,
                    f"CUDA available: {gpu_count} GPU(s), Primary: {gpu_name}"
                )
            else:
                self.check(
                    "GPU CUDA Support",
                    False,
                    "CUDA not available - ML training will be slower",
                    warning=True
                )
        except ImportError:
            self.check(
                "GPU CUDA Support",
                False,
                "PyTorch not installed - cannot check CUDA",
                warning=True
            )
            
        # Check NVIDIA-ML Python (optional detailed GPU info)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = memory_info.total / (1024**3)
            self.check(
                "GPU Memory",
                True,
                f"{gpu_name}: {memory_gb:.1f}GB VRAM"
            )
        except ImportError:
            pass  # Optional check
        except Exception:
            pass  # GPU might not be available
            
    def _check_disk_space(self):
        """Check available disk space"""
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        sufficient = free_gb >= 20
        self.check(
            "Disk Space",
            sufficient,
            f"{free_gb:.1f}GB free (20GB+ recommended)"
        )
        
    def _check_pip(self):
        """Check pip availability and version"""
        try:
            result = subprocess.run(['pip', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                self.check("Pip Package Manager", True, f"Available: {result.stdout.strip()}")
            else:
                self.check("Pip Package Manager", False, "pip command failed")
        except FileNotFoundError:
            self.check("Pip Package Manager", False, "pip not found in PATH")
            
    def _check_build_tools(self):
        """Check for C++ build tools (Windows)"""
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['cl'], capture_output=True, text=True, shell=True)
                has_cl = True
            except:
                has_cl = False
                
            self.check(
                "C++ Build Tools",
                has_cl,
                "Visual C++ Build Tools available" if has_cl else "Visual Studio Build Tools not found",
                warning=not has_cl
            )
        else:
            # Check for gcc on other platforms
            try:
                subprocess.run(['gcc', '--version'], capture_output=True)
                self.check("C++ Build Tools", True, "GCC compiler available")
            except FileNotFoundError:
                self.check("C++ Build Tools", False, "GCC not found", warning=True)
                
    def _check_talib(self):
        """Check TA-Lib binary availability"""
        try:
            import talib
            # Test a simple function
            import numpy as np
            test_data = np.random.random(50)
            _ = talib.SMA(test_data, timeperiod=10)
            self.check("TA-Lib Binary", True, "TA-Lib installed and functional")
        except ImportError:
            self.check("TA-Lib Binary", False, "TA-Lib not installed - install binary first!")
        except Exception as e:
            self.check("TA-Lib Binary", False, f"TA-Lib error: {e}")
            
    def _check_redis(self):
        """Check Redis server availability"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', 6379))
            sock.close()
            
            if result == 0:
                self.check("Redis Server", True, "Redis running on localhost:6379")
            else:
                self.check(
                    "Redis Server",
                    False,
                    "Redis not running (optional for advanced queuing)",
                    warning=True
                )
        except Exception:
            self.check("Redis Server", False, "Cannot check Redis connection", warning=True)
            
    def _check_cuda_toolkit(self):
        """Check CUDA Toolkit installation"""
        try:
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version_line = [line for line in result.stdout.split('\n') if 'release' in line.lower()]
                version_info = version_line[0] if version_line else "Version unknown"
                self.check("CUDA Toolkit", True, f"NVCC available: {version_info.strip()}")
            else:
                self.check("CUDA Toolkit", False, "NVCC not found", warning=True)
        except FileNotFoundError:
            self.check("CUDA Toolkit", False, "CUDA Toolkit not installed", warning=True)
            
    def _check_internet(self):
        """Check internet connectivity"""
        try:
            urllib.request.urlopen('https://www.google.com', timeout=5)
            self.check("Internet Connection", True, "Internet connectivity verified")
        except Exception:
            self.check("Internet Connection", False, "No internet connection detected")
            
    def _check_api_endpoints(self):
        """Check cryptocurrency API endpoints"""
        apis = [
            ('Kraken API', 'https://api.kraken.com/0/public/Time'),
            ('CoinGecko API', 'https://api.coingecko.com/api/v3/ping'),
        ]
        
        for name, url in apis:
            try:
                urllib.request.urlopen(url, timeout=10)
                self.check(f"{name}", True, f"API endpoint reachable: {url}")
            except Exception as e:
                self.check(f"{name}", False, f"API unreachable: {str(e)}", warning=True)
                
    def _check_advanced_ml_libraries(self):
        """Check optional advanced ML libraries"""
        libraries = [
            ('PyTorch', 'torch'),
            ('Transformers', 'transformers'),
            ('Scikit-learn', 'sklearn'),
            ('XGBoost', 'xgboost'),
            ('Pandas', 'pandas'),
            ('NumPy', 'numpy'),
            ('Plotly', 'plotly'),
        ]
        
        for name, module in libraries:
            try:
                __import__(module)
                self.check(f"{name}", True, f"{name} library available")
            except ImportError:
                self.check(f"{name}", False, f"{name} not installed", warning=True)
                
    def _check_monitoring_tools(self):
        """Check monitoring and observability tools"""
        tools = [
            ('Prometheus Client', 'prometheus_client'),
            ('PSUtil', 'psutil'),
            ('AioHTTP', 'aiohttp'),
            ('SetProcTitle', 'setproctitle'),
        ]
        
        for name, module in tools:
            try:
                __import__(module)
                self.check(f"{name}", True, f"{name} available for monitoring")
            except ImportError:
                self.check(f"{name}", False, f"{name} not installed", warning=True)
                
    def _print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 70)
        print("📋 SYSTEM HEALTH CHECK SUMMARY")
        print("=" * 70)
        
        total_checks = len(self.results['checks'])
        passed = self.results['passed']
        failed = self.results['failed']
        warnings = self.results['warnings']
        
        print(f"Total Checks: {total_checks}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️ Warnings: {warnings}")
        
        success_rate = (passed / total_checks * 100) if total_checks > 0 else 0
        print(f"\nSuccess Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n🎉 EXCELLENT! System is ready for CryptoSmartTrader V2")
            print("✅ All critical requirements met")
            print("✅ Proceed with installation: run install_dependencies.bat")
        elif failed <= 2:
            print("\n✅ GOOD! System mostly ready with minor issues")
            print("⚠️ Address failed checks before proceeding")
            print("📋 Review failed items above")
        else:
            print("\n❌ ATTENTION REQUIRED! Multiple critical issues found")
            print("🔧 Resolve failed checks before installation")
            print("📞 Consider system upgrade or dependency installation")
            
        # Critical failures
        critical_failures = [
            check for check in self.results['checks'] 
            if not check['passed'] and not check['warning']
        ]
        
        if critical_failures:
            print(f"\n🚨 CRITICAL ISSUES TO RESOLVE:")
            for check in critical_failures:
                print(f"   ❌ {check['name']}: {check['requirement']}")
                
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if warnings > 0:
            print(f"   📋 Review {warnings} warnings for optimal performance")
        if any('GPU' in check['name'] and not check['passed'] for check in self.results['checks']):
            print(f"   🎮 Install NVIDIA drivers and CUDA Toolkit for GPU acceleration")
        if any('TA-Lib' in check['name'] and not check['passed'] for check in self.results['checks']):
            print(f"   📊 Install TA-Lib binary before other dependencies")
            
        print(f"\n🔗 NEXT STEPS:")
        print(f"   1. Address any critical failures above")
        print(f"   2. Run: install_dependencies.bat")
        print(f"   3. Start system: start_cryptotrader.bat")

def main():
    """Main execution function"""
    print("CryptoSmartTrader V2 - System Health Check")
    print("Verifying hardware, dependencies, and system requirements...")
    print()
    
    checker = SystemHealthChecker()
    results = checker.run_all_checks()
    
    # Save results to file
    with open('health_status.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Results saved to: health_status.json")
    return results['failed'] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)