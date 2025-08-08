#!/usr/bin/env python3
"""
Preinstall Check - Comprehensive System Validation
Enterprise system health check and dependency validation
"""

import sys
import os
import json
import asyncio
import subprocess
import importlib
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CheckResult:
    """Individual check result"""
    name: str
    status: str  # 'passed', 'failed', 'warning'
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PreinstallChecker:
    """Comprehensive system validation and dependency checking"""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.error_count = 0
        self.warning_count = 0
        
    async def run_all_checks(self) -> Tuple[int, List[CheckResult]]:
        """Run all system checks and return error count and results"""
        
        logger.info("🔍 Starting comprehensive preinstall checks")
        logger.info("=" * 60)
        
        # System checks
        await self.check_python_version()
        await self.check_operating_system()
        await self.check_disk_space()
        await self.check_memory()
        
        # Directory structure checks
        await self.check_directory_structure()
        await self.check_permissions()
        
        # Python dependencies checks
        await self.check_core_dependencies()
        await self.check_ml_dependencies()
        await self.check_optional_dependencies()
        
        # Configuration checks
        await self.check_environment_config()
        await self.check_logging_setup()
        
        # Network and connectivity checks
        await self.check_network_connectivity()
        await self.check_api_endpoints()
        
        # GPU and hardware checks
        await self.check_gpu_availability()
        await self.check_cuda_support()
        
        # Security checks
        await self.check_security_configuration()
        
        # Data pipeline checks
        await self.check_data_directories()
        await self.check_cache_system()
        
        # Generate summary
        await self.generate_check_summary()
        await self.create_daily_log_entry()
        
        logger.info(f"✅ Preinstall checks completed: {self.error_count} errors, {self.warning_count} warnings")
        
        return self.error_count, self.results
    
    async def check_python_version(self):
        """Check Python version compatibility"""
        try:
            version = sys.version_info
            required_major, required_minor = 3, 9
            
            if version.major == required_major and version.minor >= required_minor:
                self.add_result("python_version", "passed", 
                              f"Python {version.major}.{version.minor}.{version.micro} ✓")
            else:
                self.add_result("python_version", "failed",
                              f"Python {required_major}.{required_minor}+ required, got {version.major}.{version.minor}.{version.micro}")
        except Exception as e:
            self.add_result("python_version", "failed", f"Failed to check Python version: {e}")
    
    async def check_operating_system(self):
        """Check operating system compatibility"""
        try:
            system = platform.system()
            machine = platform.machine()
            
            supported_systems = ["Linux", "Darwin", "Windows"]
            
            if system in supported_systems:
                self.add_result("operating_system", "passed",
                              f"{system} {machine} ✓",
                              {"system": system, "machine": machine, "version": platform.version()})
            else:
                self.add_result("operating_system", "warning",
                              f"Untested OS: {system} {machine}")
        except Exception as e:
            self.add_result("operating_system", "failed", f"Failed to check OS: {e}")
    
    async def check_disk_space(self):
        """Check available disk space"""
        try:
            import shutil
            
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            # Convert to GB
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            min_required_gb = 5.0  # Minimum 5GB free space
            
            if free_gb >= min_required_gb:
                self.add_result("disk_space", "passed",
                              f"Free space: {free_gb:.1f}GB / {total_gb:.1f}GB ✓")
            else:
                self.add_result("disk_space", "failed",
                              f"Insufficient disk space: {free_gb:.1f}GB (minimum {min_required_gb}GB required)")
        except Exception as e:
            self.add_result("disk_space", "failed", f"Failed to check disk space: {e}")
    
    async def check_memory(self):
        """Check available memory"""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            min_required_gb = 4.0  # Minimum 4GB RAM
            
            if memory_gb >= min_required_gb:
                self.add_result("memory", "passed",
                              f"RAM: {memory_gb:.1f}GB (available: {available_gb:.1f}GB) ✓")
            else:
                self.add_result("memory", "warning",
                              f"Low memory: {memory_gb:.1f}GB (recommended: {min_required_gb}GB+)")
        except Exception as e:
            self.add_result("memory", "failed", f"Failed to check memory: {e}")
    
    async def check_directory_structure(self):
        """Check required directory structure"""
        try:
            required_dirs = [
                "agents", "ml", "eval", "orchestration", "dashboards",
                "configs", "logs", "exports", "scripts", "core",
                "data", "tests"
            ]
            
            missing_dirs = []
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            if not missing_dirs:
                self.add_result("directory_structure", "passed",
                              f"All {len(required_dirs)} required directories exist ✓")
            else:
                self.add_result("directory_structure", "warning",
                              f"Created missing directories: {', '.join(missing_dirs)}")
        except Exception as e:
            self.add_result("directory_structure", "failed", f"Failed to check directories: {e}")
    
    async def check_permissions(self):
        """Check file system permissions"""
        try:
            test_file = Path("test_write_permission.tmp")
            
            # Test write permission
            test_file.write_text("test")
            test_file.unlink()
            
            # Check log directory permissions
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            test_log = log_dir / "test_log_permission.tmp"
            test_log.write_text("test")
            test_log.unlink()
            
            self.add_result("permissions", "passed", "File system permissions ✓")
        except Exception as e:
            self.add_result("permissions", "failed", f"Permission error: {e}")
    
    async def check_core_dependencies(self):
        """Check core Python dependencies"""
        core_deps = [
            "pandas", "numpy", "plotly", "streamlit",
            "asyncio", "pathlib", "json", "datetime"
        ]
        
        missing_deps = []
        for dep in core_deps:
            try:
                importlib.import_module(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if not missing_deps:
            self.add_result("core_dependencies", "passed",
                          f"All {len(core_deps)} core dependencies available ✓")
        else:
            self.add_result("core_dependencies", "failed",
                          f"Missing core dependencies: {', '.join(missing_deps)}")
    
    async def check_ml_dependencies(self):
        """Check ML/AI dependencies"""
        ml_deps = [
            ("torch", "PyTorch"),
            ("sklearn", "scikit-learn"),
            ("xgboost", "XGBoost"),
            ("transformers", "HuggingFace Transformers")
        ]
        
        available_deps = []
        missing_deps = []
        
        for module_name, display_name in ml_deps:
            try:
                importlib.import_module(module_name)
                available_deps.append(display_name)
            except ImportError:
                missing_deps.append(display_name)
        
        if not missing_deps:
            self.add_result("ml_dependencies", "passed",
                          f"ML dependencies: {', '.join(available_deps)} ✓")
        else:
            self.add_result("ml_dependencies", "warning",
                          f"Available: {', '.join(available_deps)}. Missing: {', '.join(missing_deps)}")
    
    async def check_optional_dependencies(self):
        """Check optional dependencies"""
        optional_deps = [
            ("aiohttp", "Async HTTP"),
            ("tenacity", "Retry Logic"),
            ("prometheus_client", "Monitoring"),
            ("pynvml", "GPU Monitoring")
        ]
        
        available = []
        missing = []
        
        for module_name, display_name in optional_deps:
            try:
                importlib.import_module(module_name)
                available.append(display_name)
            except ImportError:
                missing.append(display_name)
        
        status = "passed" if len(available) >= len(optional_deps) // 2 else "warning"
        
        self.add_result("optional_dependencies", status,
                      f"Optional deps available: {len(available)}/{len(optional_deps)}")
    
    async def check_environment_config(self):
        """Check environment configuration"""
        try:
            sys.path.insert(0, str(Path.cwd()))
            from configs.system_settings import get_system_settings
            
            settings = get_system_settings()
            
            # Check critical settings
            issues = []
            
            if settings.environment.value not in ["development", "staging", "production"]:
                issues.append("Invalid environment setting")
            
            if settings.max_workers < 1 or settings.max_workers > 16:
                issues.append("Invalid max_workers setting")
            
            if not issues:
                self.add_result("environment_config", "passed",
                              f"Environment: {settings.environment.value} ✓")
            else:
                self.add_result("environment_config", "warning",
                              f"Config issues: {', '.join(issues)}")
        except Exception as e:
            self.add_result("environment_config", "failed", f"Config check failed: {e}")
    
    async def check_logging_setup(self):
        """Check logging system setup"""
        try:
            # Create daily log directory structure
            today_str = datetime.now().strftime("%Y%m%d")
            daily_log_dir = Path("logs/daily") / today_str
            daily_log_dir.mkdir(parents=True, exist_ok=True)
            
            # Test log file creation
            test_log_file = daily_log_dir / "preinstall_check.json"
            test_log_data = {
                "timestamp": datetime.now().isoformat(),
                "test": "preinstall_check",
                "status": "testing"
            }
            
            with open(test_log_file, 'w') as f:
                json.dump(test_log_data, f, indent=2)
            
            self.add_result("logging_setup", "passed",
                          f"Logging system configured, daily logs: {daily_log_dir} ✓")
        except Exception as e:
            self.add_result("logging_setup", "failed", f"Logging setup failed: {e}")
    
    async def check_network_connectivity(self):
        """Check basic network connectivity"""
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname("google.com")
            
            self.add_result("network_connectivity", "passed", "Network connectivity ✓")
        except Exception as e:
            self.add_result("network_connectivity", "warning", f"Network check failed: {e}")
    
    async def check_api_endpoints(self):
        """Check API endpoint accessibility"""
        try:
            import urllib.request
            
            # Test basic HTTP connectivity
            endpoints = [
                ("https://httpbin.org/status/200", "HTTP Test"),
                ("https://api.kraken.com/0/public/Time", "Kraken API"),
            ]
            
            available = []
            failed = []
            
            for url, name in endpoints:
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'CryptoSmartTrader/2.0'})
                    with urllib.request.urlopen(req, timeout=10) as response:
                        if response.status == 200:
                            available.append(name)
                        else:
                            failed.append(f"{name} ({response.status})")
                except Exception:
                    failed.append(name)
            
            if not failed:
                self.add_result("api_endpoints", "passed",
                              f"API endpoints accessible: {', '.join(available)} ✓")
            else:
                self.add_result("api_endpoints", "warning",
                              f"Failed endpoints: {', '.join(failed)}")
        except Exception as e:
            self.add_result("api_endpoints", "warning", f"API check failed: {e}")
    
    async def check_gpu_availability(self):
        """Check GPU availability"""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                
                self.add_result("gpu_availability", "passed",
                              f"GPU available: {gpu_name} ({gpu_count} devices) ✓",
                              {"gpu_count": gpu_count, "gpu_name": gpu_name})
            else:
                self.add_result("gpu_availability", "warning",
                              "No GPU available, will use CPU")
        except ImportError:
            self.add_result("gpu_availability", "warning",
                          "PyTorch not available, cannot check GPU")
        except Exception as e:
            self.add_result("gpu_availability", "warning", f"GPU check failed: {e}")
    
    async def check_cuda_support(self):
        """Check CUDA support"""
        try:
            import torch
            
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                self.add_result("cuda_support", "passed",
                              f"CUDA {cuda_version} ✓")
            else:
                self.add_result("cuda_support", "warning",
                              "CUDA not available")
        except ImportError:
            self.add_result("cuda_support", "warning",
                          "PyTorch not available, cannot check CUDA")
        except Exception as e:
            self.add_result("cuda_support", "warning", f"CUDA check failed: {e}")
    
    async def check_security_configuration(self):
        """Check security configuration"""
        try:
            # Check for .env file
            env_file = Path(".env")
            env_example = Path(".env.example")
            
            issues = []
            
            if not env_file.exists() and not env_example.exists():
                issues.append("No environment configuration found")
            
            # Check for secrets exposure (basic check)
            if env_file.exists():
                try:
                    env_content = env_file.read_text()
                    if "your_api_key_here" in env_content or "changeme" in env_content:
                        issues.append("Default placeholder values in .env")
                except:
                    pass
            
            if not issues:
                self.add_result("security_config", "passed", "Security configuration ✓")
            else:
                self.add_result("security_config", "warning",
                              f"Security issues: {', '.join(issues)}")
        except Exception as e:
            self.add_result("security_config", "warning", f"Security check failed: {e}")
    
    async def check_data_directories(self):
        """Check data storage directories"""
        try:
            data_dirs = [
                "data/batch_output",
                "data/historical", 
                "data/shadow_trading",
                "data/dashboard_exports"
            ]
            
            for dir_path in data_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            self.add_result("data_directories", "passed",
                          f"Data directories ready ({len(data_dirs)} created) ✓")
        except Exception as e:
            self.add_result("data_directories", "failed", f"Data directory setup failed: {e}")
    
    async def check_cache_system(self):
        """Check caching system"""
        try:
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Test cache write/read
            test_cache_file = cache_dir / "test_cache.json"
            test_data = {"test": "cache_system", "timestamp": datetime.now().isoformat()}
            
            with open(test_cache_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(test_cache_file, 'r') as f:
                loaded_data = json.load(f)
            
            test_cache_file.unlink()  # Cleanup
            
            self.add_result("cache_system", "passed", "Cache system functional ✓")
        except Exception as e:
            self.add_result("cache_system", "failed", f"Cache system failed: {e}")
    
    def add_result(self, name: str, status: str, message: str, details: Optional[Dict] = None):
        """Add a check result"""
        result = CheckResult(name, status, message, details)
        self.results.append(result)
        
        if status == "failed":
            self.error_count += 1
            logger.error(f"❌ {name}: {message}")
        elif status == "warning":
            self.warning_count += 1
            logger.warning(f"⚠️  {name}: {message}")
        else:
            logger.info(f"✅ {name}: {message}")
    
    async def generate_check_summary(self):
        """Generate comprehensive check summary"""
        
        total_checks = len(self.results)
        passed_checks = len([r for r in self.results if r.status == "passed"])
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": self.error_count,
            "warnings": self.warning_count,
            "success_rate": (passed_checks / total_checks * 100) if total_checks > 0 else 0,
            "system_ready": self.error_count == 0
        }
        
        # Save summary
        summary_file = Path("logs/preinstall_check_summary.json")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Check Summary: {passed_checks}/{total_checks} passed, {self.error_count} errors, {self.warning_count} warnings")
        logger.info(f"💾 Summary saved: {summary_file}")
    
    async def create_daily_log_entry(self):
        """Create daily log entry for preinstall check"""
        
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            daily_log_dir = Path("logs/daily") / today_str
            daily_log_dir.mkdir(parents=True, exist_ok=True)
            
            log_entry = {
                "log_type": "preinstall_check",
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_checks": len(self.results),
                    "errors": self.error_count,
                    "warnings": self.warning_count,
                    "system_ready": self.error_count == 0
                },
                "checks": [asdict(result) for result in self.results]
            }
            
            log_file = daily_log_dir / f"preinstall_check_{datetime.now().strftime('%H%M%S')}.json"
            
            # Convert datetime objects to ISO strings for JSON serialization
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
            
            with open(log_file, 'w') as f:
                json.dump(log_entry, f, indent=2, default=json_serializer)
            
            logger.info(f"📝 Daily log created: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to create daily log: {e}")

async def main():
    """Main entry point for preinstall checks"""
    
    print("🚀 CryptoSmartTrader V2 - Preinstall System Check")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    checker = PreinstallChecker()
    
    try:
        error_count, results = await checker.run_all_checks()
        
        print()
        print("=" * 60)
        print(f"🏁 PREINSTALL CHECK COMPLETED")
        print(f"   Total checks: {len(results)}")
        print(f"   Errors: {error_count}")
        print(f"   Warnings: {checker.warning_count}")
        print(f"   Status: {'✅ READY' if error_count == 0 else '❌ NOT READY'}")
        print("=" * 60)
        
        return error_count
        
    except Exception as e:
        logger.error(f"❌ Preinstall check failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)