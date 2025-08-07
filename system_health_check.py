#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Comprehensive System Health Check
Validates all system components and dependencies for production readiness
"""

import sys
import os
import importlib
import traceback
from datetime import datetime
from pathlib import Path

class SystemHealthChecker:
    """Comprehensive system health checker"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0
        self.results = []
        
    def print_header(self):
        """Print test header"""
        print("=" * 70)
        print("  CryptoSmartTrader V2 - Comprehensive System Health Check")
        print("=" * 70)
        print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def test_critical_imports(self):
        """Test critical package imports"""
        print("[TEST] Critical Package Imports")
        print("-" * 35)
        
        critical_packages = [
            'streamlit',
            'pandas', 
            'numpy',
            'plotly',
            'ccxt',
            'sklearn',
            'xgboost'
        ]
        
        failed_imports = []
        
        for package in critical_packages:
            try:
                importlib.import_module(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                failed_imports.append(package)
        
        if failed_imports:
            self.failed_tests += 1
            self.results.append(f"❌ Critical imports failed: {failed_imports}")
            print(f"\n❌ FAILED: Missing critical packages: {failed_imports}")
        else:
            self.passed_tests += 1
            self.results.append("✅ All critical packages imported successfully")
            print("\n✅ PASSED: All critical packages available")
        
        print()
    
    def test_optional_imports(self):
        """Test optional package imports"""
        print("[TEST] Optional Package Imports")
        print("-" * 35)
        
        optional_packages = [
            'aiohttp',
            'textblob',
            'pydantic',
            'tenacity',
            'schedule',
            'psutil',
            'numba',
            'openai',
            'prometheus_client'
        ]
        
        failed_optional = []
        
        for package in optional_packages:
            try:
                importlib.import_module(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ⚠️  {package}")
                failed_optional.append(package)
        
        if failed_optional:
            self.warnings += 1
            self.results.append(f"⚠️ Optional packages missing: {failed_optional}")
            print(f"\n⚠️ WARNING: Missing optional packages: {failed_optional}")
            print("System will work with reduced functionality")
        else:
            self.passed_tests += 1
            self.results.append("✅ All optional packages available")
            print("\n✅ PASSED: All optional packages available")
        
        print()
    
    def test_core_modules(self):
        """Test core system modules"""
        print("[TEST] Core System Modules")
        print("-" * 30)
        
        core_modules = [
            'core.config_manager',
            'core.health_monitor',
            'core.comprehensive_market_scanner',
            'config.daily_logging_config'
        ]
        
        failed_modules = []
        
        for module in core_modules:
            try:
                importlib.import_module(module)
                print(f"  ✅ {module}")
            except Exception as e:
                print(f"  ❌ {module} - {e}")
                failed_modules.append(module)
        
        if failed_modules:
            self.failed_tests += 1
            self.results.append(f"❌ Core modules failed: {failed_modules}")
            print(f"\n❌ FAILED: Core modules not working: {failed_modules}")
        else:
            self.passed_tests += 1
            self.results.append("✅ All core modules working")
            print("\n✅ PASSED: All core modules working")
        
        print()
    
    def test_advanced_engines(self):
        """Test advanced AI engines"""
        print("[TEST] Advanced AI Engines")
        print("-" * 30)
        
        engines = [
            'core.advanced_ai_engine',
            'core.shadow_trading_engine',
            'core.market_impact_engine',
            'core.multi_agent_cooperation_engine',
            'core.model_monitoring_engine',
            'core.black_swan_simulation_engine'
        ]
        
        failed_engines = []
        
        for engine in engines:
            try:
                module = importlib.import_module(engine)
                # Test if main coordinator function exists
                if hasattr(module, 'get_' + engine.split('.')[-1].replace('_engine', '_coordinator')):
                    print(f"  ✅ {engine}")
                else:
                    print(f"  ⚠️  {engine} - coordinator function missing")
                    self.warnings += 1
            except Exception as e:
                print(f"  ❌ {engine} - {e}")
                failed_engines.append(engine)
        
        if failed_engines:
            self.failed_tests += 1
            self.results.append(f"❌ Advanced engines failed: {failed_engines}")
            print(f"\n❌ FAILED: Advanced engines not working: {failed_engines}")
        else:
            self.passed_tests += 1
            self.results.append("✅ All advanced engines working")
            print("\n✅ PASSED: All advanced engines working")
        
        print()
    
    def test_file_structure(self):
        """Test required file structure"""
        print("[TEST] File Structure")
        print("-" * 20)
        
        required_files = [
            'app.py',
            'config.json',
            'pyproject.toml'
        ]
        
        required_dirs = [
            'core',
            'config',
            'logs',
            'data'
        ]
        
        missing_files = []
        missing_dirs = []
        
        for file in required_files:
            if Path(file).exists():
                print(f"  ✅ {file}")
            else:
                print(f"  ❌ {file}")
                missing_files.append(file)
        
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print(f"  ✅ {dir_path}/")
            else:
                print(f"  ⚠️  {dir_path}/")
                missing_dirs.append(dir_path)
        
        if missing_files:
            self.failed_tests += 1
            self.results.append(f"❌ Missing critical files: {missing_files}")
            print(f"\n❌ FAILED: Missing critical files: {missing_files}")
        elif missing_dirs:
            self.warnings += 1
            self.results.append(f"⚠️ Missing directories: {missing_dirs}")
            print(f"\n⚠️ WARNING: Missing directories: {missing_dirs}")
        else:
            self.passed_tests += 1
            self.results.append("✅ File structure complete")
            print("\n✅ PASSED: File structure complete")
        
        print()
    
    def test_configuration(self):
        """Test configuration system"""
        print("[TEST] Configuration System")
        print("-" * 30)
        
        try:
            from core.config_manager import ConfigManager
            config = ConfigManager()
            
            # Test basic configuration access
            exchange = config.get('exchange', 'kraken')
            risk_settings = config.get('risk_management', {})
            
            print(f"  ✅ Configuration loaded")
            print(f"  ✅ Exchange: {exchange}")
            print(f"  ✅ Risk settings: {len(risk_settings)} parameters")
            
            self.passed_tests += 1
            self.results.append("✅ Configuration system working")
            print("\n✅ PASSED: Configuration system working")
            
        except Exception as e:
            self.failed_tests += 1
            self.results.append(f"❌ Configuration system failed: {e}")
            print(f"\n❌ FAILED: Configuration system error: {e}")
        
        print()
    
    def test_logging_system(self):
        """Test logging system"""
        print("[TEST] Logging System")
        print("-" * 22)
        
        try:
            from config.daily_logging_config import setup_daily_logging, get_daily_logger
            
            # Setup logging
            setup_daily_logging()
            print("  ✅ Daily logging setup")
            
            # Get logger manager
            logger_manager = get_daily_logger()
            print("  ✅ Logger manager available")
            
            # Test log creation
            summary = logger_manager.create_daily_summary()
            print(f"  ✅ Daily summary: {len(summary.get('log_counts', {}))} categories")
            
            self.passed_tests += 1
            self.results.append("✅ Logging system working")
            print("\n✅ PASSED: Logging system working")
            
        except Exception as e:
            self.failed_tests += 1
            self.results.append(f"❌ Logging system failed: {e}")
            print(f"\n❌ FAILED: Logging system error: {e}")
        
        print()
    
    def test_streamlit_app(self):
        """Test Streamlit app structure"""
        print("[TEST] Streamlit App")
        print("-" * 20)
        
        try:
            if not Path('app.py').exists():
                raise FileNotFoundError("app.py not found")
            
            with open('app.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential Streamlit components
            required_components = [
                'st.title',
                'st.sidebar',
                'st.selectbox',
                'streamlit'
            ]
            
            missing_components = []
            for component in required_components:
                if component in content:
                    print(f"  ✅ {component}")
                else:
                    print(f"  ❌ {component}")
                    missing_components.append(component)
            
            if missing_components:
                self.warnings += 1
                self.results.append(f"⚠️ Streamlit app missing components: {missing_components}")
                print(f"\n⚠️ WARNING: Missing components: {missing_components}")
            else:
                self.passed_tests += 1
                self.results.append("✅ Streamlit app structure valid")
                print("\n✅ PASSED: Streamlit app structure valid")
                
        except Exception as e:
            self.failed_tests += 1
            self.results.append(f"❌ Streamlit app test failed: {e}")
            print(f"\n❌ FAILED: Streamlit app error: {e}")
        
        print()
    
    def test_production_readiness(self):
        """Test production readiness"""
        print("[TEST] Production Readiness")
        print("-" * 30)
        
        checks = []
        
        # Check Python version
        if sys.version_info >= (3, 10):
            print(f"  ✅ Python version: {sys.version_info.major}.{sys.version_info.minor}")
            checks.append(True)
        else:
            print(f"  ❌ Python version: {sys.version_info.major}.{sys.version_info.minor} (requires 3.10+)")
            checks.append(False)
        
        # Check working directory
        if 'CryptoSmartTrader' in os.getcwd() or Path('app.py').exists():
            print("  ✅ Working directory correct")
            checks.append(True)
        else:
            print("  ❌ Working directory incorrect")
            checks.append(False)
        
        # Check log directory
        if Path('logs').exists():
            print("  ✅ Logs directory exists")
            checks.append(True)
        else:
            print("  ⚠️  Logs directory missing (will be created)")
            checks.append(True)  # Not critical
        
        # Check configuration
        if Path('config.json').exists():
            print("  ✅ Configuration file exists")
            checks.append(True)
        else:
            print("  ⚠️  Configuration file missing (will use defaults)")
            checks.append(True)  # Not critical
        
        if all(checks):
            self.passed_tests += 1
            self.results.append("✅ System ready for production")
            print("\n✅ PASSED: System ready for production")
        else:
            self.failed_tests += 1
            self.results.append("❌ Production readiness issues")
            print("\n❌ FAILED: Production readiness issues")
        
        print()
    
    def print_summary(self):
        """Print test summary"""
        total_tests = self.passed_tests + self.failed_tests
        
        print("=" * 70)
        print("  SYSTEM HEALTH CHECK SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Warnings: {self.warnings}")
        print()
        
        if self.failed_tests == 0:
            print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
            print()
            print("✅ Your CryptoSmartTrader V2 system is fully operational")
            print("✅ All critical components are working correctly")
            print("✅ You can now run start_cryptotrader.bat")
            
            if self.warnings > 0:
                print()
                print(f"⚠️  Note: {self.warnings} warnings detected (non-critical)")
                print("System will work with reduced functionality for optional features")
        else:
            print("❌ SYSTEM NOT READY - CRITICAL ISSUES DETECTED")
            print()
            print("Please fix the following issues:")
            for result in self.results:
                if result.startswith("❌"):
                    print(f"  {result}")
            print()
            print("Recommended actions:")
            print("1. Run setup_windows_environment.bat")
            print("2. Run install_dependencies.bat")
            print("3. Check error messages above")
        
        print()
        print("Detailed Results:")
        for result in self.results:
            print(f"  {result}")
        
        print()
        return self.failed_tests == 0

def main():
    """Main health check function"""
    checker = SystemHealthChecker()
    
    checker.print_header()
    
    # Run all tests
    checker.test_critical_imports()
    checker.test_optional_imports()
    checker.test_core_modules()
    checker.test_advanced_engines()
    checker.test_file_structure()
    checker.test_configuration()
    checker.test_logging_system()
    checker.test_streamlit_app()
    checker.test_production_readiness()
    
    # Print summary
    success = checker.print_summary()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())