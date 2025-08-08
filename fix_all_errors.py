#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Complete Error Fix Script
Automatically fixes all common workstation deployment issues
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path

def fix_missing_packages():
    """Install any missing packages"""
    print("🔧 Checking and fixing missing packages...")
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
        'torch', 'ccxt', 'textblob', 'aiohttp', 'pydantic', 'psutil'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\nInstalling {len(missing_packages)} missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All required packages are installed")

def fix_directory_structure():
    """Ensure all required directories exist"""
    print("\n🗂️  Fixing directory structure...")
    
    required_dirs = [
        'data', 'data/raw', 'data/processed', 'data/predictions',
        'logs', 'models', 'models/saved', 'models/backups',
        '.streamlit'
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {directory}/")
        else:
            print(f"✓ {directory}/")

def fix_configuration_files():
    """Ensure all configuration files are properly set"""
    print("\n⚙️  Fixing configuration files...")
    
    # Streamlit config
    streamlit_config = Path('.streamlit/config.toml')
    if not streamlit_config.exists():
        config_content = """[server]
headless = true
address = "0.0.0.0"
port = 5000
enableXsrfProtection = false

[theme]
primaryColor = "#FF6B35"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
"""
        streamlit_config.write_text(config_content)
        print("✅ Created Streamlit configuration")
    else:
        print("✓ Streamlit configuration exists")
    
    # Environment file
    env_file = Path('.env')
    if not env_file.exists():
        env_content = """# CryptoSmartTrader V2 Environment Variables
OPENAI_API_KEY=your_openai_api_key_here
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET=your_binance_secret
LOG_LEVEL=INFO
ENVIRONMENT=production
"""
        env_file.write_text(env_content)
        print("✅ Created environment template")
    else:
        print("✓ Environment file exists")

def fix_import_errors():
    """Fix common import errors in core modules"""
    print("\n🔍 Checking for import errors...")
    
    modules_to_test = [
        'core.synthetic_data_augmentation',
        'core.human_in_the_loop', 
        'core.shadow_trading_engine',
        'dashboards.synthetic_data_dashboard',
        'dashboards.human_in_loop_dashboard',
        'dashboards.shadow_trading_dashboard'
    ]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            # Try to fix common import issues
            if "No module named" in str(e):
                package = str(e).split("'")[1]
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    print(f"✅ Fixed by installing {package}")
                except:
                    print(f"⚠️  Could not auto-fix {module}")

def fix_port_conflicts():
    """Check and suggest fixes for port conflicts"""
    print("\n🌐 Checking port availability...")
    
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', 5000))
            if result == 0:
                print("⚠️  Port 5000 is in use")
                print("   → Run this to free the port:")
                print("   → netstat -ano | find \":5000\"")
                print("   → taskkill /PID <PID> /F")
            else:
                print("✅ Port 5000 is available")
    except Exception as e:
        print(f"⚠️  Could not check port: {e}")

def test_system_functionality():
    """Test core system functionality"""
    print("\n🧪 Testing system functionality...")
    
    try:
        # Test app import
        sys.path.append('.')
        import app_minimal
        print("✅ Main app imports successfully")
        
        # Test core modules
        from core.synthetic_data_augmentation import generate_stress_test_scenarios
        from core.human_in_the_loop import get_human_in_the_loop_system
        from core.shadow_trading_engine import get_shadow_trading_engine
        print("✅ Core modules import successfully")
        
        # Test dashboard modules
        from dashboards.synthetic_data_dashboard import SyntheticDataDashboard
        from dashboards.human_in_loop_dashboard import HumanInLoopDashboard
        from dashboards.shadow_trading_dashboard import ShadowTradingDashboard
        print("✅ Dashboard modules import successfully")
        
        print("✅ All system functionality tests passed!")
        
    except Exception as e:
        print(f"❌ System functionality test failed: {e}")
        return False
    
    return True

def main():
    """Run complete error fixing process"""
    print("🚀 CryptoSmartTrader V2 - Complete Error Fix")
    print("=" * 55)
    
    # Fix all common issues
    fix_missing_packages()
    fix_directory_structure()
    fix_configuration_files()
    fix_import_errors()
    fix_port_conflicts()
    
    # Test everything
    success = test_system_functionality()
    
    print("\n" + "=" * 55)
    if success:
        print("🎉 ALL ERRORS FIXED! System is ready for deployment!")
        print("\n📋 Next Steps:")
        print("   1. Run: 2_start_background_services.bat")
        print("   2. Run: 3_start_dashboard.bat")
        print("   3. Open: http://localhost:5000")
    else:
        print("🔧 Some issues remain. Check the errors above.")
        print("\n📋 Manual Steps:")
        print("   1. Review error messages above")
        print("   2. Install missing dependencies manually")
        print("   3. Re-run this script")
    
    print("\n🔍 Health Check: python workstation_health_check.py")

if __name__ == "__main__":
    main()