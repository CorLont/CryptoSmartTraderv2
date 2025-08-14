#!/usr/bin/env python3
"""
Install development tools for CI/CD pipeline
"""

import subprocess
import sys

def install_dev_tools():
    """Install CI/CD development tools"""
    
    print("🔧 Installing CI/CD development tools...")
    
    tools = [
        "ruff>=0.1.0",
        "black>=23.0.0", 
        "mypy>=1.5.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "bandit>=1.7.0",
        "pip-audit>=2.6.0"
    ]
    
    for tool in tools:
        try:
            print(f"Installing {tool}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                tool, "--quiet", "--disable-pip-version-check"
            ], check=True, capture_output=True)
            print(f"✅ {tool} installed")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {tool}: {e}")
            return False
    
    # Test tools
    print("\n🔍 Testing development tools...")
    
    tests = [
        (["ruff", "--version"], "ruff"),
        (["black", "--version"], "black"),
        (["mypy", "--version"], "mypy"),
        (["pytest", "--version"], "pytest"),
        (["bandit", "--version"], "bandit"),
        (["pip-audit", "--version"], "pip-audit")
    ]
    
    for cmd, name in tests:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {name}: {result.stdout.strip().split()[1] if len(result.stdout.split()) > 1 else 'installed'}")
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            print(f"❌ {name}: failed - {e}")
        except Exception as e:
            print(f"❌ {name}: unexpected error - {e}")
    
    print("\n✅ Development tools installation complete!")
    return True

if __name__ == "__main__":
    success = install_dev_tools()
    sys.exit(0 if success else 1)