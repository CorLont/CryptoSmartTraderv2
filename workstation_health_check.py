#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Workstation Health Check
Comprehensive system validation before deployment
"""

import sys
import os
import subprocess
import importlib
import platform
from pathlib import Path
from typing import List, Dict, Any

def check_python_version() -> Dict[str, Any]:
    """Check Python version compatibility"""
    version = sys.version_info
    result = {
        'check': 'Python Version',
        'status': 'pass' if version >= (3, 9) else 'fail',
        'details': f"{version.major}.{version.minor}.{version.micro}",
        'required': '3.9+',
        'recommendation': 'Install Python 3.11+ for best performance' if version < (3, 11) else 'OK'
    }
    return result

def check_required_packages() -> List[Dict[str, Any]]:
    """Check all required Python packages"""
    required_packages = [
        ('streamlit', 'Web framework'),
        ('pandas', 'Data manipulation'),
        ('numpy', 'Numerical computing'),
        ('plotly', 'Interactive charts'),
        ('scikit-learn', 'Machine learning'),
        ('torch', 'Deep learning'),
        ('ccxt', 'Exchange integration'),
        ('textblob', 'Text processing'),
        ('aiohttp', 'Async HTTP'),
        ('pydantic', 'Data validation'),
        ('psutil', 'System monitoring')
    ]
    
    results = []
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            status = 'pass'
            version = 'installed'
            recommendation = 'OK'
        except ImportError:
            status = 'fail'
            version = 'missing'
            recommendation = f'Run: pip install {package}'
        
        results.append({
            'check': f'Package: {package}',
            'status': status,
            'details': version,
            'description': description,
            'recommendation': recommendation
        })
    
    return results

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability for ML acceleration"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            status = 'pass'
            details = f"{gpu_count}x {gpu_name}"
            recommendation = 'GPU acceleration enabled'
        else:
            status = 'warning'
            details = 'CPU only'
            recommendation = 'Install CUDA drivers for GPU acceleration'
    except ImportError:
        status = 'fail'
        details = 'PyTorch not installed'
        recommendation = 'Install PyTorch with CUDA support'
    
    return {
        'check': 'GPU Acceleration',
        'status': status,
        'details': details,
        'recommendation': recommendation
    }

def check_system_resources() -> List[Dict[str, Any]]:
    """Check system resources"""
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_status = 'pass' if memory_gb >= 8 else 'warning' if memory_gb >= 4 else 'fail'
        memory_rec = 'OK' if memory_gb >= 16 else 'Consider upgrading to 16GB+ for optimal performance'
        
        # Disk space check
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        disk_status = 'pass' if disk_free_gb >= 10 else 'warning' if disk_free_gb >= 5 else 'fail'
        disk_rec = 'OK' if disk_free_gb >= 20 else 'Free up disk space for optimal performance'
        
        # CPU check
        cpu_count = psutil.cpu_count()
        cpu_status = 'pass' if cpu_count >= 4 else 'warning'
        cpu_rec = 'OK' if cpu_count >= 8 else 'More CPU cores recommended for background services'
        
        return [
            {
                'check': 'Memory (RAM)',
                'status': memory_status,
                'details': f'{memory_gb:.1f} GB',
                'recommendation': memory_rec
            },
            {
                'check': 'Disk Space',
                'status': disk_status,
                'details': f'{disk_free_gb:.1f} GB free',
                'recommendation': disk_rec
            },
            {
                'check': 'CPU Cores',
                'status': cpu_status,
                'details': f'{cpu_count} cores',
                'recommendation': cpu_rec
            }
        ]
    except ImportError:
        return [{
            'check': 'System Resources',
            'status': 'fail',
            'details': 'psutil not installed',
            'recommendation': 'Install psutil for system monitoring'
        }]

def check_file_structure() -> List[Dict[str, Any]]:
    """Check required file structure"""
    required_files = [
        ('app_minimal.py', 'Main application'),
        ('1_install_all_dependencies.bat', 'Dependency installer'),
        ('2_start_background_services.bat', 'Background services'),
        ('3_start_dashboard.bat', 'Dashboard launcher'),
        ('core/', 'Core modules'),
        ('dashboards/', 'Dashboard modules'),
        ('replit.md', 'Project documentation')
    ]
    
    results = []
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            status = 'pass'
            details = 'exists'
            recommendation = 'OK'
        else:
            status = 'fail'
            details = 'missing'
            recommendation = f'Create {file_path}'
        
        results.append({
            'check': f'File: {file_path}',
            'status': status,
            'details': details,
            'description': description,
            'recommendation': recommendation
        })
    
    return results

def check_network_connectivity() -> Dict[str, Any]:
    """Check network connectivity for market data"""
    try:
        import urllib.request
        urllib.request.urlopen('https://api.kraken.com/0/public/Time', timeout=5)
        status = 'pass'
        details = 'Connected'
        recommendation = 'OK'
    except Exception as e:
        status = 'warning'
        details = 'Connection issues'
        recommendation = 'Check internet connection for market data'
    
    return {
        'check': 'Network Connectivity',
        'status': status,
        'details': details,
        'recommendation': recommendation
    }

def check_port_availability() -> Dict[str, Any]:
    """Check if required ports are available"""
    try:
        import socket
        
        # Check port 5000 for Streamlit
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(('localhost', 5000))
            if result == 0:
                status = 'warning'
                details = 'Port 5000 in use'
                recommendation = 'Stop other services on port 5000 or use different port'
            else:
                status = 'pass'
                details = 'Port 5000 available'
                recommendation = 'OK'
    except Exception:
        status = 'warning'
        details = 'Cannot check port'
        recommendation = 'Manual port check needed'
    
    return {
        'check': 'Port Availability',
        'status': status,
        'details': details,
        'recommendation': recommendation
    }

def run_comprehensive_health_check() -> Dict[str, Any]:
    """Run comprehensive health check"""
    print("🔍 CryptoSmartTrader V2 - Workstation Health Check")
    print("=" * 60)
    
    all_checks = []
    
    # System checks
    print("\n📋 System Requirements...")
    all_checks.append(check_python_version())
    all_checks.extend(check_system_resources())
    all_checks.append(check_gpu_availability())
    
    # Package checks
    print("\n📦 Python Packages...")
    all_checks.extend(check_required_packages())
    
    # File structure checks
    print("\n📁 File Structure...")
    all_checks.extend(check_file_structure())
    
    # Network checks
    print("\n🌐 Network & Ports...")
    all_checks.append(check_network_connectivity())
    all_checks.append(check_port_availability())
    
    # Summary
    passed = len([c for c in all_checks if c['status'] == 'pass'])
    warned = len([c for c in all_checks if c['status'] == 'warning'])
    failed = len([c for c in all_checks if c['status'] == 'fail'])
    total = len(all_checks)
    
    print(f"\n{'='*60}")
    print("📊 HEALTH CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}/{total}")
    print(f"⚠️  Warnings: {warned}/{total}")
    print(f"❌ Failed: {failed}/{total}")
    
    # Show details
    if failed > 0:
        print(f"\n❌ CRITICAL ISSUES ({failed}):")
        for check in all_checks:
            if check['status'] == 'fail':
                print(f"   • {check['check']}: {check['details']}")
                print(f"     → {check['recommendation']}")
    
    if warned > 0:
        print(f"\n⚠️  WARNINGS ({warned}):")
        for check in all_checks:
            if check['status'] == 'warning':
                print(f"   • {check['check']}: {check['details']}")
                print(f"     → {check['recommendation']}")
    
    # Overall status
    if failed == 0:
        if warned == 0:
            print(f"\n🎉 PERFECT! System ready for deployment!")
            overall_status = 'perfect'
        else:
            print(f"\n✅ GOOD! System ready with minor recommendations.")
            overall_status = 'good'
    else:
        print(f"\n🔧 ISSUES FOUND! Please fix critical issues before deployment.")
        overall_status = 'issues'
    
    print(f"\n📋 NEXT STEPS:")
    if overall_status == 'perfect':
        print("   1. Run: 1_install_all_dependencies.bat")
        print("   2. Run: 2_start_background_services.bat")
        print("   3. Run: 3_start_dashboard.bat")
        print("   4. Open: http://localhost:5000")
    elif overall_status == 'good':
        print("   1. Address warnings above (optional)")
        print("   2. Run: 1_install_all_dependencies.bat")
        print("   3. Run: 2_start_background_services.bat")
        print("   4. Run: 3_start_dashboard.bat")
    else:
        print("   1. Fix all critical issues above")
        print("   2. Re-run this health check")
        print("   3. Proceed with deployment when all clear")
    
    return {
        'total_checks': total,
        'passed': passed,
        'warnings': warned,
        'failed': failed,
        'overall_status': overall_status,
        'checks': all_checks
    }

if __name__ == "__main__":
    health_check_results = run_comprehensive_health_check()