#!/usr/bin/env python3
"""
Replit Services Starter - Optimized for .replit run configuration
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path


def setup_environment():
    """Setup environment and ensure dependencies"""
    print("🔧 Setting up environment...")
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Set environment variables for production
    os.environ.setdefault('ENVIRONMENT', 'production')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    print("✅ Environment configured")


def start_services():
    """Start all services in background with proper coordination"""
    print("🚀 Starting CryptoSmartTrader V2 Multi-Service Architecture")
    print("=" * 60)
    
    # Start API service (port 8001)
    print("🏥 Starting Health API on port 8001...")
    api_process = subprocess.Popen([
        "python", "api/health_endpoint.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Small delay between service starts
    time.sleep(2)
    
    # Start Metrics service (port 8000)  
    print("📊 Starting Metrics Server on port 8000...")
    metrics_process = subprocess.Popen([
        "python", "metrics/metrics_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Small delay before dashboard
    time.sleep(2)
    
    # Start Dashboard service (port 5000) - this should be the main service
    print("🎯 Starting Main Dashboard on port 5000...")
    dashboard_process = subprocess.Popen([
        "streamlit", "run", "app_fixed_all_issues.py",
        "--server.port", "5000",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
    
    print("\n✅ All services started successfully!")
    print("🌐 Service URLs:")
    print("   Dashboard:  http://localhost:5000")
    print("   API:        http://localhost:8001") 
    print("   Metrics:    http://localhost:8000")
    print("   API Docs:   http://localhost:8001/api/docs")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Shutting down services...")
        try:
            api_process.terminate()
            metrics_process.terminate() 
            dashboard_process.terminate()
        except Exception:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for main dashboard process (Streamlit)
    try:
        dashboard_process.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    setup_environment()
    start_services()