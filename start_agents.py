#!/usr/bin/env python3
"""
Start all 5 CryptoSmartTrader agents in background
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path

def start_agent_background(agent_name, script_path):
    """Start an agent in background"""
    try:
        # Start process in background
        process = subprocess.Popen([
            sys.executable, script_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"✅ Started {agent_name} (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {agent_name}: {e}")
        return None

def main():
    print("🚀 Starting CryptoSmartTrader Multi-Agent System")
    print("="*60)
    
    # Create necessary directories
    Path("logs/daily").mkdir(parents=True, exist_ok=True)
    Path("exports/production").mkdir(parents=True, exist_ok=True)
    
    agents = [
        ("Data Collector", "agents/data_collector.py"),
        ("Whale Detector", "agents/whale_detector.py"),
        ("Health Monitor", "agents/health_monitor.py"),
        ("ML Predictor", "agents/ml_predictor.py"),
        ("Risk Manager", "agents/risk_manager.py")
    ]
    
    processes = []
    
    # Start all agents
    for name, script in agents:
        if Path(script).exists():
            proc = start_agent_background(name, script)
            if proc:
                processes.append((name, proc))
        else:
            print(f"⚠️ Agent script not found: {script}")
    
    print(f"\n🎯 Started {len(processes)}/5 agents successfully")
    print("Agents running in background. Press Ctrl+C to stop all agents.")
    
    # Monitor agents
    try:
        while True:
            time.sleep(30)
            
            # Check if agents are still running
            alive_count = 0
            for name, proc in processes:
                if proc.poll() is None:  # Still running
                    alive_count += 1
                else:
                    print(f"⚠️ Agent {name} stopped")
            
            print(f"Status: {alive_count}/{len(processes)} agents running")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all agents...")
        
        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=10)
                print(f"✅ Stopped {name}")
            except:
                proc.kill()
                print(f"🔨 Force killed {name}")
        
        print("All agents stopped.")

if __name__ == "__main__":
    main()