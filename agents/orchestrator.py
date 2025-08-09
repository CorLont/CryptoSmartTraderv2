# agents/orchestrator.py - Multi-agent orchestrator with 24/7 background processing
import asyncio
import multiprocessing as mp
import signal
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiAgentOrchestrator:
    """Orchestrates 5 independent agents in isolated processes"""
    
    def __init__(self):
        self.agents = {}
        self.should_stop = False
        
    def start_agent_process(self, agent_name, target_func):
        """Start an agent in isolated process"""
        try:
            process = mp.Process(target=target_func, name=f"Agent-{agent_name}")
            process.daemon = True
            process.start()
            self.agents[agent_name] = process
            logger.info(f"Started {agent_name} agent (PID: {process.pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to start {agent_name}: {e}")
            return False
    
    def monitor_agents(self):
        """Monitor agent health and restart if needed"""
        for name, process in list(self.agents.items()):
            if not process.is_alive():
                logger.warning(f"Agent {name} died, restarting...")
                process.terminate()
                process.join(timeout=5)
                
                # Restart the agent
                if name == "DataCollector":
                    self.start_data_collector()
                elif name == "WhaleDetector": 
                    self.start_whale_detector()
                elif name == "HealthMonitor":
                    self.start_health_monitor()
                elif name == "MLPredictor":
                    self.start_ml_predictor()
                elif name == "RiskManager":
                    self.start_risk_manager()
    
    def start_data_collector(self):
        """Start data collection agent"""
        def run_collector():
            from agents.data_collector import DataCollectorAgent
            agent = DataCollectorAgent()
            asyncio.run(agent.run_continuous())
        
        return self.start_agent_process("DataCollector", run_collector)
    
    def start_whale_detector(self):
        """Start whale detection agent"""
        def run_whale():
            from agents.whale_detector import WhaleDetectorAgent
            agent = WhaleDetectorAgent()
            asyncio.run(agent.run_continuous())
        
        return self.start_agent_process("WhaleDetector", run_whale)
    
    def start_health_monitor(self):
        """Start health monitoring agent"""
        def run_health():
            from agents.health_monitor import HealthMonitorAgent
            agent = HealthMonitorAgent()
            asyncio.run(agent.run_continuous())
        
        return self.start_agent_process("HealthMonitor", run_health)
    
    def start_ml_predictor(self):
        """Start ML prediction agent"""
        def run_predictor():
            from agents.ml_predictor import MLPredictorAgent
            agent = MLPredictorAgent()
            asyncio.run(agent.run_continuous())
        
        return self.start_agent_process("MLPredictor", run_predictor)
    
    def start_risk_manager(self):
        """Start risk management agent"""
        def run_risk():
            from agents.risk_manager import RiskManagerAgent
            agent = RiskManagerAgent()
            asyncio.run(agent.run_continuous())
        
        return self.start_agent_process("RiskManager", run_risk)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal, stopping all agents...")
        self.should_stop = True
        
        for name, process in self.agents.items():
            logger.info(f"Stopping {name} agent...")
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Force killing {name} agent")
                process.kill()
        
        sys.exit(0)
    
    async def run_orchestrator(self):
        """Main orchestration loop"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 Starting Multi-Agent CryptoSmartTrader System")
        
        # Start all agents
        success_count = 0
        success_count += self.start_data_collector()
        success_count += self.start_whale_detector()  
        success_count += self.start_health_monitor()
        success_count += self.start_ml_predictor()
        success_count += self.start_risk_manager()
        
        logger.info(f"Started {success_count}/5 agents successfully")
        
        if success_count < 3:
            logger.error("Failed to start minimum required agents")
            return
        
        # Main monitoring loop
        while not self.should_stop:
            try:
                self.monitor_agents()
                
                # Log system status
                alive_agents = sum(1 for p in self.agents.values() if p.is_alive())
                logger.info(f"System Status: {alive_agents}/5 agents running")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Orchestrator error: {e}")
                await asyncio.sleep(60)

def main():
    """Main entry point"""
    orchestrator = MultiAgentOrchestrator()
    
    try:
        asyncio.run(orchestrator.run_orchestrator())
    except KeyboardInterrupt:
        logger.info("Orchestrator shutdown requested")
    except Exception as e:
        logger.error(f"Orchestrator fatal error: {e}")

if __name__ == "__main__":
    main()