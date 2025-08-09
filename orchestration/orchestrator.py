#!/usr/bin/env python3
"""
Main Orchestrator
Central coordination of all agents and system components
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from core.structured_logger import get_structured_logger
from core.strict_gate import StrictConfidenceGate
from agents.data_collector_agent import DataCollectorAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.ml_agent import MLAgent
from agents.whale_detector_agent import WhaleDetectorAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
from agents.health_monitor_agent import HealthMonitorAgent

@dataclass
class OrchestratorConfig:
    """Orchestrator configuration"""
    run_interval: int = 3600  # 1 hour
    max_concurrent_agents: int = 4
    strict_confidence_threshold: float = 0.8
    enable_paper_trading: bool = True
    log_level: str = "INFO"
    health_check_interval: int = 300  # 5 minutes

@dataclass
class AgentResult:
    """Individual agent execution result"""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

@dataclass
class OrchestrationResult:
    """Complete orchestration cycle result"""
    cycle_id: str
    start_time: datetime
    end_time: datetime
    total_execution_time: float
    agent_results: List[AgentResult]
    confidence_gate_result: Dict[str, Any]
    final_recommendations: List[Dict[str, Any]]
    health_status: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class CryptoSmartTraderOrchestrator:
    """Main orchestrator for CryptoSmartTrader system"""
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self.logger = get_structured_logger("Orchestrator")
        
        # Initialize agents
        self.agents = {}
        self.confidence_gate = StrictConfidenceGate()
        
        # State tracking
        self.is_running = False
        self.current_cycle_id = None
        self.last_health_check = None
        
        # Results storage
        self.results_dir = Path("logs/orchestration")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Orchestrator initialized", config=asdict(self.config))
    
    async def initialize_agents(self):
        """Initialize all system agents"""
        
        self.logger.info("Initializing agents...")
        
        try:
            # Core data collection and analysis agents
            self.agents['data_collector'] = DataCollectorAgent()
            self.agents['sentiment'] = SentimentAgent() 
            self.agents['technical'] = TechnicalAgent()
            self.agents['ml_predictor'] = MLAgent()
            self.agents['whale_detector'] = WhaleDetectorAgent()
            
            # Risk and portfolio management
            self.agents['risk_manager'] = RiskManagerAgent()
            self.agents['portfolio_optimizer'] = PortfolioOptimizerAgent()
            
            # System monitoring
            self.agents['health_monitor'] = HealthMonitorAgent()
            
            # Initialize each agent
            for agent_name, agent in self.agents.items():
                if hasattr(agent, 'initialize'):
                    await agent.initialize()
                self.logger.info(f"Agent initialized: {agent_name}")
            
            self.logger.info(f"All {len(self.agents)} agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def execute_agent(self, agent_name: str, agent: Any, shared_data: Dict[str, Any]) -> AgentResult:
        """Execute a single agent and return result"""
        
        start_time = time.time()
        
        try:
            self.logger.debug(f"Executing agent: {agent_name}")
            
            # Execute agent with timeout
            if hasattr(agent, 'process_async'):
                result_data = await asyncio.wait_for(
                    agent.process_async(shared_data), 
                    timeout=300  # 5 minute timeout
                )
            elif hasattr(agent, 'process'):
                result_data = await asyncio.wait_for(
                    asyncio.to_thread(agent.process, shared_data),
                    timeout=300
                )
            else:
                raise ValueError(f"Agent {agent_name} has no process method")
            
            execution_time = time.time() - start_time
            
            self.logger.info(f"Agent completed: {agent_name}", 
                           execution_time=execution_time)
            
            return AgentResult(
                agent_name=agent_name,
                success=True,
                data=result_data or {},
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            error_msg = f"Agent {agent_name} timed out after {execution_time:.1f}s"
            
            self.logger.error(error_msg)
            
            return AgentResult(
                agent_name=agent_name,
                success=False,
                data={},
                execution_time=execution_time,
                error_message=error_msg
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent {agent_name} failed: {str(e)}"
            
            self.logger.error(error_msg, error=str(e))
            
            return AgentResult(
                agent_name=agent_name,
                success=False,
                data={},
                execution_time=execution_time,
                error_message=error_msg
            )
    
    async def run_orchestration_cycle(self) -> OrchestrationResult:
        """Execute one complete orchestration cycle"""
        
        cycle_id = f"cycle_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()
        
        self.current_cycle_id = cycle_id
        self.logger.info(f"Starting orchestration cycle: {cycle_id}")
        
        try:
            # Phase 1: Data Collection
            shared_data = {"cycle_id": cycle_id, "timestamp": start_time}
            
            data_result = await self.execute_agent('data_collector', self.agents['data_collector'], shared_data)
            if data_result.success:
                shared_data.update(data_result.data)
            
            # Phase 2: Parallel Analysis (Sentiment, Technical, Whale Detection)
            analysis_agents = ['sentiment', 'technical', 'whale_detector']
            analysis_tasks = []
            
            for agent_name in analysis_agents:
                if agent_name in self.agents:
                    task = self.execute_agent(agent_name, self.agents[agent_name], shared_data.copy())
                    analysis_tasks.append(task)
            
            # Execute analysis agents concurrently
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process analysis results
            all_results = [data_result]
            for result in analysis_results:
                if isinstance(result, AgentResult):
                    all_results.append(result)
                    if result.success:
                        shared_data.update(result.data)
            
            # Phase 3: ML Prediction
            ml_result = await self.execute_agent('ml_predictor', self.agents['ml_predictor'], shared_data)
            all_results.append(ml_result)
            if ml_result.success:
                shared_data.update(ml_result.data)
            
            # Phase 4: Confidence Gating
            self.logger.info("Applying strict confidence gate")
            confidence_result = await self.apply_confidence_gate(shared_data)
            
            # Phase 5: Risk Management and Portfolio Optimization
            final_agents = ['risk_manager', 'portfolio_optimizer']
            final_tasks = []
            
            for agent_name in final_agents:
                if agent_name in self.agents:
                    task = self.execute_agent(agent_name, self.agents[agent_name], shared_data.copy())
                    final_tasks.append(task)
            
            final_results = await asyncio.gather(*final_tasks, return_exceptions=True)
            
            for result in final_results:
                if isinstance(result, AgentResult):
                    all_results.append(result)
                    if result.success:
                        shared_data.update(result.data)
            
            # Phase 6: Health Check
            health_result = await self.execute_agent('health_monitor', self.agents['health_monitor'], shared_data)
            all_results.append(health_result)
            
            # Generate final recommendations
            recommendations = self.generate_recommendations(shared_data, confidence_result)
            
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            # Create orchestration result
            orchestration_result = OrchestrationResult(
                cycle_id=cycle_id,
                start_time=start_time,
                end_time=end_time,
                total_execution_time=total_time,
                agent_results=all_results,
                confidence_gate_result=confidence_result,
                final_recommendations=recommendations,
                health_status=health_result.data if health_result.success else {},
                success=True
            )
            
            # Save results
            await self.save_results(orchestration_result)
            
            self.logger.info(f"Orchestration cycle completed: {cycle_id}",
                           total_time=total_time,
                           successful_agents=sum(1 for r in all_results if r.success),
                           total_agents=len(all_results),
                           recommendations=len(recommendations))
            
            return orchestration_result
            
        except Exception as e:
            end_time = datetime.utcnow()
            total_time = (end_time - start_time).total_seconds()
            
            error_msg = f"Orchestration cycle {cycle_id} failed: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            
            return OrchestrationResult(
                cycle_id=cycle_id,
                start_time=start_time,
                end_time=end_time,
                total_execution_time=total_time,
                agent_results=[],
                confidence_gate_result={},
                final_recommendations=[],
                health_status={},
                success=False,
                error_message=error_msg
            )
    
    async def apply_confidence_gate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply strict confidence gating to predictions"""
        
        try:
            # Extract predictions from data
            predictions = data.get('predictions', [])
            
            if not predictions:
                self.logger.warning("No predictions to filter")
                return {"filtered_predictions": [], "gate_passed": False}
            
            # Apply confidence gate
            filtered_predictions = []
            for pred in predictions:
                confidence = pred.get('confidence', 0.0)
                if confidence >= self.config.strict_confidence_threshold:
                    filtered_predictions.append(pred)
            
            gate_passed = len(filtered_predictions) > 0
            pass_rate = len(filtered_predictions) / len(predictions) if predictions else 0
            
            result = {
                "original_count": len(predictions),
                "filtered_count": len(filtered_predictions),
                "pass_rate": pass_rate,
                "gate_passed": gate_passed,
                "filtered_predictions": filtered_predictions,
                "threshold_used": self.config.strict_confidence_threshold
            }
            
            self.logger.info("Confidence gate applied",
                           original_count=len(predictions),
                           filtered_count=len(filtered_predictions),
                           pass_rate=pass_rate,
                           gate_passed=gate_passed)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Confidence gate failed: {e}")
            return {"error": str(e), "gate_passed": False}
    
    def generate_recommendations(self, data: Dict[str, Any], confidence_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate final trading recommendations"""
        
        recommendations = []
        
        try:
            filtered_predictions = confidence_result.get('filtered_predictions', [])
            
            for pred in filtered_predictions:
                symbol = pred.get('symbol', 'UNKNOWN')
                confidence = pred.get('confidence', 0.0)
                predicted_direction = pred.get('direction', 'HOLD')
                
                # Get additional data
                sentiment_score = data.get('sentiment_data', {}).get(symbol, {}).get('score', 0.0)
                technical_signals = data.get('technical_data', {}).get(symbol, {})
                risk_metrics = data.get('risk_data', {}).get(symbol, {})
                
                recommendation = {
                    "symbol": symbol,
                    "action": predicted_direction,
                    "confidence": confidence,
                    "ml_prediction": pred,
                    "sentiment_score": sentiment_score,
                    "technical_signals": technical_signals,
                    "risk_metrics": risk_metrics,
                    "timestamp": datetime.utcnow().isoformat(),
                    "cycle_id": self.current_cycle_id
                }
                
                recommendations.append(recommendation)
            
            # Sort by confidence (highest first)
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"Generated {len(recommendations)} recommendations")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    async def save_results(self, result: OrchestrationResult):
        """Save orchestration results to disk"""
        
        try:
            # Create results file
            result_file = self.results_dir / f"{result.cycle_id}.json"
            
            # Convert to serializable format
            result_dict = asdict(result)
            
            # Handle datetime serialization
            for key, value in result_dict.items():
                if isinstance(value, datetime):
                    result_dict[key] = value.isoformat()
            
            # Save agent results
            for agent_result in result_dict['agent_results']:
                if 'timestamp' in agent_result and agent_result['timestamp']:
                    agent_result['timestamp'] = agent_result['timestamp'].isoformat()
            
            # Write to file
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.info(f"Results saved: {result_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
    
    async def run_continuous(self):
        """Run orchestrator continuously"""
        
        self.is_running = True
        self.logger.info("Starting continuous orchestration",
                        interval=self.config.run_interval)
        
        try:
            while self.is_running:
                # Run orchestration cycle
                result = await self.run_orchestration_cycle()
                
                # Check if we should continue based on health
                if not result.success:
                    self.logger.error("Orchestration failed, checking system health")
                    # Could implement exponential backoff here
                
                # Wait for next cycle
                if self.is_running:
                    self.logger.info(f"Waiting {self.config.run_interval}s for next cycle")
                    await asyncio.sleep(self.config.run_interval)
                    
        except KeyboardInterrupt:
            self.logger.info("Orchestration stopped by user")
        except Exception as e:
            self.logger.error(f"Continuous orchestration failed: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the orchestrator"""
        self.logger.info("Stopping orchestrator")
        self.is_running = False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        status = {
            "is_running": self.is_running,
            "current_cycle_id": self.current_cycle_id,
            "last_health_check": self.last_health_check,
            "config": asdict(self.config),
            "agents_initialized": len(self.agents),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
        
        return status

# Factory function
async def create_orchestrator(config: OrchestratorConfig = None) -> CryptoSmartTraderOrchestrator:
    """Create and initialize orchestrator"""
    
    orchestrator = CryptoSmartTraderOrchestrator(config)
    await orchestrator.initialize_agents()
    return orchestrator

# Main execution
async def main():
    """Main orchestrator execution"""
    
    config = OrchestratorConfig(
        run_interval=3600,  # 1 hour
        strict_confidence_threshold=0.8,
        enable_paper_trading=True
    )
    
    orchestrator = await create_orchestrator(config)
    
    try:
        await orchestrator.run_continuous()
    except KeyboardInterrupt:
        orchestrator.stop()
        print("Orchestrator stopped")

if __name__ == "__main__":
    asyncio.run(main())