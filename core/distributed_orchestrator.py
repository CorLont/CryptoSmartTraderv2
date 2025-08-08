#!/usr/bin/env python3
"""
Distributed Orchestrator
Coordinates isolated agent processes with async queues and Prometheus metrics
"""

import asyncio
import signal
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import core components
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger
from core.process_isolation import ProcessIsolationManager, AgentConfig
from core.async_queue_system import AsyncQueueSystem, AsyncioQueueBackend, RateLimiter, MessagePriority
from core.prometheus_metrics import (
    PrometheusMetricsServer, LatencyMetricsCollector, ErrorRatioMetricsCollector,
    CompletenessMetricsCollector, SystemMetricsCollector, metrics_collection_loop
)

class DistributedOrchestrator:
    """Main orchestrator for distributed agent system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = get_structured_logger("DistributedOrchestrator")
        
        # Core components
        self.process_manager = ProcessIsolationManager()
        self.queue_system = None
        self.metrics_server = None
        
        # State management
        self.running = False
        self.agents_config = []
        self.background_tasks: List[asyncio.Task] = []
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        
        default_config = {
            "agents": [
                {
                    "name": "DataCollector",
                    "module_path": "agents.data_collector.run_data_collector",
                    "restart_policy": "always",
                    "max_restarts": 5,
                    "restart_delay_base": 2.0,
                    "health_check_interval": 30.0
                },
                {
                    "name": "SentimentAnalyzer",
                    "module_path": "agents.sentiment_analyzer.run_sentiment_analyzer", 
                    "restart_policy": "always",
                    "max_restarts": 5,
                    "restart_delay_base": 3.0,
                    "health_check_interval": 45.0
                },
                {
                    "name": "TechnicalAnalyzer",
                    "module_path": "agents.technical_analyzer.run_technical_analyzer",
                    "restart_policy": "always", 
                    "max_restarts": 5,
                    "restart_delay_base": 2.0,
                    "health_check_interval": 30.0
                },
                {
                    "name": "MLPredictor",
                    "module_path": "agents.ml_predictor.run_ml_predictor",
                    "restart_policy": "always",
                    "max_restarts": 3,
                    "restart_delay_base": 5.0,
                    "health_check_interval": 60.0
                },
                {
                    "name": "WhaleDetector",
                    "module_path": "agents.whale_detector.run_whale_detector",
                    "restart_policy": "always",
                    "max_restarts": 5,
                    "restart_delay_base": 3.0,
                    "health_check_interval": 45.0
                },
                {
                    "name": "RiskManager",
                    "module_path": "agents.risk_manager.run_risk_manager",
                    "restart_policy": "always",
                    "max_restarts": 3,
                    "restart_delay_base": 1.0,
                    "health_check_interval": 20.0
                },
                {
                    "name": "PortfolioOptimizer",
                    "module_path": "agents.portfolio_optimizer.run_portfolio_optimizer",
                    "restart_policy": "always",
                    "max_restarts": 3,
                    "restart_delay_base": 4.0,
                    "health_check_interval": 60.0
                },
                {
                    "name": "HealthMonitor",
                    "module_path": "agents.health_monitor.run_health_monitor",
                    "restart_policy": "always",
                    "max_restarts": 10,
                    "restart_delay_base": 1.0,
                    "health_check_interval": 15.0
                }
            ],
            "queue_system": {
                "backend": "asyncio",  # or "redis"
                "rate_limit_rps": 15.0,
                "max_queue_size": 10000
            },
            "metrics": {
                "prometheus_port": 8090,
                "collection_interval": 10.0,
                "enabled": True
            },
            "orchestrator": {
                "startup_delay": 2.0,
                "shutdown_timeout": 30.0,
                "health_check_interval": 30.0
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
                    self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_file}: {e}")
        
        return default_config
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        
        self.logger.info("Initializing distributed orchestrator")
        
        try:
            # Initialize queue system
            await self._initialize_queue_system()
            
            # Initialize metrics system
            await self._initialize_metrics_system()
            
            # Initialize agents
            await self._initialize_agents()
            
            self.logger.info("Distributed orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def _initialize_queue_system(self) -> None:
        """Initialize async queue system"""
        
        queue_config = self.config["queue_system"]
        
        # Create backend
        if queue_config["backend"] == "redis":
            # Would use RedisQueueBackend in production
            backend = AsyncioQueueBackend(max_queue_size=queue_config["max_queue_size"])
            self.logger.info("Using asyncio queue backend (Redis not configured)")
        else:
            backend = AsyncioQueueBackend(max_queue_size=queue_config["max_queue_size"])
            self.logger.info("Using asyncio queue backend")
        
        # Create rate limiter
        rate_limiter = RateLimiter(requests_per_second=queue_config["rate_limit_rps"])
        
        # Create queue system
        self.queue_system = AsyncQueueSystem(backend, rate_limiter)
        
        # Register message handlers
        self._register_message_handlers()
        
        self.logger.info("Queue system initialized")
    
    async def _initialize_metrics_system(self) -> None:
        """Initialize Prometheus metrics system"""
        
        metrics_config = self.config["metrics"]
        
        if not metrics_config["enabled"]:
            self.logger.info("Metrics system disabled")
            return
        
        # Create metrics server
        self.metrics_server = PrometheusMetricsServer(port=metrics_config["prometheus_port"])
        
        # Add collectors
        self.metrics_server.add_collector(LatencyMetricsCollector())
        self.metrics_server.add_collector(ErrorRatioMetricsCollector())
        self.metrics_server.add_collector(CompletenessMetricsCollector())
        self.metrics_server.add_collector(SystemMetricsCollector())
        
        # Start metrics server
        success = self.metrics_server.start_server()
        if success:
            self.logger.info(f"Metrics server started on port {metrics_config['prometheus_port']}")
        else:
            self.logger.error("Failed to start metrics server")
    
    async def _initialize_agents(self) -> None:
        """Initialize agent processes"""
        
        for agent_config_dict in self.config["agents"]:
            try:
                # Create agent config
                agent_config = AgentConfig(
                    name=agent_config_dict["name"],
                    target_function="run_agent",
                    module_path=agent_config_dict["module_path"],
                    restart_policy=agent_config_dict.get("restart_policy", "always"),
                    max_restarts=agent_config_dict.get("max_restarts", 5),
                    restart_delay_base=agent_config_dict.get("restart_delay_base", 2.0),
                    health_check_interval=agent_config_dict.get("health_check_interval", 30.0)
                )
                
                # Register with process manager
                success = self.process_manager.register_agent(agent_config)
                if success:
                    self.agents_config.append(agent_config)
                    self.logger.info(f"Registered agent: {agent_config.name}")
                else:
                    self.logger.error(f"Failed to register agent: {agent_config.name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to initialize agent {agent_config_dict['name']}: {e}")
        
        self.logger.info(f"Initialized {len(self.agents_config)} agents")
    
    def _register_message_handlers(self) -> None:
        """Register message handlers for queue system"""
        
        async def data_collection_handler(message):
            self.logger.info(f"Processing data collection: {message.payload.get('symbol', 'unknown')}")
        
        async def ml_prediction_handler(message):
            self.logger.info(f"Processing ML prediction: {message.payload.get('model', 'unknown')}")
        
        async def sentiment_analysis_handler(message):
            self.logger.info(f"Processing sentiment analysis: {message.payload.get('text_length', 0)} chars")
        
        async def technical_analysis_handler(message):
            self.logger.info(f"Processing technical analysis: {message.payload.get('indicators', [])}")
        
        async def whale_detection_handler(message):
            self.logger.info(f"Processing whale detection: {message.payload.get('volume', 0)}")
        
        async def risk_assessment_handler(message):
            self.logger.info(f"Processing risk assessment: {message.payload.get('portfolio_value', 0)}")
        
        async def portfolio_optimization_handler(message):
            self.logger.info(f"Processing portfolio optimization: {message.payload.get('positions', [])}")
        
        async def health_check_handler(message):
            self.logger.info(f"Processing health check: {message.payload.get('component', 'unknown')}")
        
        # Register handlers
        self.queue_system.register_handler('data_collection', data_collection_handler)
        self.queue_system.register_handler('ml_prediction', ml_prediction_handler)
        self.queue_system.register_handler('sentiment_analysis', sentiment_analysis_handler)
        self.queue_system.register_handler('technical_analysis', technical_analysis_handler)
        self.queue_system.register_handler('whale_detection', whale_detection_handler)
        self.queue_system.register_handler('risk_assessment', risk_assessment_handler)
        self.queue_system.register_handler('portfolio_optimization', portfolio_optimization_handler)
        self.queue_system.register_handler('health_check', health_check_handler)
    
    async def start(self) -> bool:
        """Start the distributed system"""
        
        if self.running:
            self.logger.warning("Orchestrator already running")
            return True
        
        self.logger.info("Starting distributed orchestrator")
        
        try:
            # Start process monitoring
            await self.process_manager.start_monitoring()
            
            # Start agents with delay
            startup_delay = self.config["orchestrator"]["startup_delay"]
            await asyncio.sleep(startup_delay)
            
            success = self.process_manager.start_all_agents()
            if not success:
                self.logger.error("Failed to start all agents")
                return False
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.running = True
            self.logger.info("Distributed orchestrator started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start orchestrator: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the distributed system"""
        
        if not self.running:
            return True
        
        self.logger.info("Stopping distributed orchestrator")
        
        try:
            self.running = False
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop agents
            self.process_manager.stop_all_agents()
            
            # Stop process monitoring
            await self.process_manager.stop_monitoring()
            
            # Stop metrics server
            if self.metrics_server:
                self.metrics_server.stop_server()
            
            self.logger.info("Distributed orchestrator stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop orchestrator: {e}")
            return False
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and processing tasks"""
        
        # Queue processing task
        queue_task = asyncio.create_task(self._queue_processing_loop())
        self.background_tasks.append(queue_task)
        
        # Metrics collection task
        if self.metrics_server:
            metrics_interval = self.config["metrics"]["collection_interval"]
            metrics_task = asyncio.create_task(
                metrics_collection_loop(self.metrics_server, metrics_interval)
            )
            self.background_tasks.append(metrics_task)
        
        # Health monitoring task
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        self.logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        self.logger.info("Stopped background tasks")
    
    async def _queue_processing_loop(self) -> None:
        """Main queue processing loop"""
        
        self.logger.info("Starting queue processing loop")
        
        queues_to_process = [
            "data_pipeline", "ml_pipeline", "sentiment_pipeline",
            "technical_pipeline", "whale_pipeline", "risk_pipeline",
            "portfolio_pipeline", "health_pipeline"
        ]
        
        while self.running:
            try:
                for queue_name in queues_to_process:
                    processed = await self.queue_system.process_queue(queue_name, max_messages=10)
                    if processed > 0:
                        self.logger.debug(f"Processed {processed} messages from {queue_name}")
                
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Queue processing loop error: {e}")
                await asyncio.sleep(5.0)
        
        self.logger.info("Queue processing loop stopped")
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop"""
        
        self.logger.info("Starting health monitoring loop")
        
        health_interval = self.config["orchestrator"]["health_check_interval"]
        
        while self.running:
            try:
                # Check overall system health
                health_summary = self.process_manager.get_health_summary()
                
                if health_summary['health_percentage'] < 80:
                    self.logger.warning(f"System health degraded: {health_summary['health_percentage']:.1f}%")
                
                # Check queue system health
                if self.queue_system:
                    queue_health = await self.queue_system.health_check()
                    if not queue_health['system_healthy']:
                        self.logger.warning("Queue system unhealthy")
                
                await asyncio.sleep(health_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(health_interval)
        
        self.logger.info("Health monitoring loop stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            'orchestrator_running': self.running,
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'queues': {},
            'metrics': {}
        }
        
        # Agent status
        agent_statuses = self.process_manager.get_all_status()
        for name, agent_status in agent_statuses.items():
            status['agents'][name] = {
                'state': agent_status.state.value,
                'pid': agent_status.pid,
                'restart_count': agent_status.restart_count,
                'memory_usage_mb': agent_status.memory_usage_mb,
                'cpu_usage_percent': agent_status.cpu_usage_percent
            }
        
        # Queue status
        if self.queue_system:
            status['queues'] = self.queue_system.get_metrics()
        
        # Metrics status
        if self.metrics_server:
            status['metrics'] = self.metrics_server.get_metrics_summary()
        
        return status
    
    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.stop())

async def main():
    """Main entry point for distributed orchestrator"""
    
    print("🚀 DISTRIBUTED ORCHESTRATOR")
    print("=" * 60)
    
    # Create and initialize orchestrator
    orchestrator = DistributedOrchestrator()
    
    success = await orchestrator.initialize()
    if not success:
        print("❌ Failed to initialize orchestrator")
        return
    
    print("✅ Orchestrator initialized")
    
    # Start orchestrator
    success = await orchestrator.start()
    if not success:
        print("❌ Failed to start orchestrator")
        return
    
    print("✅ Orchestrator started")
    print("📊 System status:")
    
    try:
        # Run for demonstration
        for i in range(12):  # 60 seconds total
            await asyncio.sleep(5.0)
            
            status = orchestrator.get_system_status()
            
            # Show agent status
            running_agents = sum(1 for agent in status['agents'].values() 
                               if agent['state'] == 'running')
            total_agents = len(status['agents'])
            
            print(f"   {i+1}/12: {running_agents}/{total_agents} agents running")
            
            # Show queue metrics
            if 'messages_sent' in status['queues']:
                print(f"        Messages sent: {status['queues']['messages_sent']}")
            
            # Show metrics status
            if 'collectors_active' in status['metrics']:
                print(f"        Metrics collectors: {status['metrics']['collectors_active']}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Shutdown requested")
    
    # Stop orchestrator
    print("\n🛑 Stopping orchestrator...")
    success = await orchestrator.stop()
    print(f"   Stop result: {'✅' if success else '❌'}")
    
    print("\n✅ DISTRIBUTED ORCHESTRATOR TEST COMPLETED")

if __name__ == "__main__":
    asyncio.run(main())