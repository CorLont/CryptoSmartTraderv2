# containers.py
from dependency_injector import containers, providers
from core.config_manager import ConfigManager
from core.data_manager import DataManager
from core.health_monitor import HealthMonitor
from utils.exchange_manager import ExchangeManager
from core.cache_manager import CacheManager
from config.logging_config import setup_logging
from utils.performance_optimizer import PerformanceOptimizer
from utils.error_handler import error_handler
from utils.rate_limiter import rate_limiter
from utils.system_optimizer import SystemOptimizer
from utils.orchestrator import SystemOrchestrator
from config.security import SecurityManager
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.ml_predictor_agent import MLPredictorAgent
from agents.backtest_agent import BacktestAgent
from agents.trade_executor_agent import TradeExecutorAgent
from agents.whale_detector_agent import WhaleDetectorAgent
from models.ml_models import MLModelManager
from data.coin_registry import CoinRegistry
from utils.metrics import MetricsServer
from core.daily_analysis_scheduler import DailyAnalysisScheduler
from core.error_handler import CentralizedErrorHandler
from core.monitoring_system import ProductionMonitoringSystem
from core.openai_enhanced_analyzer import OpenAIEnhancedAnalyzer
from scripts.backup_system import AutomatedBackupSystem
from core.comprehensive_market_scanner import ComprehensiveMarketScanner
from core.gpu_accelerator import gpu_accelerator
from core.alpha_seeker import AlphaSeeker
from core.comprehensive_analyzer import ComprehensiveAnalyzer
from core.real_time_pipeline import RealTimePipeline
from core.multi_horizon_ml import MultiHorizonMLSystem
from core.system_validator import SystemValidator
from core.advanced_analytics import AdvancedAnalyticsEngine
from core.explainable_ai import PredictionExplainer
from core.performance_optimizer import PerformanceOptimizer, PerformanceMonitor
from core.deep_learning_engine import DeepLearningEngine
from core.automl_engine import AutoMLEngine
from core.gpu_accelerator import GPUAccelerator


class ApplicationContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for CryptoSmartTrader V2.
    Provides centralized management of all system dependencies.
    """
    
    # Core configuration
    config = providers.Singleton(ConfigManager)
    
    # Logging setup
    logger_setup = providers.Resource(
        setup_logging,
        config_manager=config
    )
    
    # Performance and error handling
    performance_optimizer = providers.Singleton(PerformanceOptimizer)
    
    error_handler = providers.Object(error_handler)
    
    # Enterprise-grade error handling
    centralized_error_handler = providers.Singleton(
        CentralizedErrorHandler,
        config_manager=config
    )
    
    rate_limiter = providers.Object(rate_limiter)
    
    system_optimizer = providers.Singleton(
        SystemOptimizer,
        config_manager=config
    )
    
    # Security management
    security_manager = providers.Singleton(SecurityManager)
    
    # Core infrastructure
    cache_manager = providers.Singleton(CacheManager)
    
    health_monitor = providers.Singleton(
        HealthMonitor,
        config_manager=config
    )
    
    # Production monitoring system
    monitoring_system = providers.Singleton(
        ProductionMonitoringSystem,
        config_manager=config,
        error_handler=centralized_error_handler
    )
    
    # OpenAI enhanced analyzer
    openai_analyzer = providers.Singleton(
        OpenAIEnhancedAnalyzer,
        config_manager=config
    )
    
    # Daily analysis scheduler
    daily_scheduler = providers.Singleton(
        DailyAnalysisScheduler,
        config_manager=config,
        cache_manager=cache_manager,
        health_monitor=health_monitor
    )
    
    # Automated backup system
    backup_system = providers.Singleton(
        AutomatedBackupSystem,
        config_manager=config
    )
    
    # Comprehensive market scanner
    market_scanner = providers.Singleton(
        ComprehensiveMarketScanner,
        config_manager=config,
        cache_manager=cache_manager,
        error_handler=centralized_error_handler
    )
    
    # GPU accelerator (singleton for system-wide GPU management)
    gpu_accelerator_provider = providers.Object(gpu_accelerator)
    
    # Alpha seeker for high-growth identification
    alpha_seeker = providers.Singleton(
        AlphaSeeker,
        config_manager=config,
        cache_manager=cache_manager,
        openai_analyzer=openai_analyzer
    )
    
    # Comprehensive analyzer for coordinated analysis
    comprehensive_analyzer = providers.Factory(
        ComprehensiveAnalyzer,
        container=providers.Self
    )
    
    # Real-time pipeline for strict alpha seeking
    real_time_pipeline = providers.Factory(
        RealTimePipeline,
        container=providers.Self
    )
    
    # Multi-horizon ML system
    multi_horizon_ml = providers.Factory(
        MultiHorizonMLSystem,
        container=providers.Self
    )
    
    # System validator for health checks
    system_validator = providers.Factory(
        SystemValidator,
        container=providers.Self
    )
    
    # Advanced analytics engine
    advanced_analytics = providers.Factory(
        AdvancedAnalyticsEngine,
        container=providers.Self
    )
    
    # Explainable AI system
    prediction_explainer = providers.Factory(
        PredictionExplainer,
        container=providers.Self
    )
    
    # Performance monitoring and optimization
    performance_monitor = providers.Factory(
        PerformanceMonitor,
        container=providers.Self
    )
    
    performance_optimizer = providers.Factory(
        PerformanceOptimizer,
        container=providers.Self
    )
    
    # Deep learning engine
    deep_learning_engine = providers.Factory(
        DeepLearningEngine,
        container=providers.Self
    )
    
    # AutoML engine
    automl_engine = providers.Factory(
        AutoMLEngine,
        container=providers.Self
    )
    
    # Crypto AI System
    crypto_ai_system = providers.Factory(
        lambda container: __import__('core.crypto_ai_system', fromlist=['CryptoAISystem']).CryptoAISystem(container),
        container=providers.Self(),
    )
    
    # GPU accelerator
    gpu_accelerator = providers.Factory(
        GPUAccelerator,
        container=providers.Self
    )
    
    # System orchestrator
    orchestrator = providers.Singleton(
        SystemOrchestrator,
        config_manager=config,
        health_monitor=health_monitor
    )
    
    # Exchange management
    exchange_manager = providers.Singleton(
        ExchangeManager,
        config_manager=config,
        health_monitor=health_monitor
    )
    
    # Data management
    coin_registry = providers.Singleton(
        CoinRegistry,
        config_manager=config
    )
    
    data_manager = providers.Singleton(
        DataManager,
        config_manager=config,
        exchange_manager=exchange_manager,
        health_monitor=health_monitor,
        coin_registry=coin_registry,
        cache_manager=cache_manager
    )
    
    # ML Model management
    ml_model_manager = providers.Singleton(
        MLModelManager,
        config_manager=config,
        cache_manager=cache_manager
    )
    
    # Agent dependencies
    sentiment_agent = providers.Factory(
        SentimentAgent,
        config_manager=config,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    technical_agent = providers.Factory(
        TechnicalAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    ml_predictor_agent = providers.Factory(
        MLPredictorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager,
        ml_model_manager=ml_model_manager
    )
    
    backtest_agent = providers.Factory(
        BacktestAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    trade_executor_agent = providers.Factory(
        TradeExecutorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    whale_detector_agent = providers.Factory(
        WhaleDetectorAgent,
        config_manager=config,
        data_manager=data_manager,
        health_monitor=health_monitor,
        cache_manager=cache_manager
    )
    
    # Metrics server
    metrics_server = providers.Singleton(
        MetricsServer,
        config_manager=config,
        health_monitor=health_monitor
    )