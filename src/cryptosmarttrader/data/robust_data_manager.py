#!/usr/bin/env python3
"""
Robust Data Manager
Centrale coördinatie van enterprise data ingestion met alle robuste features
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import threading

from .enterprise_data_ingestion import (
    EnterpriseDataIngestion, 
    DataRequest, 
    DataPriority, 
    create_data_ingestion,
    create_market_data_request,
    create_orderbook_request,
    create_ohlcv_request
)
from .data_scheduler import DataScheduler, create_data_scheduler
from .data_quality_validator import DataQualityValidator, create_data_quality_validator


@dataclass
class DataIngestionConfig:
    """Comprehensive data ingestion configuration"""
    # Exchange configuration
    exchanges: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'kraken': {
            'api_key': '',
            'secret': '',
            'sandbox': False,
            'rate_limit': 1.0  # requests per second
        },
        'binance': {
            'api_key': '',
            'secret': '',
            'sandbox': False,
            'rate_limit': 10.0
        }
    })
    
    # Redis configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Rate limiting configuration
    rate_limits: Dict[str, float] = field(default_factory=lambda: {
        'kraken': 1.0,
        'binance': 10.0,
        'kucoin': 3.0,
        'huobi': 5.0
    })
    
    # Symbol configuration
    critical_symbols: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD'
    ])
    
    tracked_symbols: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD',
        'AVAX/USD', 'DOT/USD', 'MATIC/USD', 'LINK/USD', 'UNI/USD',
        'ATOM/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD', 'ETC/USD'
    ])
    
    analytics_symbols: List[str] = field(default_factory=lambda: [
        'BTC/USD', 'ETH/USD', 'BNB/USD', 'ADA/USD', 'SOL/USD'
    ])
    
    # Quality validation thresholds
    completeness_threshold: float = 0.95
    accuracy_threshold: float = 0.98
    consistency_threshold: float = 0.90
    timeliness_threshold: float = 0.95
    validity_threshold: float = 0.99
    uniqueness_threshold: float = 0.98
    
    # Price bounds for validation
    price_bounds: Dict[str, tuple] = field(default_factory=lambda: {
        'BTC/USD': (1000, 200000),
        'ETH/USD': (10, 10000),
        'BNB/USD': (1, 1000),
        'ADA/USD': (0.01, 10),
        'SOL/USD': (1, 1000),
        'default': (0.0001, 100000)
    })
    
    # Symbol patterns for validation
    symbol_patterns: Dict[str, str] = field(default_factory=lambda: {
        'crypto_usd': r'^[A-Z0-9]{2,10}/USD$',
        'crypto_btc': r'^[A-Z0-9]{2,10}/BTC$',
        'crypto_eth': r'^[A-Z0-9]{2,10}/ETH$'
    })


class RobustDataManager:
    """Enterprise-grade data manager met comprehensive robustness"""
    
    def __init__(self, config: Optional[DataIngestionConfig] = None):
        self.config = config or DataIngestionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.data_ingestion: Optional[EnterpriseDataIngestion] = None
        self.scheduler: Optional[DataScheduler] = None
        self.quality_validator: Optional[DataQualityValidator] = None
        
        # State management
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time_ms': 0.0,
            'cache_hits': 0,
            'quality_scores': [],
            'uptime_seconds': 0.0
        }
        
        # Event callbacks
        self.data_callbacks: List[Callable] = []
        self.quality_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
    
    async def initialize(self):
        """Initialize all data ingestion components"""
        self.logger.info("Initializing robust data manager...")
        
        try:
            # Convert config to dict format
            config_dict = {
                'exchanges': self.config.exchanges,
                'redis_url': self.config.redis_url,
                'rate_limits': self.config.rate_limits,
                'critical_symbols': self.config.critical_symbols,
                'tracked_symbols': self.config.tracked_symbols,
                'analytics_symbols': self.config.analytics_symbols
            }
            
            # Initialize data ingestion
            self.data_ingestion = await create_data_ingestion(config_dict)
            
            # Initialize quality validator
            quality_config = {
                'completeness_threshold': self.config.completeness_threshold,
                'accuracy_threshold': self.config.accuracy_threshold,
                'consistency_threshold': self.config.consistency_threshold,
                'timeliness_threshold': self.config.timeliness_threshold,
                'validity_threshold': self.config.validity_threshold,
                'uniqueness_threshold': self.config.uniqueness_threshold,
                'tracked_symbols': self.config.tracked_symbols,
                'price_bounds': self.config.price_bounds,
                'symbol_patterns': self.config.symbol_patterns
            }
            self.quality_validator = create_data_quality_validator(quality_config)
            
            # Initialize scheduler
            self.scheduler = create_data_scheduler(self.data_ingestion, config_dict)
            
            self.logger.info("✅ Robust data manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize data manager: {e}")
            raise
    
    async def start(self):
        """Start all data ingestion services"""
        if self.is_running:
            self.logger.warning("Data manager is already running")
            return
        
        if not self.data_ingestion or not self.scheduler:
            await self.initialize()
        
        try:
            self.logger.info("Starting robust data ingestion services...")
            
            # Start scheduler
            await self.scheduler.start_scheduler()
            
            # Mark as running
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start monitoring task
            asyncio.create_task(self._monitoring_loop())
            
            self.logger.info("✅ Robust data manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start data manager: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop all data ingestion services"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping robust data manager...")
        
        try:
            # Stop scheduler
            if self.scheduler:
                await self.scheduler.stop_scheduler()
            
            # Stop data ingestion
            if self.data_ingestion:
                await self.data_ingestion.stop_workers()
            
            self.is_running = False
            self.logger.info("✅ Robust data manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping data manager: {e}")
    
    async def request_data_with_quality_check(
        self, 
        request: DataRequest,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """Request data with optional quality validation"""
        if not self.data_ingestion:
            raise RuntimeError("Data manager not initialized")
        
        start_time = datetime.now()
        
        try:
            # Execute data request
            response = await self.data_ingestion.request_data(request)
            
            # Update metrics
            self.metrics['total_requests'] += 1
            if response.status == "success":
                self.metrics['successful_requests'] += 1
            else:
                self.metrics['failed_requests'] += 1
            
            # Update average response time
            total_requests = self.metrics['total_requests']
            current_avg = self.metrics['avg_response_time_ms']
            self.metrics['avg_response_time_ms'] = (
                (current_avg * (total_requests - 1) + response.latency_ms) / total_requests
            )
            
            # Update cache metrics
            if response.cached:
                self.metrics['cache_hits'] += 1
            
            # Quality validation
            quality_report = None
            if quality_check and self.quality_validator and response.status == "success":
                quality_report = await self.quality_validator.validate_response(response)
                self.metrics['quality_scores'].append(quality_report.overall_score)
                
                # Keep only last 100 quality scores
                if len(self.metrics['quality_scores']) > 100:
                    self.metrics['quality_scores'] = self.metrics['quality_scores'][-100:]
                
                # Trigger quality callbacks
                for callback in self.quality_callbacks:
                    try:
                        await callback(quality_report)
                    except Exception as e:
                        self.logger.error(f"Quality callback error: {e}")
            
            # Trigger data callbacks
            result = {
                'response': response,
                'quality_report': quality_report,
                'request_id': request.request_id
            }
            
            for callback in self.data_callbacks:
                try:
                    await callback(result)
                except Exception as e:
                    self.logger.error(f"Data callback error: {e}")
            
            return result
            
        except Exception as e:
            self.metrics['failed_requests'] += 1
            self.logger.error(f"Data request failed: {e}")
            
            # Trigger error callbacks
            for callback in self.error_callbacks:
                try:
                    await callback(e, request)
                except Exception as cb_error:
                    self.logger.error(f"Error callback failed: {cb_error}")
            
            raise
    
    async def get_market_data(
        self, 
        exchange: str, 
        symbol: str, 
        priority: DataPriority = DataPriority.HIGH,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """Get market data with quality validation"""
        request = create_market_data_request(
            exchange=exchange,
            symbol=symbol,
            priority=priority,
            cache_ttl=30
        )
        
        return await self.request_data_with_quality_check(request, quality_check)
    
    async def get_orderbook_data(
        self, 
        exchange: str, 
        symbol: str, 
        limit: int = 20,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """Get order book data with quality validation"""
        request = create_orderbook_request(
            exchange=exchange,
            symbol=symbol,
            limit=limit,
            priority=DataPriority.CRITICAL
        )
        
        return await self.request_data_with_quality_check(request, quality_check)
    
    async def get_ohlcv_data(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: str = "1m",
        limit: int = 500,
        quality_check: bool = True
    ) -> Dict[str, Any]:
        """Get OHLCV data with quality validation"""
        request = create_ohlcv_request(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            priority=DataPriority.MEDIUM,
            cache_ttl=60
        )
        
        return await self.request_data_with_quality_check(request, quality_check)
    
    def add_data_callback(self, callback: Callable):
        """Add callback for successful data responses"""
        self.data_callbacks.append(callback)
    
    def add_quality_callback(self, callback: Callable):
        """Add callback for quality reports"""
        self.quality_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    async def _monitoring_loop(self):
        """Background monitoring and health checks"""
        while self.is_running:
            try:
                # Update uptime
                if self.start_time:
                    self.metrics['uptime_seconds'] = (
                        datetime.now() - self.start_time
                    ).total_seconds()
                
                # Log comprehensive metrics every 5 minutes
                await asyncio.sleep(300)
                await self._log_comprehensive_metrics()
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _log_comprehensive_metrics(self):
        """Log comprehensive system metrics"""
        if not self.is_running:
            return
        
        try:
            # Get system status
            status = await self.get_system_status()
            
            # Calculate derived metrics
            success_rate = (
                self.metrics['successful_requests'] / 
                max(1, self.metrics['total_requests'])
            )
            
            avg_quality = (
                sum(self.metrics['quality_scores']) / 
                max(1, len(self.metrics['quality_scores']))
            ) if self.metrics['quality_scores'] else 0.0
            
            # Log summary
            summary = {
                'uptime_hours': self.metrics['uptime_seconds'] / 3600,
                'total_requests': self.metrics['total_requests'],
                'success_rate': success_rate,
                'avg_response_time_ms': self.metrics['avg_response_time_ms'],
                'cache_hit_rate': (
                    self.metrics['cache_hits'] / 
                    max(1, self.metrics['total_requests'])
                ),
                'avg_quality_score': avg_quality,
                'system_health': status.get('health', 'unknown')
            }
            
            self.logger.info(f"📊 Data Manager Metrics: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to log metrics: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': self.metrics['uptime_seconds'],
            'metrics': self.metrics.copy()
        }
        
        # Add component statuses
        if self.data_ingestion:
            status['data_ingestion'] = self.data_ingestion.get_system_status()
        
        if self.scheduler:
            status['scheduler'] = self.scheduler.get_scheduler_status()
            status['performance'] = self.scheduler.get_performance_metrics()
        
        if self.quality_validator:
            status['quality_validation'] = self.quality_validator.get_validation_summary()
        
        # Calculate overall health
        health_score = self._calculate_health_score(status)
        status['health'] = 'healthy' if health_score > 0.8 else 'degraded' if health_score > 0.5 else 'unhealthy'
        status['health_score'] = health_score
        
        return status
    
    def _calculate_health_score(self, status: Dict[str, Any]) -> float:
        """Calculate overall system health score"""
        scores = []
        
        # Data ingestion health
        if 'data_ingestion' in status:
            di_status = status['data_ingestion']
            if di_status.get('status') == 'healthy':
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Scheduler health
        if 'scheduler' in status:
            scheduler_status = status['scheduler']
            if scheduler_status.get('active'):
                scores.append(1.0)
            else:
                scores.append(0.0)
        
        # Performance health
        if 'performance' in status:
            perf = status['performance']
            if perf.get('system_health') == 'healthy':
                scores.append(1.0)
            elif perf.get('system_health') == 'degraded':
                scores.append(0.7)
            else:
                scores.append(0.3)
        
        # Quality validation health
        if 'quality_validation' in status:
            quality = status['quality_validation']
            success_rate = quality.get('success_rate', 0)
            scores.append(success_rate)
        
        # Overall request success rate
        if self.metrics['total_requests'] > 0:
            success_rate = self.metrics['successful_requests'] / self.metrics['total_requests']
            scores.append(success_rate)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check"""
        health_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'unknown',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check if running
            if not self.is_running:
                health_results['issues'].append("Data manager is not running")
                health_results['recommendations'].append("Start the data manager")
                health_results['overall_status'] = 'critical'
                return health_results
            
            # Test data ingestion
            if self.data_ingestion:
                try:
                    test_request = create_market_data_request('kraken', 'BTC/USD')
                    test_result = await asyncio.wait_for(
                        self.data_ingestion.request_data(test_request),
                        timeout=30.0
                    )
                    health_results['components']['data_ingestion'] = {
                        'status': 'healthy' if test_result.status == 'success' else 'degraded',
                        'test_latency_ms': test_result.latency_ms
                    }
                except Exception as e:
                    health_results['components']['data_ingestion'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_results['issues'].append(f"Data ingestion test failed: {e}")
            
            # Check scheduler
            if self.scheduler:
                scheduler_status = self.scheduler.get_scheduler_status()
                health_results['components']['scheduler'] = {
                    'status': 'healthy' if scheduler_status['active'] else 'unhealthy',
                    'running_tasks': len(scheduler_status['running_tasks'])
                }
                
                if not scheduler_status['active']:
                    health_results['issues'].append("Scheduler is not active")
                    health_results['recommendations'].append("Restart the scheduler")
            
            # Check quality validator
            if self.quality_validator:
                quality_stats = self.quality_validator.get_validation_summary()
                avg_score = quality_stats.get('avg_quality_score', 0)
                health_results['components']['quality_validator'] = {
                    'status': 'healthy' if avg_score > 0.8 else 'degraded' if avg_score > 0.5 else 'unhealthy',
                    'avg_quality_score': avg_score
                }
                
                if avg_score < 0.8:
                    health_results['issues'].append(f"Low average quality score: {avg_score:.2f}")
                    health_results['recommendations'].append("Check data sources and validation thresholds")
            
            # Determine overall status
            component_statuses = [
                comp.get('status', 'unknown') 
                for comp in health_results['components'].values()
            ]
            
            if all(status == 'healthy' for status in component_statuses):
                health_results['overall_status'] = 'healthy'
            elif any(status == 'unhealthy' for status in component_statuses):
                health_results['overall_status'] = 'degraded'
            else:
                health_results['overall_status'] = 'healthy'
            
        except Exception as e:
            health_results['overall_status'] = 'critical'
            health_results['issues'].append(f"Health check failed: {e}")
            health_results['recommendations'].append("Investigate system errors")
        
        return health_results


# Factory functions
async def create_robust_data_manager(
    config: Optional[DataIngestionConfig] = None
) -> RobustDataManager:
    """Create and initialize robust data manager"""
    manager = RobustDataManager(config)
    await manager.initialize()
    return manager


def create_default_config(
    exchanges: Optional[Dict[str, Dict[str, str]]] = None
) -> DataIngestionConfig:
    """Create default configuration with optional exchange credentials"""
    config = DataIngestionConfig()
    
    if exchanges:
        for exchange_name, credentials in exchanges.items():
            if exchange_name in config.exchanges:
                config.exchanges[exchange_name].update(credentials)
    
    return config


# Usage example
async def example_usage():
    """Example usage of robust data manager"""
    # Create configuration
    config = create_default_config({
        'kraken': {
            'api_key': 'your_kraken_api_key',
            'secret': 'your_kraken_secret'
        }
    })
    
    # Create and start manager
    manager = await create_robust_data_manager(config)
    await manager.start()
    
    try:
        # Request market data
        result = await manager.get_market_data('kraken', 'BTC/USD')
        print(f"Market data: {result['response'].data}")
        
        if result['quality_report']:
            print(f"Quality score: {result['quality_report'].overall_score}")
        
        # Get system status
        status = await manager.get_system_status()
        print(f"System health: {status['health']}")
        
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())