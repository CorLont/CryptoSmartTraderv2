#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Daily Logging Configuration
Centralized daily logging system that organizes all logs by date for comprehensive monitoring
"""

import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import json

class DailyLogManager:
    """Manages daily organized logging for CryptoSmartTrader system"""
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.current_date = datetime.now().strftime('%Y-%m-%d')
        self.daily_log_dir = self.base_log_dir / self.current_date
        
        # Create daily log directory
        self.daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log categories for organized monitoring
        self.log_categories = {
            'system': 'system_health.log',
            'market_scanner': 'market_scanner.log', 
            'ml_predictor': 'ml_predictor.log',
            'sentiment_agent': 'sentiment_agent.log',
            'whale_detector': 'whale_detector.log',
            'trade_executor': 'trade_executor.log',
            'orchestrator': 'orchestrator.log',
            'errors': 'errors.log',
            'performance': 'performance.log',
            'api_calls': 'api_calls.log',
            'security': 'security.log'
        }
        
        # Setup loggers
        self.loggers = {}
        self._setup_daily_loggers()
        
        print(f"📊 Daily logging system initialized")
        print(f"📁 Log location: {self.daily_log_dir}")
        
    def _setup_daily_loggers(self):
        """Setup specialized loggers for each category"""
        
        # Custom formatter with timestamp and detailed info
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # JSON formatter for structured logging
        json_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        
        for category, filename in self.log_categories.items():
            logger = logging.getLogger(f'cryptotrader.{category}')
            logger.setLevel(logging.DEBUG)
            
            # Clear existing handlers
            logger.handlers.clear()
            
            # File handler for this category
            file_handler = logging.handlers.RotatingFileHandler(
                self.daily_log_dir / filename,
                maxBytes=50 * 1024 * 1024,  # 50MB per file
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(detailed_formatter)
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler for critical messages
            if category in ['system', 'errors', 'orchestrator']:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                console_handler.setLevel(logging.WARNING)
                logger.addHandler(console_handler)
            
            logger.addHandler(file_handler)
            self.loggers[category] = logger
            
        # Setup master daily summary logger
        self._setup_summary_logger()
        
    def _setup_summary_logger(self):
        """Setup master summary logger for daily overview"""
        summary_logger = logging.getLogger('cryptotrader.daily_summary')
        summary_logger.setLevel(logging.INFO)
        summary_logger.handlers.clear()
        
        # Daily summary file
        summary_handler = logging.FileHandler(
            self.daily_log_dir / 'daily_summary.log',
            encoding='utf-8'
        )
        summary_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        summary_logger.addHandler(summary_handler)
        self.loggers['daily_summary'] = summary_logger
        
    def get_logger(self, category: str) -> logging.Logger:
        """Get logger for specific category"""
        if category not in self.loggers:
            raise ValueError(f"Unknown log category: {category}. Available: {list(self.log_categories.keys())}")
        return self.loggers[category]
    
    def log_system_check(self, check_name: str, success: bool, details: str, error: Optional[Exception] = None):
        """Log system health check results with full details"""
        logger = self.get_logger('system')
        
        status = "SUCCESS" if success else "FAILED"
        message = f"System Check [{check_name}] {status}: {details}"
        
        if success:
            logger.info(message)
        else:
            logger.error(message)
            if error:
                logger.error(f"Error details: {error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
    def log_trading_opportunity(self, coin: str, timeframe: str, score: int, details: Dict):
        """Log detected trading opportunities"""
        logger = self.get_logger('market_scanner')
        logger.info(f"Trading opportunity: {coin} {timeframe} (score: {score}) - {json.dumps(details)}")
        
    def log_ml_prediction(self, coin: str, prediction: Dict, confidence: float, model_info: Dict):
        """Log ML predictions with confidence and model details"""
        logger = self.get_logger('ml_predictor')
        logger.info(f"ML Prediction [{coin}] Confidence: {confidence:.3f} - {json.dumps(prediction)} - Model: {json.dumps(model_info)}")
        
    def log_sentiment_analysis(self, source: str, sentiment_score: float, volume: int, details: Dict):
        """Log sentiment analysis results"""
        logger = self.get_logger('sentiment_agent')
        logger.info(f"Sentiment [{source}] Score: {sentiment_score:.3f} Volume: {volume} - {json.dumps(details)}")
        
    def log_whale_activity(self, transaction_hash: str, amount: float, coin: str, exchange: str, details: Dict):
        """Log whale detection events"""
        logger = self.get_logger('whale_detector')
        logger.warning(f"Whale Activity [{coin}] {amount:,.2f} on {exchange} - TX: {transaction_hash} - {json.dumps(details)}")
        
    def log_trade_execution(self, action: str, coin: str, amount: float, price: float, success: bool, details: Dict):
        """Log trade execution results"""
        logger = self.get_logger('trade_executor')
        level = logger.info if success else logger.error
        status = "SUCCESS" if success else "FAILED"
        level(f"Trade {action} [{coin}] {amount} @ {price} {status} - {json.dumps(details)}")
        
    def log_orchestrator_event(self, event_type: str, agent: str, status: str, details: str):
        """Log orchestrator coordination events"""
        logger = self.get_logger('orchestrator')
        logger.info(f"Orchestrator [{event_type}] Agent: {agent} Status: {status} - {details}")
        
    def log_api_call(self, endpoint: str, method: str, status_code: int, response_time: float, error: Optional[str] = None):
        """Log API call performance and status"""
        logger = self.get_logger('api_calls')
        message = f"API {method} {endpoint} - Status: {status_code} - Time: {response_time:.3f}s"
        
        if error:
            message += f" - Error: {error}"
            logger.error(message)
        elif status_code >= 400:
            logger.warning(message)
        else:
            logger.info(message)
            
    def log_performance_metric(self, metric_name: str, value: float, unit: str, context: Dict):
        """Log performance metrics"""
        logger = self.get_logger('performance')
        logger.info(f"Performance [{metric_name}] {value:.3f}{unit} - Context: {json.dumps(context)}")
        
    def log_security_event(self, event_type: str, severity: str, details: str, source_ip: Optional[str] = None):
        """Log security-related events"""
        logger = self.get_logger('security')
        message = f"Security [{event_type}] Severity: {severity} - {details}"
        if source_ip:
            message += f" - Source IP: {source_ip}"
            
        if severity.upper() in ['HIGH', 'CRITICAL']:
            logger.error(message)
        elif severity.upper() == 'MEDIUM':
            logger.warning(message)
        else:
            logger.info(message)
            
    def log_error_with_context(self, error: Exception, context: Dict, category: str = 'errors'):
        """Log errors with full context and traceback"""
        logger = self.get_logger(category)
        logger.error(f"Error: {str(error)}")
        logger.error(f"Context: {json.dumps(context)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        
    def create_daily_summary(self) -> Dict:
        """Create comprehensive daily summary report"""
        summary = {
            'date': self.current_date,
            'log_files': [],
            'file_sizes': {},
            'log_counts': {},
            'generated_at': datetime.now().isoformat()
        }
        
        # Analyze each log file
        for category, filename in self.log_categories.items():
            file_path = self.daily_log_dir / filename
            
            if file_path.exists():
                # File size
                size_bytes = file_path.stat().st_size
                summary['file_sizes'][category] = size_bytes
                summary['log_files'].append(filename)
                
                # Count log entries
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    summary['log_counts'][category] = line_count
                except Exception as e:
                    summary['log_counts'][category] = f"Error reading: {e}"
                    
        # Write summary to file
        summary_file = self.daily_log_dir / 'daily_log_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        # Log summary to daily summary logger
        summary_logger = self.get_logger('daily_summary')
        summary_logger.info(f"Daily log summary generated - {len(summary['log_files'])} active log files")
        summary_logger.info(f"Total log entries: {sum(c for c in summary['log_counts'].values() if isinstance(c, int))}")
        summary_logger.info(f"Total log size: {sum(summary['file_sizes'].values()) / 1024 / 1024:.2f} MB")
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log directories older than specified days"""
        cutoff_date = datetime.now()
        cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
        
        cleaned_dirs = []
        for log_dir in self.base_log_dir.iterdir():
            if log_dir.is_dir() and log_dir.name < cutoff_date.strftime('%Y-%m-%d'):
                try:
                    import shutil
                    shutil.rmtree(log_dir)
                    cleaned_dirs.append(log_dir.name)
                except Exception as e:
                    self.log_error_with_context(e, {'action': 'cleanup_old_logs', 'dir': str(log_dir)})
                    
        if cleaned_dirs:
            self.log_system_check('log_cleanup', True, f"Cleaned {len(cleaned_dirs)} old log directories: {cleaned_dirs}")
            
    def get_log_file_paths(self) -> Dict[str, Path]:
        """Get all current log file paths organized by category"""
        return {category: self.daily_log_dir / filename for category, filename in self.log_categories.items()}

# Global daily log manager instance
daily_logger = None

def get_daily_logger() -> DailyLogManager:
    """Get or create the global daily logger instance"""
    global daily_logger
    if daily_logger is None:
        daily_logger = DailyLogManager()
    return daily_logger

def setup_daily_logging():
    """Initialize the daily logging system"""
    return get_daily_logger()

# Convenience functions for easy logging
def log_system(check_name: str, success: bool, details: str, error: Optional[Exception] = None):
    """Convenience function for system logging"""
    get_daily_logger().log_system_check(check_name, success, details, error)

def log_trading(coin: str, timeframe: str, score: int, details: Dict):
    """Convenience function for trading opportunity logging"""
    get_daily_logger().log_trading_opportunity(coin, timeframe, score, details)

def log_ml(coin: str, prediction: Dict, confidence: float, model_info: Dict):
    """Convenience function for ML prediction logging"""
    get_daily_logger().log_ml_prediction(coin, prediction, confidence, model_info)

def log_error(error: Exception, context: Dict, category: str = 'errors'):
    """Convenience function for error logging"""
    get_daily_logger().log_error_with_context(error, context, category)

def log_performance(metric_name: str, value: float, unit: str, context: Dict):
    """Convenience function for performance logging"""
    get_daily_logger().log_performance_metric(metric_name, value, unit, context)

if __name__ == "__main__":
    # Test the daily logging system
    logger_manager = setup_daily_logging()
    
    # Test various log types
    log_system("test_check", True, "System test completed successfully")
    log_trading("BTC/USD", "15m", 4, {"rsi": 65, "volume": 1000000})
    log_ml("ETH/USD", {"direction": "up", "target": 2500}, 0.85, {"model": "LSTM", "version": "v1.0"})
    
    # Create daily summary
    summary = logger_manager.create_daily_summary()
    print(f"Daily summary created: {summary}")