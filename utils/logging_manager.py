#!/usr/bin/env python3
"""
Advanced Logging Manager for CryptoSmartTrader V2
Provides comprehensive logging with daily rotation, structured JSON logging,
and performance monitoring for all system components.
"""

import logging
import json
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import shutil
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from dataclasses import dataclass, asdict
import traceback
import time

@dataclass
class LogEntry:
    """Structured log entry for JSON logging"""
    timestamp: str
    level: str
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class AdvancedLogger:
    """Advanced logging system with daily rotation and structured logging"""
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log directory
        self.daily_log_dir = self.base_log_dir / datetime.now().strftime("%Y-%m-%d")
        self.daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers for different components
        self.loggers = {}
        self.json_handlers = {}
        self.text_handlers = {}
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_timings = {}
        
        # Setup main loggers
        self._setup_main_loggers()
        
    def _setup_main_loggers(self):
        """Setup main system loggers"""
        components = [
            'system',
            'trading',
            'ml_models', 
            'predictions',
            'api_calls',
            'performance',
            'errors',
            'user_actions',
            'data_pipeline',
            'confidence_scoring'
        ]
        
        for component in components:
            self._create_component_logger(component)
    
    def _create_component_logger(self, component: str):
        """Create logger for specific component"""
        logger = logging.getLogger(f"crypto_trader.{component}")
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # JSON handler for structured logging
        json_log_file = self.daily_log_dir / f"{component}.json"
        json_handler = TimedRotatingFileHandler(
            json_log_file, 
            when='midnight', 
            interval=1, 
            backupCount=30,
            encoding='utf-8'
        )
        json_handler.setFormatter(self._get_json_formatter())
        json_handler.setLevel(logging.DEBUG)
        
        # Text handler for human-readable logs
        text_log_file = self.daily_log_dir / f"{component}.log"
        text_handler = TimedRotatingFileHandler(
            text_log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        text_handler.setFormatter(self._get_text_formatter())
        text_handler.setLevel(logging.INFO)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_console_formatter())
        console_handler.setLevel(logging.WARNING)
        
        logger.addHandler(json_handler)
        logger.addHandler(text_handler)
        logger.addHandler(console_handler)
        
        self.loggers[component] = logger
        self.json_handlers[component] = json_handler
        self.text_handlers[component] = text_handler
        
        return logger
    
    def _get_json_formatter(self):
        """JSON formatter for structured logging"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'component': record.name.split('.')[-1] if '.' in record.name else record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra data if available
                if hasattr(record, 'data'):
                    log_entry['data'] = record.data
                if hasattr(record, 'execution_time'):
                    log_entry['execution_time'] = record.execution_time
                if hasattr(record, 'correlation_id'):
                    log_entry['correlation_id'] = record.correlation_id
                
                # Add exception info if present
                if record.exc_info:
                    log_entry['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        return JSONFormatter()
    
    def _get_text_formatter(self):
        """Text formatter for human-readable logs"""
        return logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def _get_console_formatter(self):
        """Console formatter for development"""
        return logging.Formatter(
            '%(levelname)s:%(name)s:%(message)s'
        )
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get logger for specific component"""
        if component not in self.loggers:
            self._create_component_logger(component)
        return self.loggers[component]
    
    def log_prediction(self, coin: str, prediction_data: Dict[str, Any], confidence: float):
        """Log ML prediction with structured data"""
        logger = self.get_logger('predictions')
        
        extra_data = {
            'coin': coin,
            'confidence': confidence,
            'prediction_data': prediction_data,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"ML Prediction generated for {coin} with {confidence:.3f} confidence",
            extra={'data': extra_data}
        )
    
    def log_confidence_scoring(self, coin: str, confidence_details: Dict[str, Any]):
        """Log confidence scoring details"""
        logger = self.get_logger('confidence_scoring')
        
        extra_data = {
            'coin': coin,
            'confidence_details': confidence_details,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Confidence scoring for {coin}: {confidence_details.get('final_confidence', 'N/A')}",
            extra={'data': extra_data}
        )
    
    def log_api_call(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Log API calls with performance metrics"""
        logger = self.get_logger('api_calls')
        
        extra_data = {
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"API Call: {method} {endpoint} -> {status_code} ({response_time:.3f}s)",
            extra={'data': extra_data, 'execution_time': response_time}
        )
    
    def log_performance(self, operation: str, execution_time: float, additional_metrics: Dict[str, Any] = None):
        """Log performance metrics"""
        logger = self.get_logger('performance')
        
        # Track operation timings
        if operation not in self.operation_timings:
            self.operation_timings[operation] = []
        self.operation_timings[operation].append(execution_time)
        
        extra_data = {
            'operation': operation,
            'execution_time': execution_time,
            'additional_metrics': additional_metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Performance: {operation} completed in {execution_time:.3f}s",
            extra={'data': extra_data, 'execution_time': execution_time}
        )
    
    def log_user_action(self, action: str, user_id: str = None, session_data: Dict[str, Any] = None):
        """Log user actions and interactions"""
        logger = self.get_logger('user_actions')
        
        extra_data = {
            'action': action,
            'user_id': user_id,
            'session_data': session_data or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"User Action: {action}",
            extra={'data': extra_data, 'user_id': user_id}
        )
    
    def log_error(self, component: str, error: Exception, context: Dict[str, Any] = None):
        """Log errors with full context"""
        logger = self.get_logger('errors')
        
        extra_data = {
            'component': component,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.error(
            f"Error in {component}: {error}",
            extra={'data': extra_data},
            exc_info=True
        )
    
    def log_data_pipeline(self, stage: str, records_processed: int, success_rate: float, timing: float):
        """Log data pipeline operations"""
        logger = self.get_logger('data_pipeline')
        
        extra_data = {
            'stage': stage,
            'records_processed': records_processed,
            'success_rate': success_rate,
            'timing': timing,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(
            f"Data Pipeline: {stage} processed {records_processed} records ({success_rate:.1%} success) in {timing:.3f}s",
            extra={'data': extra_data, 'execution_time': timing}
        )
    
    def generate_daily_summary(self) -> Dict[str, Any]:
        """Generate daily summary of all logged activities"""
        summary = {
            'date': date.today().isoformat(),
            'log_directory': str(self.daily_log_dir),
            'components': list(self.loggers.keys()),
            'performance_metrics': {},
            'summary_generated_at': datetime.now().isoformat()
        }
        
        # Calculate performance summaries
        for operation, timings in self.operation_timings.items():
            if timings:
                summary['performance_metrics'][operation] = {
                    'count': len(timings),
                    'avg_time': sum(timings) / len(timings),
                    'min_time': min(timings),
                    'max_time': max(timings),
                    'total_time': sum(timings)
                }
        
        # Save summary to file
        summary_file = self.daily_log_dir / "daily_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_dir in self.base_log_dir.iterdir():
            if log_dir.is_dir():
                try:
                    dir_date = datetime.strptime(log_dir.name, "%Y-%m-%d")
                    if dir_date < cutoff_date:
                        shutil.rmtree(log_dir)
                        print(f"Cleaned up old log directory: {log_dir}")
                except ValueError:
                    # Skip directories that don't match date format
                    pass

# Global logger instance
_global_logger = None
_logger_lock = threading.Lock()

def get_advanced_logger() -> AdvancedLogger:
    """Get global logger instance (singleton)"""
    global _global_logger
    
    if _global_logger is None:
        with _logger_lock:
            if _global_logger is None:
                _global_logger = AdvancedLogger()
    
    return _global_logger

# Convenience functions
def log_prediction(coin: str, prediction_data: Dict[str, Any], confidence: float):
    """Convenience function for logging predictions"""
    get_advanced_logger().log_prediction(coin, prediction_data, confidence)

def log_confidence_scoring(coin: str, confidence_details: Dict[str, Any]):
    """Convenience function for logging confidence scoring"""
    get_advanced_logger().log_confidence_scoring(coin, confidence_details)

def log_api_call(endpoint: str, method: str, status_code: int, response_time: float):
    """Convenience function for logging API calls"""
    get_advanced_logger().log_api_call(endpoint, method, status_code, response_time)

def log_performance(operation: str, execution_time: float, additional_metrics: Dict[str, Any] = None):
    """Convenience function for logging performance"""
    get_advanced_logger().log_performance(operation, execution_time, additional_metrics)

def log_user_action(action: str, user_id: str = None, session_data: Dict[str, Any] = None):
    """Convenience function for logging user actions"""
    get_advanced_logger().log_user_action(action, user_id, session_data)

def log_error(component: str, error: Exception, context: Dict[str, Any] = None):
    """Convenience function for logging errors"""
    get_advanced_logger().log_error(component, error, context)

def log_data_pipeline(stage: str, records_processed: int, success_rate: float, timing: float):
    """Convenience function for logging data pipeline operations"""
    get_advanced_logger().log_data_pipeline(stage, records_processed, success_rate, timing)

# Context manager for performance timing
class PerformanceTimer:
    """Context manager for automatic performance timing"""
    
    def __init__(self, operation_name: str, logger_component: str = 'performance'):
        self.operation_name = operation_name
        self.logger_component = logger_component
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        execution_time = self.end_time - self.start_time
        
        get_advanced_logger().log_performance(
            self.operation_name, 
            execution_time,
            {'success': exc_type is None}
        )
        
        if exc_type is not None:
            get_advanced_logger().log_error(
                self.logger_component,
                exc_val,
                {'operation': self.operation_name, 'execution_time': execution_time}
            )