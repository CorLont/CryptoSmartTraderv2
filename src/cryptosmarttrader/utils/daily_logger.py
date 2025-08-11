"""
Daily Logger - Organized logging system with daily folders
Saves all logs in structured daily directories
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import threading
from typing import Dict, Optional
import json
import time
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class DailyLogManager:
    """
    Advanced logging manager that organizes logs by date
    Creates daily folders and manages log rotation
    """
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.current_date = None
        self.current_log_dir = None
        self.loggers = {}
        self.handlers = {}
        self.lock = threading.Lock()
        
        # Ensure base directory exists
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Initialize for today
        self._setup_daily_logging()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _setup_daily_logging(self):
        """Setup logging for current date"""
        
        today = datetime.now().date()
        
        if self.current_date != today:
            with self.lock:
                # Create today's log directory
                today_str = today.strftime('%Y-%m-%d')
                self.current_log_dir = self.base_log_dir / today_str
                self.current_log_dir.mkdir(exist_ok=True)
                
                # Update current date
                self.current_date = today
                
                # Clear existing handlers to prevent duplicate logs
                self._clear_existing_handlers()
                
                # Setup new handlers for today
                self._setup_handlers()
                
                print(f"📁 Daily logging initialized for {today_str}")
                print(f"📂 Log directory: {self.current_log_dir}")
    
    def _clear_existing_handlers(self):
        """Clear existing handlers to prevent duplicates"""
        
        for logger_name, logger in self.loggers.items():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
        
        self.handlers.clear()
    
    def _setup_handlers(self):
        """Setup logging handlers for current day"""
        
        # Main application log
        self._create_handler(
            'main',
            'main_application.log',
            logging.INFO,
            format_style='detailed'
        )
        
        # Trading opportunities log
        self._create_handler(
            'trading',
            'trading_opportunities.log',
            logging.INFO,
            format_style='trading'
        )
        
        # Error log
        self._create_handler(
            'error',
            'errors.log',
            logging.ERROR,
            format_style='detailed'
        )
        
        # Security log
        self._create_handler(
            'security',
            'security.log',
            logging.WARNING,
            format_style='security'
        )
        
        # ML predictions log
        self._create_handler(
            'ml_predictions',
            'ml_predictions.log',
            logging.INFO,
            format_style='ml'
        )
        
        # API calls log
        self._create_handler(
            'api',
            'api_calls.log',
            logging.DEBUG,
            format_style='api'
        )
        
        # Performance log
        self._create_handler(
            'performance',
            'performance.log',
            logging.INFO,
            format_style='performance'
        )
        
        # System health log
        self._create_handler(
            'health',
            'system_health.log',
            logging.INFO,
            format_style='health'
        )
    
    def _create_handler(self, 
                       handler_name: str,
                       filename: str,
                       level: int,
                       format_style: str = 'detailed'):
        """Create a specific log handler"""
        
        file_path = self.current_log_dir / filename
        
        # Create rotating file handler
        handler = RotatingFileHandler(
            file_path,
            maxBytes=50 * 1024 * 1024,  # 50MB max file size
            backupCount=5,  # Keep 5 backup files
            encoding='utf-8'
        )
        
        handler.setLevel(level)
        
        # Set formatter based on style
        formatter = self._get_formatter(format_style)
        handler.setFormatter(formatter)
        
        # Store handler
        self.handlers[handler_name] = handler
        
        # Create logger if it doesn't exist
        if handler_name not in self.loggers:
            logger = logging.getLogger(f'cryptotrader.{handler_name}')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            self.loggers[handler_name] = logger
        else:
            self.loggers[handler_name].addHandler(handler)
    
    def _get_formatter(self, style: str) -> logging.Formatter:
        """Get formatter based on style"""
        
        formatters = {
            'detailed': logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'trading': logging.Formatter(
                '%(asctime)s - TRADING - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'security': logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(funcName)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'ml': logging.Formatter(
                '%(asctime)s - ML - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'api': logging.Formatter(
                '%(asctime)s - API - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'performance': logging.Formatter(
                '%(asctime)s - PERF - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ),
            'health': logging.Formatter(
                '%(asctime)s - HEALTH - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        }
        
        return formatters.get(style, formatters['detailed'])
    
    def get_logger(self, logger_type: str = 'main') -> logging.Logger:
        """Get logger for specific type"""
        
        # Check if we need to roll over to new day
        self._setup_daily_logging()
        
        if logger_type in self.loggers:
            return self.loggers[logger_type]
        else:
            # Return main logger as fallback
            return self.loggers.get('main', logging.getLogger())
    
    def log_trading_opportunity(self, 
                              coin: str,
                              timeframe: str,
                              score: int,
                              confidence: float = None,
                              details: Dict = None):
        """Log trading opportunity"""
        
        logger = self.get_logger('trading')
        
        message = f"Trading opportunity: {coin} {timeframe} (score: {score}"
        
        if confidence is not None:
            message += f", confidence: {confidence:.3f}"
        
        message += ")"
        
        if details:
            message += f" - Details: {json.dumps(details, default=str)}"
        
        logger.info(message)
    
    def log_ml_prediction(self,
                         coin: str,
                         horizon: str,
                         prediction: float,
                         confidence: float,
                         model_type: str = None):
        """Log ML prediction"""
        
        logger = self.get_logger('ml_predictions')
        
        message = f"ML Prediction: {coin} {horizon} = {prediction:.4f} (conf: {confidence:.3f})"
        
        if model_type:
            message += f" [{model_type}]"
        
        logger.info(message)
    
    def log_api_call(self,
                    exchange: str,
                    endpoint: str,
                    status: str,
                    response_time: float = None,
                    error: str = None):
        """Log API call"""
        
        logger = self.get_logger('api')
        
        message = f"API Call: {exchange} - {endpoint} - {status}"
        
        if response_time is not None:
            message += f" ({response_time:.3f}s)"
        
        if error:
            message += f" - Error: {error}"
        
        if status.lower() == 'error':
            logger.error(message)
        else:
            logger.info(message)
    
    def log_security_event(self,
                          event_type: str,
                          user: str,
                          action: str,
                          success: bool,
                          details: str = None):
        """Log security event"""
        
        logger = self.get_logger('security')
        
        status = "SUCCESS" if success else "FAILED"
        message = f"Security Event: {event_type} - {user} - {action} - {status}"
        
        if details:
            message += f" - {details}"
        
        if success:
            logger.info(message)
        else:
            logger.warning(message)
    
    def log_performance_metric(self,
                              metric_name: str,
                              value: float,
                              unit: str = None,
                              context: Dict = None):
        """Log performance metric"""
        
        logger = self.get_logger('performance')
        
        message = f"Performance: {metric_name} = {value}"
        
        if unit:
            message += f" {unit}"
        
        if context:
            message += f" - Context: {json.dumps(context, default=str)}"
        
        logger.info(message)
    
    def log_health_status(self,
                         component: str,
                         status: str,
                         details: Dict = None):
        """Log system health status"""
        
        logger = self.get_logger('health')
        
        message = f"Health Check: {component} - {status}"
        
        if details:
            message += f" - {json.dumps(details, default=str)}"
        
        if status.upper() in ['ERROR', 'CRITICAL', 'FAILED']:
            logger.error(message)
        elif status.upper() in ['WARNING', 'DEGRADED']:
            logger.warning(message)
        else:
            logger.info(message)
    
    def _start_cleanup_thread(self):
        """Start background thread for log cleanup"""
        
        def cleanup_old_logs():
            while True:
                try:
                    self._cleanup_old_logs()
                    time.sleep(3600)  # Run every hour
                except Exception as e:
                    print(f"Log cleanup error: {e}")
                    time.sleep(3600)
        
        cleanup_thread = threading.Thread(target=cleanup_old_logs, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_logs(self):
        """Clean up old log directories"""
        
        # Keep logs for 30 days
        cutoff_date = datetime.now() - timedelta(days=30)
        
        try:
            for item in self.base_log_dir.iterdir():
                if item.is_dir():
                    try:
                        # Parse directory name as date
                        dir_date = datetime.strptime(item.name, '%Y-%m-%d').date()
                        
                        if datetime.combine(dir_date, datetime.min.time()) < cutoff_date:
                            # Archive or delete old directory
                            self._archive_old_directory(item)
                            
                    except ValueError:
                        # Skip directories that don't match date format
                        continue
                        
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def _archive_old_directory(self, directory: Path):
        """Archive old log directory"""
        
        # For now, just delete old directories
        # In production, you might want to compress and archive
        import shutil
        
        try:
            shutil.rmtree(directory)
            print(f"Cleaned up old logs: {directory.name}")
        except Exception as e:
            print(f"Failed to clean up {directory.name}: {e}")
    
    def get_log_summary(self) -> Dict:
        """Get summary of current day's logs"""
        
        summary = {
            'date': self.current_date.strftime('%Y-%m-%d') if self.current_date else None,
            'log_directory': str(self.current_log_dir) if self.current_log_dir else None,
            'active_loggers': list(self.loggers.keys()),
            'log_files': []
        }
        
        if self.current_log_dir and self.current_log_dir.exists():
            for log_file in self.current_log_dir.glob('*.log'):
                try:
                    stat = log_file.stat()
                    summary['log_files'].append({
                        'name': log_file.name,
                        'size_bytes': stat.st_size,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception:
                    continue
        
        return summary

# Global logger instance
_daily_logger = None

def get_daily_logger() -> DailyLogManager:
    """Get global daily logger instance"""
    global _daily_logger
    
    if _daily_logger is None:
        _daily_logger = DailyLogManager()
    
    return _daily_logger

def setup_system_logging():
    """Setup system-wide logging with daily rotation"""
    
    daily_logger = get_daily_logger()
    
    # Configure root logger to use our system
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    print("📊 Daily logging system initialized")
    print(f"📁 Log location: {daily_logger.current_log_dir}")
    
    return daily_logger

# Convenience functions
def log_trading(coin: str, timeframe: str, score: int, **kwargs):
    """Quick trading log"""
    get_daily_logger().log_trading_opportunity(coin, timeframe, score, **kwargs)

def log_ml(coin: str, horizon: str, prediction: float, confidence: float, **kwargs):
    """Quick ML prediction log"""
    get_daily_logger().log_ml_prediction(coin, horizon, prediction, confidence, **kwargs)

def log_api(exchange: str, endpoint: str, status: str, **kwargs):
    """Quick API call log"""
    get_daily_logger().log_api_call(exchange, endpoint, status, **kwargs)

def log_security(event_type: str, user: str, action: str, success: bool, **kwargs):
    """Quick security event log"""
    get_daily_logger().log_security_event(event_type, user, action, success, **kwargs)

def log_performance(metric: str, value: float, **kwargs):
    """Quick performance metric log"""
    get_daily_logger().log_performance_metric(metric, value, **kwargs)

def log_health(component: str, status: str, **kwargs):
    """Quick health status log"""
    get_daily_logger().log_health_status(component, status, **kwargs)