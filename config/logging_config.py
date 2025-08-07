# config/logging_config.py
import logging
import logging.config
from pathlib import Path
from pythonjsonlogger import jsonlogger
from config.settings import config


def setup_logging() -> None:
    """
    Setup structured logging configuration with JSON formatting.
    Implements Dutch requirements for structured logging.
    """
    
    # Ensure logs directory exists
    logs_dir = Path(config.logs_directory)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(message)s"
            },
            "console": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "console",
                "level": config.log_level,
                "stream": "ext://sys.stdout"
            },
            "file_json": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "level": config.log_level,
                "filename": str(logs_dir / "cryptotrader.json"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "level": "ERROR",
                "filename": str(logs_dir / "errors.json"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file_json", "error_file"],
                "level": config.log_level,
                "propagate": False
            },
            "agents": {
                "handlers": ["console", "file_json"],
                "level": config.log_level,
                "propagate": False
            },
            "core": {
                "handlers": ["console", "file_json"],
                "level": config.log_level,
                "propagate": False
            },
            "utils": {
                "handlers": ["console", "file_json"],
                "level": config.log_level,
                "propagate": False
            },
            "aiohttp": {
                "level": "WARNING",
                "propagate": True
            },
            "ccxt": {
                "level": "WARNING",
                "propagate": True
            }
        }
    }
    
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Log the startup
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized", 
        extra={
            "component": "logging",
            "log_level": config.log_level,
            "app_name": config.app_name
        }
    )


def get_structured_logger(name: str, component: str = None) -> logging.Logger:
    """
    Get a logger with structured context.
    
    Args:
        name: Logger name (usually __name__)
        component: Component name for context
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if component:
        # Add component context to all log messages
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.component = component
            return record
            
        logging.setLogRecordFactory(record_factory)
    
    return logger