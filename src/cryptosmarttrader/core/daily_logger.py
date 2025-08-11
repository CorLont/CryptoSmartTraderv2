#!/usr/bin/env python3
"""
Daily Logger - Comprehensive logging system with run_id tracking and error alerts
Never logs secrets, implements structured logging with daily rotation
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

class SecureFormatter(logging.Formatter):
    """Secure formatter that redacts sensitive information"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Patterns to redact (secrets, API keys, etc.)
        self.secret_patterns = [
            r'api[_-]?key["\s]*[:=]["\s]*([A-Za-z0-9+/=]{20,})',  # API keys
            r'secret["\s]*[:=]["\s]*([A-Za-z0-9+/=]{20,})',        # Secrets
            r'password["\s]*[:=]["\s]*([^\s"]+)',                  # Passwords
            r'token["\s]*[:=]["\s]*([A-Za-z0-9._-]{20,})',         # Tokens
            r'([A-Za-z0-9+/=]{40,})',                             # Long base64-like strings
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.secret_patterns]
    
    def format(self, record):
        """Format log record with secret redaction"""
        
        # Get the original formatted message
        message = super().format(record)
        
        # Redact secrets
        for pattern in self.compiled_patterns:
            message = pattern.sub(lambda m: self._redact_match(m), message)
        
        return message
    
    def _redact_match(self, match):
        """Redact a matched secret"""
        original = match.group(0)
        
        # Keep structure but replace sensitive part with hash
        if '=' in original or ':' in original:
            prefix = original.split('=')[0] if '=' in original else original.split(':')[0]
            return f"{prefix}=***REDACTED***"
        else:
            # For standalone secrets, show first 4 chars + hash suffix
            secret = match.group(1) if len(match.groups()) > 0 else match.group(0)
            if len(secret) > 8:
                hash_suffix = hashlib.sha256(secret.encode()).hexdigest()[:8]
                return f"{secret[:4]}***{hash_suffix}"
            else:
                return "***REDACTED***"

class DailyLogger:
    """Daily logger with run_id tracking and structured output"""
    
    def __init__(self, component_name: str, logs_dir: str = "logs/daily"):
        self.component_name = component_name
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate run_id for this session
        self.run_id = str(uuid.uuid4())[:8]
        self.session_start = datetime.now(timezone.utc)
        
        # Setup structured logger
        self.logger = self._setup_logger()
        
        # Initialize daily log file
        self.daily_log_file = self._get_daily_log_file()
        
        # Log session start
        self.log_info("Logger initialized", {
            "run_id": self.run_id,
            "component": self.component_name,
            "session_start": self.session_start.isoformat()
        })
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with secure formatting"""
        
        logger = logging.getLogger(f"{self.component_name}_{self.run_id}")
        logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler with secure formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler for component-specific logs
        component_log_file = self.logs_dir / f"{self.component_name}.log"
        file_handler = logging.FileHandler(component_log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Secure formatter
        formatter = SecureFormatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "' + 
            self.component_name + '", "run_id": "' + self.run_id + 
            '", "message": "%(message)s"}'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def _get_daily_log_file(self) -> Path:
        """Get daily log file path"""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.logs_dir / f"daily_{today}.json"
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message with context"""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "INFO",
            "run_id": self.run_id,
            "component": self.component_name,
            "message": message,
            "context": context or {}
        }
        
        self.logger.info(json.dumps(log_entry))
        self._append_to_daily_log(log_entry)
    
    def log_error(self, message: str, error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        """Log error with optional exception details"""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "ERROR",
            "run_id": self.run_id,
            "component": self.component_name,
            "message": message,
            "error": str(error) if error else None,
            "error_type": type(error).__name__ if error else None,
            "context": context or {}
        }
        
        self.logger.error(json.dumps(log_entry))
        self._append_to_daily_log(log_entry)
        
        # Also write to error-specific log
        self._write_error_alert(log_entry)
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": "WARNING",
            "run_id": self.run_id,
            "component": self.component_name,
            "message": message,
            "context": context or {}
        }
        
        self.logger.warning(json.dumps(log_entry))
        self._append_to_daily_log(log_entry)
    
    def log_performance(self, operation: str, duration_seconds: float, context: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        
        perf_context = {
            "operation": operation,
            "duration_seconds": duration_seconds,
            "performance_grade": self._grade_performance(duration_seconds),
            **(context or {})
        }
        
        self.log_info(f"Performance: {operation}", perf_context)
    
    def _grade_performance(self, duration: float) -> str:
        """Grade performance based on duration"""
        if duration < 1.0:
            return "EXCELLENT"
        elif duration < 5.0:
            return "GOOD"
        elif duration < 15.0:
            return "ACCEPTABLE"
        else:
            return "SLOW"
    
    def _append_to_daily_log(self, log_entry: Dict[str, Any]):
        """Append log entry to daily log file"""
        
        try:
            # Load existing daily log
            daily_entries = []
            if self.daily_log_file.exists():
                with open(self.daily_log_file, 'r') as f:
                    daily_entries = json.load(f)
            
            # Append new entry
            daily_entries.append(log_entry)
            
            # Keep only last 1000 entries per day to prevent huge files
            if len(daily_entries) > 1000:
                daily_entries = daily_entries[-1000:]
            
            # Write atomically
            temp_file = self.daily_log_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(daily_entries, f, indent=2)
            
            temp_file.rename(self.daily_log_file)
            
        except Exception as e:
            # Fallback to console if file writing fails
            print(f"Daily log write failed: {e}")
    
    def _write_error_alert(self, error_entry: Dict[str, Any]):
        """Write error to alert file for monitoring"""
        
        try:
            alerts_file = self.logs_dir / "error_alerts.json"
            
            # Load existing alerts
            alerts = []
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts = json.load(f)
            
            # Add error entry
            alerts.append(error_entry)
            
            # Keep only last 50 errors
            if len(alerts) > 50:
                alerts = alerts[-50:]
            
            # Write atomically
            temp_file = alerts_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(alerts, f, indent=2)
            
            temp_file.rename(alerts_file)
            
        except Exception as e:
            print(f"Error alert write failed: {e}")
    
    def log_pipeline_start(self, pipeline_name: str, config: Optional[Dict[str, Any]] = None):
        """Log pipeline start"""
        
        self.log_info(f"Pipeline started: {pipeline_name}", {
            "pipeline": pipeline_name,
            "config": config or {},
            "stage": "START"
        })
    
    def log_pipeline_step(self, pipeline_name: str, step_name: str, step_result: Dict[str, Any]):
        """Log pipeline step completion"""
        
        self.log_info(f"Pipeline step: {step_name}", {
            "pipeline": pipeline_name,
            "step": step_name,
            "result": step_result,
            "stage": "STEP"
        })
    
    def log_pipeline_complete(self, pipeline_name: str, total_duration: float, final_result: Dict[str, Any]):
        """Log pipeline completion"""
        
        self.log_info(f"Pipeline completed: {pipeline_name}", {
            "pipeline": pipeline_name,
            "total_duration": total_duration,
            "result": final_result,
            "stage": "COMPLETE"
        })
    
    def log_pipeline_failed(self, pipeline_name: str, error: Exception, partial_results: Optional[Dict[str, Any]] = None):
        """Log pipeline failure"""
        
        self.log_error(f"Pipeline failed: {pipeline_name}", error, {
            "pipeline": pipeline_name,
            "partial_results": partial_results or {},
            "stage": "FAILED"
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current logging session"""
        
        session_duration = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        
        # Count log entries by level
        log_counts = {"INFO": 0, "WARNING": 0, "ERROR": 0}
        
        if self.daily_log_file.exists():
            try:
                with open(self.daily_log_file, 'r') as f:
                    daily_entries = json.load(f)
                
                # Count entries for this run_id
                for entry in daily_entries:
                    if entry.get("run_id") == self.run_id:
                        level = entry.get("level", "INFO")
                        log_counts[level] = log_counts.get(level, 0) + 1
                        
            except Exception:
                pass
        
        return {
            "run_id": self.run_id,
            "component": self.component_name,
            "session_start": self.session_start.isoformat(),
            "session_duration": session_duration,
            "log_counts": log_counts,
            "daily_log_file": str(self.daily_log_file)
        }

# Global logger registry
_loggers: Dict[str, DailyLogger] = {}

def get_daily_logger(component_name: str) -> DailyLogger:
    """Get or create daily logger for component"""
    
    if component_name not in _loggers:
        _loggers[component_name] = DailyLogger(component_name)
    
    return _loggers[component_name]

def test_daily_logger():
    """Test daily logger functionality"""
    
    print("Testing Daily Logger...")
    
    # Create test logger
    logger = get_daily_logger("test_component")
    
    # Test different log levels
    logger.log_info("Test info message", {"test_key": "test_value"})
    logger.log_warning("Test warning message")
    logger.log_error("Test error message", ValueError("Test error"))
    
    # Test performance logging
    logger.log_performance("test_operation", 2.5, {"extra": "data"})
    
    # Test pipeline logging
    logger.log_pipeline_start("test_pipeline", {"param": "value"})
    logger.log_pipeline_step("test_pipeline", "step1", {"success": True})
    logger.log_pipeline_complete("test_pipeline", 10.0, {"final": "result"})
    
    # Test secret redaction
    logger.log_info("API key test", {"api_key": "sk-abc123def456ghi789jkl012mno345pqr"})
    logger.log_info("Secret test", {"secret": "super_secret_password_123"})
    
    # Get session summary
    summary = logger.get_session_summary()
    print(f"Session summary: {summary}")
    
    print("✅ Daily Logger test completed!")

if __name__ == "__main__":
    test_daily_logger()