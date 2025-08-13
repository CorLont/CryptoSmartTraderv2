# config/security.py
import os
import hashlib
import secrets
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SecurityManager:
    """Enterprise-grade security management for CryptoSmartTrader"""
    
    def __init__(self):
        self.audit_log_path = Path("logs/security_audit.log")
        self.audit_log_path.parent.mkdir(exist_ok=True)
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
    
    def validate_input(self, input_data: Any, input_type: str) -> bool:
        """Validate and sanitize all external inputs"""
        try:
            if input_type == "symbol":
                # Cryptocurrency symbol validation
                if not isinstance(input_data, str):
                    return False
                # Only alphanumeric and common separators
                allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/-_")
                return all(c.upper() in allowed_chars for c in input_data) and len(input_data) <= 20
            
            elif input_type == "api_key":
                # API key format validation
                if not isinstance(input_data, str):
                    return False
                return 10 <= len(input_data) <= 200 and input_data.isalnum()
            
            elif input_type == "amount":
                # Numeric amount validation
                try:
                    value = float(input_data)
                    return 0 < value <= 1000000  # Reasonable trading limits
                except (ValueError, TypeError):
                    return False
            
            elif input_type == "percentage":
                # Percentage validation
                try:
                    value = float(input_data)
                    return 0 <= value <= 100
                except (ValueError, TypeError):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False
    
    def sanitize_string(self, input_str: str) -> str:
        """Sanitize string inputs to prevent injection attacks"""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '`', '$']
        sanitized = input_str
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Limit length
        return sanitized[:1000]
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for logging without exposure"""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "INFO"):
        """Log security events with audit trail"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "severity": severity,
                "details": details,
                "session_id": self._get_session_id()
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
            
            # Also log to main logger
            log_method = getattr(logger, severity.lower(), logger.info)
            log_method(f"Security event {event_type}: {details}")
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if identifier exceeds rate limits"""
        try:
            current_time = datetime.now()
            window_start = current_time - timedelta(minutes=window_minutes)
            
            # Clean old entries
            if identifier in self.failed_attempts:
                self.failed_attempts[identifier] = [
                    attempt for attempt in self.failed_attempts[identifier]
                    if attempt > window_start
                ]
            
            # Check current count
            current_count = len(self.failed_attempts.get(identifier, []))
            
            if current_count >= max_requests:
                self.log_security_event(
                    "RATE_LIMIT_EXCEEDED",
                    {"identifier": self.hash_sensitive_data(identifier), "count": current_count},
                    "WARNING"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True  # Fail open for availability
    
    def record_failed_attempt(self, identifier: str):
        """Record a failed authentication or access attempt"""
        try:
            current_time = datetime.now()
            
            if identifier not in self.failed_attempts:
                self.failed_attempts[identifier] = []
            
            self.failed_attempts[identifier].append(current_time)
            
            # Check if should be locked out
            recent_failures = [
                attempt for attempt in self.failed_attempts[identifier]
                if attempt > current_time - self.lockout_duration
            ]
            
            if len(recent_failures) >= self.max_failed_attempts:
                self.log_security_event(
                    "LOCKOUT_TRIGGERED",
                    {"identifier": self.hash_sensitive_data(identifier), "failures": len(recent_failures)},
                    "CRITICAL"
                )
            
        except Exception as e:
            logger.error(f"Failed to record failed attempt: {e}")
    
    def is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is currently locked out"""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - self.lockout_duration
            
            if identifier not in self.failed_attempts:
                return False
            
            recent_failures = [
                attempt for attempt in self.failed_attempts[identifier]
                if attempt > cutoff_time
            ]
            
            return len(recent_failures) >= self.max_failed_attempts
            
        except Exception as e:
            logger.error(f"Lockout check failed: {e}")
            return False  # Fail open
    
    def validate_api_access(self, api_key: str, source_ip: str = "unknown") -> bool:
        """Validate API access with security checks"""
        try:
            # Input validation
            if not self.validate_input(api_key, "api_key"):
                self.log_security_event(
                    "INVALID_API_KEY_FORMAT",
                    {"source_ip": source_ip},
                    "WARNING"
                )
                return False
            
            # Rate limiting
            if not self.check_rate_limit(f"api_{source_ip}"):
                return False
            
            # Check lockout status
            if self.is_locked_out(f"api_{source_ip}"):
                self.log_security_event(
                    "LOCKED_OUT_ACCESS_ATTEMPT",
                    {"source_ip": source_ip},
                    "CRITICAL"
                )
                return False
            
            # Log successful validation
            self.log_security_event(
                "API_ACCESS_VALIDATED",
                {"source_ip": source_ip, "key_hash": self.hash_sensitive_data(api_key)},
                "INFO"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"API access validation failed: {e}")
            self.log_security_event(
                "API_VALIDATION_ERROR",
                {"source_ip": source_ip, "error": str(e)},
                "ERROR"
            )
            return False
    
    def _get_session_id(self) -> str:
        """Generate or retrieve session ID for audit trail"""
        # In production, this would be managed by the session system
        return secrets.token_hex(8)
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security status summary"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)
            
            # Count recent security events
            recent_events = []
            if self.audit_log_path.exists():
                with open(self.audit_log_path, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event['timestamp'])
                            if event_time > hour_ago:
                                recent_events.append(event)
                        except Exception:
                            continue
            
            # Count by severity
            severity_counts = {}
            for event in recent_events:
                severity = event.get('severity', 'INFO')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Current lockouts
            active_lockouts = 0
            cutoff_time = current_time - self.lockout_duration
            
            for identifier, attempts in self.failed_attempts.items():
                recent_failures = [a for a in attempts if a > cutoff_time]
                if len(recent_failures) >= self.max_failed_attempts:
                    active_lockouts += 1
            
            return {
                "status": "secure" if severity_counts.get('CRITICAL', 0) == 0 else "alert",
                "events_last_hour": len(recent_events),
                "severity_breakdown": severity_counts,
                "active_lockouts": active_lockouts,
                "audit_log_size": self.audit_log_path.stat().st_size if self.audit_log_path.exists() else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security summary: {e}")
            return {"status": "error", "error": str(e)}


# Global security manager instance
security_manager = SecurityManager()