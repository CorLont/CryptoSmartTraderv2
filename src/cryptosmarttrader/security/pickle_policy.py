#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Pickle Security Policy Enforcement
Runtime enforcement of pickle usage restrictions
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Set, Optional, Any
from functools import wraps
import inspect

logger = logging.getLogger(__name__)

class PickleSecurityViolation(Exception):
    """Exception raised when pickle usage violates security policy"""
    pass

class PicklePolicyEnforcer:
    """Runtime enforcement of pickle security policies"""
    
    def __init__(self):
        self.trusted_paths = {
            'src/cryptosmarttrader',
            'models',
            'ml',
            'cache',
            'exports/models',
            'model_backup',
            'mlartifacts'
        }
        
        self.violation_log = []
        self._original_pickle_load = None
        self._original_pickle_dump = None
        self._is_monitoring = False
    
    def is_trusted_caller(self) -> bool:
        """Check if the calling code is from a trusted path"""
        frame = inspect.currentframe()
        try:
            # Walk up the call stack to find the original caller
            for i in range(10):  # Limit depth to prevent infinite loops
                frame = frame.f_back
                if frame is None:
                    break
                
                filename = frame.f_code.co_filename
                filepath = Path(filename)
                
                # Check if path is trusted
                path_str = str(filepath.resolve())
                if any(trusted in path_str for trusted in self.trusted_paths):
                    return True
            
            return False
        finally:
            del frame
    
    def log_violation(self, operation: str, caller_info: str):
        """Log security violation"""
        violation = {
            'operation': operation,
            'caller': caller_info,
            'timestamp': logger._formatTime(logger.handlers[0].formatter._fmt) if logger.handlers else 'unknown'
        }
        self.violation_log.append(violation)
        
        logger.error(f"PICKLE SECURITY VIOLATION: {operation} from {caller_info}")
        warnings.warn(
            f"Pickle security violation: {operation} from untrusted path {caller_info}. "
            f"Use JSON/msgpack for external data.",
            UserWarning,
            stacklevel=3
        )
    
    def secure_pickle_load(self, file, **kwargs):
        """Secure wrapper for pickle.load"""
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        
        if not self.is_trusted_caller():
            self.log_violation('pickle.load', caller_file)
            raise PickleSecurityViolation(
                f"pickle.load not allowed from {caller_file}. "
                f"Use secure_serialization.load_json() or load_msgpack() instead."
            )
        
        # Use original pickle.load for trusted callers
        if self._original_pickle_load:
            return self._original_pickle_load(file, **kwargs)
        else:
            import pickle
            return pickle.load(file, **kwargs)
    
    def secure_pickle_dump(self, obj, file, **kwargs):
        """Secure wrapper for pickle.dump"""
        caller_frame = inspect.currentframe().f_back
        caller_file = caller_frame.f_code.co_filename
        
        if not self.is_trusted_caller():
            self.log_violation('pickle.dump', caller_file)
            raise PickleSecurityViolation(
                f"pickle.dump not allowed from {caller_file}. "
                f"Use secure_serialization.save_json() or save_msgpack() instead."
            )
        
        # Use original pickle.dump for trusted callers
        if self._original_pickle_dump:
            return self._original_pickle_dump(obj, file, **kwargs)
        else:
            import pickle
            return pickle.dump(obj, file, **kwargs)
    
    def enable_monitoring(self):
        """Enable runtime pickle monitoring"""
        if self._is_monitoring:
            return
        
        try:
            import pickle
            
            # Store original functions
            self._original_pickle_load = pickle.load
            self._original_pickle_dump = pickle.dump
            
            # Replace with secure wrappers
            pickle.load = self.secure_pickle_load
            pickle.dump = self.secure_pickle_dump
            
            self._is_monitoring = True
            logger.info("Pickle security monitoring enabled")
            
        except ImportError:
            logger.warning("pickle module not available for monitoring")
    
    def disable_monitoring(self):
        """Disable runtime pickle monitoring"""
        if not self._is_monitoring:
            return
        
        try:
            import pickle
            
            # Restore original functions
            if self._original_pickle_load:
                pickle.load = self._original_pickle_load
            if self._original_pickle_dump:
                pickle.dump = self._original_pickle_dump
            
            self._is_monitoring = False
            logger.info("Pickle security monitoring disabled")
            
        except ImportError:
            pass
    
    def get_violations(self) -> list:
        """Get all logged violations"""
        return self.violation_log.copy()


# Global enforcer instance
_enforcer = PicklePolicyEnforcer()

def enable_pickle_security():
    """Enable global pickle security monitoring"""
    _enforcer.enable_monitoring()

def disable_pickle_security():
    """Disable global pickle security monitoring"""
    _enforcer.disable_monitoring()

def get_pickle_violations():
    """Get all pickle security violations"""
    return _enforcer.get_violations()

def secure_pickle_only(func):
    """Decorator to enforce secure pickle usage in function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not _enforcer.is_trusted_caller():
            raise PickleSecurityViolation(
                f"Function {func.__name__} uses pickle and requires trusted caller"
            )
        return func(*args, **kwargs)
    return wrapper

# Auto-enable monitoring on import if in production
if os.environ.get('CRYPTOSMARTTRADER_ENV') == 'production':
    enable_pickle_security()
    logger.info("Pickle security auto-enabled for production environment")