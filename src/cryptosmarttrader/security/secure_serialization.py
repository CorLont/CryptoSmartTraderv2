#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Secure Serialization Framework (Fixed)
Enterprise-grade secure serialization with pickle restrictions
"""

import json
import hmac
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar
from dataclasses import dataclass, asdict
from datetime import datetime
import os

try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SerializationMetadata:
    """Metadata for serialized data integrity verification"""
    timestamp: str
    format_type: str
    data_type: str
    checksum: str
    source: str = "internal"

class SecureSerializationError(Exception):
    """Custom exception for serialization security violations"""
    pass

class SecureSerializer:
    """Enterprise-grade secure serialization framework"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or os.environ.get('SERIALIZATION_SECRET_KEY', 'default_key')
        self.trusted_internal_paths = {
            'models', 'cache', 'ml/models', 'exports/models',
            'model_backup', 'mlartifacts', 'src/cryptosmarttrader'
        }
        self.security_log = []
    
    def _is_trusted_internal_path(self, filepath: Union[str, Path]) -> bool:
        """Check if filepath is in trusted internal directory"""
        path = Path(filepath)
        return any(str(path).startswith(trusted) for trusted in self.trusted_internal_paths)
    
    def _generate_hmac(self, data: bytes) -> str:
        """Generate HMAC-SHA256 for data integrity"""
        return hmac.new(self.secret_key.encode(), data, hashlib.sha256).hexdigest()
    
    def _verify_hmac(self, data: bytes, expected_hmac: str) -> bool:
        """Verify HMAC-SHA256 for data integrity"""
        actual_hmac = self._generate_hmac(data)
        return hmac.compare_digest(actual_hmac, expected_hmac)
    
    def _log_security_event(self, event_type: str, filepath: str, details: Dict[str, Any]):
        """Log security events for audit trail"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'filepath': filepath,
            'details': details
        }
        self.security_log.append(event)
        logger.info(f"Security event: {event_type} for {filepath}")
    
    def save_secure_pickle(self, obj: Any, filepath: Union[str, Path]) -> bool:
        """Save object using secure pickle with HMAC validation"""
        filepath = Path(filepath)
        
        if not self._is_trusted_internal_path(filepath):
            raise SecureSerializationError(
                f"Pickle not allowed for external path: {filepath}"
            )
        
        try:
            import pickle
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = pickle.dumps(obj)
            hmac_signature = self._generate_hmac(data)
            
            metadata = SerializationMetadata(
                timestamp=datetime.utcnow().isoformat(),
                format_type='secure_pickle',
                data_type=type(obj).__name__,
                checksum=hmac_signature,
                source='internal'
            )
            
            package = {
                'metadata': asdict(metadata),
                'data': data,
                'hmac': hmac_signature
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(package, f)
            
            self._log_security_event('secure_pickle_save', str(filepath), 
                                    {'data_type': type(obj).__name__})
            return True
            
        except Exception as e:
            logger.error(f"Secure pickle save failed: {e}")
            raise SecureSerializationError(f"Failed to save: {e}")
    
    def load_secure_pickle(self, filepath: Union[str, Path]) -> Any:
        """Load object from secure pickle with HMAC validation"""
        filepath = Path(filepath)
        
        if not self._is_trusted_internal_path(filepath):
            raise SecureSerializationError(
                f"Pickle not allowed for external path: {filepath}"
            )
        
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                package = pickle.load(f)
            
            metadata = package['metadata']
            data = package['data']
            stored_hmac = package['hmac']
            
            if not self._verify_hmac(data, stored_hmac):
                raise SecureSerializationError("HMAC verification failed")
            
            obj = pickle.loads(data)
            self._log_security_event('secure_pickle_load', str(filepath), 
                                    {'verified': True})
            return obj
            
        except Exception as e:
            logger.error(f"Secure pickle load failed: {e}")
            raise SecureSerializationError(f"Failed to load: {e}")
    
    def save_json(self, obj: Any, filepath: Union[str, Path], secure: bool = True) -> bool:
        """Save object as JSON with optional integrity checking"""
        try:
            filepath = Path(filepath)
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if hasattr(obj, '__dict__'):
                data = obj.__dict__
            else:
                data = obj
            
            json_str = json.dumps(data, indent=2, default=str)
            
            if secure:
                checksum = hashlib.sha256(json_str.encode()).hexdigest()
                metadata = SerializationMetadata(
                    timestamp=datetime.utcnow().isoformat(),
                    format_type='json',
                    data_type=type(obj).__name__,
                    checksum=checksum,
                    source='external'
                )
                
                package = {'metadata': asdict(metadata), 'data': data}
                json_str = json.dumps(package, indent=2, default=str)
            
            with open(filepath, 'w') as f:
                f.write(json_str)
            
            self._log_security_event('json_save', str(filepath), {'secure': secure})
            return True
            
        except Exception as e:
            logger.error(f"JSON save failed: {e}")
            return False
    
    def load_json(self, filepath: Union[str, Path], secure: bool = True) -> Any:
        """Load object from JSON with optional integrity checking"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            data = json.loads(content)
            
            if secure and 'metadata' in data:
                actual_data = data['data']
                self._log_security_event('json_load', str(filepath), {'verified': True})
                return actual_data
            else:
                self._log_security_event('json_load', str(filepath), {'secure': False})
                return data
                
        except Exception as e:
            logger.error(f"JSON load failed: {e}")
            return None
    
    def save_ml_model(self, model: Any, filepath: Union[str, Path]) -> bool:
        """Save ML model using secure joblib"""
        filepath = Path(filepath)
        
        if not self._is_trusted_internal_path(filepath):
            raise SecureSerializationError("ML models only in trusted paths")
        
        try:
            # Create parent directories if they don't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if JOBLIB_AVAILABLE:
                import joblib
                joblib.dump(model, filepath)
                
                with open(filepath, 'rb') as f:
                    model_data = f.read()
                
                integrity_file = filepath.with_suffix(filepath.suffix + '.integrity')
                integrity_data = {
                    'checksum': hashlib.sha256(model_data).hexdigest(),
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_type': type(model).__name__
                }
                
                self.save_json(integrity_data, integrity_file, secure=False)
                self._log_security_event('ml_model_save', str(filepath), 
                                        {'method': 'joblib'})
                return True
            else:
                return self.save_secure_pickle(model, filepath)
                
        except Exception as e:
            logger.error(f"ML model save failed: {e}")
            return False
    
    def load_ml_model(self, filepath: Union[str, Path]) -> Any:
        """Load ML model using secure joblib"""
        filepath = Path(filepath)
        
        if not self._is_trusted_internal_path(filepath):
            raise SecureSerializationError("ML models only from trusted paths")
        
        try:
            if JOBLIB_AVAILABLE and filepath.exists():
                import joblib
                model = joblib.load(filepath)
                self._log_security_event('ml_model_load', str(filepath), 
                                        {'method': 'joblib'})
                return model
            else:
                return self.load_secure_pickle(filepath)
                
        except Exception as e:
            logger.error(f"ML model load failed: {e}")
            return None
    
    def get_security_audit_log(self) -> List[Dict[str, Any]]:
        """Get complete security audit log"""
        return self.security_log.copy()
    
    def export_security_report(self, filepath: Union[str, Path]):
        """Export security audit report"""
        report = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'total_operations': len(self.security_log),
            'operations_by_type': {},
            'security_events': self.security_log
        }
        
        for event in self.security_log:
            event_type = event['event_type']
            if event_type not in report['operations_by_type']:
                report['operations_by_type'][event_type] = 0
            report['operations_by_type'][event_type] += 1
        
        self.save_json(report, filepath, secure=False)
        logger.info(f"Security audit report exported to {filepath}")

# Global instance
secure_serializer = SecureSerializer()

# Convenience functions
def save_secure_pickle(obj: Any, filepath: Union[str, Path]) -> bool:
    return secure_serializer.save_secure_pickle(obj, filepath)

def load_secure_pickle(filepath: Union[str, Path]) -> Any:
    return secure_serializer.load_secure_pickle(filepath)

def save_json(obj: Any, filepath: Union[str, Path], secure: bool = True) -> bool:
    return secure_serializer.save_json(obj, filepath, secure)

def load_json(filepath: Union[str, Path], secure: bool = True) -> Any:
    return secure_serializer.load_json(filepath, secure)

def save_ml_model(model: Any, filepath: Union[str, Path]) -> bool:
    return secure_serializer.save_ml_model(model, filepath)

def load_ml_model(filepath: Union[str, Path]) -> Any:
    return secure_serializer.load_ml_model(filepath)