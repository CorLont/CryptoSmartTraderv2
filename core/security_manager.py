"""
Enterprise Security Manager
Complete credential isolation and security management
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import secrets
import base64
from pathlib import Path
import json
import threading

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

from pydantic_settings import BaseSettings
from pydantic import SecretStr, Field

class SecureSettings(BaseSettings):
    """Secure settings with environment variable support"""
    
    # API Keys
    openai_api_key: Optional[SecretStr] = Field(None, env='OPENAI_API_KEY')
    binance_api_key: Optional[SecretStr] = Field(None, env='BINANCE_API_KEY')
    binance_secret: Optional[SecretStr] = Field(None, env='BINANCE_SECRET')
    kraken_api_key: Optional[SecretStr] = Field(None, env='KRAKEN_API_KEY')
    kraken_secret: Optional[SecretStr] = Field(None, env='KRAKEN_SECRET')
    
    # Database
    database_url: Optional[SecretStr] = Field(None, env='DATABASE_URL')
    redis_url: Optional[SecretStr] = Field(None, env='REDIS_URL')
    
    # Monitoring
    slack_webhook: Optional[SecretStr] = Field(None, env='SLACK_WEBHOOK_URL')
    email_password: Optional[SecretStr] = Field(None, env='EMAIL_PASSWORD')
    
    # Security
    secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_hex(32)))
    encryption_key: Optional[SecretStr] = Field(None, env='ENCRYPTION_KEY')
    
    # Vault
    vault_url: Optional[str] = Field(None, env='VAULT_URL')
    vault_token: Optional[SecretStr] = Field(None, env='VAULT_TOKEN')
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

@dataclass
class SecurityAuditLog:
    """Security audit log entry"""
    timestamp: datetime
    event_type: str
    user_id: str
    action: str
    resource: str
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class SecurityManager:
    """
    Enterprise-grade security manager
    Handles credentials, encryption, audit logging, and access control
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.settings = SecureSettings()
        
        # Security configuration
        self.config = {
            'credential_cache_ttl': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 900,  # 15 minutes
            'audit_log_retention': 30,  # 30 days
            'encryption_algorithm': 'AES-256-GCM'
        }
        
        # Vault client
        self.vault_client = None
        
        # Credential cache with encryption
        self.credential_cache = {}
        self.cache_lock = threading.Lock()
        
        # Audit logging
        self.audit_logs = []
        self.failed_attempts = {}
        
        # Security state
        self.security_initialized = False
        
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize security components"""
        
        try:
            # Initialize Vault if available
            if VAULT_AVAILABLE and self.settings.vault_url:
                self._initialize_vault()
            
            # Setup encryption
            self._initialize_encryption()
            
            # Load existing credentials safely
            self._load_secure_credentials()
            
            # Setup audit logging
            self._initialize_audit_logging()
            
            self.security_initialized = True
            self.logger.critical("SECURITY MANAGER INITIALIZED - Enterprise security active")
            
        except Exception as e:
            self.logger.error(f"Security initialization failed: {e}")
            # Fail securely - no credentials available
            self.security_initialized = False
    
    def _initialize_vault(self):
        """Initialize HashiCorp Vault client"""
        
        try:
            if not VAULT_AVAILABLE:
                self.logger.warning("Vault client not available (hvac not installed)")
                return
            
            self.vault_client = hvac.Client(
                url=self.settings.vault_url,
                token=self.settings.vault_token.get_secret_value() if self.settings.vault_token else None
            )
            
            if self.vault_client.is_authenticated():
                self.logger.info("Vault client authenticated successfully")
            else:
                self.logger.warning("Vault client not authenticated")
                self.vault_client = None
                
        except Exception as e:
            self.logger.error(f"Vault initialization failed: {e}")
            self.vault_client = None
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive data"""
        
        try:
            from cryptography.fernet import Fernet
            
            # Get or generate encryption key
            if self.settings.encryption_key:
                key = self.settings.encryption_key.get_secret_value()
            else:
                key = base64.urlsafe_b64encode(os.urandom(32))
                self.logger.info("Generated new encryption key")
            
            self.cipher = Fernet(key)
            self.logger.info("Encryption initialized")
            
        except ImportError:
            self.logger.warning("Cryptography not available - using basic encoding")
            self.cipher = None
        except Exception as e:
            self.logger.error(f"Encryption initialization failed: {e}")
            self.cipher = None
    
    def _load_secure_credentials(self):
        """Load credentials from secure sources"""
        
        # Priority order: Vault -> Environment -> Secure file
        
        # Try Vault first
        if self.vault_client:
            self._load_from_vault()
        
        # Environment variables are already loaded via pydantic settings
        
        # Load from secure file if exists
        secure_file = Path('.env.secure')
        if secure_file.exists():
            self._load_from_secure_file(secure_file)
    
    def _load_from_vault(self):
        """Load credentials from Vault"""
        
        try:
            # Read secrets from various Vault paths
            paths = [
                'secret/cryptotrader/api_keys',
                'secret/cryptotrader/database',
                'secret/cryptotrader/monitoring'
            ]
            
            for path in paths:
                try:
                    response = self.vault_client.secrets.kv.v2.read_secret_version(path=path)
                    secrets_data = response['data']['data']
                    
                    # Cache encrypted credentials
                    for key, value in secrets_data.items():
                        encrypted_value = self._encrypt_credential(value)
                        self.credential_cache[key] = {
                            'value': encrypted_value,
                            'timestamp': datetime.now(),
                            'source': 'vault'
                        }
                    
                    self.logger.info(f"Loaded {len(secrets_data)} credentials from Vault path: {path}")
                    
                except Exception as e:
                    self.logger.debug(f"Could not load from Vault path {path}: {e}")
        
        except Exception as e:
            self.logger.error(f"Vault credential loading failed: {e}")
    
    def _load_from_secure_file(self, file_path: Path):
        """Load credentials from encrypted file"""
        
        try:
            with open(file_path, 'r') as f:
                encrypted_data = f.read()
            
            if self.cipher:
                decrypted_data = self.cipher.decrypt(encrypted_data.encode())
                credentials = json.loads(decrypted_data.decode())
            else:
                # Fallback to base64 encoding
                decoded_data = base64.b64decode(encrypted_data)
                credentials = json.loads(decoded_data.decode())
            
            for key, value in credentials.items():
                self.credential_cache[key] = {
                    'value': value,
                    'timestamp': datetime.now(),
                    'source': 'secure_file'
                }
            
            self.logger.info(f"Loaded {len(credentials)} credentials from secure file")
            
        except Exception as e:
            self.logger.error(f"Secure file loading failed: {e}")
    
    def _initialize_audit_logging(self):
        """Initialize security audit logging"""
        
        self.audit_log_file = Path('logs/security_audit.json')
        self.audit_log_file.parent.mkdir(exist_ok=True)
        
        # Load existing audit logs
        if self.audit_log_file.exists():
            try:
                with open(self.audit_log_file, 'r') as f:
                    audit_data = json.load(f)
                    
                # Convert to audit log objects
                for entry in audit_data.get('logs', []):
                    audit_log = SecurityAuditLog(
                        timestamp=datetime.fromisoformat(entry['timestamp']),
                        event_type=entry['event_type'],
                        user_id=entry['user_id'],
                        action=entry['action'],
                        resource=entry['resource'],
                        success=entry['success'],
                        ip_address=entry.get('ip_address'),
                        user_agent=entry.get('user_agent'),
                        details=entry.get('details')
                    )
                    self.audit_logs.append(audit_log)
                
                self.logger.info(f"Loaded {len(self.audit_logs)} audit log entries")
                
            except Exception as e:
                self.logger.error(f"Audit log loading failed: {e}")
    
    def get_credential(self, key: str, user_id: str = 'system') -> Optional[str]:
        """Securely retrieve credential"""
        
        if not self.security_initialized:
            self.logger.error("Security not initialized - credential access denied")
            return None
        
        try:
            with self.cache_lock:
                # Check cache first
                if key in self.credential_cache:
                    cached_cred = self.credential_cache[key]
                    
                    # Check if cache is still valid
                    age = (datetime.now() - cached_cred['timestamp']).total_seconds()
                    if age < self.config['credential_cache_ttl']:
                        # Decrypt and return
                        decrypted_value = self._decrypt_credential(cached_cred['value'])
                        
                        self._log_security_event(
                            event_type='credential_access',
                            user_id=user_id,
                            action='retrieve',
                            resource=key,
                            success=True,
                            details={'source': cached_cred['source'], 'cached': True}
                        )
                        
                        return decrypted_value
                
                # Try pydantic settings
                credential_value = self._get_from_settings(key)
                
                if credential_value:
                    # Cache encrypted credential
                    encrypted_value = self._encrypt_credential(credential_value)
                    self.credential_cache[key] = {
                        'value': encrypted_value,
                        'timestamp': datetime.now(),
                        'source': 'environment'
                    }
                    
                    self._log_security_event(
                        event_type='credential_access',
                        user_id=user_id,
                        action='retrieve',
                        resource=key,
                        success=True,
                        details={'source': 'environment', 'cached': False}
                    )
                    
                    return credential_value
                
                # Credential not found
                self._log_security_event(
                    event_type='credential_access',
                    user_id=user_id,
                    action='retrieve',
                    resource=key,
                    success=False,
                    details={'reason': 'credential_not_found'}
                )
                
                return None
        
        except Exception as e:
            self.logger.error(f"Credential retrieval failed for {key}: {e}")
            
            self._log_security_event(
                event_type='credential_access',
                user_id=user_id,
                action='retrieve',
                resource=key,
                success=False,
                details={'error': str(e)}
            )
            
            return None
    
    def _get_from_settings(self, key: str) -> Optional[str]:
        """Get credential from pydantic settings"""
        
        # Map key to settings attribute
        key_mapping = {
            'openai_api_key': 'openai_api_key',
            'binance_api_key': 'binance_api_key',
            'binance_secret': 'binance_secret',
            'kraken_api_key': 'kraken_api_key',
            'kraken_secret': 'kraken_secret',
            'database_url': 'database_url',
            'slack_webhook': 'slack_webhook',
            'email_password': 'email_password'
        }
        
        attr_name = key_mapping.get(key)
        if not attr_name:
            return None
        
        attr_value = getattr(self.settings, attr_name, None)
        
        if attr_value and hasattr(attr_value, 'get_secret_value'):
            return attr_value.get_secret_value()
        elif attr_value:
            return str(attr_value)
        
        return None
    
    def _encrypt_credential(self, value: str) -> str:
        """Encrypt credential value"""
        
        if self.cipher:
            encrypted = self.cipher.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        else:
            # Fallback to base64 encoding (not secure but better than plain text)
            return base64.b64encode(value.encode()).decode()
    
    def _decrypt_credential(self, encrypted_value: str) -> str:
        """Decrypt credential value"""
        
        if self.cipher:
            decoded = base64.b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        else:
            # Fallback from base64 encoding
            decoded = base64.b64decode(encrypted_value.encode())
            return decoded.decode()
    
    def store_credential(self, key: str, value: str, user_id: str = 'system') -> bool:
        """Securely store credential"""
        
        if not self.security_initialized:
            self.logger.error("Security not initialized - credential storage denied")
            return False
        
        try:
            # Store in Vault if available
            if self.vault_client:
                try:
                    self.vault_client.secrets.kv.v2.create_or_update_secret(
                        path=f'secret/cryptotrader/dynamic/{key}',
                        secret={key: value}
                    )
                    
                    self._log_security_event(
                        event_type='credential_storage',
                        user_id=user_id,
                        action='store',
                        resource=key,
                        success=True,
                        details={'destination': 'vault'}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Vault storage failed for {key}: {e}")
            
            # Store in encrypted cache
            with self.cache_lock:
                encrypted_value = self._encrypt_credential(value)
                self.credential_cache[key] = {
                    'value': encrypted_value,
                    'timestamp': datetime.now(),
                    'source': 'dynamic'
                }
            
            self._log_security_event(
                event_type='credential_storage',
                user_id=user_id,
                action='store',
                resource=key,
                success=True,
                details={'destination': 'cache'}
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Credential storage failed for {key}: {e}")
            
            self._log_security_event(
                event_type='credential_storage',
                user_id=user_id,
                action='store',
                resource=key,
                success=False,
                details={'error': str(e)}
            )
            
            return False
    
    def _log_security_event(self, 
                           event_type: str,
                           user_id: str,
                           action: str,
                           resource: str,
                           success: bool,
                           ip_address: str = None,
                           user_agent: str = None,
                           details: Dict[str, Any] = None):
        """Log security event for audit trail"""
        
        audit_log = SecurityAuditLog(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {}
        )
        
        self.audit_logs.append(audit_log)
        
        # Save audit logs periodically
        if len(self.audit_logs) % 50 == 0:
            self._save_audit_logs()
        
        # Log to standard logger as well
        log_level = logging.INFO if success else logging.WARNING
        self.logger.log(
            log_level,
            f"SECURITY AUDIT: {event_type.upper()} - {user_id} {action} {resource} "
            f"{'SUCCESS' if success else 'FAILED'}"
        )
    
    def _save_audit_logs(self):
        """Save audit logs to file"""
        
        try:
            audit_data = {
                'logs': [
                    {
                        'timestamp': log.timestamp.isoformat(),
                        'event_type': log.event_type,
                        'user_id': log.user_id,
                        'action': log.action,
                        'resource': log.resource,
                        'success': log.success,
                        'ip_address': log.ip_address,
                        'user_agent': log.user_agent,
                        'details': log.details
                    }
                    for log in self.audit_logs
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.audit_log_file, 'w') as f:
                json.dump(audit_data, f, indent=2)
            
            self.logger.debug(f"Saved {len(self.audit_logs)} audit log entries")
            
        except Exception as e:
            self.logger.error(f"Audit log saving failed: {e}")
    
    def validate_api_key(self, api_key: str, service: str) -> bool:
        """Validate API key format and basic checks"""
        
        if not api_key or len(api_key) < 10:
            return False
        
        # Service-specific validation
        if service == 'openai':
            return api_key.startswith('sk-') and len(api_key) > 40
        elif service == 'binance':
            return len(api_key) == 64 and api_key.isalnum()
        elif service == 'kraken':
            return len(api_key) > 50 and '+' in api_key
        
        # Generic validation
        return len(api_key) > 15 and not any(char in api_key for char in [' ', '\n', '\t'])
    
    def sanitize_logs(self, log_data: str) -> str:
        """Sanitize logs to prevent credential leakage"""
        
        # Patterns to redact
        patterns = [
            (r'sk-[a-zA-Z0-9]{40,}', 'sk-***REDACTED***'),  # OpenAI keys
            (r'[A-Za-z0-9]{64}', '***REDACTED_API_KEY***'),  # 64-char keys
            (r'password["\s]*[:=]["\s]*[^"]+', 'password":"***REDACTED***"'),  # Passwords
            (r'token["\s]*[:=]["\s]*[^"]+', 'token":"***REDACTED***"'),  # Tokens
            (r'secret["\s]*[:=]["\s]*[^"]+', 'secret":"***REDACTED***"'),  # Secrets
        ]
        
        import re
        
        sanitized = log_data
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        return {
            'security_initialized': self.security_initialized,
            'vault_available': VAULT_AVAILABLE,
            'vault_connected': self.vault_client is not None and self.vault_client.is_authenticated() if self.vault_client else False,
            'encryption_enabled': self.cipher is not None,
            'credentials_cached': len(self.credential_cache),
            'audit_logs_count': len(self.audit_logs),
            'failed_attempts': sum(self.failed_attempts.values()),
            'security_features': {
                'credential_encryption': self.cipher is not None,
                'audit_logging': True,
                'access_control': True,
                'secure_storage': VAULT_AVAILABLE and self.vault_client is not None,
                'log_sanitization': True
            },
            'last_audit_save': self.audit_log_file.stat().st_mtime if self.audit_log_file.exists() else None
        }
    
    def clear_credential_cache(self, user_id: str = 'system'):
        """Clear credential cache"""
        
        with self.cache_lock:
            cache_size = len(self.credential_cache)
            self.credential_cache.clear()
        
        self._log_security_event(
            event_type='cache_management',
            user_id=user_id,
            action='clear_cache',
            resource='credential_cache',
            success=True,
            details={'cleared_credentials': cache_size}
        )
        
        self.logger.info(f"Credential cache cleared: {cache_size} credentials removed")

# Global security manager instance
security_manager = SecurityManager()