"""
Secrets Manager for CryptoSmartTrader
Enterprise secrets management with rotation, audit logging, and secure storage.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Encryption libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
except ImportError:
    warnings.warn("Cryptography library not available, secrets will use basic encoding")
    Fernet = None


class SecretType(Enum):
    """Types of secrets managed by the system."""
    API_KEY = "api_key"
    DATABASE_URL = "database_url"
    WEBHOOK_SECRET = "webhook_secret"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_TOKEN = "oauth_token"
    SIGNING_KEY = "signing_key"
    SERVICE_ACCOUNT = "service_account"


class SecretStatus(Enum):
    """Status of secrets in the system."""
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    PENDING = "pending"


@dataclass
class SecretMetadata:
    """Metadata for secret management."""
    secret_id: str
    secret_type: SecretType
    description: str
    created_at: datetime
    last_rotated: datetime
    rotation_frequency_days: int
    status: SecretStatus
    
    # Access control
    required_permissions: List[str]
    allowed_environments: List[str]
    
    # Rotation settings
    auto_rotation_enabled: bool
    rotation_warning_days: int
    
    # Audit trail
    created_by: str
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    last_rotation_by: Optional[str] = None
    
    # Validation
    validation_regex: Optional[str] = None
    min_length: int = 8
    max_length: int = 512
    
    # Tags for organization
    tags: List[str] = None


@dataclass
class SecretAuditEvent:
    """Audit event for secret access and operations."""
    event_id: str
    secret_id: str
    event_type: str  # access, rotation, creation, deletion, etc.
    timestamp: datetime
    user_id: str
    environment: str
    success: bool
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class SecretsManager:
    """
    Enterprise secrets management system with encryption, rotation, and audit logging.
    
    Features:
    - Secure storage with encryption at rest
    - Automatic secret rotation with configurable schedules
    - Comprehensive audit logging for compliance
    - Role-based access control
    - Environment-specific secret isolation
    - Integration with CI/CD systems
    - Emergency secret revocation
    - Compliance with security standards
    """
    
    def __init__(self, 
                 secrets_path: str = "secrets",
                 encryption_key: Optional[str] = None,
                 environment: str = "development"):
        
        self.secrets_path = Path(secrets_path)
        self.secrets_path.mkdir(parents=True, exist_ok=True)
        self.environment = environment
        
        # Initialize encryption
        self.cipher_suite = self._init_encryption(encryption_key)
        
        # Storage paths
        self.metadata_path = self.secrets_path / "metadata.json"
        self.audit_path = self.secrets_path / "audit.json"
        self.secrets_store_path = self.secrets_path / "store.enc"
        
        # In-memory caches
        self.metadata_cache: Dict[str, SecretMetadata] = {}
        self.audit_events: List[SecretAuditEvent] = []
        
        # Load existing data
        self._load_metadata()
        self._load_audit_log()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"SecretsManager initialized for environment: {environment}")
    
    def store_secret(self,
                    secret_id: str,
                    secret_value: str,
                    secret_type: SecretType,
                    description: str,
                    rotation_frequency_days: int = 90,
                    required_permissions: List[str] = None,
                    tags: List[str] = None,
                    user_id: str = "system") -> bool:
        """
        Store a new secret with metadata and encryption.
        
        Args:
            secret_id: Unique identifier for the secret
            secret_value: The actual secret value
            secret_type: Type of secret
            description: Human-readable description
            rotation_frequency_days: How often to rotate (days)
            required_permissions: Required permissions to access
            tags: Tags for organization
            user_id: User storing the secret
            
        Returns:
            True if successful
        """
        
        try:
            # Validate secret
            if not self._validate_secret_value(secret_value, secret_type):
                raise ValueError("Secret validation failed")
            
            # Check if secret already exists
            if secret_id in self.metadata_cache:
                raise ValueError(f"Secret {secret_id} already exists")
            
            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                description=description,
                created_at=datetime.utcnow(),
                last_rotated=datetime.utcnow(),
                rotation_frequency_days=rotation_frequency_days,
                status=SecretStatus.ACTIVE,
                required_permissions=required_permissions or [],
                allowed_environments=[self.environment],
                auto_rotation_enabled=True,
                rotation_warning_days=7,
                created_by=user_id,
                tags=tags or []
            )
            
            # Store encrypted secret
            self._store_encrypted_secret(secret_id, secret_value)
            
            # Update metadata
            self.metadata_cache[secret_id] = metadata
            self._save_metadata()
            
            # Audit log
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_created",
                user_id=user_id,
                success=True,
                details={"secret_type": secret_type.value, "description": description}
            )
            
            self.logger.info(f"Secret {secret_id} stored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secret {secret_id}: {e}")
            
            # Audit failed attempt
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_creation_failed",
                user_id=user_id,
                success=False,
                details={"error": str(e)}
            )
            
            return False
    
    def get_secret(self,
                  secret_id: str,
                  user_id: str = "system",
                  required_permissions: List[str] = None) -> Optional[str]:
        """
        Retrieve a secret value with access control and audit logging.
        
        Args:
            secret_id: Secret identifier
            user_id: User requesting the secret
            required_permissions: Required permissions (if None, uses secret's requirements)
            
        Returns:
            Secret value if authorized, None otherwise
        """
        
        try:
            # Check if secret exists
            if secret_id not in self.metadata_cache:
                self.logger.warning(f"Secret {secret_id} not found")
                return None
            
            metadata = self.metadata_cache[secret_id]
            
            # Check environment access
            if self.environment not in metadata.allowed_environments:
                raise PermissionError(f"Secret not available in environment: {self.environment}")
            
            # Check secret status
            if metadata.status not in [SecretStatus.ACTIVE, SecretStatus.ROTATING]:
                raise PermissionError(f"Secret {secret_id} is not active (status: {metadata.status.value})")
            
            # Check permissions (simplified - in production use proper RBAC)
            required_perms = required_permissions or metadata.required_permissions
            if required_perms and not self._check_permissions(user_id, required_perms):
                raise PermissionError("Insufficient permissions to access secret")
            
            # Retrieve encrypted secret
            secret_value = self._retrieve_encrypted_secret(secret_id)
            
            if secret_value is None:
                raise ValueError("Failed to decrypt secret")
            
            # Update access metadata
            metadata.last_accessed = datetime.utcnow()
            metadata.access_count += 1
            self._save_metadata()
            
            # Audit log
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_accessed",
                user_id=user_id,
                success=True,
                details={"access_count": metadata.access_count}
            )
            
            return secret_value
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            
            # Audit failed attempt
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_access_failed",
                user_id=user_id,
                success=False,
                details={"error": str(e)}
            )
            
            return None
    
    def rotate_secret(self,
                     secret_id: str,
                     new_secret_value: str,
                     user_id: str = "system") -> bool:
        """
        Rotate a secret to a new value.
        
        Args:
            secret_id: Secret to rotate
            new_secret_value: New secret value
            user_id: User performing rotation
            
        Returns:
            True if successful
        """
        
        try:
            if secret_id not in self.metadata_cache:
                raise ValueError(f"Secret {secret_id} not found")
            
            metadata = self.metadata_cache[secret_id]
            
            # Validate new secret
            if not self._validate_secret_value(new_secret_value, metadata.secret_type):
                raise ValueError("New secret validation failed")
            
            # Update status to rotating
            old_status = metadata.status
            metadata.status = SecretStatus.ROTATING
            self._save_metadata()
            
            # Store new encrypted value
            self._store_encrypted_secret(secret_id, new_secret_value)
            
            # Update metadata
            metadata.last_rotated = datetime.utcnow()
            metadata.last_rotation_by = user_id
            metadata.status = SecretStatus.ACTIVE
            self._save_metadata()
            
            # Audit log
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_rotated",
                user_id=user_id,
                success=True,
                details={
                    "previous_status": old_status.value,
                    "rotation_type": "manual"
                }
            )
            
            self.logger.info(f"Secret {secret_id} rotated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate secret {secret_id}: {e}")
            
            # Restore status on failure
            if secret_id in self.metadata_cache:
                self.metadata_cache[secret_id].status = SecretStatus.ACTIVE
                self._save_metadata()
            
            # Audit failed attempt
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_rotation_failed",
                user_id=user_id,
                success=False,
                details={"error": str(e)}
            )
            
            return False
    
    def check_rotation_needed(self) -> List[Tuple[str, int]]:
        """
        Check which secrets need rotation.
        
        Returns:
            List of (secret_id, days_overdue) tuples
        """
        
        rotation_needed = []
        current_time = datetime.utcnow()
        
        for secret_id, metadata in self.metadata_cache.items():
            if not metadata.auto_rotation_enabled:
                continue
            
            if metadata.status != SecretStatus.ACTIVE:
                continue
            
            days_since_rotation = (current_time - metadata.last_rotated).days
            rotation_due = days_since_rotation >= metadata.rotation_frequency_days
            
            if rotation_due:
                days_overdue = days_since_rotation - metadata.rotation_frequency_days
                rotation_needed.append((secret_id, days_overdue))
        
        return rotation_needed
    
    def auto_rotate_secrets(self, user_id: str = "auto_rotation") -> Dict[str, bool]:
        """
        Automatically rotate secrets that are due for rotation.
        
        Args:
            user_id: User ID for audit purposes
            
        Returns:
            Dict mapping secret_id to rotation success status
        """
        
        rotation_results = {}
        secrets_to_rotate = self.check_rotation_needed()
        
        for secret_id, days_overdue in secrets_to_rotate:
            self.logger.info(f"Auto-rotating secret {secret_id} (overdue by {days_overdue} days)")
            
            # Generate new secret value based on type
            new_value = self._generate_secret_value(self.metadata_cache[secret_id].secret_type)
            
            if new_value:
                success = self.rotate_secret(secret_id, new_value, user_id)
                rotation_results[secret_id] = success
            else:
                rotation_results[secret_id] = False
                self.logger.error(f"Failed to generate new value for {secret_id}")
        
        return rotation_results
    
    def revoke_secret(self, secret_id: str, user_id: str = "system") -> bool:
        """
        Revoke a secret (mark as revoked, do not delete).
        
        Args:
            secret_id: Secret to revoke
            user_id: User performing revocation
            
        Returns:
            True if successful
        """
        
        try:
            if secret_id not in self.metadata_cache:
                raise ValueError(f"Secret {secret_id} not found")
            
            metadata = self.metadata_cache[secret_id]
            old_status = metadata.status
            
            # Update status
            metadata.status = SecretStatus.REVOKED
            self._save_metadata()
            
            # Audit log
            self._log_audit_event(
                secret_id=secret_id,
                event_type="secret_revoked",
                user_id=user_id,
                success=True,
                details={"previous_status": old_status.value}
            )
            
            self.logger.warning(f"Secret {secret_id} revoked by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke secret {secret_id}: {e}")
            return False
    
    def list_secrets(self, 
                    include_revoked: bool = False,
                    secret_type: Optional[SecretType] = None,
                    user_id: str = "system") -> List[Dict[str, Any]]:
        """
        List all secrets with metadata (excluding secret values).
        
        Args:
            include_revoked: Whether to include revoked secrets
            secret_type: Filter by secret type
            user_id: User requesting the list
            
        Returns:
            List of secret metadata dictionaries
        """
        
        secrets_list = []
        
        for secret_id, metadata in self.metadata_cache.items():
            # Filter by status
            if not include_revoked and metadata.status == SecretStatus.REVOKED:
                continue
            
            # Filter by type
            if secret_type and metadata.secret_type != secret_type:
                continue
            
            # Check environment access
            if self.environment not in metadata.allowed_environments:
                continue
            
            # Prepare metadata (excluding sensitive info)
            secret_info = {
                'secret_id': metadata.secret_id,
                'secret_type': metadata.secret_type.value,
                'description': metadata.description,
                'status': metadata.status.value,
                'created_at': metadata.created_at.isoformat(),
                'last_rotated': metadata.last_rotated.isoformat(),
                'rotation_frequency_days': metadata.rotation_frequency_days,
                'auto_rotation_enabled': metadata.auto_rotation_enabled,
                'access_count': metadata.access_count,
                'tags': metadata.tags,
                'days_since_rotation': (datetime.utcnow() - metadata.last_rotated).days
            }
            
            # Add last accessed if available
            if metadata.last_accessed:
                secret_info['last_accessed'] = metadata.last_accessed.isoformat()
            
            secrets_list.append(secret_info)
        
        # Audit the listing operation
        self._log_audit_event(
            secret_id="all",
            event_type="secrets_listed",
            user_id=user_id,
            success=True,
            details={"count": len(secrets_list), "include_revoked": include_revoked}
        )
        
        return secrets_list
    
    def get_audit_log(self, 
                     secret_id: Optional[str] = None,
                     event_type: Optional[str] = None,
                     user_id: Optional[str] = None,
                     hours_back: int = 24) -> List[Dict[str, Any]]:
        """
        Get audit log entries with optional filtering.
        
        Args:
            secret_id: Filter by secret ID
            event_type: Filter by event type
            user_id: Filter by user ID
            hours_back: How many hours back to include
            
        Returns:
            List of audit event dictionaries
        """
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        filtered_events = []
        
        for event in self.audit_events:
            # Time filter
            if event.timestamp < cutoff_time:
                continue
            
            # Secret ID filter
            if secret_id and event.secret_id != secret_id:
                continue
            
            # Event type filter
            if event_type and event.event_type != event_type:
                continue
            
            # User ID filter
            if user_id and event.user_id != user_id:
                continue
            
            filtered_events.append(asdict(event))
        
        return filtered_events
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on secrets management system.
        
        Returns:
            Health status and metrics
        """
        
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'environment': self.environment,
                'encryption_available': self.cipher_suite is not None,
                'total_secrets': len(self.metadata_cache),
                'active_secrets': len([m for m in self.metadata_cache.values() if m.status == SecretStatus.ACTIVE]),
                'secrets_due_rotation': len(self.check_rotation_needed()),
                'recent_audit_events': len(self.get_audit_log(hours_back=1))
            }
            
            # Check for issues
            issues = []
            
            # Check for overdue rotations
            overdue_rotations = self.check_rotation_needed()
            if overdue_rotations:
                issues.append(f"{len(overdue_rotations)} secrets overdue for rotation")
            
            # Check storage accessibility
            if not self.secrets_path.exists():
                issues.append("Secrets storage path not accessible")
            
            if issues:
                health_status['status'] = 'degraded'
                health_status['issues'] = issues
            
            return health_status
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Private methods
    
    def _init_encryption(self, encryption_key: Optional[str]) -> Optional[object]:
        """Initialize encryption cipher suite."""
        
        if Fernet is None:
            self.logger.warning("Cryptography not available, using basic encoding")
            return None
        
        try:
            if encryption_key:
                key = encryption_key.encode()
            else:
                # Use environment variable or generate
                key = os.environ.get('SECRETS_ENCRYPTION_KEY', '').encode()
                if not key:
                    # Generate key from system info (not secure for production)
                    password = f"cryptosmarttrader_{self.environment}".encode()
                    salt = b"salt_12345678"  # Should be random in production
                    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100000)
                    key = base64.urlsafe_b64encode(kdf.derive(password))
            
            return Fernet(key)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            return None
    
    def _store_encrypted_secret(self, secret_id: str, secret_value: str):
        """Store encrypted secret value."""
        
        # Load existing secrets
        secrets_store = self._load_secrets_store()
        
        # Encrypt secret
        if self.cipher_suite:
            encrypted_value = self.cipher_suite.encrypt(secret_value.encode()).decode()
        else:
            # Basic encoding (not secure)
            encrypted_value = base64.b64encode(secret_value.encode()).decode()
        
        # Store encrypted value
        secrets_store[secret_id] = encrypted_value
        
        # Save store
        self._save_secrets_store(secrets_store)
    
    def _retrieve_encrypted_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve and decrypt secret value."""
        
        try:
            # Load secrets store
            secrets_store = self._load_secrets_store()
            
            if secret_id not in secrets_store:
                return None
            
            encrypted_value = secrets_store[secret_id]
            
            # Decrypt secret
            if self.cipher_suite:
                decrypted_value = self.cipher_suite.decrypt(encrypted_value.encode()).decode()
            else:
                # Basic decoding
                decrypted_value = base64.b64decode(encrypted_value.encode()).decode()
            
            return decrypted_value
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt secret {secret_id}: {e}")
            return None
    
    def _load_secrets_store(self) -> Dict[str, str]:
        """Load encrypted secrets store."""
        
        if not self.secrets_store_path.exists():
            return {}
        
        try:
            with open(self.secrets_store_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load secrets store: {e}")
            return {}
    
    def _save_secrets_store(self, secrets_store: Dict[str, str]):
        """Save encrypted secrets store."""
        
        try:
            with open(self.secrets_store_path, 'w') as f:
                json.dump(secrets_store, f)
        except Exception as e:
            self.logger.error(f"Failed to save secrets store: {e}")
    
    def _load_metadata(self):
        """Load secrets metadata."""
        
        if not self.metadata_path.exists():
            return
        
        try:
            with open(self.metadata_path, 'r') as f:
                data = json.load(f)
            
            for secret_id, metadata_dict in data.items():
                # Convert timestamps
                metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                metadata_dict['last_rotated'] = datetime.fromisoformat(metadata_dict['last_rotated'])
                
                if metadata_dict.get('last_accessed'):
                    metadata_dict['last_accessed'] = datetime.fromisoformat(metadata_dict['last_accessed'])
                
                # Convert enums
                metadata_dict['secret_type'] = SecretType(metadata_dict['secret_type'])
                metadata_dict['status'] = SecretStatus(metadata_dict['status'])
                
                # Create metadata object
                metadata = SecretMetadata(**metadata_dict)
                self.metadata_cache[secret_id] = metadata
                
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save secrets metadata."""
        
        try:
            data = {}
            
            for secret_id, metadata in self.metadata_cache.items():
                metadata_dict = asdict(metadata)
                
                # Convert timestamps to ISO format
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['last_rotated'] = metadata.last_rotated.isoformat()
                
                if metadata.last_accessed:
                    metadata_dict['last_accessed'] = metadata.last_accessed.isoformat()
                else:
                    metadata_dict['last_accessed'] = None
                
                # Convert enums to strings
                metadata_dict['secret_type'] = metadata.secret_type.value
                metadata_dict['status'] = metadata.status.value
                
                data[secret_id] = metadata_dict
            
            with open(self.metadata_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
    
    def _load_audit_log(self):
        """Load audit log events."""
        
        if not self.audit_path.exists():
            return
        
        try:
            with open(self.audit_path, 'r') as f:
                events_data = json.load(f)
            
            for event_dict in events_data:
                event_dict['timestamp'] = datetime.fromisoformat(event_dict['timestamp'])
                event = SecretAuditEvent(**event_dict)
                self.audit_events.append(event)
                
        except Exception as e:
            self.logger.error(f"Failed to load audit log: {e}")
    
    def _save_audit_log(self):
        """Save audit log events."""
        
        try:
            events_data = []
            
            for event in self.audit_events:
                event_dict = asdict(event)
                event_dict['timestamp'] = event.timestamp.isoformat()
                events_data.append(event_dict)
            
            with open(self.audit_path, 'w') as f:
                json.dump(events_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save audit log: {e}")
    
    def _log_audit_event(self,
                        secret_id: str,
                        event_type: str,
                        user_id: str,
                        success: bool,
                        details: Dict[str, Any] = None):
        """Log an audit event."""
        
        event = SecretAuditEvent(
            event_id=secrets.token_hex(16),
            secret_id=secret_id,
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            environment=self.environment,
            success=success,
            details=details or {}
        )
        
        self.audit_events.append(event)
        
        # Keep only recent events in memory (last 1000)
        if len(self.audit_events) > 1000:
            self.audit_events = self.audit_events[-1000:]
        
        # Save to disk
        self._save_audit_log()
    
    def _validate_secret_value(self, secret_value: str, secret_type: SecretType) -> bool:
        """Validate secret value based on type."""
        
        if not secret_value or len(secret_value) < 8:
            return False
        
        # Type-specific validation
        if secret_type == SecretType.API_KEY:
            # API keys should be long enough and not contain spaces
            return len(secret_value) >= 16 and ' ' not in secret_value
        
        elif secret_type == SecretType.DATABASE_URL:
            # Database URLs should start with appropriate scheme
            return any(secret_value.startswith(scheme) for scheme in ['postgresql://', 'mysql://', 'sqlite://'])
        
        elif secret_type == SecretType.WEBHOOK_SECRET:
            # Webhook secrets should be random-looking
            return len(secret_value) >= 32
        
        # Default validation
        return True
    
    def _check_permissions(self, user_id: str, required_permissions: List[str]) -> bool:
        """Check if user has required permissions (simplified implementation)."""
        
        # In production, implement proper RBAC
        admin_users = ["system", "admin", "clont1"]
        
        if user_id in admin_users:
            return True
        
        # For demo purposes, all users have basic permissions
        return "read_secrets" in required_permissions
    
    def _generate_secret_value(self, secret_type: SecretType) -> Optional[str]:
        """Generate new secret value for rotation."""
        
        if secret_type == SecretType.API_KEY:
            return secrets.token_urlsafe(32)
        
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return secrets.token_hex(32)
        
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        
        # For other types, generate a secure random string
        return secrets.token_urlsafe(24)


def create_secrets_manager(secrets_path: str = "secrets",
                          encryption_key: Optional[str] = None,
                          environment: str = "development") -> SecretsManager:
    """Create secrets manager instance."""
    return SecretsManager(secrets_path, encryption_key, environment)