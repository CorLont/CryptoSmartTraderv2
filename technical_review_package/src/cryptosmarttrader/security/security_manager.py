"""
Security Manager

Enterprise security system with secret management, rotation,
audit logging, and comprehensive security controls.
"""

import os
import json
import hashlib
import hmac
import base64
import secrets
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from pathlib import Path
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecretType(Enum):
    """Types of secrets"""

    API_KEY = "api_key"
    API_SECRET = "api_secret"
    WEBHOOK_SECRET = "webhook_secret"
    DATABASE_URL = "database_url"
    OAUTH_TOKEN = "oauth_token"
    PRIVATE_KEY = "private_key"


class SecurityLevel(Enum):
    """Security levels for secrets"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecretMetadata:
    """Metadata for managed secrets"""

    secret_id: str
    secret_type: SecretType
    security_level: SecurityLevel

    # Lifecycle management
    created_at: datetime
    last_rotated: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    rotation_interval_days: int = 90

    # Access control
    allowed_environments: List[str] = field(default_factory=lambda: ["development"])
    allowed_components: List[str] = field(default_factory=list)

    # Audit
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if secret is expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at

    @property
    def needs_rotation(self) -> bool:
        """Check if secret needs rotation"""
        if self.last_rotated is None:
            return True

        rotation_due = self.last_rotated + timedelta(days=self.rotation_interval_days)
        return datetime.now() > rotation_due


@dataclass
class SecurityAuditEvent:
    """Security audit event"""

    event_id: str
    event_type: str
    timestamp: datetime

    # Context
    user_id: Optional[str] = None
    component: Optional[str] = None
    environment: Optional[str] = None

    # Event details
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"

    # Security context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Enterprise security management system
    """

    def __init__(
        self,
        vault_path: str = ".vault",
        master_key: Optional[str] = None,
        environment: str = "development",
    ):
        self.vault_path = Path(vault_path)
        self.vault_path.mkdir(exist_ok=True, mode=0o700)  # Restricted permissions

        self.environment = environment

        # Initialize encryption
        self.cipher_suite = self._initialize_encryption(master_key)

        # Secret storage
        self.secrets_file = self.vault_path / "secrets.enc"
        self.metadata_file = self.vault_path / "metadata.json"
        self.audit_file = self.vault_path / "audit.log"

        # Load existing secrets and metadata
        self.secrets: Dict[str, str] = self._load_secrets()
        self.metadata: Dict[str, SecretMetadata] = self._load_metadata()

        # Audit system
        self.audit_events: List[SecurityAuditEvent] = []
        self.max_audit_events = 10000

        # Security policies
        self.password_policy = {
            "min_length": 12,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": True,
            "forbidden_patterns": ["password", "123456", "qwerty"],
        }

        # Rate limiting
        self.access_limits = {"secret_access_per_minute": 100, "failed_auth_per_hour": 5}

        self._setup_security_monitoring()

    def _initialize_encryption(self, master_key: Optional[str] = None) -> Fernet:
        """Initialize encryption system"""

        try:
            key_file = self.vault_path / "master.key"

            if master_key:
                # Use provided master key
                key = master_key.encode()
            elif key_file.exists():
                # Load existing key
                with open(key_file, "rb") as f:
                    key = f.read()
            else:
                # Generate new key
                key = Fernet.generate_key()
                with open(key_file, "wb") as f:
                    f.write(key)
                os.chmod(key_file, 0o600)  # Owner read/write only

                logger.info("Generated new encryption key")

            return Fernet(key)

        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise

    def _load_secrets(self) -> Dict[str, str]:
        """Load encrypted secrets from disk"""

        try:
            if not self.secrets_file.exists():
                return {}

            with open(self.secrets_file, "rb") as f:
                encrypted_data = f.read()

            if not encrypted_data:
                return {}

            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            secrets_dict = json.loads(decrypted_data.decode())

            logger.info(f"Loaded {len(secrets_dict)} secrets from vault")
            return secrets_dict

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            return {}

    def _save_secrets(self):
        """Save encrypted secrets to disk"""

        try:
            secrets_json = json.dumps(self.secrets)
            encrypted_data = self.cipher_suite.encrypt(secrets_json.encode())

            # Atomic write
            temp_file = self.secrets_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                f.write(encrypted_data)

            temp_file.replace(self.secrets_file)
            os.chmod(self.secrets_file, 0o600)

            logger.debug("Secrets saved to vault")

        except Exception as e:
            logger.error(f"Failed to save secrets: {e}")
            raise

    def _load_metadata(self) -> Dict[str, SecretMetadata]:
        """Load secret metadata"""

        try:
            if not self.metadata_file.exists():
                return {}

            with open(self.metadata_file, "r") as f:
                data = json.load(f)

            metadata_dict = {}
            for secret_id, meta_data in data.items():
                metadata_dict[secret_id] = SecretMetadata(
                    secret_id=meta_data["secret_id"],
                    secret_type=SecretType(meta_data["secret_type"]),
                    security_level=SecurityLevel(meta_data["security_level"]),
                    created_at=datetime.fromisoformat(meta_data["created_at"]),
                    last_rotated=datetime.fromisoformat(meta_data["last_rotated"])
                    if meta_data.get("last_rotated")
                    else None,
                    expires_at=datetime.fromisoformat(meta_data["expires_at"])
                    if meta_data.get("expires_at")
                    else None,
                    rotation_interval_days=meta_data.get("rotation_interval_days", 90),
                    allowed_environments=meta_data.get("allowed_environments", ["development"]),
                    allowed_components=meta_data.get("allowed_components", []),
                    access_count=meta_data.get("access_count", 0),
                    last_accessed=datetime.fromisoformat(meta_data["last_accessed"])
                    if meta_data.get("last_accessed")
                    else None,
                )

            return metadata_dict

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save secret metadata"""

        try:
            data = {}
            for secret_id, metadata in self.metadata.items():
                data[secret_id] = {
                    "secret_id": metadata.secret_id,
                    "secret_type": metadata.secret_type.value,
                    "security_level": metadata.security_level.value,
                    "created_at": metadata.created_at.isoformat(),
                    "last_rotated": metadata.last_rotated.isoformat()
                    if metadata.last_rotated
                    else None,
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "rotation_interval_days": metadata.rotation_interval_days,
                    "allowed_environments": metadata.allowed_environments,
                    "allowed_components": metadata.allowed_components,
                    "access_count": metadata.access_count,
                    "last_accessed": metadata.last_accessed.isoformat()
                    if metadata.last_accessed
                    else None,
                }

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

            os.chmod(self.metadata_file, 0o600)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def store_secret(
        self,
        secret_id: str,
        secret_value: str,
        secret_type: SecretType,
        security_level: SecurityLevel = SecurityLevel.MEDIUM,
        allowed_environments: Optional[List[str]] = None,
        rotation_interval_days: int = 90,
    ) -> bool:
        """Store a secret securely"""

        try:
            # Validate secret
            if not self._validate_secret(secret_value, secret_type):
                return False

            # Create metadata
            metadata = SecretMetadata(
                secret_id=secret_id,
                secret_type=secret_type,
                security_level=security_level,
                created_at=datetime.now(),
                rotation_interval_days=rotation_interval_days,
                allowed_environments=allowed_environments or [self.environment],
            )

            # Store secret and metadata
            self.secrets[secret_id] = secret_value
            self.metadata[secret_id] = metadata

            # Persist to disk
            self._save_secrets()
            self._save_metadata()

            # Audit log
            self._audit_log("secret_stored", secret_id, result="success")

            logger.info(f"Stored secret: {secret_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store secret {secret_id}: {e}")
            self._audit_log("secret_stored", secret_id, result="error", metadata={"error": str(e)})
            return False

    def get_secret(self, secret_id: str, component: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret with access control"""

        try:
            # Check if secret exists
            if secret_id not in self.secrets:
                self._audit_log("secret_access", secret_id, result="not_found", component=component)
                return None

            metadata = self.metadata.get(secret_id)
            if not metadata:
                self._audit_log(
                    "secret_access", secret_id, result="no_metadata", component=component
                )
                return None

            # Check environment access
            if self.environment not in metadata.allowed_environments:
                self._audit_log(
                    "secret_access", secret_id, result="environment_denied", component=component
                )
                logger.warning(
                    f"Secret {secret_id} access denied for environment {self.environment}"
                )
                return None

            # Check component access
            if metadata.allowed_components and component not in metadata.allowed_components:
                self._audit_log(
                    "secret_access", secret_id, result="component_denied", component=component
                )
                logger.warning(f"Secret {secret_id} access denied for component {component}")
                return None

            # Check expiration
            if metadata.is_expired:
                self._audit_log("secret_access", secret_id, result="expired", component=component)
                logger.warning(f"Secret {secret_id} has expired")
                return None

            # Update access tracking
            metadata.access_count += 1
            metadata.last_accessed = datetime.now()
            self._save_metadata()

            # Audit log
            self._audit_log("secret_access", secret_id, result="success", component=component)

            return self.secrets[secret_id]

        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_id}: {e}")
            self._audit_log(
                "secret_access",
                secret_id,
                result="error",
                component=component,
                metadata={"error": str(e)},
            )
            return None

    def rotate_secret(self, secret_id: str, new_secret_value: Optional[str] = None) -> bool:
        """Rotate a secret"""

        try:
            if secret_id not in self.secrets:
                return False

            metadata = self.metadata[secret_id]

            # Generate new secret if not provided
            if new_secret_value is None:
                new_secret_value = self._generate_secret(metadata.secret_type)

            # Validate new secret
            if not self._validate_secret(new_secret_value, metadata.secret_type):
                return False

            # Update secret and metadata
            old_secret = self.secrets[secret_id]
            self.secrets[secret_id] = new_secret_value
            metadata.last_rotated = datetime.now()

            # Persist changes
            self._save_secrets()
            self._save_metadata()

            # Audit log
            self._audit_log("secret_rotated", secret_id, result="success")

            logger.info(f"Rotated secret: {secret_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secret {secret_id}: {e}")
            self._audit_log("secret_rotated", secret_id, result="error", metadata={"error": str(e)})
            return False

    def delete_secret(self, secret_id: str) -> bool:
        """Delete a secret"""

        try:
            if secret_id not in self.secrets:
                return False

            # Remove secret and metadata
            del self.secrets[secret_id]
            if secret_id in self.metadata:
                del self.metadata[secret_id]

            # Persist changes
            self._save_secrets()
            self._save_metadata()

            # Audit log
            self._audit_log("secret_deleted", secret_id, result="success")

            logger.info(f"Deleted secret: {secret_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            self._audit_log("secret_deleted", secret_id, result="error", metadata={"error": str(e)})
            return False

    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets with metadata (without values)"""

        secrets_list = []

        for secret_id, metadata in self.metadata.items():
            secrets_list.append(
                {
                    "secret_id": secret_id,
                    "secret_type": metadata.secret_type.value,
                    "security_level": metadata.security_level.value,
                    "created_at": metadata.created_at.isoformat(),
                    "last_rotated": metadata.last_rotated.isoformat()
                    if metadata.last_rotated
                    else None,
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "needs_rotation": metadata.needs_rotation,
                    "is_expired": metadata.is_expired,
                    "access_count": metadata.access_count,
                    "last_accessed": metadata.last_accessed.isoformat()
                    if metadata.last_accessed
                    else None,
                    "allowed_environments": metadata.allowed_environments,
                }
            )

        return secrets_list

    def check_rotation_needed(self) -> List[str]:
        """Check which secrets need rotation"""

        rotation_needed = []

        for secret_id, metadata in self.metadata.items():
            if metadata.needs_rotation:
                rotation_needed.append(secret_id)

        return rotation_needed

    def auto_rotate_secrets(self) -> Dict[str, bool]:
        """Automatically rotate secrets that need rotation"""

        rotation_results = {}
        secrets_to_rotate = self.check_rotation_needed()

        for secret_id in secrets_to_rotate:
            try:
                success = self.rotate_secret(secret_id)
                rotation_results[secret_id] = success

                if success:
                    logger.info(f"Auto-rotated secret: {secret_id}")
                else:
                    logger.warning(f"Failed to auto-rotate secret: {secret_id}")

            except Exception as e:
                logger.error(f"Auto-rotation failed for {secret_id}: {e}")
                rotation_results[secret_id] = False

        return rotation_results

    def _validate_secret(self, secret_value: str, secret_type: SecretType) -> bool:
        """Validate secret according to type and security policies"""

        if not secret_value or len(secret_value.strip()) == 0:
            return False

        # Type-specific validation
        if secret_type == SecretType.API_KEY:
            # API keys should be at least 20 characters
            return len(secret_value) >= 20

        elif secret_type == SecretType.DATABASE_URL:
            # Basic URL validation
            return secret_value.startswith(("postgresql://", "mysql://", "sqlite://"))

        elif secret_type in [SecretType.API_SECRET, SecretType.WEBHOOK_SECRET]:
            # Apply password policy
            return self._validate_password(secret_value)

        return True

    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy"""

        policy = self.password_policy

        # Length check
        if len(password) < policy["min_length"]:
            return False

        # Character requirements
        if policy["require_uppercase"] and not re.search(r"[A-Z]", password):
            return False

        if policy["require_lowercase"] and not re.search(r"[a-z]", password):
            return False

        if policy["require_digits"] and not re.search(r"\d", password):
            return False

        if policy["require_special"] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False

        # Forbidden patterns
        password_lower = password.lower()
        for pattern in policy["forbidden_patterns"]:
            if pattern.lower() in password_lower:
                return False

        return True

    def _generate_secret(self, secret_type: SecretType) -> str:
        """Generate a new secret of the specified type"""

        if secret_type == SecretType.API_KEY:
            # Generate 32-character API key
            return secrets.token_hex(16)

        elif secret_type in [SecretType.API_SECRET, SecretType.WEBHOOK_SECRET]:
            # Generate strong password
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*"
            return "".join(secrets.choice(chars) for _ in range(32))

        elif secret_type == SecretType.OAUTH_TOKEN:
            # Generate OAuth-style token
            return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")

        else:
            # Default: random hex string
            return secrets.token_hex(32)

    def _audit_log(
        self,
        event_type: str,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: str = "success",
        component: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log security audit event"""

        try:
            event = SecurityAuditEvent(
                event_id=secrets.token_hex(8),
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                component=component,
                environment=self.environment,
                resource=resource,
                action=action,
                result=result,
                metadata=metadata or {},
            )

            self.audit_events.append(event)

            # Limit audit events in memory
            if len(self.audit_events) > self.max_audit_events:
                self.audit_events = self.audit_events[-self.max_audit_events // 2 :]

            # Write to audit log file
            audit_line = {
                "timestamp": event.timestamp.isoformat(),
                "event_id": event.event_id,
                "event_type": event_type,
                "result": result,
                "resource": resource,
                "component": component,
                "environment": self.environment,
                "metadata": metadata,
            }

            with open(self.audit_file, "a") as f:
                f.write(json.dumps(audit_line) + "\n")

        except Exception as e:
            logger.error(f"Audit logging failed: {e}")

    def _setup_security_monitoring(self):
        """Setup security monitoring and alerting"""

        # This would integrate with monitoring systems
        logger.info("Security monitoring initialized")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""

        total_secrets = len(self.secrets)
        rotation_needed = len(self.check_rotation_needed())
        expired_secrets = sum(1 for metadata in self.metadata.values() if metadata.is_expired)

        # Recent audit events
        recent_events = [
            e for e in self.audit_events if e.timestamp > datetime.now() - timedelta(hours=24)
        ]
        failed_events = [e for e in recent_events if e.result != "success"]

        return {
            "total_secrets": total_secrets,
            "rotation_needed": rotation_needed,
            "expired_secrets": expired_secrets,
            "recent_audit_events": len(recent_events),
            "recent_failed_events": len(failed_events),
            "vault_path": str(self.vault_path),
            "environment": self.environment,
            "security_health": "healthy"
            if rotation_needed == 0 and expired_secrets == 0
            else "needs_attention",
        }

    def export_audit_log(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Export audit log for compliance"""

        events = self.audit_events

        if start_date:
            events = [e for e in events if e.timestamp >= start_date]

        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        return [
            {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "resource": event.resource,
                "action": event.action,
                "result": event.result,
                "component": event.component,
                "environment": event.environment,
                "metadata": event.metadata,
            }
            for event in events
        ]

    def validate_environment_secrets(self) -> Dict[str, bool]:
        """Validate that all required secrets are available for current environment"""

        required_secrets = {
            "development": ["KRAKEN_API_KEY", "KRAKEN_SECRET", "OPENAI_API_KEY"],
            "staging": ["KRAKEN_API_KEY", "KRAKEN_SECRET", "OPENAI_API_KEY"],
            "production": [
                "KRAKEN_API_KEY",
                "KRAKEN_SECRET",
                "OPENAI_API_KEY",
                "SLACK_BOT_TOKEN",
                "SLACK_CHANNEL_ID",
            ],
        }

        required = required_secrets.get(self.environment, [])
        validation_results = {}

        for secret_id in required:
            # Try to get from vault first, then environment
            secret_value = self.get_secret(secret_id)
            if not secret_value:
                secret_value = os.getenv(secret_id)

            validation_results[secret_id] = (
                secret_value is not None and len(secret_value.strip()) > 0
            )

        return validation_results
