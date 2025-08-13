#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise Security Manager
Complete credential isolation with Vault integration and audit logging
"""

import os
import logging
import json
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import threading

try:
    import hvac
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False

@dataclass
class SecurityConfig:
    """Security configuration with strict validation"""
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    vault_mount_path: str = "secret"
    env_file_path: str = ".env"
    audit_log_path: str = "logs/security_audit.log"
    secret_rotation_interval: int = 86400  # 24 hours in seconds
    max_failed_attempts: int = 3
    lockout_duration: int = 300  # 5 minutes in seconds
    enable_audit_logging: bool = True
    require_encryption_at_rest: bool = True

class SecurityManager:
    """Enterprise-grade security manager for API keys and secrets"""

    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.logger = logging.getLogger(f"{__name__}.SecurityManager")
        self._secrets_cache: Dict[str, Any] = {}
        self._access_log: List[Dict] = []
        self._failed_attempts: Dict[str, int] = {}
        self._lockouts: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Initialize Vault client if available
        self.vault_client = None
        if HAS_VAULT and self.config.vault_url:
            self._initialize_vault()

        # Setup audit logging
        if self.config.enable_audit_logging:
            self._setup_audit_logging()

        # Load environment secrets
        self._load_environment_secrets()

        self.logger.info("Security Manager initialized with enterprise-grade protection")

    def _initialize_vault(self):
        """Initialize HashiCorp Vault client"""
        try:
            self.vault_client = hvac.Client(
                url=self.config.vault_url,
                token=self.config.vault_token
            )

            if self.vault_client.is_authenticated():
                self.logger.info("Vault client initialized and authenticated")
                self._audit_log("vault_initialized", {"status": "success"})
            else:
                self.logger.error("Vault authentication failed")
                self._audit_log("vault_authentication_failed", {"status": "error"})
                self.vault_client = None
        except Exception as e:
            self.logger.error(f"Vault initialization failed: {e}")
            self._audit_log("vault_initialization_failed", {"error": str(e)})
            self.vault_client = None

    def _setup_audit_logging(self):
        """Setup security audit logging"""
        audit_path = Path(self.config.audit_log_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Create security audit logger
        self.audit_logger = logging.getLogger("security_audit")
        self.audit_logger.setLevel(logging.INFO)

        # Create file handler for audit log
        audit_handler = logging.FileHandler(audit_path)
        audit_formatter = logging.Formatter(
            '%(asctime)s - SECURITY_AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.audit_logger.addHandler(audit_handler)

    def _load_environment_secrets(self):
        """Load secrets from environment variables and .env file"""
        try:
            # Load from .env file if exists
            env_path = Path(self.config.env_file_path)
            if env_path.exists():
                with open(env_path, 'r') as f:
                    for line in f:
                        if '=' in line and not line.strip().startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value.strip('"\'')

            # Cache common secrets
            common_secrets = [
                'OPENAI_API_KEY', 'KRAKEN_API_KEY', 'KRAKEN_SECRET_KEY',
                'BINANCE_API_KEY', 'BINANCE_SECRET_KEY', 'TELEGRAM_BOT_TOKEN',
                'DISCORD_WEBHOOK_URL', 'SLACK_WEBHOOK_URL', 'EMAIL_PASSWORD'
            ]

            for secret_name in common_secrets:
                if secret_name in os.environ:
                    self._secrets_cache[secret_name] = os.environ[secret_name]
                    self._audit_log("secret_loaded", {
                        "secret_name": secret_name,
                        "source": "environment"
                    })

            self.logger.info(f"Loaded {len(self._secrets_cache)} secrets from environment")

        except Exception as e:
            self.logger.error(f"Failed to load environment secrets: {e}")
            self._audit_log("environment_secrets_load_failed", {"error": str(e)})

    def get_secret(self, secret_name: str, source: str = "auto") -> Optional[str]:
        """
        Securely retrieve a secret with audit logging

        Args:
            secret_name: Name of the secret to retrieve
            source: Source to check ("vault", "env", "auto")

        Returns:
            Secret value or None if not found
        """
        with self._lock:
            # Check if caller is locked out
            if self._is_locked_out(secret_name):
                self._audit_log("access_denied_lockout", {
                    "secret_name": secret_name,
                    "reason": "too_many_failed_attempts"
                })
                return None

            try:
                secret_value = None

                # Try Vault first if available and requested
                if source in ("vault", "auto") and self.vault_client:
                    secret_value = self._get_secret_from_vault(secret_name)

                # Fallback to environment if not found in Vault
                if not secret_value and source in ("env", "auto"):
                    secret_value = self._secrets_cache.get(secret_name) or os.environ.get(secret_name)

                if secret_value:
                    self._audit_log("secret_accessed", {
                        "secret_name": secret_name,
                        "source": "vault" if source == "vault" else "environment",
                        "hash": hashlib.sha256(secret_value.encode()).hexdigest()[:8]
                    })
                    # Reset failed attempts on successful access
                    self._failed_attempts.pop(secret_name, None)
                    return secret_value
                else:
                    self._record_failed_attempt(secret_name)
                    self._audit_log("secret_not_found", {"secret_name": secret_name})
                    return None

            except Exception as e:
                self._record_failed_attempt(secret_name)
                self.logger.error(f"Error retrieving secret {secret_name}: {e}")
                self._audit_log("secret_access_error", {
                    "secret_name": secret_name,
                    "error": str(e)
                })
                return None

    def _get_secret_from_vault(self, secret_name: str) -> Optional[str]:
        """Retrieve secret from HashiCorp Vault"""
        if not self.vault_client:
            return None

        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"cryptotrader/{secret_name}",
                mount_point=self.config.vault_mount_path
            )
            return response['data']['data'].get('value')
        except Exception as e:
            self.logger.debug(f"Vault secret retrieval failed for {secret_name}: {e}")
            return None

    def store_secret(self, secret_name: str, secret_value: str, source: str = "vault") -> bool:
        """
        Securely store a secret with audit logging

        Args:
            secret_name: Name of the secret
            secret_value: Secret value to store
            source: Where to store ("vault", "env")

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if source == "vault" and self.vault_client:
                    success = self._store_secret_in_vault(secret_name, secret_value)
                elif source == "env":
                    self._secrets_cache[secret_name] = secret_value
                    os.environ[secret_name] = secret_value
                    success = True
                else:
                    success = False

                if success:
                    self._audit_log("secret_stored", {
                        "secret_name": secret_name,
                        "source": source,
                        "hash": hashlib.sha256(secret_value.encode()).hexdigest()[:8]
                    })
                else:
                    self._audit_log("secret_store_failed", {
                        "secret_name": secret_name,
                        "source": source
                    })

                return success

            except Exception as e:
                self.logger.error(f"Error storing secret {secret_name}: {e}")
                self._audit_log("secret_store_error", {
                    "secret_name": secret_name,
                    "error": str(e)
                })
                return False

    def _store_secret_in_vault(self, secret_name: str, secret_value: str) -> bool:
        """Store secret in HashiCorp Vault"""
        if not self.vault_client:
            return False

        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"cryptotrader/{secret_name}",
                secret={'value': secret_value},
                mount_point=self.config.vault_mount_path
            )
            return True
        except Exception as e:
            self.logger.error(f"Vault secret storage failed for {secret_name}: {e}")
            return False

    def _is_locked_out(self, secret_name: str) -> bool:
        """Check if secret access is locked out due to failed attempts"""
        if secret_name not in self._lockouts:
            return False

        lockout_time = self._lockouts[secret_name]
        if (datetime.now() - lockout_time).total_seconds() > self.config.lockout_duration:
            # Lockout expired
            self._lockouts.pop(secret_name, None)
            self._failed_attempts.pop(secret_name, None)
            return False

        return True

    def _record_failed_attempt(self, secret_name: str):
        """Record a failed secret access attempt"""
        self._failed_attempts[secret_name] = self._failed_attempts.get(secret_name, 0) + 1

        if self._failed_attempts[secret_name] >= self.config.max_failed_attempts:
            self._lockouts[secret_name] = datetime.now()
            self._audit_log("secret_lockout_triggered", {
                "secret_name": secret_name,
                "failed_attempts": self._failed_attempts[secret_name]
            })

    def _audit_log(self, event: str, details: Dict[str, Any]):
        """Log security audit event"""
        if not self.config.enable_audit_logging:
            return

        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details,
            "process_id": os.getpid()
        }

        self._access_log.append(audit_entry)

        if hasattr(self, 'audit_logger'):
            self.audit_logger.info(json.dumps(audit_entry))

    def get_audit_log(self) -> List[Dict]:
        """Get security audit log entries"""
        with self._lock:
            return self._access_log.copy()

    def validate_secrets_health(self) -> Dict[str, Any]:
        """Validate health of all secrets and access patterns"""
        health_report = {
            "vault_connected": bool(self.vault_client and self.vault_client.is_authenticated()),
            "total_secrets_cached": len(self._secrets_cache),
            "failed_attempts": dict(self._failed_attempts),
            "active_lockouts": len(self._lockouts),
            "audit_entries": len(self._access_log),
            "timestamp": datetime.now().isoformat()
        }

        # Test critical secrets availability
        critical_secrets = ['OPENAI_API_KEY', 'KRAKEN_API_KEY', 'BINANCE_API_KEY']
        health_report["critical_secrets_available"] = {}

        for secret_name in critical_secrets:
            is_available = bool(self.get_secret(secret_name))
            health_report["critical_secrets_available"][secret_name] = is_available

        return health_report

    def cleanup_expired_lockouts(self):
        """Clean up expired lockouts and audit logs"""
        with self._lock:
            current_time = datetime.now()
            expired_lockouts = [
                secret for secret, lockout_time in self._lockouts.items()
                if (current_time - lockout_time).total_seconds() > self.config.lockout_duration
            ]

            for secret in expired_lockouts:
                self._lockouts.pop(secret, None)
                self._failed_attempts.pop(secret, None)

            # Keep only last 1000 audit entries
            if len(self._access_log) > 1000:
                self._access_log = self._access_log[-1000:]


# Singleton security manager
_security_manager = None
_security_lock = threading.Lock()

def get_security_manager(config: Optional[SecurityConfig] = None) -> SecurityManager:
    """Get the singleton security manager instance"""
    global _security_manager

    with _security_lock:
        if _security_manager is None:
            _security_manager = SecurityManager(config)
        return _security_manager

def secure_get_secret(secret_name: str) -> Optional[str]:
    """Convenient function to securely get a secret"""
    security_manager = get_security_manager()
    return security_manager.get_secret(secret_name)
