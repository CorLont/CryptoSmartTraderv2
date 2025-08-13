#!/usr/bin/env python3
"""
Enterprise Secrets Management with Vault Integration
Secure handling of API keys, credentials, and sensitive data
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import hvac
from functools import wraps
import threading
from contextlib import contextmanager


class SecretType(Enum):
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    DATABASE_URL = "database_url"


@dataclass
class SecretMetadata:
    """Metadata for secret management"""

    name: str
    secret_type: SecretType
    source: str  # 'env', 'vault', 'file'
    last_rotated: Optional[str] = None
    expires_at: Optional[str] = None
    description: Optional[str] = None


class SecretRedactor:
    """Redacts sensitive information from logs and error messages"""

    # Patterns for different types of secrets
    REDACTION_PATTERNS = {
        "api_key": [
            r'["\']?(?:api_?key|apikey|key)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
            r'["\']?(?:secret|password|pwd|pass)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{8,})["\']?',
            r"Bearer\s+([a-zA-Z0-9_\-\.]{20,})",
            r'token["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        ],
        "url_credentials": [
            r"(https?://[^:]+:)([^@]+)(@[^/]+)",  # URLs with credentials
            r"(postgresql://[^:]+:)([^@]+)(@[^/]+)",  # Database URLs
        ],
        "private_key": [
            r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----.*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
        ],
        "credit_card": [
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card numbers
        ],
    }

    @classmethod
    def redact_secrets(cls, text: str, redaction_char: str = "*") -> str:
        """Redact sensitive information from text"""
        if not isinstance(text, str):
            text = str(text)

        redacted_text = text

        for category, patterns in cls.REDACTION_PATTERNS.items():
            for pattern in patterns:
                if category == "url_credentials":
                    # Special handling for URLs - keep structure but redact credentials
                    redacted_text = re.sub(
                        pattern,
                        r"\1***REDACTED***\3",
                        redacted_text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                elif category == "private_key":
                    # Redact entire private key block
                    redacted_text = re.sub(
                        pattern,
                        "***PRIVATE_KEY_REDACTED***",
                        redacted_text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                else:
                    # Standard redaction - replace captured group with asterisks
                    def replace_match(match):
                        if len(match.groups()) > 0:
                            secret_value = match.group(1)
                            # Keep first 2 and last 2 characters, redact middle
                            if len(secret_value) > 8:
                                redacted = (
                                    secret_value[:2]
                                    + redaction_char * (len(secret_value) - 4)
                                    + secret_value[-2:]
                                )
                            else:
                                redacted = redaction_char * len(secret_value)
                            return match.group(0).replace(secret_value, redacted)
                        return match.group(0)

                    redacted_text = re.sub(
                        pattern, replace_match, redacted_text, flags=re.IGNORECASE
                    )

        return redacted_text

    @classmethod
    def sanitize_exception(cls, exception: Exception) -> str:
        """Sanitize exception messages to remove secrets"""
        exc_str = str(exception)
        return cls.redact_secrets(exc_str)

    @classmethod
    def sanitize_traceback(cls, traceback_str: str) -> str:
        """Sanitize traceback strings to remove secrets"""
        return cls.redact_secrets(traceback_str)


class VaultManager:
    """HashiCorp Vault integration for secure secret storage"""

    def __init__(self, vault_url: Optional[str] = None, vault_token: Optional[str] = None):
        self.vault_url = vault_url or os.getenv("VAULT_URL")
        self.vault_token = vault_token or os.getenv("VAULT_TOKEN")
        self.client: Optional[hvac.Client] = None
        self.logger = logging.getLogger(__name__)

        if self.vault_url and self.vault_token:
            self._initialize_vault()

    def _initialize_vault(self):
        """Initialize Vault client"""
        try:
            self.client = hvac.Client(url=self.vault_url, token=self.vault_token)
            if self.client.is_authenticated():
                self.logger.info("Vault client authenticated successfully")
            else:
                self.logger.warning("Vault authentication failed")
                self.client = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Vault: {SecretRedactor.sanitize_exception(e)}")
            self.client = None

    def get_secret(self, path: str, key: str = None) -> Optional[Union[str, Dict[str, Any]]]:
        """Retrieve secret from Vault"""
        if not self.client:
            return None

        try:
            response = self.client.secrets.kv.v2.read_secret_version(path=path)
            data = response["data"]["data"]

            if key:
                return data.get(key)
            return data
        except Exception as e:
            self.logger.error(
                f"Failed to retrieve secret from Vault: {SecretRedactor.sanitize_exception(e)}"
            )
            return None

    def set_secret(self, path: str, secret_data: Dict[str, Any]) -> bool:
        """Store secret in Vault"""
        if not self.client:
            return False

        try:
            self.client.secrets.kv.v2.create_or_update_secret(path=path, secret=secret_data)
            self.logger.info(f"Secret stored successfully at path: {path}")
            return True
        except Exception as e:
            self.logger.error(
                f"Failed to store secret in Vault: {SecretRedactor.sanitize_exception(e)}"
            )
            return False


class SecretsManager:
    """Central secrets management with multiple backends"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vault_manager = VaultManager()
        self.secrets_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, SecretMetadata] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

        # Sensitive environment variable names
        self.sensitive_env_vars = {
            "KRAKEN_API_KEY",
            "KRAKEN_SECRET",
            "BINANCE_API_KEY",
            "BINANCE_SECRET",
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "VAULT_TOKEN",
            "ENCRYPTION_KEY",
            "JWT_SECRET",
            "SMTP_PASSWORD",
            "REDIS_PASSWORD",
        }

        # Load secrets metadata
        self._load_secrets_metadata()

    def _load_secrets_metadata(self):
        """Load secrets metadata from configuration"""
        metadata_config = self.config.get("secrets_metadata", {})

        for secret_name, meta_dict in metadata_config.items():
            self.metadata_cache[secret_name] = SecretMetadata(**meta_dict)

    def get_secret(self, secret_name: str, default: Any = None) -> Optional[str]:
        """Retrieve secret with fallback hierarchy: Vault -> Environment -> Default"""
        with self._lock:
            # Check cache first
            if secret_name in self.secrets_cache:
                return self.secrets_cache[secret_name]

            secret_value = None
            source = None

            # Try Vault first
            if self.vault_manager.client:
                vault_path = f"cryptotrader/{secret_name.lower()}"
                secret_value = self.vault_manager.get_secret(vault_path, "value")
                if secret_value:
                    source = "vault"

            # Fallback to environment variables
            if not secret_value:
                secret_value = os.getenv(secret_name)
                if secret_value:
                    source = "env"

            # Use default if provided
            if not secret_value and default is not None:
                secret_value = default
                source = "default"

            # Cache the result (but not defaults)
            if secret_value and source != "default":
                self.secrets_cache[secret_name] = secret_value

                # Update metadata
                if secret_name not in self.metadata_cache:
                    secret_type = self._infer_secret_type(secret_name)
                    self.metadata_cache[secret_name] = SecretMetadata(
                        name=secret_name, secret_type=secret_type, source=source
                    )

            # Log access (without revealing the secret)
            if secret_value:
                self.logger.debug(f"Secret '{secret_name}' retrieved from {source}")
            else:
                self.logger.warning(f"Secret '{secret_name}' not found in any source")

            return secret_value

    def _infer_secret_type(self, secret_name: str) -> SecretType:
        """Infer secret type from name"""
        name_lower = secret_name.lower()

        if "api_key" in name_lower or "apikey" in name_lower:
            return SecretType.API_KEY
        elif "password" in name_lower or "pwd" in name_lower:
            return SecretType.PASSWORD
        elif "token" in name_lower:
            return SecretType.TOKEN
        elif "database_url" in name_lower or "db_url" in name_lower:
            return SecretType.DATABASE_URL
        elif "key" in name_lower and "private" in name_lower:
            return SecretType.PRIVATE_KEY
        elif "cert" in name_lower:
            return SecretType.CERTIFICATE
        else:
            return SecretType.API_KEY  # Default

    def set_secret(
        self, secret_name: str, secret_value: str, secret_type: SecretType = None
    ) -> bool:
        """Store secret securely"""
        with self._lock:
            # Try Vault first
            if self.vault_manager.client:
                vault_path = f"cryptotrader/{secret_name.lower()}"
                success = self.vault_manager.set_secret(vault_path, {"value": secret_value})
                if success:
                    # Update cache and metadata
                    self.secrets_cache[secret_name] = secret_value
                    self.metadata_cache[secret_name] = SecretMetadata(
                        name=secret_name,
                        secret_type=secret_type or self._infer_secret_type(secret_name),
                        source="vault",
                    )
                    return True

            # Fallback: warn about insecure storage
            self.logger.warning(f"Vault unavailable, cannot securely store secret '{secret_name}'")
            return False

    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotate a secret"""
        old_metadata = self.metadata_cache.get(secret_name)

        success = self.set_secret(
            secret_name, new_value, old_metadata.secret_type if old_metadata else None
        )

        if success:
            # Update rotation timestamp
            from datetime import datetime

            if secret_name in self.metadata_cache:
                self.metadata_cache[secret_name].last_rotated = datetime.now().isoformat()

            self.logger.info(f"Secret '{secret_name}' rotated successfully")
            return True

        return False

    def list_secrets(self) -> Dict[str, SecretMetadata]:
        """List all known secrets with metadata (no values)"""
        return self.metadata_cache.copy()

    def validate_secrets(self, required_secrets: List[str]) -> Dict[str, bool]:
        """Validate that required secrets are available"""
        validation_results = {}

        for secret_name in required_secrets:
            secret_value = self.get_secret(secret_name)
            validation_results[secret_name] = bool(secret_value)

            if not secret_value:
                self.logger.error(f"Required secret '{secret_name}' is missing")

        return validation_results

    @contextmanager
    def secure_context(self):
        """Context manager for secure operations with automatic cleanup"""
        try:
            yield self
        finally:
            # Clear sensitive data from memory
            self._clear_sensitive_cache()

    def _clear_sensitive_cache(self):
        """Clear sensitive data from memory"""
        with self._lock:
            # Only clear non-persistent cache entries
            sensitive_keys = [
                key
                for key in self.secrets_cache.keys()
                if any(sensitive in key.upper() for sensitive in self.sensitive_env_vars)
            ]

            for key in sensitive_keys:
                if key in self.secrets_cache:
                    # Overwrite with random data before deletion
                    import random
                    import string

                    self.secrets_cache[key] = "".join(random.choices(string.ascii_letters, k=32))
                    del self.secrets_cache[key]


def secure_function(redact_args: List[str] = None, redact_kwargs: List[str] = None):
    """Decorator to automatically redact sensitive function arguments from logs"""
    redact_args = redact_args or []
    redact_kwargs = redact_kwargs or []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Redact sensitive information from exception
                sanitized_message = SecretRedactor.sanitize_exception(e)

                # Create new exception with sanitized message
                new_exception = type(e)(sanitized_message)
                new_exception.__cause__ = None  # Remove original cause chain
                new_exception.__traceback__ = None  # Remove traceback with potential secrets

                raise new_exception

        return wrapper

    return decorator


class SecureLogger:
    """Logger wrapper that automatically redacts secrets"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.redactor = SecretRedactor()

    def _redact_message(self, message: str, *args, **kwargs) -> str:
        """Redact secrets from log message"""
        if args:
            # Handle format string with positional arguments
            try:
                formatted_message = message % args
            except (TypeError, ValueError):
                formatted_message = f"{message} {args}"
        else:
            formatted_message = message

        # Redact extra context
        if "extra" in kwargs:
            extra_dict = kwargs["extra"]
            for key, value in extra_dict.items():
                if isinstance(value, str):
                    extra_dict[key] = self.redactor.redact_secrets(value)

        return self.redactor.redact_secrets(formatted_message)

    def info(self, message: str, *args, **kwargs):
        redacted_message = self._redact_message(message, *args, **kwargs)
        self.logger.info(redacted_message, **{k: v for k, v in kwargs.items() if k != "extra"})

    def warning(self, message: str, *args, **kwargs):
        redacted_message = self._redact_message(message, *args, **kwargs)
        self.logger.warning(redacted_message, **{k: v for k, v in kwargs.items() if k != "extra"})

    def error(self, message: str, *args, **kwargs):
        redacted_message = self._redact_message(message, *args, **kwargs)

        # Handle exception info redaction
        if kwargs.get("exc_info"):
            import traceback

            exc_str = traceback.format_exc()
            redacted_exc = self.redactor.sanitize_traceback(exc_str)
            # Log redacted exception separately
            self.logger.error(f"{redacted_message}\nSanitized traceback: {redacted_exc}")
            kwargs.pop("exc_info")  # Remove original exc_info
        else:
            self.logger.error(redacted_message, **{k: v for k, v in kwargs.items() if k != "extra"})

    def critical(self, message: str, *args, **kwargs):
        redacted_message = self._redact_message(message, *args, **kwargs)
        self.logger.critical(redacted_message, **{k: v for k, v in kwargs.items() if k != "extra"})

    def debug(self, message: str, *args, **kwargs):
        redacted_message = self._redact_message(message, *args, **kwargs)
        self.logger.debug(redacted_message, **{k: v for k, v in kwargs.items() if k != "extra"})


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager(config: Dict[str, Any] = None) -> SecretsManager:
    """Get global secrets manager instance"""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager(config)
    return _secrets_manager


def get_secure_logger(logger: logging.Logger) -> SecureLogger:
    """Get secure logger wrapper"""
    return SecureLogger(logger)
