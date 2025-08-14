"""
Log Sanitizer for CryptoSmartTrader
Enterprise log hygiene system to prevent PII/secrets leakage in logs.
"""

import json
import logging
import re
import hashlib
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
import warnings


class SensitivityLevel(Enum):
    """Levels of data sensitivity for log sanitization."""

    PUBLIC = "public"  # Safe to log
    INTERNAL = "internal"  # Internal use only
    CONFIDENTIAL = "confidential"  # Sensitive data
    SECRET = "secret"  # Highly sensitive (API keys, etc.)
    TOP_SECRET = "top_secret"  # Must never appear in logs


@dataclass
class SanitizationRule:
    """Rule for sanitizing sensitive data in logs."""

    name: str
    pattern: re.Pattern
    replacement: str
    sensitivity: SensitivityLevel
    description: str
    enabled: bool = True

    def sanitize(self, text: str) -> str:
        """Apply sanitization rule to text."""
        if not self.enabled:
            return text
        return self.pattern.sub(self.replacement, text)


class LogSanitizer:
    """
    Enterprise log sanitization system for preventing data leakage.

    Features:
    - Comprehensive pattern matching for sensitive data
    - Configurable sanitization rules
    - Exchange ToS compliance monitoring
    - PII detection and removal
    - API key and credential protection
    - Performance optimized for high-throughput logging
    - Audit trail for sanitization actions
    """

    def __init__(self, config_path: Optional[str] = None, enable_audit: bool = True):
        self.enable_audit = enable_audit
        self.sanitization_rules: List[SanitizationRule] = []
        self.known_secrets: Set[str] = set()
        self.sanitization_stats: Dict[str, int] = {}

        # Initialize default rules
        self._init_default_rules()

        # Load custom config if provided
        if config_path:
            self._load_config(config_path)

        self.logger = logging.getLogger(__name__)

    def sanitize_log_message(self, message: str, level: str = "INFO") -> str:
        """
        Sanitize a log message to remove sensitive information.

        Args:
            message: Original log message
            level: Log level (affects sanitization strictness)

        Returns:
            Sanitized log message
        """

        if not message:
            return message

        sanitized_message = message
        applied_rules = []

        # Apply all sanitization rules
        for rule in self.sanitization_rules:
            if not rule.enabled:
                continue

            # Check if rule was applied
            if rule.pattern.search(sanitized_message):
                sanitized_message = rule.sanitize(sanitized_message)
                applied_rules.append(rule.name)

                # Update stats
                self.sanitization_stats[rule.name] = self.sanitization_stats.get(rule.name, 0) + 1

        # Additional checks for unknown secrets
        sanitized_message = self._sanitize_unknown_secrets(sanitized_message)

        # Audit sanitization if rules were applied
        if applied_rules and self.enable_audit:
            self._audit_sanitization(message, sanitized_message, applied_rules, level)

        return sanitized_message

    def sanitize_dict(self, data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
        """
        Sanitize a dictionary recursively.

        Args:
            data: Dictionary to sanitize
            deep: Whether to sanitize nested structures

        Returns:
            Sanitized dictionary
        """

        if not isinstance(data, dict):
            return data

        sanitized = {}

        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_log_message(str(key))

            # Sanitize value based on type
            if isinstance(value, str):
                sanitized_value = self.sanitize_log_message(value)
            elif isinstance(value, dict) and deep:
                sanitized_value = self.sanitize_dict(value, deep)
            elif isinstance(value, list) and deep:
                sanitized_value = self.sanitize_list(value, deep)
            else:
                sanitized_value = value

            sanitized[sanitized_key] = sanitized_value

        return sanitized

    def sanitize_list(self, data: List[Any], deep: bool = True) -> List[Any]:
        """
        Sanitize a list recursively.

        Args:
            data: List to sanitize
            deep: Whether to sanitize nested structures

        Returns:
            Sanitized list
        """

        if not isinstance(data, list):
            return data

        sanitized = []

        for item in data:
            if isinstance(item, str):
                sanitized_item = self.sanitize_log_message(item)
            elif isinstance(item, dict) and deep:
                sanitized_item = self.sanitize_dict(item, deep)
            elif isinstance(item, list) and deep:
                sanitized_item = self.sanitize_list(item, deep)
            else:
                sanitized_item = item

            sanitized.append(sanitized_item)

        return sanitized

    def register_secret(self, secret: str, identifier: str = None):
        """
        Register a known secret for sanitization.

        Args:
            secret: Secret value to sanitize
            identifier: Optional identifier for the secret
        """

        if secret and len(secret) >= 8:  # Only register substantial secrets
            secret_hash = hashlib.sha256(secret.encode()).hexdigest()[:16]
            self.known_secrets.add(secret)

            if self.enable_audit:
                self.logger.debug(
                    f"Registered secret for sanitization: {identifier or secret_hash}"
                )

    def add_sanitization_rule(self, rule: SanitizationRule):
        """Add a custom sanitization rule."""
        self.sanitization_rules.append(rule)
        self.logger.info(f"Added sanitization rule: {rule.name}")

    def get_sanitization_stats(self) -> Dict[str, Any]:
        """Get sanitization statistics."""
        return {
            "total_rules": len(self.sanitization_rules),
            "enabled_rules": len([r for r in self.sanitization_rules if r.enabled]),
            "known_secrets": len(self.known_secrets),
            "rule_applications": dict(self.sanitization_stats),
            "total_sanitizations": sum(self.sanitization_stats.values()),
        }

    def _init_default_rules(self):
        """Initialize default sanitization rules."""

        # API Keys and Tokens
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="kraken_api_key",
                    pattern=re.compile(r"[A-Za-z0-9+/]{56}={0,2}", re.IGNORECASE),
                    replacement="[KRAKEN_API_KEY_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="Kraken API keys",
                ),
                SanitizationRule(
                    name="binance_api_key",
                    pattern=re.compile(r"[A-Za-z0-9]{64}", re.IGNORECASE),
                    replacement="[BINANCE_API_KEY_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="Binance API keys",
                ),
                SanitizationRule(
                    name="generic_api_key",
                    pattern=re.compile(
                        r'(?:api[_-]?key|apikey|access[_-]?token)[\s]*[:=][\s]*["\']?([A-Za-z0-9+/]{20,})["\']?',
                        re.IGNORECASE,
                    ),
                    replacement=r'api_key="[API_KEY_REDACTED]"',
                    sensitivity=SensitivityLevel.SECRET,
                    description="Generic API keys and access tokens",
                ),
                SanitizationRule(
                    name="bearer_token",
                    pattern=re.compile(r"bearer\s+([A-Za-z0-9+/]{20,})", re.IGNORECASE),
                    replacement="Bearer [TOKEN_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="Bearer tokens",
                ),
            ]
        )

        # Exchange-specific sensitive data
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="exchange_secret",
                    pattern=re.compile(
                        r'(?:secret|signature)[\s]*[:=][\s]*["\']?([A-Za-z0-9+/]{20,})["\']?',
                        re.IGNORECASE,
                    ),
                    replacement='secret="[SECRET_REDACTED]"',
                    sensitivity=SensitivityLevel.SECRET,
                    description="Exchange API secrets",
                ),
                SanitizationRule(
                    name="webhook_signature",
                    pattern=re.compile(r"X-Signature:\s*([A-Za-z0-9+/]{20,})", re.IGNORECASE),
                    replacement="X-Signature: [SIGNATURE_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="Webhook signatures",
                ),
            ]
        )

        # Financial data that may be sensitive
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="large_amounts",
                    pattern=re.compile(
                        r"\$[\d,]+\.?\d*(?:\s*million|\s*M|\s*billion|\s*B)?(?=\s|$)", re.IGNORECASE
                    ),
                    replacement="$[AMOUNT_REDACTED]",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Large monetary amounts",
                ),
                SanitizationRule(
                    name="btc_amounts",
                    pattern=re.compile(r"\d+\.?\d*\s*BTC(?=\s|$)", re.IGNORECASE),
                    replacement="[BTC_AMOUNT_REDACTED] BTC",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Bitcoin amounts",
                ),
            ]
        )

        # Personal Identifiable Information (PII)
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="email_addresses",
                    pattern=re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
                    replacement="[EMAIL_REDACTED]",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Email addresses",
                ),
                SanitizationRule(
                    name="ip_addresses",
                    pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
                    replacement="[IP_REDACTED]",
                    sensitivity=SensitivityLevel.INTERNAL,
                    description="IP addresses",
                ),
                SanitizationRule(
                    name="phone_numbers",
                    pattern=re.compile(r"[\+]?[1-9]?[\d\s\-\(\)]{10,15}"),
                    replacement="[PHONE_REDACTED]",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Phone numbers",
                ),
            ]
        )

        # Cryptocurrency addresses
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="btc_addresses",
                    pattern=re.compile(r"\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b"),
                    replacement="[BTC_ADDRESS_REDACTED]",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Bitcoin addresses",
                ),
                SanitizationRule(
                    name="eth_addresses",
                    pattern=re.compile(r"\b0x[a-fA-F0-9]{40}\b"),
                    replacement="[ETH_ADDRESS_REDACTED]",
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="Ethereum addresses",
                ),
            ]
        )

        # Database and system information
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="database_urls",
                    pattern=re.compile(r"(?:postgresql|mysql|mongodb)://[^\s]+", re.IGNORECASE),
                    replacement="[DATABASE_URL_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="Database connection URLs",
                ),
                SanitizationRule(
                    name="jwt_tokens",
                    pattern=re.compile(r"eyJ[A-Za-z0-9+/]+\.eyJ[A-Za-z0-9+/]+\.[A-Za-z0-9+/]+"),
                    replacement="[JWT_TOKEN_REDACTED]",
                    sensitivity=SensitivityLevel.SECRET,
                    description="JWT tokens",
                ),
            ]
        )

        # Exchange-specific compliance rules
        self.sanitization_rules.extend(
            [
                SanitizationRule(
                    name="order_ids",
                    pattern=re.compile(
                        r'(?:order[_-]?id|orderId)[\s]*[:=][\s]*["\']?([A-Za-z0-9-]{10,})["\']?',
                        re.IGNORECASE,
                    ),
                    replacement='order_id="[ORDER_ID_REDACTED]"',
                    sensitivity=SensitivityLevel.INTERNAL,
                    description="Trading order IDs",
                ),
                SanitizationRule(
                    name="user_ids",
                    pattern=re.compile(
                        r'(?:user[_-]?id|userId|customer[_-]?id)[\s]*[:=][\s]*["\']?([A-Za-z0-9-]{8,})["\']?',
                        re.IGNORECASE,
                    ),
                    replacement='user_id="[USER_ID_REDACTED]"',
                    sensitivity=SensitivityLevel.CONFIDENTIAL,
                    description="User and customer IDs",
                ),
            ]
        )

    def _sanitize_unknown_secrets(self, text: str) -> str:
        """Sanitize known secrets that were registered separately."""

        sanitized_text = text

        for secret in self.known_secrets:
            if secret in sanitized_text:
                # Create a hash-based replacement
                secret_hash = hashlib.sha256(secret.encode()).hexdigest()[:8]
                replacement = f"[SECRET_{secret_hash.upper()}_REDACTED]"
                sanitized_text = sanitized_text.replace(secret, replacement)

                # Update stats
                self.sanitization_stats["known_secrets"] = (
                    self.sanitization_stats.get("known_secrets", 0) + 1
                )

        return sanitized_text

    def _audit_sanitization(
        self, original: str, sanitized: str, applied_rules: List[str], log_level: str
    ):
        """Audit sanitization actions for compliance."""

        if original == sanitized:
            return  # No sanitization occurred

        audit_entry = {
            "timestamp": "%(asctime)s",
            "log_level": log_level,
            "original_length": len(original),
            "sanitized_length": len(sanitized),
            "applied_rules": applied_rules,
            "sanitization_ratio": len(sanitized) / len(original) if original else 1.0,
        }

        # Log audit entry (safely)
        audit_logger = logging.getLogger("log_sanitization_audit")
        audit_logger.info(f"Sanitization applied: {json.dumps(audit_entry)}")

    def _load_config(self, config_path: str):
        """Load custom sanitization configuration."""

        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Load custom rules
            for rule_config in config.get("custom_rules", []):
                rule = SanitizationRule(
                    name=rule_config["name"],
                    pattern=re.compile(rule_config["pattern"], re.IGNORECASE),
                    replacement=rule_config["replacement"],
                    sensitivity=SensitivityLevel(rule_config["sensitivity"]),
                    description=rule_config.get("description", ""),
                    enabled=rule_config.get("enabled", True),
                )
                self.sanitization_rules.append(rule)

            # Load known secrets (hashed)
            for secret_hash in config.get("known_secret_hashes", []):
                self.known_secrets.add(secret_hash)

            self.logger.info(f"Loaded sanitization config from {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load sanitization config: {e}")


class SanitizedLogger:
    """
    Logger wrapper that automatically sanitizes log messages.
    """

    def __init__(self, logger: logging.Logger, sanitizer: LogSanitizer):
        self.logger = logger
        self.sanitizer = sanitizer

    def debug(self, msg, *args, **kwargs):
        sanitized_msg = self.sanitizer.sanitize_log_message(str(msg), "DEBUG")
        self.logger.debug(sanitized_msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        sanitized_msg = self.sanitizer.sanitize_log_message(str(msg), "INFO")
        self.logger.info(sanitized_msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        sanitized_msg = self.sanitizer.sanitize_log_message(str(msg), "WARNING")
        self.logger.warning(sanitized_msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        sanitized_msg = self.sanitizer.sanitize_log_message(str(msg), "ERROR")
        self.logger.error(sanitized_msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        sanitized_msg = self.sanitizer.sanitize_log_message(str(msg), "CRITICAL")
        self.logger.critical(sanitized_msg, *args, **kwargs)


class ExchangeComplianceMonitor:
    """
    Monitor compliance with exchange Terms of Service for logging.
    """

    def __init__(self):
        self.compliance_rules = self._init_compliance_rules()
        self.violations: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def check_compliance(self, log_message: str, exchange: str = None) -> List[str]:
        """
        Check log message for ToS compliance violations.

        Args:
            log_message: Log message to check
            exchange: Specific exchange to check against

        Returns:
            List of violation descriptions
        """

        violations = []

        # Check all applicable rules
        for rule_name, rule_func in self.compliance_rules.items():
            if exchange and exchange not in rule_name and "general" not in rule_name:
                continue

            if rule_func(log_message):
                violations.append(rule_name)

                # Record violation
                violation_record = {
                    "timestamp": "%(asctime)s",
                    "rule": rule_name,
                    "exchange": exchange,
                    "message_length": len(log_message),
                }
                self.violations.append(violation_record)

        return violations

    def _init_compliance_rules(self) -> Dict[str, Callable[[str], bool]]:
        """Initialize compliance checking rules."""

        return {
            "kraken_data_redistribution": lambda msg: self._check_kraken_data_redistribution(msg),
            "binance_user_data": lambda msg: self._check_binance_user_data(msg),
            "general_user_privacy": lambda msg: self._check_user_privacy(msg),
            "general_market_manipulation": lambda msg: self._check_market_manipulation_indicators(
                msg
            ),
        }

    def _check_kraken_data_redistribution(self, msg: str) -> bool:
        """Check for potential Kraken data redistribution violations."""

        # Look for patterns that might indicate unauthorized data sharing
        redistribution_patterns = [
            r"sharing.*kraken.*data",
            r"redistribute.*market.*data",
            r"selling.*kraken.*feed",
        ]

        return any(re.search(pattern, msg, re.IGNORECASE) for pattern in redistribution_patterns)

    def _check_binance_user_data(self, msg: str) -> bool:
        """Check for Binance user data handling violations."""

        user_data_patterns = [
            r"user.*balance.*binance",
            r"personal.*trading.*history",
            r"individual.*portfolio",
        ]

        return any(re.search(pattern, msg, re.IGNORECASE) for pattern in user_data_patterns)

    def _check_user_privacy(self, msg: str) -> bool:
        """Check for general user privacy violations."""

        privacy_patterns = [
            r"user.*[0-9]{6,}",  # User IDs
            r"customer.*[A-Za-z0-9]{8,}",  # Customer identifiers
            r"account.*balance.*[0-9]+",  # Account balances
        ]

        return any(re.search(pattern, msg, re.IGNORECASE) for pattern in privacy_patterns)

    def _check_market_manipulation_indicators(self, msg: str) -> bool:
        """Check for potential market manipulation indicators."""

        manipulation_patterns = [r"pump.*dump", r"artificial.*volume", r"coordinated.*trading"]

        return any(re.search(pattern, msg, re.IGNORECASE) for pattern in manipulation_patterns)


def create_log_sanitizer(
    config_path: Optional[str] = None, enable_audit: bool = True
) -> LogSanitizer:
    """Create log sanitizer instance."""
    return LogSanitizer(config_path, enable_audit)


def create_sanitized_logger(logger_name: str, sanitizer: LogSanitizer) -> SanitizedLogger:
    """Create a sanitized logger wrapper."""
    logger = logging.getLogger(logger_name)
    return SanitizedLogger(logger, sanitizer)
