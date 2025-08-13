# utils/secrets.py
from typing import Dict, Optional
import os
import logging
from config.settings import config

try:
    import hvac
    HVAC_AVAILABLE = True
except ImportError:
    HVAC_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretManager:
    """
    Secure secret management with HashiCorp Vault integration.
    Falls back to environment variables if Vault is not configured.
    """

    def __init__(self):
        self.vault_client = None
        if config.vault_configured and HVAC_AVAILABLE:
            try:
                self.vault_client = hvac.Client(
                    url=config.vault_addr,
                    token=config.vault_token
                )
                if not self.vault_client.is_authenticated():
                    logger.warning("Vault authentication failed, falling back to env vars")
                    self.vault_client = None
                else:
                    logger.info("Successfully connected to HashiCorp Vault")
            except Exception as e:
                logger.error(f"Failed to connect to Vault: {e}")
                self.vault_client = None

    def get_secret(self, path: str, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a secret from Vault or environment variables.

        Args:
            path: Vault secret path
            key: Secret key name
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Try Vault first if available
        if self.vault_client:
            try:
                response = self.vault_client.secrets.kv.v2.read_secret_version(path=path)
                secret_data = response["data"]["data"]
                if key in secret_data:
                    logger.debug(f"Retrieved secret '{key}' from Vault")
                    return secret_data[key]
            except Exception as e:
                logger.warning(f"Failed to retrieve secret from Vault: {e}")

        # Fall back to environment variables
        env_key = f"{path.replace('/', '_').upper()}_{key.upper()}"
        value = os.getenv(env_key, default)

        if value:
            logger.debug(f"Retrieved secret '{key}' from environment")

        return value

    def get_database_credentials(self) -> Dict[str, Optional[str]]:
        """Get database connection credentials"""
        return {
            "host": self.get_secret("database", "host", "localhost"),
            "port": self.get_secret("database", "port", "5432"),
            "database": self.get_secret("database", "name", "cryptotrader"),
            "username": self.get_secret("database", "username"),
            "password": self.get_secret("database", "password"),
        }

    def get_exchange_credentials(self, exchange: str) -> Dict[str, Optional[str]]:
        """Get exchange API credentials"""
        return {
            "api_key": self.get_secret(f"exchanges/{exchange}", "api_key"),
            "secret": self.get_secret(f"exchanges/{exchange}", "secret"),
            "passphrase": self.get_secret(f"exchanges/{exchange}", "passphrase"),
            "sandbox": self.get_secret(f"exchanges/{exchange}", "sandbox", "false").lower() == "true"
        }

    def get_notification_credentials(self) -> Dict[str, Optional[str]]:
        """Get notification service credentials"""
        return {
            "smtp_host": self.get_secret("notifications", "smtp_host"),
            "smtp_port": self.get_secret("notifications", "smtp_port", "587"),
            "smtp_username": self.get_secret("notifications", "smtp_username"),
            "smtp_password": self.get_secret("notifications", "smtp_password"),
            "from_email": self.get_secret("notifications", "from_email"),
        }


# Global secret manager instance
secret_manager = SecretManager()
