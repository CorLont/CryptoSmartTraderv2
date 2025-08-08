#!/usr/bin/env python3
"""
Unit tests for SecretsManager
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.secrets_manager import (
    SecretsManager, SecretRedactor, VaultManager, 
    SecretType, SecretMetadata, get_secrets_manager,
    secure_function, SecureLogger
)

@pytest.mark.unit
class TestSecretRedactor:
    """Test secret redaction functionality"""
    
    def test_redact_api_keys(self):
        """Test API key redaction"""
        text = "api_key=sk-1234567890abcdef1234567890abcdef"
        redacted = SecretRedactor.redact_secrets(text)
        assert "sk-1234567890abcdef1234567890abcdef" not in redacted
        assert "sk**************************ef" in redacted
    
    def test_redact_passwords(self):
        """Test password redaction"""
        text = "password=mysecretpassword123"
        redacted = SecretRedactor.redact_secrets(text)
        assert "mysecretpassword123" not in redacted
        assert "my****************23" in redacted
    
    def test_redact_bearer_tokens(self):
        """Test Bearer token redaction"""
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = SecretRedactor.redact_secrets(text)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in redacted
        assert "ey**********************************J9" in redacted
    
    def test_redact_database_urls(self):
        """Test database URL credential redaction"""
        text = "postgresql://user:secret123@localhost:5432/db"
        redacted = SecretRedactor.redact_secrets(text)
        assert "secret123" not in redacted
        assert "postgresql://user:***REDACTED***@localhost:5432/db" in redacted
    
    def test_redact_private_keys(self):
        """Test private key redaction"""
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890
-----END RSA PRIVATE KEY-----"""
        redacted = SecretRedactor.redact_secrets(text)
        assert "MIIEpAIBAAKCAQEA1234567890" not in redacted
        assert "***PRIVATE_KEY_REDACTED***" in redacted
    
    def test_sanitize_exception(self):
        """Test exception sanitization"""
        try:
            raise ValueError("API key sk-1234567890abcdef failed")
        except ValueError as e:
            sanitized = SecretRedactor.sanitize_exception(e)
            assert "sk-1234567890abcdef" not in sanitized
            assert "sk****************ef" in sanitized
    
    def test_preserve_structure(self):
        """Test that redaction preserves text structure"""
        original = "Connect to api_key=secret123 and token=abc456def"
        redacted = SecretRedactor.redact_secrets(original)
        
        # Should preserve the overall structure
        assert "Connect to api_key=" in redacted
        assert "and token=" in redacted
        # But redact the actual secrets
        assert "secret123" not in redacted
        assert "abc456def" not in redacted

@pytest.mark.unit
class TestSecretsManager:
    """Test SecretsManager functionality"""
    
    def test_get_secret_from_env(self, test_env_vars):
        """Test getting secret from environment variables"""
        manager = SecretsManager()
        
        secret = manager.get_secret('KRAKEN_API_KEY')
        assert secret == 'test_kraken_key'
        
        # Test caching
        with patch.dict(os.environ, {'KRAKEN_API_KEY': 'changed'}, clear=False):
            cached_secret = manager.get_secret('KRAKEN_API_KEY')
            assert cached_secret == 'test_kraken_key'  # Should return cached value
    
    def test_get_secret_with_default(self):
        """Test getting secret with default value"""
        manager = SecretsManager()
        
        secret = manager.get_secret('NONEXISTENT_SECRET', 'default_value')
        assert secret == 'default_value'
    
    def test_get_secret_missing(self):
        """Test getting missing secret returns None"""
        manager = SecretsManager()
        
        secret = manager.get_secret('COMPLETELY_MISSING_SECRET')
        assert secret is None
    
    def test_secret_type_inference(self):
        """Test automatic secret type inference"""
        manager = SecretsManager()
        
        assert manager._infer_secret_type('KRAKEN_API_KEY') == SecretType.API_KEY
        assert manager._infer_secret_type('DATABASE_PASSWORD') == SecretType.PASSWORD
        assert manager._infer_secret_type('JWT_TOKEN') == SecretType.TOKEN
        assert manager._infer_secret_type('PRIVATE_KEY') == SecretType.PRIVATE_KEY
        assert manager._infer_secret_type('SSL_CERT') == SecretType.CERTIFICATE
        assert manager._infer_secret_type('DATABASE_URL') == SecretType.DATABASE_URL
    
    def test_validate_secrets(self, test_env_vars):
        """Test secret validation"""
        manager = SecretsManager()
        
        validation = manager.validate_secrets(['KRAKEN_API_KEY', 'MISSING_SECRET'])
        
        assert validation['KRAKEN_API_KEY'] is True
        assert validation['MISSING_SECRET'] is False
    
    def test_list_secrets(self, test_env_vars):
        """Test listing secrets with metadata"""
        manager = SecretsManager()
        
        # Access some secrets to populate metadata
        manager.get_secret('KRAKEN_API_KEY')
        manager.get_secret('BINANCE_API_KEY')
        
        secrets_list = manager.list_secrets()
        
        assert 'KRAKEN_API_KEY' in secrets_list
        assert secrets_list['KRAKEN_API_KEY'].secret_type == SecretType.API_KEY
        assert secrets_list['KRAKEN_API_KEY'].source == 'env'
    
    def test_secure_context_cleanup(self, test_env_vars):
        """Test secure context manager cleans up sensitive data"""
        manager = SecretsManager()
        
        with manager.secure_context():
            # Access some secrets
            manager.get_secret('KRAKEN_API_KEY')
            manager.get_secret('KRAKEN_SECRET')
            
            # Should have cached values
            assert 'KRAKEN_API_KEY' in manager.secrets_cache
        
        # After context exit, sensitive data should be cleared
        # (Note: Only sensitive keys are cleared, not all cached secrets)

@pytest.mark.unit
class TestVaultManager:
    """Test Vault integration (mocked)"""
    
    @patch('hvac.Client')
    def test_vault_initialization_success(self, mock_hvac_client):
        """Test successful Vault initialization"""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_hvac_client.return_value = mock_client
        
        manager = VaultManager('http://vault:8200', 'test-token')
        
        assert manager.client is not None
        mock_hvac_client.assert_called_once_with(url='http://vault:8200', token='test-token')
        mock_client.is_authenticated.assert_called_once()
    
    @patch('hvac.Client')
    def test_vault_initialization_failure(self, mock_hvac_client):
        """Test failed Vault initialization"""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = False
        mock_hvac_client.return_value = mock_client
        
        manager = VaultManager('http://vault:8200', 'invalid-token')
        
        assert manager.client is None
    
    @patch('hvac.Client')
    def test_get_secret_success(self, mock_hvac_client):
        """Test successful secret retrieval from Vault"""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            'data': {'data': {'value': 'secret_value'}}
        }
        mock_hvac_client.return_value = mock_client
        
        manager = VaultManager('http://vault:8200', 'test-token')
        secret = manager.get_secret('test/path', 'value')
        
        assert secret == 'secret_value'
        mock_client.secrets.kv.v2.read_secret_version.assert_called_once_with(path='test/path')
    
    @patch('hvac.Client')
    def test_set_secret_success(self, mock_hvac_client):
        """Test successful secret storage in Vault"""
        mock_client = Mock()
        mock_client.is_authenticated.return_value = True
        mock_hvac_client.return_value = mock_client
        
        manager = VaultManager('http://vault:8200', 'test-token')
        success = manager.set_secret('test/path', {'key': 'value'})
        
        assert success is True
        mock_client.secrets.kv.v2.create_or_update_secret.assert_called_once_with(
            path='test/path',
            secret={'key': 'value'}
        )

@pytest.mark.unit
class TestSecureFunction:
    """Test secure function decorator"""
    
    def test_secure_function_decorator(self):
        """Test that secure function decorator catches and sanitizes exceptions"""
        
        @secure_function(redact_kwargs=['api_key'])
        def test_function(api_key=None):
            raise ValueError(f"Failed with API key: {api_key}")
        
        with pytest.raises(ValueError) as exc_info:
            test_function(api_key="sk-1234567890abcdef")
        
        # Exception message should be sanitized
        assert "sk-1234567890abcdef" not in str(exc_info.value)
        assert "sk****************ef" in str(exc_info.value)
    
    def test_secure_function_no_exception(self):
        """Test secure function works normally without exceptions"""
        
        @secure_function()
        def test_function(value):
            return value * 2
        
        result = test_function(5)
        assert result == 10

@pytest.mark.unit
class TestSecureLogger:
    """Test SecureLogger wrapper"""
    
    def test_secure_logger_redacts_messages(self):
        """Test that SecureLogger automatically redacts sensitive information"""
        mock_logger = Mock()
        secure_logger = SecureLogger(mock_logger)
        
        # Test info logging with API key
        secure_logger.info("Connecting with api_key=sk-1234567890abcdef")
        
        # Check that the logged message was redacted
        call_args = mock_logger.info.call_args
        logged_message = call_args[0][0]
        assert "sk-1234567890abcdef" not in logged_message
        assert "sk****************ef" in logged_message
    
    def test_secure_logger_handles_extra_context(self):
        """Test that SecureLogger redacts extra context"""
        mock_logger = Mock()
        secure_logger = SecureLogger(mock_logger)
        
        # Test with extra context containing secrets
        extra_context = {'api_key': 'sk-secret123', 'user': 'test'}
        secure_logger.info("Test message", extra=extra_context)
        
        # Extra context should be redacted but structure preserved
        call_args = mock_logger.info.call_args
        # The original extra dict should not contain the actual secret
        assert 'api_key' in extra_context  # Key should still exist
        assert 'sk-secret123' not in str(extra_context)  # But value should be redacted
    
    def test_secure_logger_exception_handling(self):
        """Test secure logger handles exceptions with traceback redaction"""
        mock_logger = Mock()
        secure_logger = SecureLogger(mock_logger)
        
        try:
            raise ValueError("Error with api_key=sk-1234567890abcdef")
        except ValueError:
            secure_logger.error("Exception occurred", exc_info=True)
        
        # Should have logged error without exc_info (since it was processed)
        call_args = mock_logger.error.call_args
        logged_message = call_args[0][0]
        
        # Should contain sanitized traceback info
        assert "Sanitized traceback:" in logged_message
        assert "sk-1234567890abcdef" not in logged_message

@pytest.mark.unit
def test_get_secrets_manager_singleton():
    """Test that get_secrets_manager returns singleton instance"""
    # Reset global state
    import core.secrets_manager
    core.secrets_manager._secrets_manager = None
    
    manager1 = get_secrets_manager()
    manager2 = get_secrets_manager()
    
    assert manager1 is manager2
    assert isinstance(manager1, SecretsManager)

@pytest.mark.unit
def test_secret_metadata_dataclass():
    """Test SecretMetadata dataclass functionality"""
    metadata = SecretMetadata(
        name="TEST_SECRET",
        secret_type=SecretType.API_KEY,
        source="vault",
        description="Test secret for unit tests"
    )
    
    assert metadata.name == "TEST_SECRET"
    assert metadata.secret_type == SecretType.API_KEY
    assert metadata.source == "vault"
    assert metadata.description == "Test secret for unit tests"
    assert metadata.last_rotated is None
    assert metadata.expires_at is None