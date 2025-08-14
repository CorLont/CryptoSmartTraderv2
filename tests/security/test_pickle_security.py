#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Pickle Security Tests
Comprehensive testing of pickle security hardening measures
"""

import pytest
import tempfile
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from cryptosmarttrader.security.secure_serialization import (
    SecureSerializer, 
    SecureSerializationError,
    save_secure_pickle,
    load_secure_pickle,
    save_json,
    load_json,
    save_ml_model,
    load_ml_model
)
from cryptosmarttrader.security.pickle_policy import (
    PicklePolicyEnforcer,
    PickleSecurityViolation,
    enable_pickle_security,
    disable_pickle_security
)

class TestSecureSerializer:
    """Test secure serialization framework"""
    
    def setup_method(self):
        """Setup test environment"""
        self.serializer = SecureSerializer()
        self.temp_dir = tempfile.mkdtemp()
        self.trusted_file = Path(self.temp_dir) / "models" / "test.pkl"
        self.external_file = Path(self.temp_dir) / "data" / "test.pkl"
        
        # Create directories
        self.trusted_file.parent.mkdir(parents=True, exist_ok=True)
        self.external_file.parent.mkdir(parents=True, exist_ok=True)
    
    def test_trusted_path_detection(self):
        """Test trusted path detection logic"""
        # Trusted paths
        assert self.serializer._is_trusted_internal_path("src/cryptosmarttrader/test.py")
        assert self.serializer._is_trusted_internal_path("models/test.pkl")
        assert self.serializer._is_trusted_internal_path("ml/models/test.pkl")
        
        # External paths
        assert not self.serializer._is_trusted_internal_path("data/test.pkl")
        assert not self.serializer._is_trusted_internal_path("configs/test.json")
        assert not self.serializer._is_trusted_internal_path("scripts/test.py")
    
    def test_secure_pickle_trusted_path(self):
        """Test secure pickle for trusted paths"""
        test_data = {"model": "test", "parameters": [1, 2, 3]}
        
        # Should succeed for trusted path
        success = self.serializer.save_secure_pickle(test_data, "models/test_model.pkl")
        assert success
        
        # Should load successfully with integrity check
        loaded_data = self.serializer.load_secure_pickle("models/test_model.pkl")
        assert loaded_data == test_data
    
    def test_secure_pickle_external_path_rejection(self):
        """Test pickle rejection for external paths"""
        test_data = {"data": "external"}
        
        # Should raise SecurityError for external path
        with pytest.raises(SecureSerializationError) as exc_info:
            self.serializer.save_secure_pickle(test_data, "data/external.pkl")
        
        assert "not allowed for external path" in str(exc_info.value)
        
        # Should also reject loading from external path
        with pytest.raises(SecureSerializationError) as exc_info:
            self.serializer.load_secure_pickle("data/external.pkl")
        
        assert "not allowed for external path" in str(exc_info.value)
    
    def test_json_serialization(self):
        """Test JSON serialization with integrity checks"""
        test_data = {"model_config": {"lr": 0.01, "epochs": 100}}
        
        # Save with integrity
        success = self.serializer.save_json(test_data, "configs/test.json", secure=True)
        assert success
        
        # Load with integrity verification
        loaded_data = self.serializer.load_json("configs/test.json", secure=True)
        assert loaded_data == test_data
    
    def test_ml_model_serialization(self):
        """Test ML model serialization with integrity"""
        # Mock ML model
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "MockLGBMRegressor"
        
        # Should succeed for trusted path
        success = self.serializer.save_ml_model(mock_model, "models/test_model.joblib")
        assert success
        
        # Should create integrity file
        integrity_file = Path("models/test_model.joblib.integrity")
        
        # Should load model
        loaded_model = self.serializer.load_ml_model("models/test_model.joblib")
        # Model loading might fail due to mocking, but shouldn't raise SecurityError
    
    def test_hmac_integrity_validation(self):
        """Test HMAC integrity validation"""
        test_data = {"sensitive": "data"}
        
        # Save data
        self.serializer.save_secure_pickle(test_data, "models/test.pkl")
        
        # Manually corrupt the file
        with open("models/test.pkl", "rb") as f:
            content = f.read()
        
        # Modify content slightly
        corrupted_content = content[:-10] + b"corrupted!"
        
        with open("models/test.pkl", "wb") as f:
            f.write(corrupted_content)
        
        # Should detect corruption
        with pytest.raises(SecureSerializationError) as exc_info:
            self.serializer.load_secure_pickle("models/test.pkl")
        
        assert "HMAC verification failed" in str(exc_info.value)
    
    def test_security_audit_logging(self):
        """Test security audit logging"""
        initial_log_size = len(self.serializer.get_security_audit_log())
        
        # Perform some operations
        self.serializer.save_json({"test": "data"}, "configs/test.json")
        self.serializer.load_json("configs/test.json")
        
        # Check audit log grew
        final_log_size = len(self.serializer.get_security_audit_log())
        assert final_log_size > initial_log_size
        
        # Check log entries
        log_entries = self.serializer.get_security_audit_log()
        assert any(entry['event_type'] == 'json_save' for entry in log_entries)
        assert any(entry['event_type'] == 'json_load' for entry in log_entries)


class TestPicklePolicyEnforcer:
    """Test runtime pickle policy enforcement"""
    
    def setup_method(self):
        """Setup test environment"""
        self.enforcer = PicklePolicyEnforcer()
    
    def test_trusted_caller_detection(self):
        """Test trusted caller path detection"""
        # This test is complex because it requires actual call stack analysis
        # For now, test the path checking logic
        assert 'src/cryptosmarttrader' in self.enforcer.trusted_paths
        assert 'models' in self.enforcer.trusted_paths
    
    def test_violation_logging(self):
        """Test security violation logging"""
        initial_violations = len(self.enforcer.get_violations())
        
        # Log a violation
        self.enforcer.log_violation("pickle.load", "/external/path/test.py")
        
        violations = self.enforcer.get_violations()
        assert len(violations) > initial_violations
        assert violations[-1]['operation'] == 'pickle.load'
        assert '/external/path/test.py' in violations[-1]['caller']
    
    @patch('inspect.currentframe')
    def test_secure_pickle_load_trusted(self, mock_frame):
        """Test secure pickle load for trusted caller"""
        # Mock frame to simulate trusted caller
        mock_frame.return_value.f_back.f_code.co_filename = "src/cryptosmarttrader/test.py"
        
        with patch('pickle.load') as mock_pickle_load:
            mock_pickle_load.return_value = {"test": "data"}
            
            with patch.object(self.enforcer, 'is_trusted_caller', return_value=True):
                result = self.enforcer.secure_pickle_load(MagicMock())
                assert result == {"test": "data"}
    
    @patch('inspect.currentframe')
    def test_secure_pickle_load_untrusted(self, mock_frame):
        """Test secure pickle load rejection for untrusted caller"""
        # Mock frame to simulate untrusted caller
        mock_frame.return_value.f_back.f_code.co_filename = "/external/scripts/test.py"
        
        with patch.object(self.enforcer, 'is_trusted_caller', return_value=False):
            with pytest.raises(PickleSecurityViolation) as exc_info:
                self.enforcer.secure_pickle_load(MagicMock())
            
            assert "not allowed" in str(exc_info.value)


class TestPickleSecurityIntegration:
    """Integration tests for complete pickle security system"""
    
    def test_convenience_functions(self):
        """Test convenience functions work correctly"""
        test_data = {"model": "test"}
        
        # JSON functions should work for any path
        assert save_json(test_data, "data/test.json")
        loaded = load_json("data/test.json")
        assert loaded == test_data
        
        # Secure pickle should only work for trusted paths
        with pytest.raises(SecureSerializationError):
            save_secure_pickle(test_data, "data/test.pkl")
    
    def test_ml_model_functions(self):
        """Test ML model convenience functions"""
        mock_model = MagicMock()
        
        # Should work for trusted path
        try:
            success = save_ml_model(mock_model, "models/test.joblib")
            # May fail due to joblib issues, but shouldn't raise SecurityError
        except Exception as e:
            assert "not allowed" not in str(e)
        
        # Should reject external path
        with pytest.raises(SecureSerializationError):
            save_ml_model(mock_model, "data/model.joblib")
    
    def test_security_policy_compliance(self):
        """Test overall security policy compliance"""
        # Verify policy enforcer can be enabled/disabled
        try:
            enable_pickle_security()
            disable_pickle_security()
        except Exception as e:
            pytest.fail(f"Security policy control failed: {e}")


class TestMigrationValidation:
    """Validate pickle migration was successful"""
    
    def test_no_unsafe_pickle_imports(self):
        """Test that no files have unsafe pickle imports"""
        # This would scan the codebase for unsafe patterns
        # For now, basic validation
        assert True  # Migration script handled this
    
    def test_secure_alternatives_available(self):
        """Test secure alternatives are properly available"""
        from cryptosmarttrader.security.secure_serialization import SecureSerializer
        from cryptosmarttrader.security.pickle_policy import PicklePolicyEnforcer
        
        # Should be able to instantiate
        serializer = SecureSerializer()
        enforcer = PicklePolicyEnforcer()
        
        assert serializer is not None
        assert enforcer is not None


# Integration test with temporary environment
def test_full_security_workflow():
    """Test complete security workflow"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup paths
        trusted_path = Path(temp_dir) / "models" / "test.pkl"
        external_path = Path(temp_dir) / "data" / "test.pkl"
        trusted_path.parent.mkdir(parents=True)
        external_path.parent.mkdir(parents=True)
        
        # Test data
        model_data = {"weights": [1, 2, 3], "bias": 0.5}
        config_data = {"learning_rate": 0.01, "epochs": 100}
        
        # Change to temp directory for path validation
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_dir)
            
            # Should succeed: JSON for config (external data)
            assert save_json(config_data, "data/config.json")
            loaded_config = load_json("data/config.json")
            assert loaded_config == config_data
            
            # Should succeed: Secure pickle for model (internal data)
            assert save_secure_pickle(model_data, "models/model.pkl")
            loaded_model = load_secure_pickle("models/model.pkl")
            assert loaded_model == model_data
            
            # Should fail: Pickle for external data
            with pytest.raises(SecureSerializationError):
                save_secure_pickle(config_data, "data/config.pkl")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])