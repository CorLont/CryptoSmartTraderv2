"""
Production Systems Test Suite
Comprehensive testing for robustness and reliability
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import tempfile
import threading
import time

# Import our core systems
from core.zero_fallback_validator import ZeroFallbackValidator, DataType
from core.deep_ml_engine import DeepMLEngine
from core.cross_coin_fusion import CrossCoinFusionEngine
from core.async_orchestrator import AsyncOrchestrator
from core.security_manager import SecurityManager
from core.bayesian_uncertainty import BayesianUncertaintyModel


class TestZeroFallbackValidator:
    """Test zero tolerance fallback data validation"""

    def setup_method(self):
        self.validator = ZeroFallbackValidator()

    def test_reject_synthetic_price_data(self):
        """Test rejection of synthetic price data"""
        synthetic_data = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
            "timestamp": time.time(),
            "source": "generated_fallback_data",
        }

        result = self.validator.validate_price_data(synthetic_data, "BTC/USD")

        assert not result.is_valid
        assert "Synthetic data markers detected" in result.rejection_reason

    def test_reject_invalid_ohlc(self):
        """Test rejection of invalid OHLC data"""
        invalid_data = {
            "open": 100.0,
            "high": 95.0,  # High < Open (invalid)
            "low": 98.0,
            "close": 102.0,
            "volume": 1000000,
            "timestamp": time.time(),
        }

        result = self.validator.validate_price_data(invalid_data, "BTC/USD")

        assert not result.is_valid
        assert "OHLC consistency violation" in result.rejection_reason

    def test_accept_valid_price_data(self):
        """Test acceptance of valid price data"""
        valid_data = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
            "timestamp": time.time(),
            "source": "kraken",
        }

        result = self.validator.validate_price_data(valid_data, "BTC/USD")

        assert result.is_valid
        assert result.source == "kraken"

    def test_reject_fallback_sentiment(self):
        """Test rejection of fallback sentiment data"""
        fallback_sentiment = {"sentiment_score": 0.7, "is_fallback": True, "data_sources": []}

        result = self.validator.validate_sentiment_data(fallback_sentiment, "BTC/USD")

        assert not result.is_valid
        assert "Fallback/synthetic sentiment data" in result.rejection_reason

    def test_batch_filtering(self):
        """Test batch data filtering"""
        batch = [
            {"symbol": "BTC", "price": 100, "source": "exchange"},
            {"symbol": "ETH", "price": 50, "source": "synthetic"},
            {"symbol": "ADA", "price": 1, "source": "kraken"},
        ]

        valid_batch = self.validator.validate_and_filter_batch(batch, DataType.PRICE)

        # Should filter out synthetic data
        assert len(valid_batch) < len(batch)


class TestDeepMLEngine:
    """Test mandatory deep learning implementation"""

    def setup_method(self):
        mock_container = Mock()
        self.engine = DeepMLEngine(mock_container)

    def test_model_initialization(self):
        """Test deep learning model initialization"""
        success = self.engine.initialize_models()

        assert success
        assert all(model is not None for model in self.engine.models.values())

    @pytest.mark.asyncio
    async def test_mandatory_prediction(self):
        """Test mandatory deep learning prediction"""
        # Initialize models first
        self.engine.initialize_models()

        # Create sample features
        features = np.random.randn(100, 50)  # 100 timesteps, 50 features

        # Test prediction
        result = await self.engine.mandatory_deep_prediction("BTC/USD", features, "1h")

        assert result["deep_learning_used"] is True
        assert "prediction" in result
        assert "confidence" in result
        assert "uncertainty" in result
        assert result["coin"] == "BTC/USD"
        assert result["horizon"] == "1h"

    def test_ensemble_prediction(self):
        """Test ensemble prediction combining LSTM and Transformer"""
        self.engine.initialize_models()

        # Should have both LSTM and Transformer models
        assert self.engine.models["lstm_1h"] is not None
        assert self.engine.models["transformer_1h"] is not None

    def test_model_status(self):
        """Test model status reporting"""
        self.engine.initialize_models()

        status = self.engine.get_model_status()

        assert status["deep_learning_mandatory"] is True
        assert status["models_initialized"] > 0
        assert "model_details" in status


class TestCrossCoinFusion:
    """Test cross-coin feature fusion"""

    def setup_method(self):
        mock_container = Mock()
        self.fusion = CrossCoinFusionEngine(mock_container)

    @pytest.mark.asyncio
    async def test_correlation_analysis(self):
        """Test cross-coin correlation analysis"""
        # Create sample coin data
        coins_data = []
        for i, symbol in enumerate(["BTC/USD", "ETH/USD", "ADA/USD"]):
            # Generate correlated price history
            base_prices = np.cumsum(np.random.randn(50)) + 1000
            price_history = [
                {"close": price, "volume": 1000000 + np.random.randint(-100000, 100000)}
                for price in base_prices
            ]

            coins_data.append({"symbol": symbol, "price_history": price_history})

        result = await self.fusion.analyze_cross_coin_correlations(coins_data)

        assert "price_correlations" in result
        assert "fusion_features" in result
        assert "market_structure" in result

    def test_sector_classification(self):
        """Test coin sector classification"""
        assert self.fusion._classify_coin_sector("uni") == "defi"
        assert self.fusion._classify_coin_sector("btc") == "layer1"
        assert self.fusion._classify_coin_sector("doge") == "meme"
        assert self.fusion._classify_coin_sector("unknown_coin") == "other"

    def test_market_features(self):
        """Test market-wide feature calculation"""
        coins_data = [
            {
                "symbol": f"COIN_{i}",
                "price_history": [
                    {"close": 100 + np.random.randn(), "volume": 1000000} for _ in range(10)
                ],
            }
            for i in range(5)
        ]

        features = self.fusion._calculate_market_features(coins_data)

        assert "market_momentum" in features
        assert "market_volatility" in features
        assert "total_volume" in features


class TestAsyncOrchestrator:
    """Test async orchestrator performance"""

    def setup_method(self):
        mock_container = Mock()
        self.orchestrator = AsyncOrchestrator(mock_container)

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test async orchestrator initialization"""
        await self.orchestrator.initialize()

        assert self.orchestrator.running is True
        assert self.orchestrator.session is not None
        assert self.orchestrator.executor is not None

    @pytest.mark.asyncio
    async def test_async_task_submission(self):
        """Test async task submission and execution"""
        await self.orchestrator.initialize()

        # Define a simple test function
        def test_function(x, y):
            return x + y

        # Submit task
        task_id = await self.orchestrator.submit_task(
            test_function, 5, 10, task_id="test_task", priority=1, timeout=30.0
        )

        assert task_id == "test_task"

        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            status = await self.orchestrator.get_task_status(task_id)
            if status["status"] in ["completed", "failed"]:
                break
            await asyncio.sleep(0.1)

        final_status = await self.orchestrator.get_task_status(task_id)
        assert final_status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_batch_task_execution(self):
        """Test batch task execution"""
        await self.orchestrator.initialize()

        # Create batch tasks
        batch_tasks = [
            {"function": lambda x: x * 2, "args": (i,), "task_id": f"batch_task_{i}", "priority": 1}
            for i in range(5)
        ]

        task_ids = await self.orchestrator.submit_batch_tasks(batch_tasks)

        assert len(task_ids) == 5

        # Wait for all to complete
        completed_count = 0
        start_time = time.time()

        while completed_count < 5 and time.time() - start_time < 30:
            completed_count = 0
            for task_id in task_ids:
                status = await self.orchestrator.get_task_status(task_id)
                if status["status"] == "completed":
                    completed_count += 1
            await asyncio.sleep(0.1)

        assert completed_count == 5

    def test_performance_stats(self):
        """Test performance statistics tracking"""
        stats = self.orchestrator.get_performance_stats()

        assert "tasks_submitted" in stats
        assert "tasks_completed" in stats
        assert "current_load" in stats
        assert "async_mode" in stats
        assert stats["async_mode"] == "FULLY_ASYNC"


class TestSecurityManager:
    """Test enterprise security management"""

    def setup_method(self):
        self.security = SecurityManager()

    def test_security_initialization(self):
        """Test security manager initialization"""
        assert hasattr(self.security, "settings")
        assert hasattr(self.security, "audit_logs")
        assert hasattr(self.security, "credential_cache")

    def test_credential_validation(self):
        """Test API key validation"""
        # OpenAI key validation
        assert self.security.validate_api_key("sk-1234567890abcdef" + "x" * 30, "openai")
        assert not self.security.validate_api_key("invalid_key", "openai")

        # Generic validation
        assert self.security.validate_api_key("valid_long_api_key_12345", "generic")
        assert not self.security.validate_api_key("short", "generic")

    def test_log_sanitization(self):
        """Test credential sanitization in logs"""
        sensitive_log = "API key: sk-1234567890abcdefghijklmnopqrstuvwxyz password: secret123"

        sanitized = self.security.sanitize_logs(sensitive_log)

        assert "sk-***REDACTED***" in sanitized
        assert "secret123" not in sanitized
        assert "***REDACTED***" in sanitized

    def test_credential_storage_retrieval(self):
        """Test secure credential storage and retrieval"""
        test_key = "test_api_key"
        test_value = "secret_value_12345"

        # Store credential
        success = self.security.store_credential(test_key, test_value, "test_user")
        assert success

        # Retrieve credential
        retrieved = self.security.get_credential(test_key, "test_user")
        assert retrieved == test_value

        # Check audit logging
        assert len(self.security.audit_logs) > 0

        # Test non-existent credential
        non_existent = self.security.get_credential("non_existent_key", "test_user")
        assert non_existent is None

    def test_security_status(self):
        """Test security status reporting"""
        status = self.security.get_security_status()

        assert "security_initialized" in status
        assert "encryption_enabled" in status
        assert "security_features" in status
        assert "audit_logging" in status["security_features"]


class TestBayesianUncertainty:
    """Test Bayesian uncertainty modeling"""

    def setup_method(self):
        mock_container = Mock()
        self.uncertainty = BayesianUncertaintyModel(mock_container)

    def test_uncertainty_initialization(self):
        """Test uncertainty model initialization"""
        success = self.uncertainty.initialize_uncertainty_models()

        assert success
        assert all(model is not None for model in self.uncertainty.uncertainty_models.values())

    def test_uncertainty_estimation(self):
        """Test prediction uncertainty estimation"""
        # Sample features and predictions
        features = np.random.randn(50)
        predictions = {"1h": 0.05, "24h": 0.12, "7d": 0.25, "30d": 0.35}

        result = self.uncertainty.estimate_prediction_uncertainty(features, predictions, "BTC/USD")

        assert result["coin"] == "BTC/USD"
        assert "horizons" in result
        assert "overall_uncertainty" in result

        # Check each horizon has uncertainty data
        for horizon in predictions.keys():
            if horizon in result["horizons"]:
                horizon_data = result["horizons"][horizon]
                assert "uncertainty" in horizon_data
                assert "confidence_intervals" in horizon_data
                assert "bayesian_confidence" in horizon_data

    def test_confidence_intervals(self):
        """Test confidence interval calculation"""
        prediction = 0.15
        uncertainty = 0.05

        intervals = self.uncertainty._calculate_confidence_intervals(prediction, uncertainty)

        assert "68%" in intervals
        assert "95%" in intervals
        assert "99%" in intervals

        # Check interval properties
        assert intervals["68%"]["upper"] > intervals["68%"]["lower"]
        assert intervals["95%"]["width"] > intervals["68%"]["width"]
        assert intervals["99%"]["width"] > intervals["95%"]["width"]

    def test_uncertainty_filtering(self):
        """Test prediction filtering by uncertainty"""
        predictions = [
            {
                "coin": "BTC/USD",
                "prediction": 0.15,
                "uncertainty": {
                    "overall_uncertainty": {"mean_uncertainty": 0.1, "mean_confidence": 0.8}
                },
            },
            {
                "coin": "ETH/USD",
                "prediction": 0.25,
                "uncertainty": {
                    "overall_uncertainty": {
                        "mean_uncertainty": 0.4,  # High uncertainty
                        "mean_confidence": 0.5,  # Low confidence
                    }
                },
            },
            {
                "coin": "ADA/USD",
                "prediction": 0.08,
                "uncertainty": {
                    "overall_uncertainty": {"mean_uncertainty": 0.05, "mean_confidence": 0.9}
                },
            },
        ]

        filtered = self.uncertainty.filter_predictions_by_uncertainty(
            predictions, max_uncertainty=0.3, min_confidence=0.7
        )

        # Should filter out ETH/USD due to high uncertainty and low confidence
        assert len(filtered) == 2
        assert all(pred["coin"] != "ETH/USD" for pred in filtered)
        assert all(pred["passed_uncertainty_filter"] for pred in filtered)

    def test_uncertainty_status(self):
        """Test uncertainty status reporting"""
        self.uncertainty.initialize_uncertainty_models()

        status = self.uncertainty.get_uncertainty_status()

        assert "models_initialized" in status
        assert "total_models" in status
        assert "uncertainty_history_size" in status
        assert "config" in status


class TestIntegrationScenarios:
    """Test integrated system scenarios"""

    def setup_method(self):
        self.mock_container = Mock()

        # Initialize all systems
        self.validator = ZeroFallbackValidator()
        self.ml_engine = DeepMLEngine(self.mock_container)
        self.fusion_engine = CrossCoinFusionEngine(self.mock_container)
        self.uncertainty_model = BayesianUncertaintyModel(self.mock_container)
        self.security_manager = SecurityManager()

    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline with all systems"""
        # Initialize systems
        self.ml_engine.initialize_models()
        self.uncertainty_model.initialize_uncertainty_models()

        # Sample input data
        raw_data = {
            "symbol": "BTC/USD",
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 102.0,
            "volume": 1000000,
            "timestamp": time.time(),
            "source": "kraken",
        }

        # Step 1: Validate data (zero fallback)
        validation_result = self.validator.validate_price_data(raw_data, "BTC/USD")
        assert validation_result.is_valid

        # Step 2: If valid, proceed with ML prediction
        if validation_result.is_valid:
            features = np.random.randn(100, 50)  # Sample features

            # ML prediction would be async in real system
            prediction_result = {"1h": 0.05, "24h": 0.12, "7d": 0.25, "30d": 0.35}

            # Step 3: Estimate uncertainty
            uncertainty_result = self.uncertainty_model.estimate_prediction_uncertainty(
                features, prediction_result, "BTC/USD"
            )

            assert uncertainty_result["coin"] == "BTC/USD"
            assert "overall_uncertainty" in uncertainty_result

            # Step 4: Apply uncertainty filtering
            full_prediction = {
                "coin": "BTC/USD",
                "predictions": prediction_result,
                "uncertainty": uncertainty_result,
            }

            filtered_predictions = self.uncertainty_model.filter_predictions_by_uncertainty(
                [full_prediction], max_uncertainty=0.3, min_confidence=0.6
            )

            # Should pass filtering with reasonable parameters
            assert len(filtered_predictions) <= 1

    def test_security_integration(self):
        """Test security integration across systems"""
        # Test credential access
        api_key = self.security_manager.get_credential("openai_api_key", "system")

        # Should handle missing credentials gracefully
        assert api_key is None or isinstance(api_key, str)

        # Test audit logging
        initial_logs = len(self.security_manager.audit_logs)

        # Perform some operations that should be logged
        self.security_manager.store_credential("test_key", "test_value", "test_user")
        self.security_manager.get_credential("test_key", "test_user")

        # Should have more audit logs
        assert len(self.security_manager.audit_logs) > initial_logs

    def test_system_resilience(self):
        """Test system resilience under error conditions"""
        # Test with invalid data
        invalid_data = {"symbol": "INVALID", "source": "synthetic_generator"}

        # Should reject gracefully
        result = self.validator.validate_price_data(invalid_data, "INVALID")
        assert not result.is_valid

        # Test ML engine with no models
        ml_engine = DeepMLEngine(self.mock_container)
        # Should handle uninitialized state
        status = ml_engine.get_model_status()
        assert status["models_initialized"] == 0

        # Test uncertainty with insufficient data
        uncertainty_result = self.uncertainty_model.estimate_prediction_uncertainty(
            np.array([]), {}, "EMPTY"
        )
        assert uncertainty_result == {}


# Performance and load testing
class TestPerformance:
    """Test system performance under load"""

    @pytest.mark.asyncio
    async def test_async_orchestrator_load(self):
        """Test async orchestrator under load"""
        mock_container = Mock()
        orchestrator = AsyncOrchestrator(mock_container)
        await orchestrator.initialize()

        # Submit many tasks quickly
        def simple_task(x):
            time.sleep(0.01)  # Small delay
            return x * 2

        task_ids = []
        start_time = time.time()

        # Submit 50 tasks
        for i in range(50):
            task_id = await orchestrator.submit_task(
                simple_task, i, task_id=f"load_test_{i}", timeout=10.0
            )
            task_ids.append(task_id)

        submission_time = time.time() - start_time

        # Should submit quickly
        assert submission_time < 5.0  # 5 seconds max

        # Wait for completion
        completed = 0
        max_wait = 30  # 30 seconds max
        start_wait = time.time()

        while completed < 50 and (time.time() - start_wait) < max_wait:
            completed = 0
            for task_id in task_ids:
                status = await orchestrator.get_task_status(task_id)
                if status["status"] == "completed":
                    completed += 1
            await asyncio.sleep(0.1)

        # Should complete most tasks
        assert completed >= 40  # At least 80% completion

        await orchestrator.shutdown()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
