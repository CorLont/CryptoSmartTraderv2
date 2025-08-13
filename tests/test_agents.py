# tests/test_agents.py
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from agents.ml_predictor_agent import MLPredictorAgent
from core.config_manager import ConfigManager
from core.health_monitor import HealthMonitor
from utils.cache_manager import CacheManager


@pytest.fixture
def mock_config_manager():
    """Mock configuration manager for tests"""
    config = Mock(spec=ConfigManager)
    config.get.return_value = "test_value"
    config.get_agent_config.return_value = {"enabled": True}
    return config


@pytest.fixture
def mock_health_monitor():
    """Mock health monitor for tests"""
    monitor = Mock(spec=HealthMonitor)
    monitor.record_agent_status.return_value = None
    return monitor


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for tests"""
    cache = Mock(spec=CacheManager)
    cache.get.return_value = None
    cache.set.return_value = True
    return cache


class TestSentimentAgent:
    """Test cases for SentimentAgent"""

    @pytest.fixture
    def sentiment_agent(self, mock_config_manager, mock_health_monitor, mock_cache_manager):
        return SentimentAgent(
            config_manager=mock_config_manager,
            health_monitor=mock_health_monitor,
            cache_manager=mock_cache_manager,
        )

    def test_agent_initialization(self, sentiment_agent):
        """Test agent initializes properly"""
        assert sentiment_agent.agent_name == "SentimentAgent"
        assert sentiment_agent.status == "initialized"

    @pytest.mark.asyncio
    async def test_sentiment_analysis_basic(self, sentiment_agent):
        """Test basic sentiment analysis"""
        with patch.object(sentiment_agent, "_analyze_text_sentiment") as mock_analyze:
            mock_analyze.return_value = {"score": 0.8, "confidence": 0.9}

            result = await sentiment_agent.analyze_sentiment("Bitcoin is looking bullish!")

            assert result["score"] == 0.8
            assert result["confidence"] == 0.9
            mock_analyze.assert_called_once()

    def test_sentiment_score_validation(self, sentiment_agent):
        """Test sentiment score validation"""
        # Valid scores
        assert sentiment_agent._validate_sentiment_score(0.5) is True
        assert sentiment_agent._validate_sentiment_score(0.0) is True
        assert sentiment_agent._validate_sentiment_score(1.0) is True

        # Invalid scores
        assert sentiment_agent._validate_sentiment_score(-0.1) is False
        assert sentiment_agent._validate_sentiment_score(1.1) is False
        assert sentiment_agent._validate_sentiment_score(None) is False


class TestTechnicalAgent:
    """Test cases for TechnicalAgent"""

    @pytest.fixture
    def technical_agent(self, mock_config_manager, mock_health_monitor, mock_cache_manager):
        mock_data_manager = Mock()
        return TechnicalAgent(
            config_manager=mock_config_manager,
            data_manager=mock_data_manager,
            health_monitor=mock_health_monitor,
            cache_manager=mock_cache_manager,
        )

    def test_technical_analysis_initialization(self, technical_agent):
        """Test technical analysis agent initialization"""
        assert technical_agent.agent_name == "TechnicalAgent"
        assert hasattr(technical_agent, "indicators")

    @pytest.mark.asyncio
    async def test_calculate_indicators(self, technical_agent):
        """Test technical indicator calculation"""
        # Mock market data
        mock_data = {
            "close": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }

        with patch.object(technical_agent, "_fetch_market_data") as mock_fetch:
            mock_fetch.return_value = mock_data

            result = await technical_agent.analyze_symbol("BTC")

            assert "indicators" in result
            assert "signals" in result

    def test_signal_generation(self, technical_agent):
        """Test trading signal generation"""
        indicators = {
            "rsi": 30,  # Oversold
            "macd": {"signal": "buy"},
            "bb_position": 0.1,  # Near lower band
        }

        signals = technical_agent._generate_signals(indicators)

        assert isinstance(signals, list)
        assert any(signal["action"] == "buy" for signal in signals)


class TestMLPredictorAgent:
    """Test cases for MLPredictorAgent"""

    @pytest.fixture
    def ml_agent(self, mock_config_manager, mock_health_monitor, mock_cache_manager):
        mock_data_manager = Mock()
        mock_ml_manager = Mock()
        return MLPredictorAgent(
            config_manager=mock_config_manager,
            data_manager=mock_data_manager,
            health_monitor=mock_health_monitor,
            cache_manager=mock_cache_manager,
            ml_model_manager=mock_ml_manager,
        )

    def test_ml_agent_initialization(self, ml_agent):
        """Test ML agent initialization"""
        assert ml_agent.agent_name == "MLPredictorAgent"
        assert hasattr(ml_agent, "models")

    @pytest.mark.asyncio
    async def test_price_prediction(self, ml_agent):
        """Test price prediction functionality"""
        with patch.object(ml_agent.ml_model_manager, "predict") as mock_predict:
            mock_predict.return_value = {
                "1h": {"price": 105.0, "confidence": 0.85},
                "24h": {"price": 110.0, "confidence": 0.75},
            }

            result = await ml_agent.predict_price("BTC", ["1h", "24h"])

            assert "1h" in result
            assert "24h" in result
            assert result["1h"]["price"] == 105.0

    def test_feature_preparation(self, ml_agent):
        """Test feature preparation for ML models"""
        raw_data = {
            "prices": [100, 101, 102, 103, 104],
            "volumes": [1000, 1100, 1200, 1300, 1400],
            "indicators": {"rsi": 65, "macd": 0.5},
        }

        features = ml_agent._prepare_features(raw_data)

        assert isinstance(features, dict)
        assert "price_features" in features
        assert "volume_features" in features


# Integration tests
@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for agent coordination"""

    @pytest.mark.asyncio
    async def test_multi_agent_analysis(self):
        """Test multiple agents working together"""
        # This would test the full agent coordination
        # In a real scenario, you'd test with actual agent instances
        pass

    @pytest.mark.asyncio
    async def test_agent_health_monitoring(self):
        """Test agent health monitoring integration"""
        # Test that agents properly report health status
        pass


# Utility functions for tests
class MockResponse:
    """Mock HTTP response for testing"""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    async def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error {self.status_code}")


@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        "BTC": {
            "price": 50000,
            "volume": 1000000,
            "change_24h": 0.05,
            "high_24h": 51000,
            "low_24h": 49000,
        },
        "ETH": {
            "price": 3000,
            "volume": 500000,
            "change_24h": 0.03,
            "high_24h": 3100,
            "low_24h": 2950,
        },
    }
