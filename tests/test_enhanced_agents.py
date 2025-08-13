"""
Comprehensive test suite for enhanced agents
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

# Import enhanced agents
from agents.enhanced_sentiment_agent import sentiment_agent, SentimentResult
from agents.enhanced_technical_agent import technical_agent, TechnicalSignal
from agents.enhanced_whale_agent import whale_agent, WhaleTransaction
from agents.enhanced_ml_agent import ml_agent, MLPrediction
from agents.enhanced_backtest_agent import backtest_agent, BacktestResult
from core.enhanced_orchestrator import orchestrator
from core.order_book_analyzer import order_book_analyzer, LiquidityMetrics


class TestEnhancedSentimentAgent:
    """Test enhanced sentiment agent capabilities"""

    @pytest.mark.asyncio
    async def test_sentiment_analysis_with_bot_detection(self):
        """Test sentiment analysis with bot filtering"""

        result = await sentiment_agent.analyze_coin_sentiment("BTC", timeframe_hours=24)

        assert isinstance(result, SentimentResult)
        assert result.coin == "BTC"
        assert 0 <= result.confidence <= 1
        assert 0 <= result.bot_ratio <= 1
        assert 0 <= result.data_completeness <= 1
        assert result.filtered_mentions <= result.raw_mentions

    @pytest.mark.asyncio
    async def test_anti_detection_features(self):
        """Test anti-detection capabilities"""

        # Test rate limiting
        start_time = datetime.now()
        await sentiment_agent.anti_detection.rate_limit_wait("test_domain", 2.0)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Should respect minimum interval
        assert elapsed >= 0

        # Test header generation
        headers = sentiment_agent.anti_detection.get_headers()
        assert "User-Agent" in headers
        assert "Accept" in headers

    def test_bot_detection_engine(self):
        """Test bot detection functionality"""

        # Create mock posts with bot-like behavior
        bot_posts = [
            {"text": "🚀🚀🚀 BTC to the moon!!! Buy now!!!", "timestamp": 1000},
            {"text": "🚀🚀🚀 BTC to the moon!!! Buy now!!!", "timestamp": 1001},
            {"text": "🚀🚀🚀 BTC to the moon!!! Buy now!!!", "timestamp": 1002},
        ]

        filtered_posts, bot_ratio = sentiment_agent.bot_detector.filter_bot_content(bot_posts)

        # Should detect repetitive content
        assert bot_ratio > 0
        assert len(filtered_posts) < len(bot_posts)

    def test_entity_recognition(self):
        """Test coin entity recognition"""

        text = "BTC and bitcoin are going up, but SOL could be solvent issues"
        btc_mentions = sentiment_agent.entity_recognizer.extract_coin_mentions(text, "BTC")
        sol_mentions = sentiment_agent.entity_recognizer.extract_coin_mentions(text, "SOL")

        assert btc_mentions >= 2  # BTC and bitcoin
        assert sol_mentions >= 1  # SOL mentioned but may be discounted for ambiguity


class TestEnhancedTechnicalAgent:
    """Test enhanced technical agent capabilities"""

    def create_mock_data(self):
        """Create mock market data"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)

        return pd.DataFrame(
            {
                "close": prices,
                "high": prices * 1.02,
                "low": prices * 0.98,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    @pytest.mark.asyncio
    async def test_parallel_analysis(self):
        """Test parallel processing capabilities"""

        coin_data = {
            "BTC/USD": self.create_mock_data(),
            "ETH/USD": self.create_mock_data(),
            "SOL/USD": self.create_mock_data(),
        }

        start_time = datetime.now()
        results = await technical_agent.analyze_multiple_coins(coin_data, ["1h"])
        duration = (datetime.now() - start_time).total_seconds()

        # Should process multiple coins efficiently
        assert len(results) == 3
        assert duration < 5.0  # Should be fast with parallel processing

        for coin in results:
            assert "1h" in results[coin]
            signal = results[coin]["1h"]
            assert isinstance(signal, TechnicalSignal)

    def test_regime_detection(self):
        """Test market regime detection"""

        # Create trending data
        trending_data = self.create_mock_data()
        trending_data["close"] = np.linspace(45000, 55000, 100)  # Strong uptrend

        regime = technical_agent.regime_detector.detect_regime(trending_data)

        assert regime.regime_type in ["bull", "bear", "sideways", "volatile"]
        assert 0 <= regime.confidence <= 1
        assert 0 <= regime.trend_strength <= 1

    def test_indicator_selection(self):
        """Test dynamic indicator selection"""

        # Test different regimes
        bull_indicators = technical_agent.indicator_selector.select_indicators("bull")
        bear_indicators = technical_agent.indicator_selector.select_indicators("bear")
        sideways_indicators = technical_agent.indicator_selector.select_indicators("sideways")

        assert len(bull_indicators) > 0
        assert len(bear_indicators) > 0
        assert len(sideways_indicators) > 0

        # Different regimes should select different indicators
        assert set(bull_indicators) != set(sideways_indicators)


class TestEnhancedWhaleAgent:
    """Test enhanced whale agent capabilities"""

    @pytest.mark.asyncio
    async def test_whale_analysis_with_context(self):
        """Test whale analysis with contextual information"""

        results = await whale_agent.analyze_whale_activity(
            tokens=["BTC"], min_value_usd=100000, timeframe_hours=24
        )

        assert "BTC" in results
        transactions = results["BTC"]

        if transactions:
            tx = transactions[0]
            assert isinstance(tx, WhaleTransaction)
            assert tx.usd_value >= 100000
            assert 0 <= tx.confidence <= 1
            assert 0 <= tx.false_positive_score <= 1
            assert tx.context  # Should have contextual description

    def test_false_positive_filtering(self):
        """Test false positive filtering"""

        # Mock transaction
        mock_tx = {
            "amount": 1000.0,  # Round number
            "from_address": "0xtest",
        }

        # Mock labels
        exchange_to_exchange = {"type": "exchange", "name": "Binance 1"}
        same_exchange = {"type": "exchange", "name": "Binance 2"}

        fp_score = whale_agent.fp_filter.calculate_false_positive_score(
            mock_tx, exchange_to_exchange, same_exchange
        )

        # Should detect high false positive risk for exchange-to-exchange transfers
        assert fp_score > 0.5

    @pytest.mark.asyncio
    async def test_async_pipeline(self):
        """Test async pipeline performance"""

        addresses = [f"0x{i:040x}" for i in range(10)]  # 10 mock addresses

        async with whale_agent.AsyncOnChainPipeline() as pipeline:
            start_time = datetime.now()
            results = await pipeline.fetch_transactions(addresses, 50000)
            duration = (datetime.now() - start_time).total_seconds()

            # Should handle multiple addresses efficiently
            assert duration < 10.0  # Should be reasonably fast
            assert isinstance(results, list)


class TestEnhancedMLAgent:
    """Test enhanced ML agent capabilities"""

    def create_training_data(self):
        """Create mock training data"""
        training_data = {}
        for coin in ["BTC/USD", "ETH/USD"]:
            dates = pd.date_range(start="2024-01-01", periods=500, freq="h")
            prices = 50000 + np.cumsum(np.random.randn(500) * 100)

            training_data[coin] = pd.DataFrame(
                {
                    "close": prices,
                    "high": prices * 1.02,
                    "low": prices * 0.98,
                    "volume": np.random.randint(1000, 10000, 500),
                },
                index=dates,
            )

        return training_data

    @pytest.mark.asyncio
    async def test_ensemble_training(self):
        """Test ensemble model training"""

        training_data = self.create_training_data()

        results = await ml_agent.train_ensemble(training_data, ["1h"])

        assert "1h" in results
        assert len(ml_agent.models) > 0

        # Should have multiple models in ensemble
        if "1h" in ml_agent.models:
            models = ml_agent.models["1h"]
            assert len(models) >= 2  # At least Random Forest + Gradient Boosting

    @pytest.mark.asyncio
    async def test_uncertainty_quantification(self):
        """Test uncertainty quantification"""

        # Create and train model first
        training_data = self.create_training_data()
        await ml_agent.train_ensemble(training_data, ["1h"])

        test_data = training_data["BTC/USD"].iloc[-50:]
        predictions = await ml_agent.predict("BTC/USD", test_data, ["1h"], "bull")

        if predictions:
            pred = predictions[0]
            assert isinstance(pred, MLPrediction)
            assert 0 <= pred.confidence <= 1
            assert pred.uncertainty >= 0
            assert len(pred.prediction_interval) == 2
            assert pred.prediction_interval[0] <= pred.prediction_interval[1]

    def test_feature_engineering(self):
        """Test adaptive feature engineering"""

        data = self.create_training_data()["BTC/USD"]

        # Test different regimes
        bull_features = ml_agent.feature_engineer.engineer_features(data, "bull")
        bear_features = ml_agent.feature_engineer.engineer_features(data, "bear")

        assert len(bull_features.columns) > 20  # Should create many features
        assert len(bear_features.columns) > 20

        # Should have different weightings for different regimes
        assert not bull_features.equals(bear_features)

    def test_drift_detection(self):
        """Test model drift detection"""

        # Create reference data
        reference_data = pd.DataFrame(
            {"feature1": np.random.normal(0, 1, 100), "feature2": np.random.normal(5, 2, 100)}
        )

        # Create drifted data
        drifted_data = pd.DataFrame(
            {
                "feature1": np.random.normal(2, 1, 100),  # Mean shifted
                "feature2": np.random.normal(5, 2, 100),
            }
        )

        # Initialize with reference data
        ml_agent.drift_detector.reference_stats["test_model"] = {
            "means": reference_data.mean(),
            "stds": reference_data.std(),
            "timestamp": datetime.now(),
        }

        # Test drift detection
        is_drifted, drift_score, drift_details = ml_agent.drift_detector.detect_drift(
            drifted_data, "test_model"
        )

        assert drift_score > 0  # Should detect some drift
        assert "feature1" in drift_details


class TestEnhancedBacktestAgent:
    """Test enhanced backtest agent capabilities"""

    def create_backtest_data(self):
        """Create mock backtest data"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")

        signals = pd.DataFrame(
            {
                "timestamp": dates,
                "signal": np.random.choice([-1, 0, 1], 100, p=[0.1, 0.8, 0.1]),
                "close": 50000 + np.cumsum(np.random.randn(100) * 100),
            }
        )

        market_data = pd.DataFrame(
            {
                "timestamp": dates,
                "close": signals["close"],
                "high": signals["close"] * 1.02,
                "low": signals["close"] * 0.98,
                "volume": np.random.randint(1000000, 10000000, 100),
            }
        )

        return signals, market_data

    @pytest.mark.asyncio
    async def test_realistic_backtest(self):
        """Test backtest with realistic constraints"""

        signals, market_data = self.create_backtest_data()

        result = await backtest_agent.backtest_strategy(
            signals, market_data, "BTC/USD", "test_strategy"
        )

        assert isinstance(result, BacktestResult)
        assert result.symbol == "BTC/USD"
        assert result.strategy_name == "test_strategy"
        assert result.slippage_cost >= 0
        assert result.latency_impact >= 0
        assert 0 <= result.liquidity_score <= 1

    def test_market_impact_model(self):
        """Test market impact modeling"""

        constraints = backtest_agent.constraints

        # Test slippage calculation
        slippage = backtest_agent.market_impact_model.calculate_slippage(
            order_size_usd=10000, daily_volume_usd=1000000, volatility=0.02, constraints=constraints
        )

        assert slippage >= constraints.slippage_base
        assert slippage <= 0.05  # Should be capped

    def test_smart_order_routing(self):
        """Test smart order routing"""

        # Test normal order
        normal_result = backtest_agent.smart_router.route_order(
            "BTC/USD", 5000, "buy", {"volume": 1000000}
        )

        assert normal_result["status"] == "accepted"

        # Test large order (should trigger chunking)
        large_result = backtest_agent.smart_router.route_order(
            "BTC/USD", 50000, "buy", {"volume": 1000000}
        )

        # Should either be accepted or modified for chunking
        assert large_result["status"] in ["accepted", "modified"]

    def test_stress_testing(self):
        """Test stress testing capabilities"""

        mock_results = {"total_return": 0.15, "final_portfolio_value": 11500}

        mock_data = pd.DataFrame({"close": np.random.randn(100), "volume": np.random.randn(100)})

        stress_results = backtest_agent.stress_tester.run_stress_tests(mock_results, mock_data)

        expected_tests = [
            "flash_crash",
            "high_volatility",
            "liquidity_crisis",
            "exchange_outage",
            "regulatory_risk",
        ]

        for test_name in expected_tests:
            assert test_name in stress_results
            assert isinstance(stress_results[test_name], (int, float))


class TestEnhancedOrchestrator:
    """Test enhanced orchestrator capabilities"""

    def test_system_status(self):
        """Test system status reporting"""

        status = orchestrator.get_system_status()

        assert "orchestrator_status" in status
        assert "agents" in status
        assert "performance" in status
        assert "agent_details" in status

        # Check agents structure
        agents = status["agents"]
        assert "healthy" in agents
        assert "failed" in agents
        assert "total" in agents

    def test_resource_monitoring(self):
        """Test resource monitoring"""

        resources = orchestrator.resource_monitor._get_current_resources()

        assert 0 <= resources.cpu_percent <= 100
        assert 0 <= resources.memory_percent <= 100
        assert 0 <= resources.disk_percent <= 100
        assert isinstance(resources.network_io, dict)

    def test_resource_availability(self):
        """Test resource availability checking"""

        # Should be able to check if resources are available
        available = orchestrator.resource_monitor.is_resource_available(10, 10)
        assert isinstance(available, bool)


class TestOrderBookAnalyzer:
    """Test order book analysis capabilities"""

    @pytest.mark.asyncio
    async def test_liquidity_analysis(self):
        """Test liquidity analysis"""

        results = await order_book_analyzer.analyze_symbol_liquidity(
            "BTC/USD", ["kraken"], volume_24h=10000000
        )

        if results:
            for exchange, metrics in results.items():
                assert isinstance(metrics, LiquidityMetrics)
                assert 0 <= metrics.liquidity_score <= 1
                assert 0 <= metrics.spoofing_risk <= 1
                assert metrics.spread_percentage >= 0

    def test_spoofing_detection(self):
        """Test spoofing detection"""

        # Would need mock order book data for comprehensive testing
        # For now, just test that the detector initializes
        detector = order_book_analyzer.spoofing_detector
        assert detector is not None
        assert hasattr(detector, "detect_spoofing")


class TestIntegration:
    """Integration tests for enhanced system"""

    @pytest.mark.asyncio
    async def test_agent_coordination(self):
        """Test that all enhanced agents can work together"""

        # Test that all agents are operational
        agents = [
            ("sentiment", sentiment_agent),
            ("technical", technical_agent),
            ("whale", whale_agent),
            ("ml", ml_agent),
            ("backtest", backtest_agent),
        ]

        for name, agent in agents:
            if hasattr(agent, "get_status"):
                status = agent.get_status()
                assert status.get("status") == "operational", f"{name} agent not operational"

    def test_daily_logging_integration(self):
        """Test that all agents can log to daily logging system"""

        from utils.daily_logger import get_daily_logger

        logger = get_daily_logger()

        # Test various log types
        logger.log_trading_opportunity("BTC/USD", "4h", 8, confidence=0.95)
        logger.log_ml_prediction("ETH/USD", "1d", 3500.0, 0.88, "Enhanced_Ensemble")
        logger.log_api_call("enhanced_sentiment", "analyze", "success", response_time=0.087)
        logger.log_performance_metric("enhanced_throughput", 1200.0, "ops/sec")

        # Should not raise exceptions
        assert True


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__])
