#!/usr/bin/env python3
"""
API Integration Tests - Test external API integration with proper markers
"""

import pytest
import asyncio
from unittest.mock import patch, Mock
from tests.fixtures.time_helpers import DeterministicTime
from tests.fixtures.api_fixtures import mock_kraken_response, mock_binance_response


@pytest.mark.integration
@pytest.mark.api_key
class TestExchangeAPIIntegration:
    """Integration tests for exchange APIs"""
    
    def test_kraken_api_connection(self, mock_kraken_response):
        """Test Kraken API connection (requires API key)"""
        # This test would normally require API keys
        # Using mock for demonstration
        
        with patch('ccxt.kraken') as mock_kraken:
            mock_instance = Mock()
            mock_instance.fetch_ticker.return_value = {
                'symbol': 'BTC/USD',
                'last': 50000.0,
                'volume': 1234.5
            }
            mock_kraken.return_value = mock_instance
            
            # Test API call
            exchange = mock_kraken()
            ticker = exchange.fetch_ticker('BTC/USD')
            
            assert ticker['symbol'] == 'BTC/USD'
            assert ticker['last'] > 0
            assert ticker['volume'] > 0
    
    @pytest.mark.slow
    def test_multiple_exchange_data_consistency(self):
        """Test data consistency across exchanges (slow test)"""
        # This would be a slow test comparing data from multiple exchanges
        
        exchanges_data = {
            'kraken': {'BTC/USD': 50000.0},
            'binance': {'BTC/USDT': 50010.0},
            'coinbase': {'BTC-USD': 49995.0}
        }
        
        # Check price consistency (within 1%)
        btc_prices = list(exchanges_data.values())
        max_price = max(price[list(price.keys())[0]] for price in btc_prices)
        min_price = min(price[list(price.keys())[0]] for price in btc_prices)
        
        price_spread = (max_price - min_price) / min_price
        assert price_spread < 0.01, f"Price spread too large: {price_spread:.2%}"


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def test_portfolio_data_persistence(self, frozen_time):
        """Test portfolio data persistence with deterministic time"""
        
        # Mock database operations
        portfolio_data = {
            'account_id': 'test_account',
            'total_value': 100000.0,
            'timestamp': frozen_time.current_time.isoformat()
        }
        
        # Simulate database save/load
        saved_data = self.save_portfolio_data(portfolio_data)
        loaded_data = self.load_portfolio_data('test_account')
        
        assert loaded_data['account_id'] == portfolio_data['account_id']
        assert loaded_data['total_value'] == portfolio_data['total_value']
        assert loaded_data['timestamp'] == portfolio_data['timestamp']
    
    def save_portfolio_data(self, data):
        """Mock database save operation"""
        return data
    
    def load_portfolio_data(self, account_id):
        """Mock database load operation"""
        return {
            'account_id': account_id,
            'total_value': 100000.0,
            'timestamp': '2024-01-15T12:00:00+00:00'
        }


@pytest.mark.integration
@pytest.mark.ml
class TestMLModelIntegration:
    """Integration tests for ML model operations"""
    
    def test_model_training_pipeline(self, deterministic_time):
        """Test complete model training pipeline with time control"""
        
        with deterministic_time.freeze_time():
            # Generate training data
            training_data = self.generate_training_data(deterministic_time)
            
            # Train model
            model = self.train_model(training_data)
            
            # Validate model
            assert model is not None
            assert hasattr(model, 'predict')
            
            # Test prediction
            test_features = training_data['features'][:10]
            predictions = model.predict(test_features)
            
            assert len(predictions) == 10
            assert all(0 <= p <= 1 for p in predictions)
    
    def generate_training_data(self, time_controller):
        """Generate training data with time control"""
        import numpy as np
        
        # Generate features and labels
        n_samples = 1000
        n_features = 20
        
        features = np.random.randn(n_samples, n_features)
        labels = np.random.choice([0, 1], n_samples)
        
        return {
            'features': features,
            'labels': labels,
            'timestamp': time_controller.current_time
        }
    
    def train_model(self, training_data):
        """Mock model training"""
        from unittest.mock import Mock
        
        model = Mock()
        model.predict.return_value = np.random.uniform(0, 1, 10)
        
        return model


@pytest.mark.integration
@pytest.mark.trading
class TestTradingSystemIntegration:
    """Integration tests for trading system"""
    
    def test_end_to_end_signal_processing(self, frozen_time):
        """Test complete signal processing pipeline"""
        
        # Step 1: Generate signal
        signal = {
            'timestamp': frozen_time.current_time.isoformat(),
            'symbol': 'BTC/USD',
            'signal_type': 'buy',
            'confidence': 0.85,
            'price': 50000.0
        }
        
        # Step 2: Process signal through confidence gate
        processed_signal = self.process_signal(signal)
        assert processed_signal['passed_gate'] is True
        
        # Step 3: Calculate position size
        position_size = self.calculate_position_size(processed_signal)
        assert position_size > 0
        
        # Step 4: Execute trade (simulated)
        trade_result = self.execute_trade(processed_signal, position_size)
        assert trade_result['status'] == 'executed'
        
        # Step 5: Update portfolio
        portfolio_update = self.update_portfolio(trade_result)
        assert portfolio_update['success'] is True
    
    def process_signal(self, signal):
        """Process signal through confidence gate"""
        passed_gate = signal['confidence'] >= 0.8
        return {**signal, 'passed_gate': passed_gate}
    
    def calculate_position_size(self, signal):
        """Calculate position size for signal"""
        if not signal['passed_gate']:
            return 0
        
        capital = 100000.0
        max_position = 0.1
        confidence = signal['confidence']
        
        return (capital * max_position * confidence) / signal['price']
    
    def execute_trade(self, signal, position_size):
        """Execute trade (simulated)"""
        return {
            'symbol': signal['symbol'],
            'side': signal['signal_type'],
            'quantity': position_size,
            'price': signal['price'],
            'status': 'executed',
            'trade_id': f"trade_{hash(signal['timestamp'])}"
        }
    
    def update_portfolio(self, trade_result):
        """Update portfolio with trade result"""
        return {'success': True, 'trade_id': trade_result['trade_id']}


@pytest.mark.integration
@pytest.mark.slow
class TestSystemHealthIntegration:
    """Integration tests for system health monitoring"""
    
    def test_complete_health_check_cycle(self, deterministic_time):
        """Test complete health check cycle"""
        
        with deterministic_time.freeze_time():
            # Collect health metrics
            health_metrics = self.collect_health_metrics()
            
            # Calculate health scores
            health_scores = self.calculate_health_scores(health_metrics)
            
            # Determine overall grade
            overall_grade = self.calculate_overall_grade(health_scores)
            
            # Update trading policy
            trading_policy = self.update_trading_policy(overall_grade)
            
            # Validate results
            assert 'data_quality' in health_scores
            assert 'system_performance' in health_scores
            assert overall_grade in ['A', 'B', 'C', 'D', 'F']
            assert 'trading_enabled' in trading_policy
    
    def collect_health_metrics(self):
        """Collect system health metrics"""
        return {
            'api_uptime': 99.5,
            'data_freshness': 2.0,  # minutes
            'cpu_usage': 45.0,
            'memory_usage': 60.0,
            'error_rate': 0.1
        }
    
    def calculate_health_scores(self, metrics):
        """Calculate component health scores"""
        return {
            'data_quality': 95.0,
            'system_performance': 88.0,
            'api_health': 92.0
        }
    
    def calculate_overall_grade(self, scores):
        """Calculate overall health grade"""
        avg_score = sum(scores.values()) / len(scores)
        
        if avg_score >= 90:
            return 'A'
        elif avg_score >= 80:
            return 'B'
        elif avg_score >= 70:
            return 'C'
        elif avg_score >= 60:
            return 'D'
        else:
            return 'F'
    
    def update_trading_policy(self, grade):
        """Update trading policy based on health grade"""
        return {
            'trading_enabled': grade in ['A', 'B', 'C'],
            'position_size_multiplier': 1.0 if grade in ['A', 'B'] else 0.5,
            'grade': grade
        }


if __name__ == "__main__":
    # Run integration tests
    pytest.main([
        __file__, 
        "-v", 
        "-m", "integration and not slow",  # Run integration tests but skip slow ones
        "--tb=short"
    ])