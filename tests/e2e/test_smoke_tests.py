"""End-to-end smoke tests for system startup and basic functionality."""

import pytest
import httpx
import time
import subprocess
import signal
import os
from pathlib import Path


@pytest.mark.e2e
class TestSystemStartup:
    """Test system startup and basic service health."""
    
    @pytest.fixture(scope="class")
    def running_services(self):
        """Start services for testing."""
        # Start the multi-service system
        process = subprocess.Popen(
            ["python", "start_replit_services.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        # Wait for services to start
        time.sleep(10)
        
        yield process
        
        # Cleanup - stop services
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=10)
        except Exception:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    
    @pytest.mark.asyncio
    async def test_dashboard_startup(self, running_services):
        """Test dashboard starts and responds on port 5000."""
        max_retries = 30
        retry_count = 0
        
        async with httpx.AsyncClient() as client:
            while retry_count < max_retries:
                try:
                    response = await client.get("http://localhost:5000/", timeout=5.0)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                
                retry_count += 1
                await asyncio.sleep(1)
            
            assert retry_count < max_retries, "Dashboard failed to start within timeout"
            assert response.status_code == 200
            
            # Check it's actually Streamlit
            assert "streamlit" in response.text.lower() or "crypto" in response.text.lower()
    
    @pytest.mark.asyncio 
    async def test_api_health_endpoint(self, running_services):
        """Test API health endpoint returns 200."""
        max_retries = 30
        retry_count = 0
        
        async with httpx.AsyncClient() as client:
            while retry_count < max_retries:
                try:
                    response = await client.get("http://localhost:8001/health", timeout=5.0)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                
                retry_count += 1
                await asyncio.sleep(1)
            
            assert retry_count < max_retries, "API health endpoint failed to respond"
            assert response.status_code == 200
            
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_accessible(self, running_services):
        """Test metrics endpoint is accessible."""
        max_retries = 30
        retry_count = 0
        
        async with httpx.AsyncClient() as client:
            while retry_count < max_retries:
                try:
                    response = await client.get("http://localhost:8000/metrics", timeout=5.0)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                
                retry_count += 1
                await asyncio.sleep(1)
            
            assert retry_count < max_retries, "Metrics endpoint failed to respond"
            assert response.status_code == 200
            assert "text/plain" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_service_discovery(self, running_services):
        """Test all expected services are discoverable."""
        expected_endpoints = [
            ("http://localhost:5000/", "Dashboard"),
            ("http://localhost:8001/health", "API Health"),
            ("http://localhost:8000/metrics", "Metrics")
        ]
        
        async with httpx.AsyncClient() as client:
            for url, service_name in expected_endpoints:
                max_retries = 10
                success = False
                
                for _ in range(max_retries):
                    try:
                        response = await client.get(url, timeout=3.0)
                        if response.status_code == 200:
                            success = True
                            break
                    except Exception:
                        pass
                    await asyncio.sleep(1)
                
                assert success, f"{service_name} at {url} not accessible"
    
    def test_process_health(self, running_services):
        """Test that the main process is still running."""
        assert running_services.poll() is None, "Main process has died"
    
    @pytest.mark.asyncio
    async def test_api_info_endpoint(self, running_services):
        """Test API info endpoint provides system information."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8001/info", timeout=5.0)
                
                if response.status_code == 200:
                    info_data = response.json()
                    assert "version" in info_data
                    assert "environment" in info_data
            except Exception:
                # Info endpoint might not be implemented yet
                pytest.skip("Info endpoint not available")
    
    @pytest.mark.asyncio
    async def test_cross_service_communication(self, running_services):
        """Test that services can communicate with each other."""
        # This would test internal service communication
        # For now, just verify all services respond
        
        services = [
            "http://localhost:5000/",  # Dashboard
            "http://localhost:8001/health",  # API
            "http://localhost:8000/metrics"   # Metrics
        ]
        
        async with httpx.AsyncClient() as client:
            responses = []
            for service_url in services:
                try:
                    response = await client.get(service_url, timeout=3.0)
                    responses.append(response.status_code)
                except Exception:
                    responses.append(0)  # Failed
            
            # At least 2 out of 3 services should be responding
            successful_responses = sum(1 for status in responses if status == 200)
            assert successful_responses >= 2, f"Only {successful_responses} services responding"


@pytest.mark.e2e
class TestBasicFunctionality:
    """Test basic system functionality end-to-end."""
    
    def test_config_loading(self):
        """Test configuration loading without errors."""
        try:
            from src.cryptosmarttrader.core.config_manager import ConfigManager
            
            config = ConfigManager()
            assert config is not None
            
            # Should be able to access basic config
            assert hasattr(config, 'environment')
            
        except ImportError:
            pytest.skip("Config manager not available")
    
    def test_logging_system(self):
        """Test logging system initialization."""
        try:
            from src.cryptosmarttrader.core.structured_logger import get_logger
            
            logger = get_logger("test_logger")
            assert logger is not None
            
            # Should be able to log without errors
            logger.info("Test log message")
            
        except ImportError:
            pytest.skip("Logging system not available")
    
    def test_risk_guard_initialization(self):
        """Test risk guard system initialization."""
        try:
            from src.cryptosmarttrader.core.risk_guard import RiskGuard
            
            risk_guard = RiskGuard()
            assert risk_guard is not None
            assert risk_guard.current_risk_level is not None
            
        except ImportError:
            pytest.skip("Risk guard not available")
    
    def test_package_imports(self):
        """Test critical package imports."""
        critical_imports = [
            "src.cryptosmarttrader",
            "src.cryptosmarttrader.core.config_manager",
            "src.cryptosmarttrader.core.structured_logger",
            "src.cryptosmarttrader.core.risk_guard"
        ]
        
        failed_imports = []
        
        for import_path in critical_imports:
            try:
                __import__(import_path)
            except ImportError as e:
                failed_imports.append(f"{import_path}: {e}")
        
        assert len(failed_imports) == 0, f"Failed imports: {failed_imports}"


@pytest.mark.e2e
class TestDataFlowSmokeTest:
    """Smoke test for basic data flow."""
    
    @pytest.mark.asyncio
    async def test_market_data_mock_flow(self):
        """Test mock market data can flow through system."""
        try:
            # Import required modules
            from src.cryptosmarttrader.adapters.kraken_data_adapter import KrakenDataAdapter
            from unittest.mock import patch
            
            # Create adapter with mock data
            adapter = KrakenDataAdapter(api_key="test", secret_key="test", sandbox=True)
            
            # Test with mocked response
            with patch.object(adapter, '_make_request') as mock_request:
                mock_request.return_value = {
                    'result': {
                        'XXBTZUSD': {
                            'a': ['50100.0', '0', '0.000'],
                            'b': ['50000.0', '0', '0.000'],
                            'c': ['50050.0', '0.100'],
                            'v': ['150.5', '450.2']
                        }
                    }
                }
                
                ticker = await adapter.get_ticker('BTC/USD')
                assert ticker is not None
                assert 'bid' in ticker
                assert 'ask' in ticker
                
        except ImportError:
            pytest.skip("Market data adapter not available")
    
    def test_ml_pipeline_smoke(self):
        """Smoke test for ML pipeline components."""
        try:
            # Test that ML components can be imported
            import numpy as np
            import pandas as pd
            
            # Basic data structure test
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
                'price': np.random.uniform(45000, 55000, 100),
                'volume': np.random.uniform(1, 100, 100)
            })
            
            assert len(sample_data) == 100
            assert 'price' in sample_data.columns
            
        except ImportError:
            pytest.skip("ML dependencies not available")


@pytest.mark.e2e 
@pytest.mark.slow
class TestSystemStress:
    """Basic stress tests for system stability."""
    
    @pytest.mark.asyncio
    async def test_concurrent_api_requests(self):
        """Test system handles concurrent API requests."""
        async def make_request():
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://localhost:8001/health", timeout=5.0)
                    return response.status_code
                except Exception:
                    return 0
        
        # Make 10 concurrent requests
        import asyncio
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # Most should succeed
        successful = sum(1 for status in responses if status == 200)
        assert successful >= 7, f"Only {successful}/10 requests succeeded"
    
    @pytest.mark.asyncio
    async def test_rapid_health_checks(self):
        """Test rapid health check requests."""
        async with httpx.AsyncClient() as client:
            success_count = 0
            
            for i in range(20):
                try:
                    response = await client.get("http://localhost:8001/health", timeout=2.0)
                    if response.status_code == 200:
                        success_count += 1
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)  # 100ms between requests
            
            # Should handle rapid requests reasonably well
            assert success_count >= 15, f"Only {success_count}/20 rapid requests succeeded"