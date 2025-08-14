#!/usr/bin/env python3
"""
Demo Script - Hardened Data Layer
Demonstrates enterprise HTTP client and data orchestrator functionality
"""

import asyncio
import logging
import time
from datetime import datetime

from src.cryptosmarttrader.infrastructure.hardened_http_client import (
    HardenedHTTPClient, HTTPConfig, create_hardened_client
)
from src.cryptosmarttrader.infrastructure.data_orchestrator import (
    DataOrchestrator, DataJob
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_hardened_http_client():
    """Demo hardened HTTP client with circuit breaker and rate limiting"""
    logger.info("🚀 Testing Hardened HTTP Client...")
    
    # Create custom configuration
    config = HTTPConfig(
        base_timeout=5.0,
        max_timeout=10.0,
        max_retries=2,
        circuit_breaker_threshold=3,
        rate_limit_per_minute=30
    )
    
    async with HardenedHTTPClient(config) as client:
        
        # Test successful requests
        logger.info("✅ Testing successful requests...")
        try:
            response1 = await client.get(
                "https://api.kraken.com/0/public/Ticker?pair=BTCUSD",
                source="kraken"
            )
            logger.info(f"Request 1 successful: {response1['latency_ms']:.0f}ms")
            
            response2 = await client.get(
                "https://api.kraken.com/0/public/Time",
                source="kraken"
            )
            logger.info(f"Request 2 successful: {response2['latency_ms']:.0f}ms")
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
        
        # Test rate limiting
        logger.info("⏱️  Testing rate limiting...")
        start_time = time.time()
        
        # Make multiple rapid requests
        tasks = []
        for i in range(5):
            task = client.get(
                "https://api.kraken.com/0/public/Time",
                source="kraken_rate_test"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time
        
        successful = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Rate limiting test: {successful}/5 successful in {duration:.1f}s")
        
        # Test invalid endpoint for circuit breaker
        logger.info("🔴 Testing circuit breaker with invalid endpoint...")
        for i in range(4):
            try:
                await client.get(
                    "https://api.kraken.com/0/public/NonExistentEndpoint",
                    source="kraken_error_test"
                )
            except Exception as e:
                logger.info(f"Expected error {i+1}: {str(e)[:100]}...")
        
        # Show health status
        health = client.get_health_status()
        logger.info("📊 Health Status:")
        for source, status in health.items():
            logger.info(f"  {source}: {status['circuit_breaker_state']} - Success rate: {status['success_rate']:.1%}")


async def demo_data_orchestrator():
    """Demo data orchestrator with scheduled jobs"""
    logger.info("🚀 Testing Data Orchestrator...")
    
    config = {
        'http_timeout': 10.0,
        'rate_limits': {
            'kraken': 60
        }
    }
    
    orchestrator = DataOrchestrator(config)
    
    try:
        # Start orchestrator
        await orchestrator.start()
        logger.info("✅ Orchestrator started")
        
        # Add custom test job
        test_job = DataJob(
            job_id="test_ticker",
            name="Test BTC Ticker",
            source="kraken",
            endpoint="https://api.kraken.com/0/public/Ticker?pair=BTCUSD",
            schedule="*/10 * * * * *",  # Every 10 seconds
            timeout_seconds=5.0,
            rate_limit_per_minute=30
        )
        
        await orchestrator.add_job(test_job)
        logger.info("✅ Test job added")
        
        # Let it run for a short time
        logger.info("⏳ Running orchestrator for 30 seconds...")
        await asyncio.sleep(30)
        
        # Get status
        status = orchestrator.get_status()
        logger.info("📊 Orchestrator Status:")
        logger.info(f"  Total executions: {status['orchestrator']['total_executions']}")
        logger.info(f"  Success rate: {status['orchestrator']['success_rate']:.1%}")
        
        # Show job details
        for job_id, job_status in status['jobs'].items():
            logger.info(f"  Job {job_status['name']}: {job_status['success_count']} successes, {job_status['failure_count']} failures")
        
        # Show source health
        for source, health in status['sources'].items():
            logger.info(f"  Source {source}: {health['status']} - Error rate: {health['error_rate']:.1%}")
        
    finally:
        await orchestrator.stop()
        logger.info("✅ Orchestrator stopped")


async def main():
    """Main demo function"""
    logger.info("🎯 Starting Hardened Data Layer Demo")
    
    print("\n" + "="*60)
    print("  CryptoSmartTrader V2 - Hardened Data Layer Demo")
    print("="*60)
    
    try:
        # Demo HTTP client
        await demo_hardened_http_client()
        
        print("\n" + "-"*60)
        
        # Demo orchestrator
        await demo_data_orchestrator()
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())