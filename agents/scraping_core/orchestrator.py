#!/usr/bin/env python3
"""
Scraping Orchestrator - Coordinates Multi-Source Data Collection
Manages parallel scraping across all sources with completeness validation
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

from .async_client import AsyncScrapeClient, get_async_client
from .data_sources import AVAILABLE_SOURCES, ScrapeResult

@dataclass
class ScrapingTask:
    """Individual scraping task configuration"""
    symbol: str
    sources: List[str]
    limit_per_source: int = 50
    timeout: int = 60
    require_all_sources: bool = False  # No-fallback mode flag

@dataclass
class ScrapingResults:
    """Results from orchestrated scraping operation"""
    symbol: str
    total_results: int
    results_per_source: Dict[str, int]
    successful_sources: List[str]
    failed_sources: List[str]
    completeness_percentage: float
    execution_time: float
    data: List[ScrapeResult]
    errors: Dict[str, str]

class ScrapingOrchestrator:
    """Orchestrates multi-source scraping with completeness validation"""
    
    def __init__(self, no_fallback_mode: bool = True):
        self.no_fallback_mode = no_fallback_mode
        self.client: Optional[AsyncScrapeClient] = None
        
        # Source scrapers
        self.scrapers: Dict[str, Any] = {}
        
        # Completeness tracking
        self.completeness_requirements = {
            "twitter": 80.0,    # 80% success rate required
            "reddit": 75.0,     # 75% success rate required
            "news": 85.0,       # 85% success rate required
            "telegram": 60.0,   # 60% success rate required (limited API)
            "discord": 60.0     # 60% success rate required (limited API)
        }
        
        # Minimum sources required in no-fallback mode
        self.minimum_sources_required = 3
    
    async def initialize(self):
        """Initialize the orchestrator and scrapers"""
        
        # Get async client
        self.client = await get_async_client(
            max_connections=100,
            max_connections_per_host=20,
            timeout=30
        )
        
        # Initialize all available scrapers
        for source_name, scraper_class in AVAILABLE_SOURCES.items():
            try:
                self.scrapers[source_name] = scraper_class(self.client)
                self.client.logger.info(f"Initialized {source_name} scraper")
            except Exception as e:
                self.client.logger.error(f"Failed to initialize {source_name} scraper: {e}")
        
        self.client.logger.info(f"Scraping orchestrator initialized with {len(self.scrapers)} sources")
    
    async def scrape_symbol(self, 
                           symbol: str,
                           sources: Optional[List[str]] = None,
                           limit_per_source: int = 50,
                           timeout: int = 60) -> ScrapingResults:
        """Scrape data for a single symbol across multiple sources"""
        
        if not self.client:
            await self.initialize()
        
        # Use all available sources if none specified
        if sources is None:
            sources = list(self.scrapers.keys())
        
        # Filter to available sources
        available_sources = [s for s in sources if s in self.scrapers]
        
        start_time = time.time()
        
        # Parallel scraping tasks
        scraping_tasks = []
        for source in available_sources:
            task = self._scrape_source(
                source=source,
                symbol=symbol,
                limit=limit_per_source,
                timeout=timeout
            )
            scraping_tasks.append(task)
        
        self.client.logger.info(f"Starting parallel scraping for {symbol}",
                               sources=available_sources,
                               limit_per_source=limit_per_source)
        
        # Execute all scraping tasks with timeout
        try:
            scraping_results = await asyncio.wait_for(
                asyncio.gather(*scraping_tasks, return_exceptions=True),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            self.client.logger.error(f"Scraping timeout for {symbol} after {timeout}s")
            scraping_results = [Exception("Timeout")] * len(scraping_tasks)
        
        # Process results
        all_data = []
        results_per_source = {}
        successful_sources = []
        failed_sources = []
        errors = {}
        
        for i, result in enumerate(scraping_results):
            source = available_sources[i]
            
            if isinstance(result, Exception):
                failed_sources.append(source)
                errors[source] = str(result)
                results_per_source[source] = 0
            else:
                successful_sources.append(source)
                results_per_source[source] = len(result)
                all_data.extend(result)
        
        execution_time = time.time() - start_time
        
        # Calculate completeness
        completeness_percentage = self._calculate_completeness(
            successful_sources, 
            failed_sources,
            available_sources
        )
        
        results = ScrapingResults(
            symbol=symbol,
            total_results=len(all_data),
            results_per_source=results_per_source,
            successful_sources=successful_sources,
            failed_sources=failed_sources,
            completeness_percentage=completeness_percentage,
            execution_time=execution_time,
            data=all_data,
            errors=errors
        )
        
        # Apply completeness validation
        if self.no_fallback_mode:
            self._validate_completeness(results)
        
        self.client.logger.info(f"Scraping completed for {symbol}",
                               total_results=len(all_data),
                               successful_sources=len(successful_sources),
                               failed_sources=len(failed_sources),
                               completeness=completeness_percentage,
                               execution_time=execution_time)
        
        return results
    
    async def batch_scrape_symbols(self, 
                                  symbols: List[str],
                                  sources: Optional[List[str]] = None,
                                  limit_per_source: int = 50,
                                  max_concurrent: int = 10,
                                  timeout: int = 60) -> Dict[str, ScrapingResults]:
        """Scrape data for multiple symbols in parallel"""
        
        if not self.client:
            await self.initialize()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_single_symbol(symbol):
            async with semaphore:
                return await self.scrape_symbol(
                    symbol=symbol,
                    sources=sources,
                    limit_per_source=limit_per_source,
                    timeout=timeout
                )
        
        self.client.logger.info(f"Starting batch scraping for {len(symbols)} symbols",
                               max_concurrent=max_concurrent)
        
        start_time = time.time()
        
        # Execute batch scraping
        batch_tasks = [scrape_single_symbol(symbol) for symbol in symbols]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        # Process batch results
        results = {}
        successful_symbols = 0
        failed_symbols = 0
        
        for i, result in enumerate(batch_results):
            symbol = symbols[i]
            
            if isinstance(result, Exception):
                failed_symbols += 1
                self.client.logger.error(f"Batch scraping failed for {symbol}: {result}")
                # Create empty result for failed symbol
                results[symbol] = ScrapingResults(
                    symbol=symbol,
                    total_results=0,
                    results_per_source={},
                    successful_sources=[],
                    failed_sources=list(sources) if sources else list(self.scrapers.keys()),
                    completeness_percentage=0.0,
                    execution_time=0.0,
                    data=[],
                    errors={"batch_error": str(result)}
                )
            else:
                successful_symbols += 1
                results[symbol] = result
        
        self.client.logger.info(f"Batch scraping completed",
                               total_symbols=len(symbols),
                               successful_symbols=successful_symbols,
                               failed_symbols=failed_symbols,
                               execution_time=execution_time)
        
        return results
    
    async def _scrape_source(self, 
                            source: str, 
                            symbol: str, 
                            limit: int,
                            timeout: int) -> List[ScrapeResult]:
        """Scrape single source for symbol"""
        
        if source not in self.scrapers:
            raise ValueError(f"Source {source} not available")
        
        scraper = self.scrapers[source]
        
        try:
            # Execute scraping with timeout
            results = await asyncio.wait_for(
                scraper.scrape_symbol_mentions(symbol, limit),
                timeout=timeout
            )
            
            self.client.logger.debug(f"Source {source} scraped {len(results)} results for {symbol}")
            return results
        
        except asyncio.TimeoutError:
            raise Exception(f"Timeout scraping {source} for {symbol}")
        except Exception as e:
            self.client.logger.error(f"Scraping failed for {source}/{symbol}: {e}")
            raise
    
    def _calculate_completeness(self, 
                               successful_sources: List[str],
                               failed_sources: List[str],
                               total_sources: List[str]) -> float:
        """Calculate overall completeness percentage"""
        
        if not total_sources:
            return 0.0
        
        # Weighted completeness based on source importance
        source_weights = {
            "twitter": 0.3,
            "reddit": 0.25,
            "news": 0.25,
            "telegram": 0.1,
            "discord": 0.1
        }
        
        total_weight = 0.0
        successful_weight = 0.0
        
        for source in total_sources:
            weight = source_weights.get(source, 0.1)
            total_weight += weight
            
            if source in successful_sources:
                successful_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return (successful_weight / total_weight) * 100
    
    def _validate_completeness(self, results: ScrapingResults):
        """Validate completeness in no-fallback mode"""
        
        if not self.no_fallback_mode:
            return
        
        # Check minimum successful sources
        if len(results.successful_sources) < self.minimum_sources_required:
            raise Exception(
                f"No-fallback mode: Only {len(results.successful_sources)} sources successful, "
                f"minimum {self.minimum_sources_required} required"
            )
        
        # Check individual source completeness requirements
        for source in results.successful_sources:
            if source in self.completeness_requirements:
                required_completeness = self.completeness_requirements[source]
                
                # Get source metrics from client
                source_metrics = self.client.get_source_metrics(source)
                if source_metrics and source_metrics.completeness_percentage < required_completeness:
                    raise Exception(
                        f"No-fallback mode: Source {source} completeness "
                        f"{source_metrics.completeness_percentage:.1f}% < required {required_completeness}%"
                    )
        
        # Check overall completeness
        if results.completeness_percentage < 75.0:  # Minimum 75% overall
            raise Exception(
                f"No-fallback mode: Overall completeness {results.completeness_percentage:.1f}% < required 75.0%"
            )
    
    async def generate_completeness_report(self) -> Dict[str, Any]:
        """Generate completeness report for all sources"""
        
        if not self.client:
            await self.initialize()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "no_fallback_mode": self.no_fallback_mode,
            "sources": {},
            "overall_health": "unknown"
        }
        
        healthy_sources = 0
        total_sources = len(self.scrapers)
        
        for source_name in self.scrapers.keys():
            metrics = self.client.get_source_metrics(source_name)
            
            if metrics:
                required_completeness = self.completeness_requirements.get(source_name, 70.0)
                is_healthy = metrics.completeness_percentage >= required_completeness
                
                if is_healthy:
                    healthy_sources += 1
                
                report["sources"][source_name] = {
                    "completeness_percentage": metrics.completeness_percentage,
                    "required_completeness": required_completeness,
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "last_success": metrics.last_success_timestamp.isoformat() if metrics.last_success_timestamp else None,
                    "last_error": metrics.last_error_timestamp.isoformat() if metrics.last_error_timestamp else None,
                    "status": "healthy" if is_healthy else "unhealthy",
                    "average_response_time": metrics.average_response_time
                }
            else:
                report["sources"][source_name] = {
                    "status": "no_data",
                    "completeness_percentage": 0.0
                }
        
        # Determine overall health
        health_percentage = (healthy_sources / total_sources * 100) if total_sources > 0 else 0
        
        if health_percentage >= 80:
            overall_health = "healthy"
        elif health_percentage >= 60:
            overall_health = "degraded"
        else:
            overall_health = "unhealthy"
        
        report["overall_health"] = overall_health
        report["health_percentage"] = health_percentage
        report["healthy_sources"] = healthy_sources
        report["total_sources"] = total_sources
        
        return report
    
    async def save_completeness_report(self) -> Path:
        """Save completeness report to daily logs"""
        
        # Generate report
        report = await self.generate_completeness_report()
        
        # Create daily log directory
        today_str = datetime.now().strftime("%Y%m%d")
        daily_log_dir = Path("logs/daily") / today_str
        daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save report
        timestamp_str = datetime.now().strftime("%H%M%S")
        report_file = daily_log_dir / f"completeness_report_{timestamp_str}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest
        latest_file = daily_log_dir / "completeness_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.client:
            self.client.logger.info(f"Completeness report saved: {report_file}")
        
        return report_file
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client._close_session()

# Global orchestrator instance
_orchestrator: Optional[ScrapingOrchestrator] = None

async def get_scraping_orchestrator(no_fallback_mode: bool = True) -> ScrapingOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = ScrapingOrchestrator(no_fallback_mode=no_fallback_mode)
        await _orchestrator.initialize()
    
    return _orchestrator