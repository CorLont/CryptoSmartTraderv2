#!/usr/bin/env python3
"""
Scraping Core Module - Enterprise Data Collection Framework
"""

from .async_client import AsyncScrapeClient, get_async_client, close_async_client
from .data_sources import (
    TwitterScraper, RedditScraper, NewsScraper,
    TelegramScraper, DiscordScraper, ScrapeResult,
    AVAILABLE_SOURCES
)
from .orchestrator import ScrapingOrchestrator, get_scraping_orchestrator, ScrapingResults

__all__ = [
    'AsyncScrapeClient',
    'get_async_client',
    'close_async_client',
    'TwitterScraper',
    'RedditScraper',
    'NewsScraper',
    'TelegramScraper',
    'DiscordScraper',
    'ScrapeResult',
    'AVAILABLE_SOURCES',
    'ScrapingOrchestrator',
    'get_scraping_orchestrator',
    'ScrapingResults'
]
