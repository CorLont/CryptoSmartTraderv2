#!/usr/bin/env python3
"""
Coverage Audit Manager - Ensures 100% Exchange Coverage
Implements daily coverage audits and missing coin alerts for complete market coverage
"""

import asyncio
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger
from core.secrets_manager import get_secrets_manager, secure_function
from core.data_quality_manager import get_data_quality_manager

class CoverageStatus(str, Enum):
    """Coverage status levels"""
    COMPLETE = "complete"      # 100% coverage
    PARTIAL = "partial"        # Missing some coins
    CRITICAL = "critical"      # Missing many coins
    FAILED = "failed"          # Audit failed

class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class CoinListing:
    """Exchange coin listing information"""
    symbol: str
    base_currency: str
    quote_currency: str
    listing_date: datetime
    status: str  # 'active', 'delisted', 'suspended'
    min_order_size: float
    trading_fees: Dict[str, float]
    last_updated: datetime

@dataclass
class CoverageGap:
    """Missing coin coverage gap"""
    symbol: str
    exchange: str
    listing_date: datetime
    gap_duration_hours: float
    impact_score: float  # Estimated impact of missing this coin
    volume_24h: Optional[float] = None
    market_cap_rank: Optional[int] = None

@dataclass
class CoverageAuditResult:
    """Coverage audit results"""
    audit_id: str
    timestamp: datetime
    exchange: str
    total_live_coins: int
    analyzed_coins: int
    missing_coins: int
    coverage_percentage: float
    status: CoverageStatus
    missing_coin_symbols: List[str]
    coverage_gaps: List[CoverageGap]
    new_listings: List[str]
    delisted_coins: List[str]
    audit_duration_seconds: float

@dataclass
class CoverageAlert:
    """Coverage monitoring alert"""
    alert_id: str
    alert_type: str
    severity: AlertSeverity
    exchange: str
    symbol: Optional[str]
    message: str
    impact_score: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ExchangeConnector:
    """Secure exchange connector for coverage audits"""
    
    def __init__(self):
        self.logger = get_logger()
        self.secrets_manager = get_secrets_manager()
        self.exchanges = {}
        
    @secure_function(redact_kwargs=['apiKey', 'secret'])
    async def initialize_exchange(self, exchange_name: str) -> bool:
        """Initialize exchange connection securely"""
        
        try:
            if exchange_name.lower() == 'kraken':
                api_key = self.secrets_manager.get_secret('KRAKEN_API_KEY')
                secret = self.secrets_manager.get_secret('KRAKEN_SECRET')
                
                if not api_key or not secret:
                    self.logger.error("Kraken API credentials not available")
                    return False
                
                self.exchanges[exchange_name] = ccxt_async.kraken({
                    'apiKey': api_key,
                    'secret': secret,
                    'enableRateLimit': True,
                    'sandbox': False
                })
                
                # Test connection
                await self.exchanges[exchange_name].load_markets()
                self.logger.info(f"Exchange {exchange_name} connected successfully")
                return True
                
            else:
                self.logger.warning(f"Exchange {exchange_name} not supported yet")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {exchange_name}: {e}")
            return False
    
    async def get_live_markets(self, exchange_name: str) -> Dict[str, CoinListing]:
        """Get all live markets from exchange"""
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not initialized")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            # Get fresh market data
            markets = await exchange.load_markets(reload=True)
            
            live_listings = {}
            
            for symbol, market in markets.items():
                # Only include active trading pairs
                if market.get('active', False) and market.get('type') == 'spot':
                    listing = CoinListing(
                        symbol=symbol,
                        base_currency=market.get('base', ''),
                        quote_currency=market.get('quote', ''),
                        listing_date=datetime.now(),  # Placeholder - real implementation would track this
                        status='active',
                        min_order_size=market.get('limits', {}).get('amount', {}).get('min', 0),
                        trading_fees=market.get('fees', {}),
                        last_updated=datetime.now()
                    )
                    live_listings[symbol] = listing
            
            self.logger.info(
                f"Retrieved {len(live_listings)} live markets from {exchange_name}",
                extra={'exchange': exchange_name, 'markets_count': len(live_listings)}
            )
            
            return live_listings
            
        except Exception as e:
            self.logger.error(f"Failed to get markets from {exchange_name}: {e}")
            raise
    
    async def get_coin_volume_data(self, exchange_name: str, symbols: List[str]) -> Dict[str, float]:
        """Get 24h volume data for coins"""
        
        if exchange_name not in self.exchanges:
            return {}
        
        exchange = self.exchanges[exchange_name]
        volume_data = {}
        
        try:
            tickers = await exchange.fetch_tickers(symbols)
            
            for symbol, ticker in tickers.items():
                volume_24h = ticker.get('quoteVolume', 0)
                volume_data[symbol] = volume_24h
                
        except Exception as e:
            self.logger.warning(f"Failed to get volume data: {e}")
        
        return volume_data
    
    async def cleanup(self):
        """Cleanup exchange connections"""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass

class CoverageAuditor:
    """Main coverage audit engine"""
    
    def __init__(self):
        self.logger = get_logger()
        self.exchange_connector = ExchangeConnector()
        self.data_quality_manager = get_data_quality_manager()
        
        # Audit state
        self.audit_history: List[CoverageAuditResult] = []
        self.coverage_alerts: List[CoverageAlert] = []
        self.baseline_coverage: Dict[str, Set[str]] = {}  # exchange -> symbols
        
        # Configuration
        self.audit_schedule_hours = 24  # Daily audits
        self.critical_coverage_threshold = 0.95  # 95% minimum coverage
        self.warning_coverage_threshold = 0.98   # 98% warning threshold
        
    async def run_daily_coverage_audit(self, exchange: str = 'kraken') -> CoverageAuditResult:
        """Run comprehensive daily coverage audit"""
        
        audit_start = datetime.now()
        audit_id = f"coverage_audit_{exchange}_{audit_start.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting coverage audit for {exchange}")
        
        try:
            # Initialize exchange connection
            if not await self.exchange_connector.initialize_exchange(exchange):
                raise Exception(f"Failed to initialize {exchange}")
            
            # Get live markets from exchange
            live_markets = await self.exchange_connector.get_live_markets(exchange)
            live_symbols = set(live_markets.keys())
            
            # Get currently analyzed symbols
            analyzed_symbols = await self._get_analyzed_symbols(exchange)
            
            # Calculate coverage metrics
            missing_symbols = live_symbols - analyzed_symbols
            extra_symbols = analyzed_symbols - live_symbols  # Symbols we track but exchange doesn't have
            
            coverage_percentage = len(analyzed_symbols & live_symbols) / len(live_symbols) if live_symbols else 0
            
            # Determine coverage status
            if coverage_percentage >= self.critical_coverage_threshold:
                if coverage_percentage >= self.warning_coverage_threshold:
                    status = CoverageStatus.COMPLETE
                else:
                    status = CoverageStatus.PARTIAL
            else:
                status = CoverageStatus.CRITICAL
            
            # Analyze coverage gaps
            coverage_gaps = await self._analyze_coverage_gaps(
                exchange, missing_symbols, live_markets
            )
            
            # Detect new listings
            baseline_symbols = self.baseline_coverage.get(exchange, set())
            new_listings = list(live_symbols - baseline_symbols)
            delisted_coins = list(baseline_symbols - live_symbols)
            
            # Update baseline
            self.baseline_coverage[exchange] = live_symbols.copy()
            
            audit_duration = (datetime.now() - audit_start).total_seconds()
            
            # Create audit result
            audit_result = CoverageAuditResult(
                audit_id=audit_id,
                timestamp=audit_start,
                exchange=exchange,
                total_live_coins=len(live_symbols),
                analyzed_coins=len(analyzed_symbols & live_symbols),
                missing_coins=len(missing_symbols),
                coverage_percentage=coverage_percentage,
                status=status,
                missing_coin_symbols=list(missing_symbols),
                coverage_gaps=coverage_gaps,
                new_listings=new_listings,
                delisted_coins=delisted_coins,
                audit_duration_seconds=audit_duration
            )
            
            # Store audit result
            self.audit_history.append(audit_result)
            if len(self.audit_history) > 100:  # Keep last 100 audits
                self.audit_history = self.audit_history[-100:]
            
            # Generate alerts
            await self._generate_coverage_alerts(audit_result)
            
            # Log audit summary
            self.logger.info(
                f"Coverage audit completed for {exchange}",
                extra={
                    'audit_id': audit_id,
                    'coverage_percentage': coverage_percentage,
                    'status': status.value,
                    'missing_coins': len(missing_symbols),
                    'new_listings': len(new_listings),
                    'audit_duration_seconds': audit_duration
                }
            )
            
            return audit_result
            
        except Exception as e:
            # Create failed audit result
            audit_duration = (datetime.now() - audit_start).total_seconds()
            
            failed_result = CoverageAuditResult(
                audit_id=audit_id,
                timestamp=audit_start,
                exchange=exchange,
                total_live_coins=0,
                analyzed_coins=0,
                missing_coins=0,
                coverage_percentage=0.0,
                status=CoverageStatus.FAILED,
                missing_coin_symbols=[],
                coverage_gaps=[],
                new_listings=[],
                delisted_coins=[],
                audit_duration_seconds=audit_duration
            )
            
            self.audit_history.append(failed_result)
            
            self.logger.error(f"Coverage audit failed for {exchange}: {e}")
            
            # Generate failure alert
            await self._generate_failure_alert(exchange, str(e))
            
            return failed_result
        
        finally:
            await self.exchange_connector.cleanup()
    
    async def _get_analyzed_symbols(self, exchange: str) -> Set[str]:
        """Get symbols currently being analyzed by our system"""
        
        try:
            # Get analyzed symbols from data quality manager
            quality_summary = self.data_quality_manager.get_quality_summary()
            
            # Extract symbols from tracked coins
            tracked_coins = quality_summary.get('tracked_coins', [])
            analyzed_symbols = set()
            
            for coin_info in tracked_coins:
                if isinstance(coin_info, dict):
                    symbol = coin_info.get('symbol', '')
                    if symbol and exchange.lower() in symbol.lower():
                        analyzed_symbols.add(symbol)
                elif isinstance(coin_info, str):
                    analyzed_symbols.add(coin_info)
            
            # Also check recent data files for additional symbols
            # This would integrate with your data storage system
            additional_symbols = await self._scan_data_files_for_symbols(exchange)
            analyzed_symbols.update(additional_symbols)
            
            self.logger.info(
                f"Found {len(analyzed_symbols)} analyzed symbols for {exchange}",
                extra={'exchange': exchange, 'symbols_count': len(analyzed_symbols)}
            )
            
            return analyzed_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to get analyzed symbols: {e}")
            return set()
    
    async def _scan_data_files_for_symbols(self, exchange: str) -> Set[str]:
        """Scan data files to find additional symbols being tracked"""
        
        symbols = set()
        
        try:
            # Scan data directory for recent files
            data_dir = Path('data')
            if data_dir.exists():
                # Look for recent data files
                cutoff_time = datetime.now() - timedelta(days=7)
                
                for file_path in data_dir.glob('**/*.json'):
                    try:
                        if file_path.stat().st_mtime > cutoff_time.timestamp():
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                            # Extract symbols from data structure
                            if isinstance(data, dict):
                                for key in data.keys():
                                    if '/' in key and exchange.lower() in key.lower():
                                        symbols.add(key)
                                        
                    except Exception:
                        continue  # Skip problematic files
            
        except Exception as e:
            self.logger.warning(f"Failed to scan data files: {e}")
        
        return symbols
    
    async def _analyze_coverage_gaps(
        self, 
        exchange: str, 
        missing_symbols: Set[str], 
        live_markets: Dict[str, CoinListing]
    ) -> List[CoverageGap]:
        """Analyze missing coin coverage gaps"""
        
        coverage_gaps = []
        
        # Get volume data for missing coins to assess impact
        volume_data = await self.exchange_connector.get_coin_volume_data(
            exchange, list(missing_symbols)
        )
        
        for symbol in missing_symbols:
            listing = live_markets.get(symbol)
            if not listing:
                continue
            
            # Calculate gap duration (simplified - would need real listing tracking)
            gap_duration = 24.0  # Assume 24 hours for now
            
            # Calculate impact score based on volume
            volume_24h = volume_data.get(symbol, 0)
            impact_score = min(volume_24h / 1000000, 1.0)  # Normalize to 0-1 based on $1M volume
            
            gap = CoverageGap(
                symbol=symbol,
                exchange=exchange,
                listing_date=listing.listing_date,
                gap_duration_hours=gap_duration,
                impact_score=impact_score,
                volume_24h=volume_24h
            )
            
            coverage_gaps.append(gap)
        
        # Sort by impact score descending
        coverage_gaps.sort(key=lambda x: x.impact_score, reverse=True)
        
        return coverage_gaps
    
    async def _generate_coverage_alerts(self, audit_result: CoverageAuditResult):
        """Generate alerts based on audit results"""
        
        alerts = []
        
        # Coverage percentage alert
        if audit_result.status == CoverageStatus.CRITICAL:
            alerts.append(CoverageAlert(
                alert_id=f"coverage_critical_{audit_result.audit_id}",
                alert_type="coverage_critical",
                severity=AlertSeverity.CRITICAL,
                exchange=audit_result.exchange,
                symbol=None,
                message=f"CRITICAL: Coverage only {audit_result.coverage_percentage:.1%} - missing {audit_result.missing_coins} coins",
                impact_score=1.0 - audit_result.coverage_percentage,
                timestamp=audit_result.timestamp
            ))
        elif audit_result.status == CoverageStatus.PARTIAL:
            alerts.append(CoverageAlert(
                alert_id=f"coverage_partial_{audit_result.audit_id}",
                alert_type="coverage_partial",
                severity=AlertSeverity.WARNING,
                exchange=audit_result.exchange,
                symbol=None,
                message=f"WARNING: Coverage {audit_result.coverage_percentage:.1%} - missing {audit_result.missing_coins} coins",
                impact_score=1.0 - audit_result.coverage_percentage,
                timestamp=audit_result.timestamp
            ))
        
        # High-impact missing coins
        for gap in audit_result.coverage_gaps[:5]:  # Top 5 missing coins
            if gap.impact_score > 0.3:  # High impact threshold
                alerts.append(CoverageAlert(
                    alert_id=f"missing_coin_{gap.symbol}_{audit_result.audit_id}",
                    alert_type="missing_high_impact_coin",
                    severity=AlertSeverity.CRITICAL if gap.impact_score > 0.7 else AlertSeverity.WARNING,
                    exchange=audit_result.exchange,
                    symbol=gap.symbol,
                    message=f"High-impact coin missing: {gap.symbol} (impact: {gap.impact_score:.2f}, volume: ${gap.volume_24h:,.0f})",
                    impact_score=gap.impact_score,
                    timestamp=audit_result.timestamp
                ))
        
        # New listings alert
        if audit_result.new_listings:
            alerts.append(CoverageAlert(
                alert_id=f"new_listings_{audit_result.audit_id}",
                alert_type="new_listings_detected",
                severity=AlertSeverity.INFO,
                exchange=audit_result.exchange,
                symbol=None,
                message=f"New listings detected: {', '.join(audit_result.new_listings[:5])}{'...' if len(audit_result.new_listings) > 5 else ''}",
                impact_score=len(audit_result.new_listings) / 10,  # Impact based on number of new listings
                timestamp=audit_result.timestamp
            ))
        
        # Store alerts
        self.coverage_alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                self.logger.critical(alert.message, extra={'alert_id': alert.alert_id})
            elif alert.severity == AlertSeverity.WARNING:
                self.logger.warning(alert.message, extra={'alert_id': alert.alert_id})
            else:
                self.logger.info(alert.message, extra={'alert_id': alert.alert_id})
    
    async def _generate_failure_alert(self, exchange: str, error_message: str):
        """Generate alert for audit failure"""
        
        failure_alert = CoverageAlert(
            alert_id=f"audit_failure_{exchange}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            alert_type="audit_failure",
            severity=AlertSeverity.EMERGENCY,
            exchange=exchange,
            symbol=None,
            message=f"Coverage audit failed for {exchange}: {error_message}",
            impact_score=1.0,
            timestamp=datetime.now()
        )
        
        self.coverage_alerts.append(failure_alert)
        
        self.logger.critical(
            f"AUDIT FAILURE: {failure_alert.message}",
            extra={'alert_id': failure_alert.alert_id}
        )
    
    def get_coverage_summary(self, exchange: str = 'kraken') -> Dict[str, Any]:
        """Get comprehensive coverage summary"""
        
        latest_audit = None
        for audit in reversed(self.audit_history):
            if audit.exchange == exchange:
                latest_audit = audit
                break
        
        active_alerts = [a for a in self.coverage_alerts if not a.resolved and a.exchange == exchange]
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'exchange': exchange,
            'latest_audit': {
                'audit_id': latest_audit.audit_id,
                'timestamp': latest_audit.timestamp.isoformat(),
                'coverage_percentage': latest_audit.coverage_percentage,
                'status': latest_audit.status.value,
                'total_live_coins': latest_audit.total_live_coins,
                'analyzed_coins': latest_audit.analyzed_coins,
                'missing_coins': latest_audit.missing_coins,
                'new_listings_count': len(latest_audit.new_listings)
            } if latest_audit else None,
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            'audit_history_count': len([a for a in self.audit_history if a.exchange == exchange]),
            'baseline_coverage_symbols': len(self.baseline_coverage.get(exchange, set()))
        }
        
        return summary
    
    def get_missing_coins_report(self, exchange: str = 'kraken') -> Dict[str, Any]:
        """Get detailed missing coins report"""
        
        latest_audit = None
        for audit in reversed(self.audit_history):
            if audit.exchange == exchange:
                latest_audit = audit
                break
        
        if not latest_audit:
            return {'error': 'No audit data available'}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'exchange': exchange,
            'audit_id': latest_audit.audit_id,
            'missing_coins_count': latest_audit.missing_coins,
            'missing_symbols': latest_audit.missing_coin_symbols,
            'high_impact_gaps': [
                {
                    'symbol': gap.symbol,
                    'impact_score': gap.impact_score,
                    'volume_24h': gap.volume_24h,
                    'gap_duration_hours': gap.gap_duration_hours
                }
                for gap in latest_audit.coverage_gaps[:10]  # Top 10
            ],
            'coverage_percentage': latest_audit.coverage_percentage,
            'recommended_actions': self._get_coverage_recommendations(latest_audit)
        }
    
    def _get_coverage_recommendations(self, audit_result: CoverageAuditResult) -> List[str]:
        """Get actionable recommendations to improve coverage"""
        
        recommendations = []
        
        if audit_result.status == CoverageStatus.CRITICAL:
            recommendations.append("URGENT: Add missing high-volume coins to analysis pipeline")
            recommendations.append("Review data collection processes for systematic gaps")
        
        if audit_result.missing_coins > 10:
            recommendations.append("Implement batch addition process for multiple missing coins")
        
        if audit_result.new_listings:
            recommendations.append("Set up automated new listing detection and addition")
        
        # Specific coin recommendations
        high_impact_gaps = [g for g in audit_result.coverage_gaps if g.impact_score > 0.5]
        if high_impact_gaps:
            top_gaps = high_impact_gaps[:3]
            recommendations.append(
                f"Priority coins to add: {', '.join(gap.symbol for gap in top_gaps)}"
            )
        
        if not recommendations:
            recommendations.append("Coverage is complete - maintain current monitoring")
        
        return recommendations

# Global instance
_coverage_auditor = None

def get_coverage_auditor() -> CoverageAuditor:
    """Get global coverage auditor instance"""
    global _coverage_auditor
    if _coverage_auditor is None:
        _coverage_auditor = CoverageAuditor()
    return _coverage_auditor