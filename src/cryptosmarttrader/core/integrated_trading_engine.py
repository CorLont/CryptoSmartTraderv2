#!/usr/bin/env python3
"""
Integrated Trading Engine - Verbindt alle componenten voor live trading
Gebruikt DataFlowOrchestrator voor strakke data discipline en component integratie
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path
import json

from ..core.structured_logger import get_logger
from ..core.data_flow_orchestrator import DataFlowOrchestrator, DataQualityGate, DataFlowState
# from ..adapters.kraken_data_adapter import KrakenDataAdapter  # Would be imported when available
from ..observability.unified_metrics import UnifiedMetrics


class TradingEngineState(Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY = "emergency"


@dataclass
class TradingSession:
    """Trading session information."""
    session_id: str
    start_time: datetime
    symbols_tracked: List[str]
    signals_processed: int = 0
    orders_executed: int = 0
    orders_rejected: int = 0
    total_pnl: float = 0.0
    state: TradingEngineState = TradingEngineState.STOPPED


class IntegratedTradingEngine:
    """
    Integrated trading engine combining all components for live trading.
    
    Features:
    - Strict data validation through DataFlowOrchestrator
    - Real-time signal processing
    - Risk-aware execution
    - Regime-adaptive positioning
    - Kelly-optimal sizing
    - Comprehensive monitoring
    """

    def __init__(self, 
                 symbols: List[str],
                 data_quality_gate: DataQualityGate = DataQualityGate.STRICT,
                 max_concurrent_signals: int = 10):
        """Initialize integrated trading engine."""
        
        self.logger = get_logger("integrated_trading_engine")
        self.symbols = symbols
        self.max_concurrent_signals = max_concurrent_signals
        
        # Core orchestrator
        self.data_orchestrator = DataFlowOrchestrator(
            strict_mode=True,
            quality_gate=data_quality_gate
        )
        
        # Data adapter (placeholder for now)
        self.data_adapter = None  # Would initialize KrakenDataAdapter when available
        
        # Metrics and monitoring
        self.metrics = UnifiedMetrics("integrated_trading_engine")
        
        # Engine state
        self.current_state = TradingEngineState.STOPPED
        self.current_session: Optional[TradingSession] = None
        
        # Signal processing
        self.signal_queue = asyncio.Queue(maxsize=100)
        self.processing_semaphore = asyncio.Semaphore(max_concurrent_signals)
        
        # Performance tracking
        self.performance_stats = {
            'signals_per_minute': 0.0,
            'average_processing_time_ms': 0.0,
            'pipeline_success_rate': 0.0,
            'execution_success_rate': 0.0,
            'data_quality_score': 0.0
        }
        
        # Threading
        self._shutdown_event = asyncio.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._signal_processor_task: Optional[asyncio.Task] = None
        
        # Persistence
        self.data_path = Path("data/trading_engine")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("IntegratedTradingEngine initialized",
                        symbols=len(symbols),
                        quality_gate=data_quality_gate.value,
                        max_concurrent=max_concurrent_signals)

    async def start_trading_session(self, session_id: Optional[str] = None) -> str:
        """Start new trading session."""
        
        if self.current_state != TradingEngineState.STOPPED:
            raise RuntimeError(f"Cannot start session in state: {self.current_state.value}")
        
        self.current_state = TradingEngineState.STARTING
        
        # Create session
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = TradingSession(
            session_id=session_id,
            start_time=datetime.now(),
            symbols_tracked=self.symbols.copy()
        )
        
        try:
            # Start background tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._signal_processor_task = asyncio.create_task(self._signal_processor_loop())
            
            # Verify data flow readiness
            flow_status = self.data_orchestrator.get_flow_status()
            if flow_status['flow_state'] not in ['active', 'degraded']:
                self.logger.warning("Data flow not optimal for trading", 
                                  flow_state=flow_status['flow_state'])
            
            self.current_state = TradingEngineState.ACTIVE
            
            self.logger.info("Trading session started",
                           session_id=session_id,
                           symbols=len(self.symbols),
                           flow_state=flow_status['flow_state'])
            
            # Record session start
            self.metrics.record_signal("system", 1.0)  # Session start signal
            
            return session_id
            
        except Exception as e:
            self.current_state = TradingEngineState.STOPPED
            self.logger.error("Failed to start trading session", error=str(e))
            raise

    async def stop_trading_session(self) -> Dict[str, Any]:
        """Stop current trading session."""
        
        if self.current_state == TradingEngineState.STOPPED:
            return {'message': 'No active session to stop'}
        
        self.logger.info("Stopping trading session",
                        session_id=self.current_session.session_id if self.current_session else None)
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for tasks to complete
        if self._monitoring_task:
            await self._monitoring_task
        if self._signal_processor_task:
            await self._signal_processor_task
        
        # Finalize session
        session_summary = {}
        if self.current_session:
            session_duration = datetime.now() - self.current_session.start_time
            session_summary = {
                'session_id': self.current_session.session_id,
                'duration_minutes': session_duration.total_seconds() / 60,
                'signals_processed': self.current_session.signals_processed,
                'orders_executed': self.current_session.orders_executed,
                'orders_rejected': self.current_session.orders_rejected,
                'total_pnl': self.current_session.total_pnl,
                'performance_stats': self.performance_stats.copy()
            }
        
        # Reset state
        self.current_state = TradingEngineState.STOPPED
        self.current_session = None
        self._shutdown_event.clear()
        
        self.logger.info("Trading session stopped", session_summary=session_summary)
        return session_summary

    async def process_market_signal(self, symbol: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual market signal through complete pipeline."""
        
        if self.current_state != TradingEngineState.ACTIVE:
            return {
                'success': False,
                'reason': f'Engine not active: {self.current_state.value}'
            }
        
        signal_start = time.time()
        
        try:
            # Get current market data
            market_data = await self._get_market_data(symbol)
            if not market_data:
                return {
                    'success': False,
                    'reason': 'Failed to get market data'
                }
            
            # Process through data flow orchestrator
            async with self.processing_semaphore:
                pipeline_result = await self.data_orchestrator.process_market_signal(
                    symbol=symbol,
                    market_data=market_data,
                    signal_data=signal_data
                )
            
            processing_time = (time.time() - signal_start) * 1000
            
            # Update session stats
            if self.current_session:
                self.current_session.signals_processed += 1
                
                if pipeline_result['pipeline_success']:
                    # Would execute order here in real trading
                    self.current_session.orders_executed += 1
                    self.logger.info("Signal approved for execution",
                                   symbol=symbol,
                                   position_size=pipeline_result.get('final_position_size', 0),
                                   processing_time_ms=processing_time)
                else:
                    self.current_session.orders_rejected += 1
                    self.logger.info("Signal rejected",
                                   symbol=symbol,
                                   rejection_reasons=pipeline_result.get('components_failed', []),
                                   processing_time_ms=processing_time)
            
            # Update performance stats
            self._update_performance_stats(processing_time, pipeline_result)
            
            # Record metrics
            if pipeline_result['pipeline_success']:
                self.metrics.record_order('approved', symbol, 'signal_processed')
            else:
                self.metrics.record_order('rejected', symbol, 'signal_processed')
            
            return {
                'success': pipeline_result['pipeline_success'],
                'processing_time_ms': processing_time,
                'pipeline_result': pipeline_result,
                'execution_recommendation': pipeline_result.get('execution_recommendation', 'REJECT')
            }
            
        except Exception as e:
            processing_time = (time.time() - signal_start) * 1000
            self.logger.error("Signal processing failed",
                            symbol=symbol,
                            error=str(e),
                            processing_time_ms=processing_time)
            
            return {
                'success': False,
                'reason': f'Processing error: {str(e)}',
                'processing_time_ms': processing_time
            }

    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for symbol."""
        
        try:
            # Get market data from adapter (mock for now)
            if self.data_adapter:
                ticker_data = await self.data_adapter.get_ticker_data(symbol)
                orderbook_data = await self.data_adapter.get_orderbook_data(symbol)
            else:
                # Mock data for demonstration
                ticker_data = {
                    'last': 45000.0,
                    'volume': 1000000,
                    'change': 0.02,
                    'spread_bps': 25,
                    'volatility': 0.025
                }
                orderbook_data = {
                    'depth': 15000,
                    'spread': 0.0025
                }
            
            if not ticker_data:
                return None
            
            # Combine data
            market_data = {
                'symbol': symbol,
                'price': ticker_data.get('last', 0),
                'volume_24h': ticker_data.get('volume', 0),
                'price_change_24h': ticker_data.get('change', 0),
                'spread_bps': ticker_data.get('spread_bps', 50),
                'volatility_24h': ticker_data.get('volatility', 0.02),
                'timestamp': datetime.now(),
                'portfolio_value': 100000  # Would come from portfolio manager
            }
            
            # Add orderbook data if available
            if orderbook_data:
                market_data['orderbook_depth'] = orderbook_data.get('depth', 10000)
                market_data['bid_ask_spread'] = orderbook_data.get('spread', 0.001)
            
            return market_data
            
        except Exception as e:
            self.logger.error("Failed to get market data", symbol=symbol, error=str(e))
            return None

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        
        self.logger.info("Starting monitoring loop")
        
        try:
            while not self._shutdown_event.is_set():
                # Monitor data flow health
                flow_status = self.data_orchestrator.get_flow_status()
                
                # Check for degraded state
                if flow_status['flow_state'] == 'emergency':
                    self.logger.warning("Data flow in emergency state - pausing trading")
                    self.current_state = TradingEngineState.EMERGENCY
                elif flow_status['flow_state'] == 'degraded':
                    self.logger.warning("Data flow degraded - monitoring closely")
                
                # Update performance metrics
                self._calculate_performance_metrics()
                
                # Record monitoring metrics
                self.metrics.update_drawdown(0.0)  # Would come from portfolio
                
                # Sleep before next check
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("Monitoring loop cancelled")
        except Exception as e:
            self.logger.error("Monitoring loop error", error=str(e))

    async def _signal_processor_loop(self) -> None:
        """Background signal processor loop."""
        
        self.logger.info("Starting signal processor loop")
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for signals (with timeout)
                    signal_data = await asyncio.wait_for(
                        self.signal_queue.get(), 
                        timeout=5.0
                    )
                    
                    # Process signal
                    result = await self.process_market_signal(
                        signal_data['symbol'],
                        signal_data['signal']
                    )
                    
                    # Mark task done
                    self.signal_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No signals to process - continue monitoring
                    continue
                    
        except asyncio.CancelledError:
            self.logger.info("Signal processor loop cancelled")
        except Exception as e:
            self.logger.error("Signal processor loop error", error=str(e))

    def _update_performance_stats(self, processing_time_ms: float, pipeline_result: Dict[str, Any]) -> None:
        """Update performance statistics."""
        
        # Update processing time (exponential moving average)
        alpha = 0.1
        self.performance_stats['average_processing_time_ms'] = (
            alpha * processing_time_ms + 
            (1 - alpha) * self.performance_stats['average_processing_time_ms']
        )
        
        # Update success rate
        if pipeline_result['pipeline_success']:
            success = 1.0
        else:
            success = 0.0
            
        self.performance_stats['pipeline_success_rate'] = (
            alpha * success + 
            (1 - alpha) * self.performance_stats['pipeline_success_rate']
        )
        
        # Update data quality score
        quality_score = pipeline_result.get('quality_metrics', {}).get('data_quality_score', 0.5)
        self.performance_stats['data_quality_score'] = (
            alpha * quality_score + 
            (1 - alpha) * self.performance_stats['data_quality_score']
        )

    def _calculate_performance_metrics(self) -> None:
        """Calculate overall performance metrics."""
        
        if not self.current_session:
            return
        
        # Calculate signals per minute
        session_duration = datetime.now() - self.current_session.start_time
        minutes = session_duration.total_seconds() / 60
        
        if minutes > 0:
            self.performance_stats['signals_per_minute'] = self.current_session.signals_processed / minutes
        
        # Calculate execution success rate
        total_orders = self.current_session.orders_executed + self.current_session.orders_rejected
        if total_orders > 0:
            self.performance_stats['execution_success_rate'] = self.current_session.orders_executed / total_orders

    async def queue_signal(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Queue signal for processing."""
        
        try:
            await self.signal_queue.put({
                'symbol': symbol,
                'signal': signal_data,
                'timestamp': datetime.now()
            })
            return True
            
        except asyncio.QueueFull:
            self.logger.warning("Signal queue full - dropping signal", symbol=symbol)
            return False

    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        
        flow_status = self.data_orchestrator.get_flow_status()
        
        status = {
            'engine_state': self.current_state.value,
            'data_flow_state': flow_status['flow_state'],
            'performance_stats': self.performance_stats.copy(),
            'queue_size': self.signal_queue.qsize(),
            'symbols_tracked': len(self.symbols)
        }
        
        if self.current_session:
            session_duration = datetime.now() - self.current_session.start_time
            status['current_session'] = {
                'session_id': self.current_session.session_id,
                'duration_minutes': round(session_duration.total_seconds() / 60, 2),
                'signals_processed': self.current_session.signals_processed,
                'orders_executed': self.current_session.orders_executed,
                'orders_rejected': self.current_session.orders_rejected
            }
        
        return status

    def get_detailed_statistics(self) -> Dict[str, Any]:
        """Get detailed engine and pipeline statistics."""
        
        flow_stats = self.data_orchestrator.get_pipeline_statistics()
        flow_status = self.data_orchestrator.get_flow_status()
        
        return {
            'engine_performance': self.performance_stats,
            'data_flow_statistics': flow_stats,
            'data_flow_status': flow_status,
            'component_performance': flow_status.get('performance', {}),
            'session_summary': {
                'current_session': self.current_session.__dict__ if self.current_session else None,
                'engine_state': self.current_state.value
            }
        }