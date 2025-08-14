#!/usr/bin/env python3
"""
WHALE EXECUTION INTEGRATION
Directe koppeling tussen whale detection en execution gates voor real-time protective actions

Features:
- Real-time whale alert processing
- Automatic protective order generation  
- Position limit adjustments
- Emergency halt mechanisms
- Comprehensive audit trail
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from .enterprise_whale_detection import WhaleAlert, enterprise_whale_detector
from ..core.mandatory_execution_gateway import MandatoryExecutionGateway, UniversalOrderRequest, GatewayResult
from ..risk.central_risk_guard import CentralRiskGuard, RiskDecision
from ..observability.centralized_metrics import centralized_metrics


logger = logging.getLogger(__name__)


@dataclass
class ProtectiveAction:
    """Protective action taken in response to whale activity"""
    action_id: str
    whale_alert_id: str
    action_type: str  # reduce_position, halt_trading, emergency_exit
    symbol: str
    timestamp: datetime
    
    # Action details
    target_reduction: float  # percentage
    executed_reduction: float  # actual percentage executed
    orders_created: int
    orders_successful: int
    
    # Results
    success: bool
    total_value_protected: Decimal
    execution_time_ms: float
    error_message: Optional[str] = None


class WhaleExecutionIntegrator:
    """Integration layer tussen whale detection en execution system"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WhaleExecutionIntegrator")
        self.execution_gateway = MandatoryExecutionGateway()
        self.risk_guard = CentralRiskGuard()
        
        self.protective_actions: Dict[str, ProtectiveAction] = {}
        self.symbol_restrictions: Dict[str, Dict] = {}
        
        # Configuration
        self.max_auto_reduction = 0.3  # Maximum 30% automatic reduction
        self.emergency_halt_threshold = 20000000  # $20M triggers emergency halt
        
    async def start_whale_integration(self):
        """Start whale detection integration service"""
        
        self.logger.info("Starting whale execution integration service")
        
        # Start whale detector
        detector_task = asyncio.create_task(enterprise_whale_detector.start_continuous_monitoring())
        
        # Start integration processor
        integration_task = asyncio.create_task(self._process_whale_alerts())
        
        await asyncio.gather(detector_task, integration_task)
        
    async def _process_whale_alerts(self):
        """Process whale alerts for execution integration"""
        
        while True:
            try:
                # Check for new critical alerts
                critical_alerts = [
                    alert for alert in enterprise_whale_detector.active_alerts.values()
                    if alert.severity in ['critical', 'high'] and 
                    alert.alert_id not in self.protective_actions
                ]
                
                for alert in critical_alerts:
                    await self._handle_critical_whale_alert(alert)
                    
                # Check for alert expiration and cleanup
                await self._cleanup_expired_actions()
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in whale alert processing: {e}")
                await asyncio.sleep(60)
                
    async def _handle_critical_whale_alert(self, alert: WhaleAlert):
        """Handle critical whale alert met protective actions"""
        
        self.logger.warning(f"Processing critical whale alert {alert.alert_id} for {alert.symbol}")
        
        start_time = datetime.utcnow()
        
        try:
            # Determine protective action needed
            action_type = self._determine_action_type(alert)
            
            if action_type == 'emergency_halt':
                protective_action = await self._execute_emergency_halt(alert, start_time)
            elif action_type == 'reduce_position':
                protective_action = await self._execute_position_reduction(alert, start_time)
            elif action_type == 'halt_new_orders':
                protective_action = await self._execute_trading_halt(alert, start_time)
            else:
                self.logger.info(f"No protective action needed for alert {alert.alert_id}")
                return
                
            # Store action
            self.protective_actions[alert.alert_id] = protective_action
            
            # Log metrics
            centralized_metrics.whale_protective_action.labels(
                symbol=alert.symbol,
                action_type=action_type,
                success=str(protective_action.success)
            ).inc()
            
            if protective_action.success:
                centralized_metrics.whale_protection_value.labels(
                    symbol=alert.symbol
                ).observe(float(protective_action.total_value_protected))
                
        except Exception as e:
            self.logger.error(f"Error handling whale alert {alert.alert_id}: {e}")
            
            # Create failed action record
            failed_action = ProtectiveAction(
                action_id=f"failed_{alert.alert_id}",
                whale_alert_id=alert.alert_id,
                action_type="failed",
                symbol=alert.symbol,
                timestamp=start_time,
                target_reduction=0.0,
                executed_reduction=0.0,
                orders_created=0,
                orders_successful=0,
                success=False,
                total_value_protected=Decimal('0'),
                execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )
            
            self.protective_actions[alert.alert_id] = failed_action
            
    def _determine_action_type(self, alert: WhaleAlert) -> str:
        """Determine appropriate protective action based on alert"""
        
        # Emergency situations
        if (alert.total_value_usd > self.emergency_halt_threshold or 
            alert.severity == 'critical' and alert.estimated_price_impact > 0.05):
            return 'emergency_halt'
            
        # Large selling pressure - reduce positions
        if (alert.alert_type == 'massive_sell' and 
            alert.recommended_action == 'reduce_exposure'):
            return 'reduce_position'
            
        # Coordinated activity - halt new orders temporarily
        if alert.alert_type == 'coordinated_action':
            return 'halt_new_orders'
            
        return 'monitor_only'
        
    async def _execute_emergency_halt(self, alert: WhaleAlert, start_time: datetime) -> ProtectiveAction:
        """Execute emergency halt procedures"""
        
        self.logger.critical(f"EMERGENCY HALT triggered for {alert.symbol} due to whale alert {alert.alert_id}")
        
        action_id = f"emergency_{alert.alert_id}"
        
        try:
            # 1. Halt all new orders for this symbol
            await self._add_symbol_restriction(alert.symbol, 'emergency_halt', 3600)  # 1 hour
            
            # 2. Get current position for symbol
            current_position = await self._get_current_position(alert.symbol)
            
            if current_position > 0:
                # 3. Create emergency exit orders
                exit_orders = await self._create_emergency_exit_orders(alert.symbol, current_position)
                
                successful_orders = 0
                total_executed = Decimal('0')
                
                for order in exit_orders:
                    result = await self.execution_gateway.process_order_request(order)
                    if result.approved:
                        successful_orders += 1
                        total_executed += Decimal(str(result.approved_size))
                        
                executed_reduction = float(total_executed / current_position) if current_position > 0 else 0.0
                
            else:
                exit_orders = []
                successful_orders = 0
                executed_reduction = 0.0
                total_executed = Decimal('0')
                
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProtectiveAction(
                action_id=action_id,
                whale_alert_id=alert.alert_id,
                action_type='emergency_halt',
                symbol=alert.symbol,
                timestamp=start_time,
                target_reduction=1.0,  # 100% exit target
                executed_reduction=executed_reduction,
                orders_created=len(exit_orders),
                orders_successful=successful_orders,
                success=successful_orders > 0 or current_position == 0,
                total_value_protected=total_executed * Decimal('2000'),  # Mock price
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in emergency halt execution: {e}")
            raise
            
    async def _execute_position_reduction(self, alert: WhaleAlert, start_time: datetime) -> ProtectiveAction:
        """Execute gradual position reduction"""
        
        self.logger.warning(f"Executing position reduction for {alert.symbol} (target: {alert.max_position_reduction*100:.1f}%)")
        
        action_id = f"reduce_{alert.alert_id}"
        
        try:
            # Get current position
            current_position = await self._get_current_position(alert.symbol)
            
            if current_position <= 0:
                return ProtectiveAction(
                    action_id=action_id,
                    whale_alert_id=alert.alert_id,
                    action_type='reduce_position',
                    symbol=alert.symbol,
                    timestamp=start_time,
                    target_reduction=alert.max_position_reduction,
                    executed_reduction=0.0,
                    orders_created=0,
                    orders_successful=0,
                    success=True,  # No position to reduce
                    total_value_protected=Decimal('0'),
                    execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
                
            # Calculate reduction amount
            target_reduction = min(alert.max_position_reduction, self.max_auto_reduction)
            reduction_size = current_position * target_reduction
            
            # Create reduction orders (split into smaller chunks)
            reduction_orders = await self._create_reduction_orders(alert.symbol, reduction_size)
            
            successful_orders = 0
            total_executed = Decimal('0')
            
            for order in reduction_orders:
                result = await self.execution_gateway.process_order_request(order)
                if result.approved:
                    successful_orders += 1
                    total_executed += Decimal(str(result.approved_size))
                    
                # Small delay between orders to avoid market impact
                await asyncio.sleep(1)
                
            executed_reduction = float(total_executed / current_position) if current_position > 0 else 0.0
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProtectiveAction(
                action_id=action_id,
                whale_alert_id=alert.alert_id,
                action_type='reduce_position',
                symbol=alert.symbol,
                timestamp=start_time,
                target_reduction=target_reduction,
                executed_reduction=executed_reduction,
                orders_created=len(reduction_orders),
                orders_successful=successful_orders,
                success=executed_reduction >= target_reduction * 0.8,  # 80% success threshold
                total_value_protected=total_executed * Decimal('2000'),  # Mock price
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in position reduction execution: {e}")
            raise
            
    async def _execute_trading_halt(self, alert: WhaleAlert, start_time: datetime) -> ProtectiveAction:
        """Execute temporary trading halt for new orders"""
        
        self.logger.warning(f"Executing trading halt for {alert.symbol} (duration: {alert.suggested_timeframe} minutes)")
        
        action_id = f"halt_{alert.alert_id}"
        
        try:
            # Add temporary restriction
            duration_seconds = alert.suggested_timeframe * 60 if alert.suggested_timeframe > 0 else 1800  # 30 min default
            await self._add_symbol_restriction(alert.symbol, 'trading_halt', duration_seconds)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ProtectiveAction(
                action_id=action_id,
                whale_alert_id=alert.alert_id,
                action_type='halt_new_orders',
                symbol=alert.symbol,
                timestamp=start_time,
                target_reduction=0.0,
                executed_reduction=0.0,
                orders_created=0,
                orders_successful=0,
                success=True,
                total_value_protected=Decimal('0'),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in trading halt execution: {e}")
            raise
            
    async def _get_current_position(self, symbol: str) -> float:
        """Get current position size for symbol"""
        
        # In production: integrate met portfolio manager
        # Mock implementation
        mock_positions = {
            'ETH': 10.5,
            'BTC': 0.25,
            'USDT': 50000,
            'USDC': 25000
        }
        
        return mock_positions.get(symbol, 0.0)
        
    async def _create_emergency_exit_orders(self, symbol: str, position_size: float) -> List[UniversalOrderRequest]:
        """Create emergency exit orders voor complete position exit"""
        
        orders = []
        
        # Split large positions into multiple orders
        chunk_size = min(position_size / 3, position_size)  # Max 3 orders
        remaining = position_size
        
        order_count = 0
        while remaining > 0.001 and order_count < 5:  # Max 5 orders
            order_size = min(chunk_size, remaining)
            
            order = UniversalOrderRequest(
                symbol=symbol,
                side='sell',
                size=order_size,
                order_type='market',
                strategy_id='whale_emergency_exit',
                max_slippage_bps=100.0,  # Allow higher slippage in emergency
                source_module='whale_execution_integration',
                source_function='emergency_exit'
            )
            
            orders.append(order)
            remaining -= order_size
            order_count += 1
            
        return orders
        
    async def _create_reduction_orders(self, symbol: str, reduction_size: float) -> List[UniversalOrderRequest]:
        """Create position reduction orders"""
        
        orders = []
        
        # Split reduction into smaller orders to minimize market impact
        max_chunks = 5
        chunk_size = reduction_size / max_chunks
        
        for i in range(max_chunks):
            if chunk_size > 0.001:  # Minimum order size
                order = UniversalOrderRequest(
                    symbol=symbol,
                    side='sell',
                    size=chunk_size,
                    order_type='market',
                    strategy_id='whale_position_reduction',
                    max_slippage_bps=50.0,  # Moderate slippage allowance
                    source_module='whale_execution_integration',
                    source_function='position_reduction'
                )
                
                orders.append(order)
                
        return orders
        
    async def _add_symbol_restriction(self, symbol: str, restriction_type: str, duration_seconds: int):
        """Add temporary trading restriction for symbol"""
        
        expiry_time = datetime.utcnow() + timedelta(seconds=duration_seconds)
        
        self.symbol_restrictions[symbol] = {
            'type': restriction_type,
            'expiry': expiry_time,
            'reason': 'whale_activity_protection'
        }
        
        self.logger.info(f"Added {restriction_type} restriction for {symbol} until {expiry_time}")
        
        # Integrate met execution gateway to enforce restrictions
        # In production: zou dit direct RiskGuard rules updaten
        centralized_metrics.whale_trading_restriction.labels(
            symbol=symbol,
            restriction_type=restriction_type
        ).inc()
        
    async def _cleanup_expired_actions(self):
        """Cleanup expired protective actions and restrictions"""
        
        current_time = datetime.utcnow()
        
        # Cleanup expired restrictions
        expired_symbols = []
        for symbol, restriction in self.symbol_restrictions.items():
            if current_time > restriction['expiry']:
                expired_symbols.append(symbol)
                
        for symbol in expired_symbols:
            self.logger.info(f"Removing expired restriction for {symbol}")
            del self.symbol_restrictions[symbol]
            
            centralized_metrics.whale_restriction_expired.labels(
                symbol=symbol
            ).inc()
            
        # Cleanup old protective actions (keep last 50)
        if len(self.protective_actions) > 50:
            sorted_actions = sorted(
                self.protective_actions.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            
            # Keep only latest 50
            self.protective_actions = dict(sorted_actions[:50])
            
    def is_trading_restricted(self, symbol: str) -> bool:
        """Check if trading is currently restricted for symbol"""
        
        if symbol not in self.symbol_restrictions:
            return False
            
        restriction = self.symbol_restrictions[symbol]
        
        if datetime.utcnow() > restriction['expiry']:
            # Restriction expired
            del self.symbol_restrictions[symbol]
            return False
            
        return True
        
    def get_restriction_info(self, symbol: str) -> Optional[Dict]:
        """Get current restriction info for symbol"""
        
        if not self.is_trading_restricted(symbol):
            return None
            
        return self.symbol_restrictions[symbol].copy()
        
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        
        active_restrictions = len(self.symbol_restrictions)
        total_actions = len(self.protective_actions)
        successful_actions = len([a for a in self.protective_actions.values() if a.success])
        
        return {
            "status": "operational",
            "whale_detector_connected": True,
            "execution_gateway_connected": True,
            "active_restrictions": active_restrictions,
            "restricted_symbols": list(self.symbol_restrictions.keys()),
            "total_protective_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": successful_actions / total_actions if total_actions > 0 else 0.0,
            "emergency_halt_threshold": self.emergency_halt_threshold,
            "max_auto_reduction": self.max_auto_reduction
        }


# Global singleton instance
whale_execution_integrator = WhaleExecutionIntegrator()