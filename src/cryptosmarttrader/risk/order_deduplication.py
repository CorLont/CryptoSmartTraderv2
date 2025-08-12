"""
Order Deduplication System

Ensures order idempotency and prevents duplicate executions
through comprehensive order state tracking and retry management.
"""

import uuid
import hashlib
import threading
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class OrderState(Enum):
    """Order processing states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    ERROR = "error"
    EXPIRED = "expired"


class DuplicateReason(Enum):
    """Reasons for duplicate detection"""
    IDENTICAL_PARAMS = "identical_params"
    RETRY_DETECTED = "retry_detected"
    NETWORK_TIMEOUT = "network_timeout"
    CLIENT_ID_REUSE = "client_id_reuse"
    RAPID_SUBMISSION = "rapid_submission"


@dataclass
class DuplicateCheck:
    """Duplicate check result"""
    is_duplicate: bool
    reason: Optional[DuplicateReason] = None
    original_order_id: Optional[str] = None
    time_since_original: Optional[timedelta] = None
    recommendation: str = ""


@dataclass
class OrderRecord:
    """Complete order record for deduplication"""
    client_order_id: str
    order_hash: str
    
    # Order parameters
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    
    # State tracking
    state: OrderState = OrderState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Exchange interaction
    exchange_order_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Retry tracking
    retry_count: int = 0
    retry_parent_id: Optional[str] = None
    retry_children: Set[str] = field(default_factory=set)
    
    # Network tracking
    request_timeout: bool = False
    response_received: bool = False
    network_error_count: int = 0
    
    # Results
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    fees: float = 0.0
    
    @property
    def is_terminal_state(self) -> bool:
        """Check if order is in terminal state"""
        return self.state in [
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED
        ]
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.state in [
            OrderState.PENDING,
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED
        ]


class OrderDeduplicator:
    """
    Enterprise order deduplication system
    """
    
    def __init__(self,
                 duplicate_window_seconds: int = 300,  # 5 minutes
                 max_retries: int = 3,
                 rapid_submission_threshold_ms: int = 100):
        
        self.duplicate_window = timedelta(seconds=duplicate_window_seconds)
        self.max_retries = max_retries
        self.rapid_threshold = timedelta(milliseconds=rapid_submission_threshold_ms)
        
        # Order tracking
        self.orders: Dict[str, OrderRecord] = {}  # client_order_id -> OrderRecord
        self.order_hashes: Dict[str, List[str]] = defaultdict(list)  # hash -> [client_order_ids]
        self.symbol_orders: Dict[str, List[str]] = defaultdict(list)  # symbol -> [client_order_ids]
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.duplicate_stats = {
            "total_orders": 0,
            "duplicates_detected": 0,
            "duplicates_prevented": 0,
            "retries_processed": 0,
            "network_timeouts": 0
        }
        
        # Cleanup configuration
        self.cleanup_interval_hours = 24
        self.retention_days = 7
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def generate_client_order_id(self, prefix: str = "CST") -> str:
        """Generate unique client order ID"""
        timestamp = int(datetime.now().timestamp() * 1000000)  # microseconds
        unique_id = str(uuid.uuid4()).split('-')[0]
        return f"{prefix}_{timestamp}_{unique_id}"
    
    def check_duplicate(self, 
                       symbol: str,
                       side: str,
                       order_type: str,
                       quantity: float,
                       price: Optional[float] = None,
                       client_order_id: Optional[str] = None) -> DuplicateCheck:
        """Check if order is a duplicate"""
        
        with self._lock:
            try:
                # Generate order hash for comparison
                order_hash = self._generate_order_hash(symbol, side, order_type, quantity, price)
                
                # Check for existing client order ID
                if client_order_id and client_order_id in self.orders:
                    existing_order = self.orders[client_order_id]
                    
                    if existing_order.is_active:
                        return DuplicateCheck(
                            is_duplicate=True,
                            reason=DuplicateReason.CLIENT_ID_REUSE,
                            original_order_id=client_order_id,
                            recommendation="Client order ID already in use for active order"
                        )
                
                # Check for identical order parameters
                recent_orders = self._get_recent_orders_by_hash(order_hash)
                
                for existing_order_id in recent_orders:
                    existing_order = self.orders.get(existing_order_id)
                    if not existing_order:
                        continue
                    
                    time_diff = datetime.now() - existing_order.created_at
                    
                    # Check if within duplicate window
                    if time_diff <= self.duplicate_window:
                        
                        # Check for rapid submission
                        if time_diff <= self.rapid_threshold:
                            return DuplicateCheck(
                                is_duplicate=True,
                                reason=DuplicateReason.RAPID_SUBMISSION,
                                original_order_id=existing_order_id,
                                time_since_original=time_diff,
                                recommendation=f"Rapid submission detected - {time_diff.total_seconds()*1000:.0f}ms since last order"
                            )
                        
                        # Check if existing order is still active
                        if existing_order.is_active:
                            return DuplicateCheck(
                                is_duplicate=True,
                                reason=DuplicateReason.IDENTICAL_PARAMS,
                                original_order_id=existing_order_id,
                                time_since_original=time_diff,
                                recommendation="Identical order parameters with active order"
                            )
                        
                        # Check for network timeout retry pattern
                        if existing_order.request_timeout and not existing_order.response_received:
                            # This might be a legitimate retry after timeout
                            if existing_order.retry_count < self.max_retries:
                                return DuplicateCheck(
                                    is_duplicate=False,  # Allow retry
                                    reason=DuplicateReason.NETWORK_TIMEOUT,
                                    original_order_id=existing_order_id,
                                    time_since_original=time_diff,
                                    recommendation="Potential legitimate retry after network timeout"
                                )
                            else:
                                return DuplicateCheck(
                                    is_duplicate=True,
                                    reason=DuplicateReason.RETRY_DETECTED,
                                    original_order_id=existing_order_id,
                                    time_since_original=time_diff,
                                    recommendation="Maximum retries exceeded"
                                )
                
                # No duplicate detected
                return DuplicateCheck(
                    is_duplicate=False,
                    recommendation="Order cleared for submission"
                )
                
            except Exception as e:
                logger.error(f"Duplicate check failed: {e}")
                # Fail safe - assume not duplicate to avoid blocking valid orders
                return DuplicateCheck(
                    is_duplicate=False,
                    recommendation=f"Duplicate check error - allowing order: {e}"
                )
    
    def register_order(self,
                      client_order_id: str,
                      symbol: str,
                      side: str,
                      order_type: str,
                      quantity: float,
                      price: Optional[float] = None,
                      is_retry: bool = False,
                      parent_order_id: Optional[str] = None) -> OrderRecord:
        """Register new order for tracking"""
        
        with self._lock:
            try:
                order_hash = self._generate_order_hash(symbol, side, order_type, quantity, price)
                
                # Create order record
                order_record = OrderRecord(
                    client_order_id=client_order_id,
                    order_hash=order_hash,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    retry_parent_id=parent_order_id if is_retry else None
                )
                
                # Handle retry logic
                if is_retry and parent_order_id:
                    parent_order = self.orders.get(parent_order_id)
                    if parent_order:
                        parent_order.retry_children.add(client_order_id)
                        order_record.retry_count = parent_order.retry_count + 1
                
                # Store order
                self.orders[client_order_id] = order_record
                self.order_hashes[order_hash].append(client_order_id)
                self.symbol_orders[symbol].append(client_order_id)
                
                # Update statistics
                self.duplicate_stats["total_orders"] += 1
                if is_retry:
                    self.duplicate_stats["retries_processed"] += 1
                
                logger.debug(f"Registered order: {client_order_id} ({symbol} {side} {quantity})")
                
                return order_record
                
            except Exception as e:
                logger.error(f"Order registration failed: {e}")
                raise
    
    def update_order_state(self,
                          client_order_id: str,
                          new_state: OrderState,
                          exchange_order_id: Optional[str] = None,
                          filled_quantity: Optional[float] = None,
                          average_price: Optional[float] = None,
                          network_timeout: bool = False) -> bool:
        """Update order state"""
        
        with self._lock:
            try:
                order = self.orders.get(client_order_id)
                if not order:
                    logger.warning(f"Order not found for update: {client_order_id}")
                    return False
                
                # Update state
                old_state = order.state
                order.state = new_state
                order.last_updated = datetime.now()
                
                # Update specific fields
                if exchange_order_id:
                    order.exchange_order_id = exchange_order_id
                
                if filled_quantity is not None:
                    order.filled_quantity = filled_quantity
                
                if average_price is not None:
                    order.average_price = average_price
                
                # Handle network timeout
                if network_timeout:
                    order.request_timeout = True
                    order.network_error_count += 1
                    self.duplicate_stats["network_timeouts"] += 1
                
                # Handle state transitions
                if new_state == OrderState.SUBMITTED and old_state == OrderState.PENDING:
                    order.submitted_at = datetime.now()
                elif new_state == OrderState.ACKNOWLEDGED and old_state == OrderState.SUBMITTED:
                    order.acknowledged_at = datetime.now()
                    order.response_received = True
                
                logger.debug(f"Updated order {client_order_id}: {old_state.value} -> {new_state.value}")
                
                return True
                
            except Exception as e:
                logger.error(f"Order state update failed: {e}")
                return False
    
    def handle_network_timeout(self, client_order_id: str) -> DuplicateCheck:
        """Handle network timeout scenario"""
        
        with self._lock:
            try:
                order = self.orders.get(client_order_id)
                if not order:
                    return DuplicateCheck(
                        is_duplicate=False,
                        recommendation="Order not found - safe to retry"
                    )
                
                # Mark as timeout
                order.request_timeout = True
                order.network_error_count += 1
                
                # Check retry eligibility
                if order.retry_count >= self.max_retries:
                    return DuplicateCheck(
                        is_duplicate=True,
                        reason=DuplicateReason.RETRY_DETECTED,
                        original_order_id=client_order_id,
                        recommendation="Maximum retries exceeded - manual intervention required"
                    )
                
                # Generate new client order ID for retry
                retry_id = self.generate_client_order_id(f"RETRY_{order.retry_count + 1}")
                
                return DuplicateCheck(
                    is_duplicate=False,
                    recommendation=f"Safe to retry with new client order ID: {retry_id}"
                )
                
            except Exception as e:
                logger.error(f"Network timeout handling failed: {e}")
                return DuplicateCheck(
                    is_duplicate=False,
                    recommendation=f"Timeout handling error - manual review required: {e}"
                )
    
    def _generate_order_hash(self,
                           symbol: str,
                           side: str,
                           order_type: str,
                           quantity: float,
                           price: Optional[float] = None) -> str:
        """Generate deterministic hash for order parameters"""
        
        # Create normalized parameter string
        price_str = f"{price:.8f}" if price is not None else "MARKET"
        param_string = f"{symbol}|{side}|{order_type}|{quantity:.8f}|{price_str}"
        
        # Generate hash
        hash_object = hashlib.sha256(param_string.encode())
        return hash_object.hexdigest()[:16]  # Use first 16 characters
    
    def _get_recent_orders_by_hash(self, order_hash: str) -> List[str]:
        """Get recent orders with the same hash"""
        
        cutoff_time = datetime.now() - self.duplicate_window
        recent_orders = []
        
        for client_order_id in self.order_hashes.get(order_hash, []):
            order = self.orders.get(client_order_id)
            if order and order.created_at >= cutoff_time:
                recent_orders.append(client_order_id)
        
        # Sort by creation time (most recent first)
        recent_orders.sort(key=lambda oid: self.orders[oid].created_at, reverse=True)
        
        return recent_orders
    
    def get_order_status(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive order status"""
        
        with self._lock:
            order = self.orders.get(client_order_id)
            if not order:
                return None
            
            return {
                "client_order_id": order.client_order_id,
                "exchange_order_id": order.exchange_order_id,
                "state": order.state.value,
                "symbol": order.symbol,
                "side": order.side,
                "order_type": order.order_type,
                "quantity": order.quantity,
                "price": order.price,
                "filled_quantity": order.filled_quantity,
                "average_price": order.average_price,
                "created_at": order.created_at.isoformat(),
                "last_updated": order.last_updated.isoformat(),
                "retry_count": order.retry_count,
                "network_error_count": order.network_error_count,
                "is_active": order.is_active,
                "is_terminal": order.is_terminal_state
            }
    
    def get_duplicate_prevention_stats(self) -> Dict[str, Any]:
        """Get duplicate prevention statistics"""
        
        with self._lock:
            active_orders = sum(1 for order in self.orders.values() if order.is_active)
            
            return {
                "total_orders": self.duplicate_stats["total_orders"],
                "duplicates_detected": self.duplicate_stats["duplicates_detected"],
                "duplicates_prevented": self.duplicate_stats["duplicates_prevented"],
                "retries_processed": self.duplicate_stats["retries_processed"],
                "network_timeouts": self.duplicate_stats["network_timeouts"],
                "active_orders": active_orders,
                "unique_order_hashes": len(self.order_hashes),
                "duplicate_prevention_rate": (
                    self.duplicate_stats["duplicates_prevented"] / 
                    max(self.duplicate_stats["total_orders"], 1) * 100
                )
            }
    
    def force_network_timeout_test(self, symbol: str = "BTC/USD") -> Tuple[str, str]:
        """Force network timeout scenario for testing"""
        
        # Create initial order
        client_order_id_1 = self.generate_client_order_id("TEST")
        
        # Register order
        order_record = self.register_order(
            client_order_id=client_order_id_1,
            symbol=symbol,
            side="buy",
            order_type="limit",
            quantity=0.1,
            price=50000.0
        )
        
        # Simulate submission
        self.update_order_state(client_order_id_1, OrderState.SUBMITTED)
        
        # Simulate network timeout
        self.update_order_state(client_order_id_1, OrderState.ERROR, network_timeout=True)
        
        # Now try to submit identical order (should be detected as potential retry)
        duplicate_check = self.check_duplicate(
            symbol=symbol,
            side="buy",
            order_type="limit",
            quantity=0.1,
            price=50000.0
        )
        
        if not duplicate_check.is_duplicate:
            # Create retry order
            client_order_id_2 = self.generate_client_order_id("RETRY")
            retry_order = self.register_order(
                client_order_id=client_order_id_2,
                symbol=symbol,
                side="buy",
                order_type="limit",
                quantity=0.1,
                price=50000.0,
                is_retry=True,
                parent_order_id=client_order_id_1
            )
            
            return client_order_id_1, client_order_id_2
        
        return client_order_id_1, ""
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        
        def cleanup_old_orders():
            import time
            while True:
                try:
                    self._cleanup_old_orders()
                    time.sleep(self.cleanup_interval_hours * 3600)  # Sleep for cleanup interval
                except Exception as e:
                    logger.error(f"Order cleanup error: {e}")
                    time.sleep(3600)  # Sleep 1 hour on error
        
        cleanup_thread = threading.Thread(target=cleanup_old_orders, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_old_orders(self):
        """Clean up old completed orders"""
        
        with self._lock:
            try:
                cutoff_time = datetime.now() - timedelta(days=self.retention_days)
                orders_to_remove = []
                
                for client_order_id, order in self.orders.items():
                    if order.is_terminal_state and order.last_updated < cutoff_time:
                        orders_to_remove.append(client_order_id)
                
                # Remove old orders
                for client_order_id in orders_to_remove:
                    order = self.orders.pop(client_order_id)
                    
                    # Clean up hash mapping
                    if order.order_hash in self.order_hashes:
                        self.order_hashes[order.order_hash] = [
                            oid for oid in self.order_hashes[order.order_hash] 
                            if oid != client_order_id
                        ]
                        if not self.order_hashes[order.order_hash]:
                            del self.order_hashes[order.order_hash]
                    
                    # Clean up symbol mapping
                    if order.symbol in self.symbol_orders:
                        self.symbol_orders[order.symbol] = [
                            oid for oid in self.symbol_orders[order.symbol] 
                            if oid != client_order_id
                        ]
                        if not self.symbol_orders[order.symbol]:
                            del self.symbol_orders[order.symbol]
                
                if orders_to_remove:
                    logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
                
            except Exception as e:
                logger.error(f"Order cleanup failed: {e}")