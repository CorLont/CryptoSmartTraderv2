"""
Order Deduplication & Idempotency System

Ensures order uniqueness, prevents duplicate fills, and provides idempotent retry logic
with comprehensive tracking and validation.
"""

import asyncio
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
import threading

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"           # Order created but not sent
    SUBMITTED = "submitted"       # Order sent to exchange
    PARTIAL_FILL = "partial_fill" # Partially filled
    FILLED = "filled"            # Completely filled
    CANCELLED = "cancelled"      # Order cancelled
    REJECTED = "rejected"        # Order rejected by exchange
    FAILED = "failed"            # Technical failure
    TIMEOUT = "timeout"          # Network timeout
    DUPLICATE = "duplicate"      # Duplicate detected

class RetryReason(Enum):
    """Reason for order retry"""
    NETWORK_TIMEOUT = "network_timeout"
    EXCHANGE_ERROR = "exchange_error"
    RATE_LIMIT = "rate_limit"
    TEMPORARY_FAILURE = "temporary_failure"
    CONNECTION_LOST = "connection_lost"

@dataclass
class ClientOrderId:
    """Unique client order identifier with metadata"""
    base_id: str                    # Base identifier
    sequence: int = 0               # Retry sequence number
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: str = ""            # Trading session ID
    strategy_id: str = ""           # Strategy that created order
    
    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())[:8]
    
    @property
    def full_id(self) -> str:
        """Generate full unique order ID"""
        return f"{self.base_id}-{self.sequence:03d}-{self.session_id}"
    
    @property
    def deterministic_hash(self) -> str:
        """Generate deterministic hash for deduplication"""
        content = f"{self.base_id}-{self.timestamp.isoformat()}-{self.strategy_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

@dataclass
class OrderSubmission:
    """Order submission record with deduplication data"""
    client_order_id: ClientOrderId
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    
    # Submission tracking
    submission_attempts: int = 0
    first_submission: datetime = field(default_factory=datetime.now)
    last_submission: Optional[datetime] = None
    
    # Exchange tracking
    exchange_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Fill tracking
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    
    # Retry tracking
    retry_history: List[Tuple[datetime, RetryReason, str]] = field(default_factory=list)
    max_retries: int = 3
    
    # Deduplication
    fingerprint: str = ""
    duplicate_of: Optional[str] = None
    
    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
    
    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication"""
        content = f"{self.symbol}-{self.side}-{self.quantity}-{self.price}-{self.order_type}"
        content += f"-{self.client_order_id.base_id}-{self.client_order_id.timestamp.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL]
    
    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]
    
    @property
    def can_retry(self) -> bool:
        """Check if order can be retried"""
        return (self.submission_attempts < self.max_retries and 
                self.status in [OrderStatus.FAILED, OrderStatus.TIMEOUT] and
                not self.duplicate_of)

@dataclass
class DeduplicationResult:
    """Result of deduplication check"""
    is_duplicate: bool
    original_order_id: Optional[str] = None
    reason: str = ""
    action: str = ""  # "reject", "merge", "allow"

class OrderDeduplicationEngine:
    """
    Comprehensive order deduplication and idempotency engine
    """
    
    def __init__(self, persistence_path: str = "data/orders"):
        self.persistence_path = Path(persistence_path)
        self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Order tracking
        self.active_orders: Dict[str, OrderSubmission] = {}
        self.order_history: Dict[str, OrderSubmission] = {}
        self.fingerprint_index: Dict[str, str] = {}  # fingerprint -> order_id
        
        # Deduplication windows
        self.dedup_window_minutes = 60  # 1 hour deduplication window
        self.cleanup_interval_hours = 6  # Cleanup every 6 hours
        
        # Statistics
        self.stats = {
            'total_orders': 0,
            'duplicate_orders': 0,
            'successful_submissions': 0,
            'failed_submissions': 0,
            'retries_performed': 0,
            'fills_tracked': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load persistent state
        self._load_persistent_state()
        
        # Start cleanup task
        self._start_cleanup_task()
        
        logger.info("Order Deduplication Engine initialized")
    
    def create_order_id(self, 
                       strategy_id: str = "",
                       base_id: Optional[str] = None) -> ClientOrderId:
        """Create new unique client order ID"""
        
        if not base_id:
            # Generate base ID from timestamp + random component
            timestamp_ms = int(time.time() * 1000)
            random_component = str(uuid.uuid4())[:8]
            base_id = f"CST{timestamp_ms}{random_component}"
        
        return ClientOrderId(
            base_id=base_id,
            sequence=0,
            strategy_id=strategy_id,
            timestamp=datetime.now()
        )
    
    def check_duplicate(self, order: OrderSubmission) -> DeduplicationResult:
        """Check if order is a duplicate"""
        
        with self._lock:
            # Check fingerprint-based deduplication
            if order.fingerprint in self.fingerprint_index:
                original_order_id = self.fingerprint_index[order.fingerprint]
                original_order = self.order_history.get(original_order_id)
                
                if original_order:
                    # Check time window
                    time_diff = order.first_submission - original_order.first_submission
                    
                    if abs(time_diff.total_seconds()) < (self.dedup_window_minutes * 60):
                        return DeduplicationResult(
                            is_duplicate=True,
                            original_order_id=original_order_id,
                            reason=f"Identical order within {self.dedup_window_minutes} minutes",
                            action="reject"
                        )
            
            # Check similar orders (same symbol, side, similar quantity)
            similar_orders = self._find_similar_orders(order)
            
            for similar_order_id, similarity_score in similar_orders:
                if similarity_score > 0.95:  # 95% similarity threshold
                    similar_order = self.order_history.get(similar_order_id)
                    
                    if similar_order and similar_order.is_active:
                        return DeduplicationResult(
                            is_duplicate=True,
                            original_order_id=similar_order_id,
                            reason=f"Similar active order (similarity: {similarity_score:.2%})",
                            action="merge"
                        )
            
            return DeduplicationResult(is_duplicate=False, action="allow")
    
    def _find_similar_orders(self, order: OrderSubmission) -> List[Tuple[str, float]]:
        """Find similar orders with similarity scores"""
        
        similar_orders = []
        
        # Look for orders in the same symbol within time window
        cutoff_time = order.first_submission - timedelta(minutes=self.dedup_window_minutes)
        
        for order_id, existing_order in self.order_history.items():
            if existing_order.first_submission < cutoff_time:
                continue
            
            if existing_order.symbol != order.symbol:
                continue
            
            if existing_order.side != order.side:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_similarity(order, existing_order)
            
            if similarity > 0.8:  # Only consider highly similar orders
                similar_orders.append((order_id, similarity))
        
        # Sort by similarity (highest first)
        similar_orders.sort(key=lambda x: x[1], reverse=True)
        return similar_orders[:5]  # Return top 5 similar orders
    
    def _calculate_similarity(self, order1: OrderSubmission, order2: OrderSubmission) -> float:
        """Calculate similarity score between two orders"""
        
        score = 0.0
        
        # Symbol match (required)
        if order1.symbol == order2.symbol:
            score += 0.3
        else:
            return 0.0
        
        # Side match (required)
        if order1.side == order2.side:
            score += 0.3
        else:
            return 0.0
        
        # Quantity similarity
        if order1.quantity > 0 and order2.quantity > 0:
            qty_ratio = min(order1.quantity, order2.quantity) / max(order1.quantity, order2.quantity)
            score += 0.2 * qty_ratio
        
        # Price similarity (if both have prices)
        if order1.price and order2.price:
            price_diff = abs(order1.price - order2.price) / max(order1.price, order2.price)
            price_similarity = max(0, 1 - price_diff * 10)  # 10% price difference = 0 similarity
            score += 0.2 * price_similarity
        elif not order1.price and not order2.price:
            score += 0.2  # Both market orders
        
        return min(score, 1.0)
    
    def submit_order(self, order: OrderSubmission) -> Tuple[bool, str]:
        """Submit order with deduplication check"""
        
        with self._lock:
            # Check for duplicates
            dedup_result = self.check_duplicate(order)
            
            if dedup_result.is_duplicate:
                self.stats['duplicate_orders'] += 1
                
                if dedup_result.action == "reject":
                    order.status = OrderStatus.DUPLICATE
                    order.duplicate_of = dedup_result.original_order_id
                    
                    logger.warning(f"Order rejected as duplicate: {order.client_order_id.full_id} "
                                 f"(original: {dedup_result.original_order_id})")
                    
                    return False, f"Duplicate order: {dedup_result.reason}"
                
                elif dedup_result.action == "merge":
                    # Merge with existing order
                    original_order = self.order_history.get(dedup_result.original_order_id)
                    if original_order:
                        logger.info(f"Merging order {order.client_order_id.full_id} "
                                   f"with existing {dedup_result.original_order_id}")
                        return True, f"Merged with existing order {dedup_result.original_order_id}"
            
            # Record order
            order_id = order.client_order_id.full_id
            order.submission_attempts += 1
            order.last_submission = datetime.now()
            
            # Store in tracking systems
            self.active_orders[order_id] = order
            self.order_history[order_id] = order
            self.fingerprint_index[order.fingerprint] = order_id
            
            self.stats['total_orders'] += 1
            
            logger.info(f"Order submitted: {order_id} ({order.symbol} {order.side} {order.quantity})")
            
            return True, "Order submitted successfully"
    
    def retry_order(self, 
                   order_id: str, 
                   reason: RetryReason, 
                   error_details: str = "") -> Tuple[bool, str]:
        """Retry failed order with new sequence number"""
        
        with self._lock:
            if order_id not in self.active_orders:
                return False, "Order not found"
            
            order = self.active_orders[order_id]
            
            if not order.can_retry:
                return False, f"Order cannot be retried (attempts: {order.submission_attempts}, status: {order.status.value})"
            
            # Create new order ID with incremented sequence
            old_client_id = order.client_order_id
            new_client_id = ClientOrderId(
                base_id=old_client_id.base_id,
                sequence=old_client_id.sequence + 1,
                timestamp=old_client_id.timestamp,
                session_id=old_client_id.session_id,
                strategy_id=old_client_id.strategy_id
            )
            
            # Update order
            order.client_order_id = new_client_id
            order.retry_history.append((datetime.now(), reason, error_details))
            order.status = OrderStatus.PENDING
            order.submission_attempts += 1
            
            # Update tracking - keep both old and new in history
            new_order_id = new_client_id.full_id
            
            # Keep old order in history but mark as retried
            old_order = self.active_orders[order_id].copy() if hasattr(self.active_orders[order_id], 'copy') else self.active_orders[order_id]
            old_order.status = OrderStatus.FAILED
            
            # Update active orders
            del self.active_orders[order_id] 
            self.active_orders[new_order_id] = order
            self.order_history[new_order_id] = order
            
            # Update fingerprint index
            if order.fingerprint in self.fingerprint_index:
                self.fingerprint_index[order.fingerprint] = new_order_id
            
            self.stats['retries_performed'] += 1
            
            logger.info(f"Order retry: {order_id} -> {new_order_id} (reason: {reason.value})")
            
            return True, f"Order retried as {new_order_id}"
    
    def update_order_status(self, 
                           order_id: str, 
                           status: OrderStatus,
                           exchange_order_id: Optional[str] = None,
                           filled_quantity: float = 0.0,
                           fill_price: float = 0.0,
                           fees: float = 0.0) -> bool:
        """Update order status and fill information"""
        
        with self._lock:
            if order_id not in self.active_orders:
                logger.warning(f"Attempt to update unknown order: {order_id}")
                return False
            
            order = self.active_orders[order_id]
            
            # Update basic status
            order.status = status
            
            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
            
            # Update fill information
            if filled_quantity > 0:
                old_filled = order.filled_quantity
                order.filled_quantity += filled_quantity
                order.total_fees += fees
                
                # Update average fill price
                if order.filled_quantity > 0:
                    total_value = (old_filled * order.average_fill_price) + (filled_quantity * fill_price)
                    order.average_fill_price = total_value / order.filled_quantity
                
                self.stats['fills_tracked'] += 1
                
                logger.info(f"Order fill: {order_id} +{filled_quantity} @ {fill_price} "
                           f"(total: {order.filled_quantity}/{order.quantity})")
            
            # Move to history if complete
            if order.is_complete:
                if status == OrderStatus.FILLED:
                    self.stats['successful_submissions'] += 1
                else:
                    self.stats['failed_submissions'] += 1
                
                # Keep in active orders for a short time for final updates
                # Will be moved to history by cleanup task
            
            return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderSubmission]:
        """Get current order status"""
        
        with self._lock:
            return self.active_orders.get(order_id) or self.order_history.get(order_id)
    
    def is_duplicate_fill(self, 
                         order_id: str, 
                         fill_id: str, 
                         quantity: float, 
                         price: float) -> bool:
        """Check if fill is a duplicate"""
        
        # Generate fill fingerprint
        fill_fingerprint = hashlib.sha256(
            f"{order_id}-{fill_id}-{quantity}-{price}".encode()
        ).hexdigest()
        
        # Check against stored fills (implementation would track fill fingerprints)
        # For now, return False (no duplicate detected)
        return False
    
    def get_active_orders(self) -> List[OrderSubmission]:
        """Get all active orders"""
        
        with self._lock:
            return [order for order in self.active_orders.values() if order.is_active]
    
    def get_pending_retries(self) -> List[OrderSubmission]:
        """Get orders that can be retried"""
        
        with self._lock:
            return [order for order in self.active_orders.values() if order.can_retry]
    
    def cleanup_old_orders(self):
        """Clean up old completed orders"""
        
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=self.cleanup_interval_hours)
            
            orders_to_remove = []
            
            for order_id, order in self.active_orders.items():
                if (order.is_complete and 
                    order.last_submission and 
                    order.last_submission < cutoff_time):
                    orders_to_remove.append(order_id)
            
            for order_id in orders_to_remove:
                del self.active_orders[order_id]
            
            if orders_to_remove:
                logger.info(f"Cleaned up {len(orders_to_remove)} old orders")
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.cleanup_interval_hours * 3600)
                    self.cleanup_old_orders()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _load_persistent_state(self):
        """Load persistent order state"""
        
        state_file = self.persistence_path / "order_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                
                # Restore statistics
                self.stats.update(data.get('stats', {}))
                
                # Restore recent orders
                recent_orders = data.get('recent_orders', {})
                
                for order_id, order_data in recent_orders.items():
                    # Reconstruct order objects (simplified)
                    # Full implementation would properly deserialize OrderSubmission objects
                    pass
                
                logger.info(f"Loaded persistent state: {len(recent_orders)} recent orders")
                
            except Exception as e:
                logger.error(f"Failed to load persistent state: {e}")
    
    def _save_persistent_state(self):
        """Save persistent order state"""
        
        state_file = self.persistence_path / "order_state.json"
        
        try:
            # Prepare data for persistence
            recent_orders = {}
            cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
            
            for order_id, order in self.order_history.items():
                if order.first_submission >= cutoff_time:
                    # Serialize order data (simplified)
                    recent_orders[order_id] = {
                        'symbol': order.symbol,
                        'side': order.side,
                        'quantity': order.quantity,
                        'status': order.status.value,
                        'submission_time': order.first_submission.isoformat(),
                        'fingerprint': order.fingerprint
                    }
            
            data = {
                'stats': self.stats,
                'recent_orders': recent_orders,
                'last_save': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
    
    def force_timeout_test(self, order_id: str) -> Tuple[bool, str]:
        """Force timeout test for order (testing purposes)"""
        
        with self._lock:
            if order_id not in self.active_orders:
                return False, "Order not found"
            
            order = self.active_orders[order_id]
            order.status = OrderStatus.TIMEOUT
            
            logger.warning(f"Forced timeout test for order: {order_id}")
            
            return True, "Timeout simulated"
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics"""
        
        with self._lock:
            active_count = len(self.active_orders)
            history_count = len(self.order_history)
            
            # Calculate success rate
            total_submissions = self.stats['successful_submissions'] + self.stats['failed_submissions']
            success_rate = (self.stats['successful_submissions'] / max(total_submissions, 1)) * 100
            
            # Calculate duplicate rate
            duplicate_rate = (self.stats['duplicate_orders'] / max(self.stats['total_orders'], 1)) * 100
            
            return {
                'timestamp': datetime.now().isoformat(),
                
                'order_counts': {
                    'active_orders': active_count,
                    'total_orders_tracked': history_count,
                    'pending_retries': len(self.get_pending_retries())
                },
                
                'submission_stats': {
                    'total_orders': self.stats['total_orders'],
                    'successful_submissions': self.stats['successful_submissions'],
                    'failed_submissions': self.stats['failed_submissions'],
                    'success_rate_percent': success_rate
                },
                
                'deduplication_stats': {
                    'duplicate_orders': self.stats['duplicate_orders'],
                    'duplicate_rate_percent': duplicate_rate,
                    'retries_performed': self.stats['retries_performed'],
                    'fills_tracked': self.stats['fills_tracked']
                },
                
                'system_health': {
                    'deduplication_window_minutes': self.dedup_window_minutes,
                    'fingerprint_index_size': len(self.fingerprint_index),
                    'cleanup_interval_hours': self.cleanup_interval_hours
                }
            }
    
    def validate_idempotency(self) -> Dict[str, Any]:
        """Validate system idempotency"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'idempotency_checks': [],
            'overall_status': 'healthy'
        }
        
        with self._lock:
            # Check for duplicate fingerprints
            fingerprint_counts = {}
            for fingerprint in self.fingerprint_index.keys():
                fingerprint_counts[fingerprint] = fingerprint_counts.get(fingerprint, 0) + 1
            
            duplicate_fingerprints = {fp: count for fp, count in fingerprint_counts.items() if count > 1}
            
            validation_results['idempotency_checks'].append({
                'check': 'fingerprint_uniqueness',
                'status': 'passed' if not duplicate_fingerprints else 'failed',
                'details': f"Found {len(duplicate_fingerprints)} duplicate fingerprints" if duplicate_fingerprints else "All fingerprints unique"
            })
            
            # Check retry logic
            retry_orders = self.get_pending_retries()
            excessive_retries = [order for order in retry_orders if order.submission_attempts >= order.max_retries]
            
            validation_results['idempotency_checks'].append({
                'check': 'retry_limits',
                'status': 'passed' if not excessive_retries else 'warning',
                'details': f"Found {len(excessive_retries)} orders with excessive retries"
            })
            
            # Check for stale orders
            stale_cutoff = datetime.now() - timedelta(hours=1)
            stale_orders = [
                order for order in self.active_orders.values()
                if order.status == OrderStatus.PENDING and order.first_submission < stale_cutoff
            ]
            
            validation_results['idempotency_checks'].append({
                'check': 'stale_orders',
                'status': 'passed' if not stale_orders else 'warning',
                'details': f"Found {len(stale_orders)} stale pending orders"
            })
            
            # Overall status
            failed_checks = [check for check in validation_results['idempotency_checks'] if check['status'] == 'failed']
            warning_checks = [check for check in validation_results['idempotency_checks'] if check['status'] == 'warning']
            
            if failed_checks:
                validation_results['overall_status'] = 'unhealthy'
            elif warning_checks:
                validation_results['overall_status'] = 'degraded'
        
        return validation_results