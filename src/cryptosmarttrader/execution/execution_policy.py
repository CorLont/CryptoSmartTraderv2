#!/usr/bin/env python3
"""
Execution Policy Engine
Enforces execution gates: spread/depth/volume checks, slippage budget, idempotent orders
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    GTC = "gtc"         # Good Till Canceled
    IOC = "ioc"         # Immediate or Cancel
    FOK = "fok"         # Fill or Kill
    POST_ONLY = "post_only"  # Post only (maker)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class ExecutionGates:
    """Execution gate thresholds"""
    # Spread gates
    max_spread_bps: int = 50        # 50 basis points max spread
    max_spread_pct: float = 0.005   # 0.5% max spread
    
    # Depth gates
    min_bid_depth_usd: float = 10000.0   # $10k min bid depth
    min_ask_depth_usd: float = 10000.0   # $10k min ask depth
    min_total_depth_usd: float = 50000.0 # $50k min total depth
    
    # Volume gates (1 minute)
    min_volume_1m_usd: float = 100000.0  # $100k min 1m volume
    min_trades_1m: int = 10              # 10 min trades per minute
    
    # Slippage budget
    max_slippage_bps: int = 25           # 25 bps max slippage
    slippage_budget_daily_bps: int = 200 # 200 bps daily budget
    
    # Order limits
    max_order_size_usd: float = 50000.0  # $50k max order size
    max_order_retries: int = 3           # 3 max retries
    order_timeout_seconds: float = 30.0  # 30s order timeout
    
    # Market conditions
    min_market_hours_pct: float = 0.8    # 80% of expected volume
    max_volatility_1h_pct: float = 0.15  # 15% max 1h volatility


@dataclass
class MarketData:
    """Market data for execution decisions"""
    symbol: str
    timestamp: datetime
    
    # Price data
    bid: float
    ask: float
    mid: float
    last: float
    
    # Depth data
    bid_depth_usd: float
    ask_depth_usd: float
    total_depth_usd: float
    
    # Volume data
    volume_1m_usd: float
    trades_1m: int
    volume_24h_usd: float
    
    # Volatility
    volatility_1h_pct: float
    volatility_24h_pct: float
    
    # Spread metrics
    spread_bps: int
    spread_pct: float


@dataclass
class OrderRequest:
    """Order request with execution policy checks"""
    # Core order data
    symbol: str
    side: str           # 'buy' or 'sell'
    size: float
    order_type: OrderType
    price: Optional[float] = None
    
    # Execution policy
    time_in_force: TimeInForce = TimeInForce.GTC
    post_only: bool = False
    reduce_only: bool = False
    
    # Risk management
    max_slippage_bps: Optional[int] = None
    min_fill_size: Optional[float] = None
    
    # Idempotency
    client_order_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionResult:
    """Execution result with metrics"""
    order_id: str
    client_order_id: str
    status: OrderStatus
    
    # Fill data
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    total_fee: float = 0.0
    
    # Execution metrics
    execution_latency_ms: float = 0.0
    slippage_bps: float = 0.0
    effective_spread_bps: float = 0.0
    
    # Timestamps
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Metadata
    error_message: Optional[str] = None
    retry_count: int = 0
    exchange_order_id: Optional[str] = None


@dataclass
class SlippageTracker:
    """Daily slippage tracking"""
    date: datetime
    used_slippage_bps: float = 0.0
    total_executions: int = 0
    avg_slippage_bps: float = 0.0


class ExecutionPolicy:
    """Execution policy engine with gates and idempotency"""
    
    def __init__(self, gates: ExecutionGates):
        self.gates = gates
        self.logger = logging.getLogger(__name__)
        
        # Order tracking
        self.active_orders: Dict[str, OrderRequest] = {}
        self.order_history: Dict[str, ExecutionResult] = {}
        self.client_order_map: Dict[str, str] = {}  # client_id -> order_id
        
        # Slippage tracking
        self.slippage_tracker = SlippageTracker(date=datetime.now().date())
        
        # Market data cache
        self.market_data_cache: Dict[str, MarketData] = {}
        self.cache_timeout_seconds = 5.0
        
        # Request deduplication
        self.request_hashes: Dict[str, datetime] = {}
        self.request_timeout_minutes = 5
    
    def update_market_data(self, symbol: str, data: MarketData):
        """Update market data for symbol"""
        self.market_data_cache[symbol] = data
        self.logger.debug(f"Market data updated for {symbol}: spread={data.spread_bps}bps")
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get cached market data if fresh"""
        data = self.market_data_cache.get(symbol)
        if data:
            age = (datetime.now() - data.timestamp).total_seconds()
            if age <= self.cache_timeout_seconds:
                return data
            else:
                self.logger.warning(f"Market data for {symbol} is stale: {age:.1f}s")
        return None
    
    def generate_client_order_id(self, request: OrderRequest) -> str:
        """Generate idempotent client order ID"""
        if request.client_order_id:
            return request.client_order_id
        
        # Generate based on request content for idempotency
        content = f"{request.symbol}_{request.side}_{request.size}_{request.price}_{request.timestamp.isoformat()}"
        hash_obj = hashlib.md5(content.encode())
        return f"CST_{hash_obj.hexdigest()[:16]}"
    
    def check_duplicate_request(self, request: OrderRequest) -> Tuple[bool, str]:
        """Check for duplicate request"""
        # Generate request hash
        request_content = {
            'symbol': request.symbol,
            'side': request.side,
            'size': request.size,
            'price': request.price,
            'order_type': request.order_type.value
        }
        request_str = json.dumps(request_content, sort_keys=True)
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        # Check if we've seen this request recently
        if request_hash in self.request_hashes:
            last_seen = self.request_hashes[request_hash]
            if (datetime.now() - last_seen).total_seconds() < self.request_timeout_minutes * 60:
                return True, f"Duplicate request detected: {request_hash[:8]}"
        
        # Record this request
        self.request_hashes[request_hash] = datetime.now()
        
        # Cleanup old hashes
        cutoff = datetime.now() - timedelta(minutes=self.request_timeout_minutes * 2)
        self.request_hashes = {
            h: t for h, t in self.request_hashes.items() if t > cutoff
        }
        
        return False, "Request is unique"
    
    def check_execution_gates(self, request: OrderRequest) -> Tuple[bool, List[str]]:
        """Check all execution gates"""
        errors = []
        
        # Get market data
        market_data = self.get_market_data(request.symbol)
        if not market_data:
            errors.append(f"No fresh market data for {request.symbol}")
            return False, errors
        
        # Check spread gates
        if market_data.spread_bps > self.gates.max_spread_bps:
            errors.append(f"Spread too wide: {market_data.spread_bps}bps > {self.gates.max_spread_bps}bps")
        
        if market_data.spread_pct > self.gates.max_spread_pct:
            errors.append(f"Spread percentage too high: {market_data.spread_pct:.2%} > {self.gates.max_spread_pct:.2%}")
        
        # Check depth gates
        if request.side.lower() == 'buy' and market_data.ask_depth_usd < self.gates.min_ask_depth_usd:
            errors.append(f"Insufficient ask depth: ${market_data.ask_depth_usd:,.0f} < ${self.gates.min_ask_depth_usd:,.0f}")
        
        if request.side.lower() == 'sell' and market_data.bid_depth_usd < self.gates.min_bid_depth_usd:
            errors.append(f"Insufficient bid depth: ${market_data.bid_depth_usd:,.0f} < ${self.gates.min_bid_depth_usd:,.0f}")
        
        if market_data.total_depth_usd < self.gates.min_total_depth_usd:
            errors.append(f"Insufficient total depth: ${market_data.total_depth_usd:,.0f} < ${self.gates.min_total_depth_usd:,.0f}")
        
        # Check volume gates
        if market_data.volume_1m_usd < self.gates.min_volume_1m_usd:
            errors.append(f"Insufficient 1m volume: ${market_data.volume_1m_usd:,.0f} < ${self.gates.min_volume_1m_usd:,.0f}")
        
        if market_data.trades_1m < self.gates.min_trades_1m:
            errors.append(f"Insufficient 1m trades: {market_data.trades_1m} < {self.gates.min_trades_1m}")
        
        # Check volatility gates
        if market_data.volatility_1h_pct > self.gates.max_volatility_1h_pct:
            errors.append(f"Volatility too high: {market_data.volatility_1h_pct:.1%} > {self.gates.max_volatility_1h_pct:.1%}")
        
        # Check order size
        order_notional = request.size * (request.price or market_data.mid)
        if order_notional > self.gates.max_order_size_usd:
            errors.append(f"Order too large: ${order_notional:,.0f} > ${self.gates.max_order_size_usd:,.0f}")
        
        # Check slippage budget
        if self.slippage_tracker.used_slippage_bps >= self.gates.slippage_budget_daily_bps:
            errors.append(f"Daily slippage budget exhausted: {self.slippage_tracker.used_slippage_bps:.1f}bps")
        
        return len(errors) == 0, errors
    
    def estimate_slippage(self, request: OrderRequest, market_data: MarketData) -> float:
        """Estimate execution slippage"""
        if request.order_type == OrderType.LIMIT and request.price:
            # For limit orders, slippage is price improvement/degradation
            if request.side.lower() == 'buy':
                slippage_pct = (request.price - market_data.ask) / market_data.ask
            else:
                slippage_pct = (market_data.bid - request.price) / market_data.bid
            return slippage_pct * 10000  # Convert to bps
        
        elif request.order_type == OrderType.MARKET:
            # For market orders, estimate based on spread and order size
            base_slippage = market_data.spread_bps / 2
            
            # Add impact based on order size vs depth
            order_notional = request.size * market_data.mid
            relevant_depth = market_data.ask_depth_usd if request.side.lower() == 'buy' else market_data.bid_depth_usd
            
            if relevant_depth > 0:
                impact_factor = min(order_notional / relevant_depth, 1.0)
                impact_slippage = impact_factor * market_data.spread_bps
                return base_slippage + impact_slippage
            
            return base_slippage
        
        return 0.0
    
    async def validate_order(self, request: OrderRequest) -> Tuple[bool, List[str], str]:
        """Validate order request through all gates"""
        errors = []
        client_order_id = self.generate_client_order_id(request)
        
        # Check for duplicate
        is_duplicate, dup_msg = self.check_duplicate_request(request)
        if is_duplicate:
            errors.append(dup_msg)
            return False, errors, client_order_id
        
        # Check if client order ID already exists
        if client_order_id in self.client_order_map:
            existing_order_id = self.client_order_map[client_order_id]
            existing_result = self.order_history.get(existing_order_id)
            if existing_result and existing_result.status not in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.FAILED]:
                errors.append(f"Client order ID already exists: {client_order_id}")
                return False, errors, client_order_id
        
        # Check execution gates
        gates_ok, gate_errors = self.check_execution_gates(request)
        errors.extend(gate_errors)
        
        # Estimate and check slippage
        market_data = self.get_market_data(request.symbol)
        if market_data:
            estimated_slippage = self.estimate_slippage(request, market_data)
            max_allowed_slippage = request.max_slippage_bps or self.gates.max_slippage_bps
            
            if estimated_slippage > max_allowed_slippage:
                errors.append(f"Estimated slippage too high: {estimated_slippage:.1f}bps > {max_allowed_slippage}bps")
        
        # Set client order ID on request
        request.client_order_id = client_order_id
        
        return len(errors) == 0, errors, client_order_id
    
    def register_order(self, request: OrderRequest, order_id: str):
        """Register order for tracking"""
        self.active_orders[order_id] = request
        self.client_order_map[request.client_order_id] = order_id
        self.logger.info(f"Order registered: {order_id} ({request.client_order_id})")
    
    def update_order_result(self, order_id: str, result: ExecutionResult):
        """Update order execution result"""
        self.order_history[order_id] = result
        
        # Update slippage tracking
        if result.status == OrderStatus.FILLED and result.slippage_bps > 0:
            self._update_slippage_tracking(result.slippage_bps)
        
        # Remove from active orders if final state
        if result.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.FAILED]:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        self.logger.info(f"Order result updated: {order_id} - {result.status.value}")
    
    def _update_slippage_tracking(self, slippage_bps: float):
        """Update daily slippage tracking"""
        today = datetime.now().date()
        
        # Reset if new day
        if self.slippage_tracker.date != today:
            self.slippage_tracker = SlippageTracker(date=today)
        
        # Update tracker
        self.slippage_tracker.used_slippage_bps += slippage_bps
        self.slippage_tracker.total_executions += 1
        self.slippage_tracker.avg_slippage_bps = (
            self.slippage_tracker.used_slippage_bps / self.slippage_tracker.total_executions
        )
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[ExecutionResult]:
        """Get order result by client order ID"""
        order_id = self.client_order_map.get(client_order_id)
        if order_id:
            return self.order_history.get(order_id)
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution policy status"""
        return {
            'gates': {
                'max_spread_bps': self.gates.max_spread_bps,
                'min_depth_usd': self.gates.min_bid_depth_usd,
                'min_volume_1m_usd': self.gates.min_volume_1m_usd,
                'max_slippage_bps': self.gates.max_slippage_bps
            },
            'slippage_tracking': {
                'date': self.slippage_tracker.date.isoformat(),
                'used_slippage_bps': self.slippage_tracker.used_slippage_bps,
                'budget_remaining_bps': self.gates.slippage_budget_daily_bps - self.slippage_tracker.used_slippage_bps,
                'total_executions': self.slippage_tracker.total_executions,
                'avg_slippage_bps': self.slippage_tracker.avg_slippage_bps
            },
            'orders': {
                'active_count': len(self.active_orders),
                'total_processed': len(self.order_history),
                'client_order_count': len(self.client_order_map)
            },
            'market_data': {
                'cached_symbols': list(self.market_data_cache.keys()),
                'cache_timeout_seconds': self.cache_timeout_seconds
            }
        }


# Singleton instance for global access
_execution_policy_instance: Optional[ExecutionPolicy] = None

def get_execution_policy() -> ExecutionPolicy:
    """Get singleton execution policy instance"""
    global _execution_policy_instance
    if _execution_policy_instance is None:
        gates = ExecutionGates()
        _execution_policy_instance = ExecutionPolicy(gates)
    return _execution_policy_instance

def initialize_execution_policy(gates: ExecutionGates) -> ExecutionPolicy:
    """Initialize execution policy with custom gates"""
    global _execution_policy_instance
    _execution_policy_instance = ExecutionPolicy(gates)
    return _execution_policy_instance