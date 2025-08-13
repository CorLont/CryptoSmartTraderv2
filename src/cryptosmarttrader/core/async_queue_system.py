#!/usr/bin/env python3
"""
Async Queue System
Redis/asyncio.Queue based dataflow with centralized rate limiting
"""

import asyncio
import time
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QueueMessage:
    """Standard queue message format"""
    id: str
    queue_name: str
    message_type: str
    payload: Dict[str, Any]
    priority: MessagePriority
    timestamp: datetime
    sender: str
    ttl_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create from dictionary"""
        data['priority'] = MessagePriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class RateLimiter:
    """Centralized rate limiter"""

    def __init__(self, requests_per_second: float = 15.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.request_count = 0
        self.logger = get_structured_logger("RateLimiter")

    async def acquire(self, operation_name: str = "unknown") -> None:
        """Acquire rate limit token"""

        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            self.logger.debug(f"Rate limiting {operation_name}: waiting {wait_time:.3f}s")
            await asyncio.sleep(wait_time)

        self.last_request_time = time.time()
        self.request_count += 1

        if self.request_count % 100 == 0:
            self.logger.info(f"Rate limiter processed {self.request_count} requests")

class AsyncQueueBackend:
    """Base class for queue backends"""

    async def put(self, queue_name: str, message: QueueMessage) -> bool:
        """Put message in queue"""
        raise NotImplementedError

    async def get(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Get message from queue"""
        raise NotImplementedError

    async def size(self, queue_name: str) -> int:
        """Get queue size"""
        raise NotImplementedError

    async def clear(self, queue_name: str) -> bool:
        """Clear queue"""
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Check backend health"""
        raise NotImplementedError

class AsyncioQueueBackend(AsyncQueueBackend):
    """Asyncio-based queue backend (in-memory)"""

    def __init__(self, max_queue_size: int = 10000):
        self.queues: Dict[str, asyncio.Queue] = {}
        self.max_queue_size = max_queue_size
        self.logger = get_structured_logger("AsyncioQueueBackend")

    def _get_queue(self, queue_name: str) -> asyncio.Queue:
        """Get or create queue"""
        if queue_name not in self.queues:
            self.queues[queue_name] = asyncio.Queue(maxsize=self.max_queue_size)
        return self.queues[queue_name]

    async def put(self, queue_name: str, message: QueueMessage) -> bool:
        """Put message in queue"""
        try:
            queue = self._get_queue(queue_name)

            # Check TTL
            if message.ttl_seconds:
                age = (datetime.now() - message.timestamp).total_seconds()
                if age > message.ttl_seconds:
                    self.logger.warning(f"Message {message.id} expired (age: {age:.1f}s)")
                    return False

            await queue.put(message)
            self.logger.debug(f"Message {message.id} queued to {queue_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to queue message {message.id}: {e}")
            return False

    async def get(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Get message from queue"""
        try:
            queue = self._get_queue(queue_name)

            message = await asyncio.wait_for(queue.get(), timeout=timeout)
            self.logger.debug(f"Message {message.id} dequeued from {queue_name}")
            return message

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Failed to dequeue from {queue_name}: {e}")
            return None

    async def size(self, queue_name: str) -> int:
        """Get queue size"""
        if queue_name in self.queues:
            return self.queues[queue_name].qsize()
        return 0

    async def clear(self, queue_name: str) -> bool:
        """Clear queue"""
        try:
            if queue_name in self.queues:
                queue = self.queues[queue_name]
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear queue {queue_name}: {e}")
            return False

    async def health_check(self) -> bool:
        """Check backend health"""
        return True

class RedisQueueBackend(AsyncQueueBackend):
    """Redis-based queue backend (distributed)"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.logger = get_structured_logger("RedisQueueBackend")
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available, using mock backend")
            return False

        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            self.connected = True
            self.logger.info("Connected to Redis")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
            self.connected = False
            self.logger.info("Disconnected from Redis")

    async def put(self, queue_name: str, message: QueueMessage) -> bool:
        """Put message in Redis queue"""
        if not self.connected:
            return False

        try:
            message_data = json.dumps(message.to_dict())

            # Use priority score for sorted set
            score = message.priority.value * 1000000 + int(time.time())

            await self.redis.zadd(f"queue:{queue_name}", {message_data: score})
            self.logger.debug(f"Message {message.id} queued to Redis {queue_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to queue message to Redis: {e}")
            return False

    async def get(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Get message from Redis queue"""
        if not self.connected:
            return None

        try:
            # Get highest priority message
            result = await self.redis.zpopmax(f"queue:{queue_name}")

            if result:
                message_data, score = result[0]
                data = json.loads(message_data)
                message = QueueMessage.from_dict(data)

                self.logger.debug(f"Message {message.id} dequeued from Redis {queue_name}")
                return message

            return None

        except Exception as e:
            self.logger.error(f"Failed to dequeue from Redis: {e}")
            return None

    async def size(self, queue_name: str) -> int:
        """Get Redis queue size"""
        if not self.connected:
            return 0

        try:
            return await self.redis.zcard(f"queue:{queue_name}")
        except Exception as e:
            self.logger.error(f"Failed to get Redis queue size: {e}")
            return 0

    async def clear(self, queue_name: str) -> bool:
        """Clear Redis queue"""
        if not self.connected:
            return False

        try:
            await self.redis.delete(f"queue:{queue_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear Redis queue: {e}")
            return False

    async def health_check(self) -> bool:
        """Check Redis health"""
        if not self.connected:
            return False

        try:
            await self.redis.ping()
            return True
        except Exception:
            return False

class AsyncQueueSystem:
    """Central async queue system with rate limiting"""

    def __init__(self, backend: AsyncQueueBackend, rate_limiter: RateLimiter):
        self.backend = backend
        self.rate_limiter = rate_limiter
        self.logger = get_structured_logger("AsyncQueueSystem")
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'queues_active': set()
        }
        self.message_handlers: Dict[str, Callable] = {}

    async def send_message(self, queue_name: str, message_type: str,
                          payload: Dict[str, Any], sender: str,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          ttl_seconds: Optional[int] = None) -> bool:
        """Send message to queue"""

        # Apply rate limiting
        await self.rate_limiter.acquire(f"send_to_{queue_name}")

        try:
            message = QueueMessage(
                id=str(uuid.uuid4()),
                queue_name=queue_name,
                message_type=message_type,
                payload=payload,
                priority=priority,
                timestamp=datetime.now(),
                sender=sender,
                ttl_seconds=ttl_seconds
            )

            success = await self.backend.put(queue_name, message)

            if success:
                self.metrics['messages_sent'] += 1
                self.metrics['queues_active'].add(queue_name)
                self.logger.debug(f"Sent message {message.id} to {queue_name}")
            else:
                self.metrics['messages_failed'] += 1
                self.logger.error(f"Failed to send message to {queue_name}")

            return success

        except Exception as e:
            self.metrics['messages_failed'] += 1
            self.logger.error(f"Error sending message to {queue_name}: {e}")
            return False

    async def receive_message(self, queue_name: str,
                            timeout: float = 1.0) -> Optional[QueueMessage]:
        """Receive message from queue"""

        # Apply rate limiting
        await self.rate_limiter.acquire(f"receive_from_{queue_name}")

        try:
            message = await self.backend.get(queue_name, timeout)

            if message:
                self.metrics['messages_received'] += 1
                self.logger.debug(f"Received message {message.id} from {queue_name}")

            return message

        except Exception as e:
            self.logger.error(f"Error receiving message from {queue_name}: {e}")
            return None

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler"""
        self.message_handlers[message_type] = handler
        self.logger.info(f"Registered handler for message type: {message_type}")

    async def process_queue(self, queue_name: str, max_messages: int = 100) -> int:
        """Process messages in queue using registered handlers"""

        processed = 0

        try:
            while processed < max_messages:
                message = await self.receive_message(queue_name, timeout=0.1)

                if not message:
                    break

                # Find and execute handler
                if message.message_type in self.message_handlers:
                    try:
                        handler = self.message_handlers[message.message_type]

                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)

                        processed += 1
                        self.logger.debug(f"Processed message {message.id}")

                    except Exception as e:
                        self.logger.error(f"Handler failed for message {message.id}: {e}")

                        # Retry logic
                        if message.retry_count < message.max_retries:
                            message.retry_count += 1
                            await self.backend.put(queue_name, message)
                            self.logger.info(f"Retrying message {message.id} (attempt {message.retry_count})")
                        else:
                            self.logger.error(f"Message {message.id} exhausted retries")
                else:
                    self.logger.warning(f"No handler for message type: {message.message_type}")

            return processed

        except Exception as e:
            self.logger.error(f"Error processing queue {queue_name}: {e}")
            return processed

    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size"""
        return await self.backend.size(queue_name)

    async def clear_queue(self, queue_name: str) -> bool:
        """Clear queue"""
        return await self.backend.clear(queue_name)

    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            'messages_sent': self.metrics['messages_sent'],
            'messages_received': self.metrics['messages_received'],
            'messages_failed': self.metrics['messages_failed'],
            'active_queues': len(self.metrics['queues_active']),
            'queue_names': list(self.metrics['queues_active']),
            'rate_limit_rps': self.rate_limiter.requests_per_second,
            'total_requests': self.rate_limiter.request_count
        }

    async def health_check(self) -> Dict[str, Any]:
        """System health check"""

        backend_healthy = await self.backend.health_check()

        return {
            'backend_healthy': backend_healthy,
            'system_healthy': backend_healthy,
            'metrics': self.get_metrics()
        }

# Message handlers for testing
async def data_collection_handler(message: QueueMessage) -> None:
    """Handler for data collection messages"""
    logger = get_structured_logger("DataCollectionHandler")
    logger.info(f"Processing data collection: {message.payload.get('symbol', 'unknown')}")

async def ml_prediction_handler(message: QueueMessage) -> None:
    """Handler for ML prediction messages"""
    logger = get_structured_logger("MLPredictionHandler")
    logger.info(f"Processing ML prediction: {message.payload.get('model', 'unknown')}")

if __name__ == "__main__":
    async def test_async_queue_system():
        """Test async queue system"""

        print("🔍 TESTING ASYNC QUEUE SYSTEM")
        print("=" * 60)

        # Create queue system with asyncio backend
        backend = AsyncioQueueBackend(max_queue_size=1000)
        rate_limiter = RateLimiter(requests_per_second=20.0)
        queue_system = AsyncQueueSystem(backend, rate_limiter)

        # Register handlers
        queue_system.register_handler('data_collection', data_collection_handler)
        queue_system.register_handler('ml_prediction', ml_prediction_handler)

        print("📤 Sending test messages...")

        # Send various messages
        messages_sent = 0

        # Data collection messages
        for i in range(5):
            success = await queue_system.send_message(
                queue_name="data_pipeline",
                message_type="data_collection",
                payload={"symbol": f"BTC{i}", "exchange": "kraken"},
                sender="test_producer",
                priority=MessagePriority.HIGH
            )
            if success:
                messages_sent += 1

        # ML prediction messages
        for i in range(3):
            success = await queue_system.send_message(
                queue_name="ml_pipeline",
                message_type="ml_prediction",
                payload={"model": f"ensemble_{i}", "horizon": "30d"},
                sender="test_producer",
                priority=MessagePriority.NORMAL
            )
            if success:
                messages_sent += 1

        print(f"   Messages sent: {messages_sent}")

        # Check queue sizes
        data_queue_size = await queue_system.get_queue_size("data_pipeline")
        ml_queue_size = await queue_system.get_queue_size("ml_pipeline")

        print(f"   Data queue size: {data_queue_size}")
        print(f"   ML queue size: {ml_queue_size}")

        print("\n📥 Processing messages...")

        # Process data pipeline
        processed_data = await queue_system.process_queue("data_pipeline", max_messages=10)
        print(f"   Processed data messages: {processed_data}")

        # Process ML pipeline
        processed_ml = await queue_system.process_queue("ml_pipeline", max_messages=10)
        print(f"   Processed ML messages: {processed_ml}")

        # System metrics
        print("\n📊 System metrics:")
        metrics = queue_system.get_metrics()
        for key, value in metrics.items():
            print(f"   {key}: {value}")

        # Health check
        print("\n🏥 Health check:")
        health = await queue_system.health_check()
        for key, value in health.items():
            if key != 'metrics':
                print(f"   {key}: {value}")

        print("\n✅ ASYNC QUEUE SYSTEM TEST COMPLETED")

    # Run test
    asyncio.run(test_async_queue_system())
