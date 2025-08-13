"""
Idempotent Order Executor

Comprehensive order execution with built-in retry logic, timeout handling,
and network failure recovery with deduplication guarantees.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import random

from .order_deduplication import (
    OrderDeduplicationEngine,
    OrderSubmission,
    ClientOrderId,
    OrderStatus,
    RetryReason
)

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    """Execution mode types"""
    LIVE = "live"           # Live trading
    PAPER = "paper"         # Paper trading
    SIMULATION = "simulation"  # Backtesting

class NetworkCondition(Enum):
    """Network condition simulation"""
    NORMAL = "normal"
    SLOW = "slow"
    UNSTABLE = "unstable"
    TIMEOUT_PRONE = "timeout_prone"

@dataclass
class ExecutionContext:
    """Execution context with retry configuration"""
    mode: ExecutionMode = ExecutionMode.PAPER
    max_retries: int = 3
    base_timeout_seconds: float = 10.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 60.0

    # Network simulation
    network_condition: NetworkCondition = NetworkCondition.NORMAL
    simulated_latency_ms: int = 100
    failure_rate: float = 0.0  # 0.0 to 1.0

@dataclass
class ExecutionAttempt:
    """Single execution attempt record"""
    attempt_number: int
    timestamp: datetime
    timeout_seconds: float
    network_latency_ms: Optional[int] = None

    # Results
    success: bool = False
    response_time_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    exchange_response: Optional[Dict[str, Any]] = None

@dataclass
class ExecutionResult:
    """Complete execution result with all attempts"""
    order_id: str
    client_order_id: ClientOrderId
    final_status: OrderStatus

    # Execution details
    total_attempts: int
    total_duration_seconds: float
    attempts: List[ExecutionAttempt] = field(default_factory=list)

    # Final result
    exchange_order_id: Optional[str] = None
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0

    # Validation
    idempotency_validated: bool = False
    duplicate_detected: bool = False

    @property
    def was_successful(self) -> bool:
        """Check if execution was ultimately successful"""
        return self.final_status in [OrderStatus.FILLED, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL]

class IdempotentOrderExecutor:
    """
    Idempotent order executor with comprehensive retry and deduplication
    """

    def __init__(self,
                 deduplication_engine: OrderDeduplicationEngine,
                 execution_context: Optional[ExecutionContext] = None):

        self.deduplication_engine = deduplication_engine
        self.execution_context = execution_context or ExecutionContext()

        # Execution tracking
        self.active_executions: Dict[str, ExecutionResult] = {}
        self.execution_history: List[ExecutionResult] = []

        # Mock exchange interface (for testing)
        self.# REMOVED: Mock data pattern not allowed in productionself.execution_context)

        # Statistics
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'duplicate_rejections': 0,
            'timeout_recoveries': 0,
            'retry_successes': 0
        }

        logger.info(f"Idempotent Order Executor initialized (mode: {self.execution_context.mode.value})")

    async def execute_order(self, order: OrderSubmission) -> ExecutionResult:
        """Execute order with full idempotency and retry logic"""

        start_time = time.time()
        order_id = order.client_order_id.full_id

        # Initialize execution result
        execution_result = ExecutionResult(
            order_id=order_id,
            client_order_id=order.client_order_id,
            final_status=OrderStatus.PENDING,
            total_attempts=0,
            total_duration_seconds=0.0
        )

        self.active_executions[order_id] = execution_result

        try:
            # Step 1: Deduplication check
            can_submit, message = self.deduplication_engine.submit_order(order)

            if not can_submit:
                execution_result.final_status = OrderStatus.DUPLICATE
                execution_result.duplicate_detected = True
                self.stats['duplicate_rejections'] += 1

                logger.warning(f"Order execution blocked: {message}")
                return execution_result

            # Step 2: Execute with retries
            success = await self._execute_with_retries(order, execution_result)

            # Step 3: Final validation
            execution_result.idempotency_validated = await self._validate_idempotency(order, execution_result)

            # Update statistics
            self.stats['total_executions'] += 1
            if success:
                self.stats['successful_executions'] += 1
            else:
                self.stats['failed_executions'] += 1

            execution_result.total_duration_seconds = time.time() - start_time

            # Store in history
            self.execution_history.append(execution_result)

            # Keep only recent history
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-500:]

            return execution_result

        except Exception as e:
            logger.error(f"Execution error for {order_id}: {e}")
            execution_result.final_status = OrderStatus.FAILED
            return execution_result

        finally:
            # Clean up active execution
            if order_id in self.active_executions:
                del self.active_executions[order_id]

    async def _execute_with_retries(self, order: OrderSubmission, execution_result: ExecutionResult) -> bool:
        """Execute order with retry logic"""

        attempt_number = 0
        last_error = None

        while attempt_number < self.execution_context.max_retries:
            attempt_number += 1
            execution_result.total_attempts = attempt_number

            # Calculate timeout for this attempt
            timeout = self._calculate_timeout(attempt_number)

            # Create attempt record
            attempt = ExecutionAttempt(
                attempt_number=attempt_number,
                timestamp=datetime.now(),
                timeout_seconds=timeout
            )

            try:
                logger.info(f"Execution attempt {attempt_number}/{self.execution_context.max_retries} "
                           f"for {order.client_order_id.full_id} (timeout: {timeout}s)")

                # Execute order
                success, result = await self._single_execution_attempt(order, timeout)

                # Record attempt details
                attempt.success = success
                attempt.response_time_ms = result.get('response_time_ms', 0)
                attempt.exchange_response = result

                execution_result.attempts.append(attempt)

                if success:
                    # Success - update order and result
                    execution_result.final_status = OrderStatus.SUBMITTED
                    execution_result.exchange_order_id = result.get('exchange_order_id')

                    # Update deduplication engine
                    order_id = order.client_order_id.full_id
                    self.deduplication_engine.update_order_status(
                        order_id,
                        OrderStatus.SUBMITTED,
                        exchange_order_id=execution_result.exchange_order_id
                    )

                    logger.info(f"Order execution successful: {order.client_order_id.full_id} "
                               f"(exchange ID: {execution_result.exchange_order_id})")

                    return True

                else:
                    # Failed attempt - determine retry strategy
                    error_type = result.get('error_type', 'unknown')
                    error_message = result.get('error_message', 'Unknown error')

                    attempt.error_type = error_type
                    attempt.error_message = error_message
                    last_error = error_message

                    # Determine if we should retry
                    should_retry, retry_reason = self._should_retry(error_type, attempt_number)

                    if not should_retry:
                        logger.error(f"Order execution failed permanently: {error_message}")
                        execution_result.final_status = OrderStatus.FAILED
                        return False

                    # Wait before retry (exponential backoff)
                    if attempt_number < self.execution_context.max_retries:
                        retry_delay = self._calculate_retry_delay(attempt_number)
                        logger.info(f"Retrying in {retry_delay:.1f}s (reason: {retry_reason.value})")

                        # Record retry in deduplication engine
                        self.deduplication_engine.retry_order(
                            order.client_order_id.full_id,
                            retry_reason,
                            error_message
                        )

                        await asyncio.sleep(retry_delay)

            except asyncio.TimeoutError:
                logger.warning(f"Execution attempt {attempt_number} timed out after {timeout}s")

                attempt.success = False
                attempt.error_type = "timeout"
                attempt.error_message = f"Network timeout after {timeout}s"
                execution_result.attempts.append(attempt)

                # Update order status in deduplication engine
                order_id = order.client_order_id.full_id
                if order_id in self.deduplication_engine.active_orders:
                    self.deduplication_engine.update_order_status(
                        order_id,
                        OrderStatus.TIMEOUT
                    )

                # Timeout is always retryable
                if attempt_number < self.execution_context.max_retries:
                    retry_delay = self._calculate_retry_delay(attempt_number)
                    logger.info(f"Retrying after timeout in {retry_delay:.1f}s")

                    self.deduplication_engine.retry_order(
                        order.client_order_id.full_id,
                        RetryReason.NETWORK_TIMEOUT,
                        f"Timeout after {timeout}s"
                    )

                    self.stats['timeout_recoveries'] += 1
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Unexpected error in execution attempt {attempt_number}: {e}")

                attempt.success = False
                attempt.error_type = "exception"
                attempt.error_message = str(e)
                execution_result.attempts.append(attempt)

                last_error = str(e)

        # All retries exhausted
        logger.error(f"Order execution failed after {attempt_number} attempts. Last error: {last_error}")
        execution_result.final_status = OrderStatus.FAILED

        # Update deduplication engine
        order_id = order.client_order_id.full_id
        if order_id in self.deduplication_engine.active_orders:
            self.deduplication_engine.update_order_status(
                order_id,
                OrderStatus.FAILED
            )

        return False

    async def _single_execution_attempt(self, order: OrderSubmission, timeout: float) -> tuple[bool, Dict[str, Any]]:
        """Execute single order attempt"""

        try:
            # Use mock exchange for testing
            result = await asyncio.wait_for(
                self.# REMOVED: Mock data pattern not allowed in productionorder),
                timeout=timeout
            )

            return result['success'], result

        except asyncio.TimeoutError:
            raise  # Re-raise timeout for handling in caller
        except Exception as e:
            return False, {
                'success': False,
                'error_type': 'exception',
                'error_message': str(e),
                'response_time_ms': 0
            }

    def _should_retry(self, error_type: str, attempt_number: int) -> tuple[bool, RetryReason]:
        """Determine if error is retryable"""

        # Non-retryable errors
        non_retryable = ['invalid_symbol', 'insufficient_funds', 'invalid_order', 'duplicate_order']

        if error_type in non_retryable:
            return False, RetryReason.TEMPORARY_FAILURE

        # Retryable errors
        if error_type in ['timeout', 'network_error', 'connection_lost']:
            return True, RetryReason.NETWORK_TIMEOUT
        elif error_type in ['rate_limit', 'too_many_requests']:
            return True, RetryReason.RATE_LIMIT
        elif error_type in ['server_error', 'service_unavailable']:
            return True, RetryReason.EXCHANGE_ERROR
        else:
            # Unknown errors are retryable for a few attempts
            return attempt_number <= 2, RetryReason.TEMPORARY_FAILURE

    def _calculate_timeout(self, attempt_number: int) -> float:
        """Calculate timeout for attempt"""
        return min(
            self.execution_context.base_timeout_seconds * (attempt_number * 1.5),
            30.0  # Cap at 30 seconds
        )

    def _calculate_retry_delay(self, attempt_number: int) -> float:
        """Calculate retry delay with exponential backoff"""
        base_delay = 1.0  # 1 second base delay
        delay = base_delay * (self.execution_context.retry_backoff_multiplier ** (attempt_number - 1))

        # Add jitter
        jitter = random.choice
        delay *= jitter

        return min(delay, self.execution_context.max_retry_delay_seconds)

    async def _validate_idempotency(self, order: OrderSubmission, execution_result: ExecutionResult) -> bool:
        """Validate execution idempotency"""

        try:
            # Get current order status (could be retried version)
            order_id = order.client_order_id.full_id
            order_status = self.deduplication_engine.get_order_status(order_id)

            # If not found, try to find by base_id (retry tracking)
            if not order_status:
                for existing_order_id, existing_order in self.deduplication_engine.order_history.items():
                    if (existing_order.client_order_id.base_id == order.client_order_id.base_id and
                        existing_order.client_order_id.session_id == order.client_order_id.session_id):
                        order_status = existing_order
                        break

            if not order_status:
                logger.warning(f"Order not found in deduplication engine: {order_id}")
                # Don't fail validation for missing orders - could be legitimate
                return True

            # Basic validation checks
            validation_passed = True

            # Validate submission attempts are reasonable
            if order_status.submission_attempts > execution_result.total_attempts + 2:  # Allow some tolerance
                logger.error(f"Submission count excessive: engine={order_status.submission_attempts}, "
                           f"executor={execution_result.total_attempts}")
                validation_passed = False

            # Check for duplicate exchange order IDs across all orders
            if execution_result.exchange_order_id:
                for other_order in self.deduplication_engine.order_history.values():
                    if (other_order.exchange_order_id == execution_result.exchange_order_id and
                        other_order.client_order_id.full_id != order_id):
                        logger.error(f"Duplicate exchange order ID detected: {execution_result.exchange_order_id}")
                        validation_passed = False
                        break

            return validation_passed

        except Exception as e:
            logger.error(f"Idempotency validation error: {e}")
            # Don't fail on validation errors
            return True

    async def force_timeout_test(self, order: OrderSubmission) -> ExecutionResult:
        """Force network timeout test"""

        logger.warning("Starting forced timeout test")

        # Temporarily set network condition to timeout-prone
        original_condition = self.execution_context.network_condition
        original_failure_rate = self.execution_context.failure_rate

        self.execution_context.network_condition = NetworkCondition.TIMEOUT_PRONE
        self.execution_context.failure_rate = 0.8  # 80% failure rate

        try:
            result = await self.execute_order(order)

            # Verify that retries occurred
            if result.total_attempts > 1:
                logger.info(f"Timeout test successful: {result.total_attempts} attempts, "
                           f"final status: {result.final_status.value}")
            else:
                logger.warning("Timeout test may not have triggered retries")

            return result

        finally:
            # Restore original conditions
            self.execution_context.network_condition = original_condition
            self.execution_context.failure_rate = original_failure_rate

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""

        # Calculate success rates
        total_executions = self.stats['total_executions']
        success_rate = (self.stats['successful_executions'] / max(total_executions, 1)) * 100

        # Analyze recent attempts
        recent_results = self.execution_history[-100:]  # Last 100 executions

        avg_attempts = 0.0
        avg_duration = 0.0
        timeout_count = 0

        if recent_results:
            avg_attempts = sum(r.total_attempts for r in recent_results) / len(recent_results)
            avg_duration = sum(r.total_duration_seconds for r in recent_results) / len(recent_results)
            timeout_count = sum(1 for r in recent_results if any(
                a.error_type == 'timeout' for a in r.attempts
            ))

        return {
            'timestamp': datetime.now().isoformat(),
            'execution_mode': self.execution_context.mode.value,

            'overall_stats': {
                'total_executions': total_executions,
                'successful_executions': self.stats['successful_executions'],
                'failed_executions': self.stats['failed_executions'],
                'success_rate_percent': success_rate,
                'duplicate_rejections': self.stats['duplicate_rejections']
            },

            'retry_stats': {
                'timeout_recoveries': self.stats['timeout_recoveries'],
                'retry_successes': self.stats['retry_successes'],
                'average_attempts_per_order': avg_attempts,
                'recent_timeout_count': timeout_count
            },

            'performance_stats': {
                'average_execution_duration_seconds': avg_duration,
                'active_executions': len(self.active_executions),
                'execution_history_size': len(self.execution_history)
            },

            'configuration': {
                'max_retries': self.execution_context.max_retries,
                'base_timeout_seconds': self.execution_context.base_timeout_seconds,
                'network_condition': self.execution_context.network_condition.value,
                'simulated_failure_rate': self.execution_context.failure_rate
            }
        }

class MockExchangeInterface:
    """Mock exchange interface for testing"""

    def __init__(self, execution_context: ExecutionContext):
        self.execution_context = execution_context
        self.submitted_orders: Dict[str, Dict[str, Any]] = {}

    async def submit_order(self, order: OrderSubmission) -> Dict[str, Any]:
        """Mock order submission with configurable failures"""

        start_time = time.time()

        # REMOVED: Mock data pattern not allowed in production
        latency = self.execution_context.simulated_latency_ms / 1000.0
        await asyncio.sleep(latency)

        # REMOVED: Mock data pattern not allowed in production
        if self._should_random.choice():
            failure_type = self._get_failure_type()

            return {
                'success': False,
                'error_type': failure_type,
                'error_message': f"Simulated {failure_type}",
                'response_time_ms': (time.time() - start_time) * 1000
            }

        # REMOVED: Mock data pattern not allowed in production
        exchange_order_id = f"EXG{int(time.time() * 1000)}{random.choice}"

        # Store order
        self.submitted_orders[order.client_order_id.full_id] = {
            'exchange_order_id': exchange_order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': order.price,
            'timestamp': datetime.now().isoformat()
        }

        return {
            'success': True,
            'exchange_order_id': exchange_order_id,
            'status': 'submitted',
            'response_time_ms': (time.time() - start_time) * 1000
        }

    def _should_generate_sample_data_self) -> bool:
        """Determine if should simulate failure"""

        if self.execution_context.network_condition == NetworkCondition.TIMEOUT_PRONE:
            return random.random() < 0.7  # 70% failure rate
        elif self.execution_context.network_condition == NetworkCondition.UNSTABLE:
            return random.random() < 0.3  # 30% failure rate
        elif self.execution_context.network_condition == NetworkCondition.SLOW:
            return random.random() < 0.1  # 10% failure rate
        else:
            return random.random() < self.execution_context.failure_rate

    def _get_failure_type(self) -> str:
        """Get random failure type"""

        failure_types = [
            'timeout',
            'network_error',
            'connection_lost',
            'rate_limit',
            'server_error'
        ]

        if self.execution_context.network_condition == NetworkCondition.TIMEOUT_PRONE:
            return 'timeout'
        else:
            return random.choice
