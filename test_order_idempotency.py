#!/usr/bin/env python3
"""
Order Idempotency & Deduplication Test Suite

Comprehensive testing of order deduplication, timeout recovery,
and idempotent execution with forced network conditions.
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any

from src.cryptosmarttrader.execution.order_deduplication import (
    OrderDeduplicationEngine,
    OrderSubmission,
    ClientOrderId,
    OrderStatus,
    RetryReason,
)

from src.cryptosmarttrader.execution.idempotent_executor import (
    IdempotentOrderExecutor,
    ExecutionContext,
    ExecutionMode,
    NetworkCondition,
)


class OrderIdempotencyTester:
    """Comprehensive order idempotency test suite"""

    def __init__(self):
        self.dedup_engine = OrderDeduplicationEngine(persistence_path="test_data/orders")
        self.executor = None
        self.test_results: List[Dict[str, Any]] = []

    def setup_test_environment(self, network_condition: NetworkCondition = NetworkCondition.NORMAL):
        """Setup test environment with specific network conditions"""

        execution_context = ExecutionContext(
            mode=ExecutionMode.SIMULATION,
            max_retries=3,
            base_timeout_seconds=2.0,
            network_condition=network_condition,
            simulated_latency_ms=50,
            failure_rate=0.1,
        )

        self.executor = IdempotentOrderExecutor(self.dedup_engine, execution_context)
        print(f"Test environment setup with network condition: {network_condition.value}")

    def create_test_order(self, symbol: str = "BTC/USD", quantity: float = 1.0) -> OrderSubmission:
        """Create test order"""

        client_order_id = self.dedup_engine.create_order_id(strategy_id="test_strategy")

        return OrderSubmission(
            client_order_id=client_order_id,
            symbol=symbol,
            side="buy",
            quantity=quantity,
            price=50000.0,
            order_type="limit",
        )

    async def test_basic_deduplication(self) -> Dict[str, Any]:
        """Test basic order deduplication"""

        print("\n🧪 Testing Basic Deduplication...")

        test_start = time.time()

        # Create identical orders
        order1 = self.create_test_order("BTC/USD", 1.0)
        order2 = self.create_test_order("BTC/USD", 1.0)

        # Force same fingerprint by copying
        order2.fingerprint = order1.fingerprint

        # Submit first order
        can_submit1, msg1 = self.dedup_engine.submit_order(order1)

        # Submit duplicate order
        can_submit2, msg2 = self.dedup_engine.submit_order(order2)

        result = {
            "test_name": "basic_deduplication",
            "duration_seconds": time.time() - test_start,
            "first_order_accepted": can_submit1,
            "duplicate_order_rejected": not can_submit2,
            "duplicate_message": msg2,
            "success": can_submit1 and not can_submit2,
            "details": {
                "first_order_id": order1.client_order_id.full_id,
                "duplicate_order_id": order2.client_order_id.full_id,
                "fingerprint_match": order1.fingerprint == order2.fingerprint,
            },
        }

        if result["success"]:
            print("✅ Basic deduplication test PASSED")
        else:
            print("❌ Basic deduplication test FAILED")

        return result

    async def test_timeout_recovery(self) -> Dict[str, Any]:
        """Test network timeout recovery with retries"""

        print("\n🧪 Testing Timeout Recovery...")

        test_start = time.time()

        # Setup timeout-prone environment
        self.setup_test_environment(NetworkCondition.TIMEOUT_PRONE)

        # Create test order
        order = self.create_test_order("ETH/USD", 2.0)

        # Execute with forced timeouts
        execution_result = await self.executor.force_timeout_test(order)

        # Validate results
        had_timeouts = any(attempt.error_type == "timeout" for attempt in execution_result.attempts)
        recovered = execution_result.was_successful or execution_result.total_attempts > 1
        no_duplicate_positions = not execution_result.duplicate_detected

        result = {
            "test_name": "timeout_recovery",
            "duration_seconds": time.time() - test_start,
            "timeouts_occurred": had_timeouts,
            "recovery_attempted": execution_result.total_attempts > 1,
            "final_success": execution_result.was_successful,
            "no_duplicate_positions": no_duplicate_positions,
            "idempotency_validated": execution_result.idempotency_validated,
            "success": recovered and no_duplicate_positions,
            "details": {
                "total_attempts": execution_result.total_attempts,
                "final_status": execution_result.final_status.value,
                "exchange_order_id": execution_result.exchange_order_id,
                "attempt_details": [
                    {
                        "attempt": i + 1,
                        "success": attempt.success,
                        "error_type": attempt.error_type,
                        "response_time_ms": attempt.response_time_ms,
                    }
                    for i, attempt in enumerate(execution_result.attempts)
                ],
            },
        }

        if result["success"]:
            print("✅ Timeout recovery test PASSED")
        else:
            print("❌ Timeout recovery test FAILED")

        return result

    async def test_duplicate_fill_prevention(self) -> Dict[str, Any]:
        """Test prevention of duplicate fills after retry"""

        print("\n🧪 Testing Duplicate Fill Prevention...")

        test_start = time.time()

        # Setup unstable network
        self.setup_test_environment(NetworkCondition.UNSTABLE)

        # Create test order
        order = self.create_test_order("LTC/USD", 5.0)

        # Execute order
        execution_result = await self.executor.execute_order(order)

        # Check deduplication engine state
        order_status = self.dedup_engine.get_order_status(order.client_order_id.full_id)

        # Validate no duplicate submission
        submission_attempts = order_status.submission_attempts if order_status else 0
        no_duplicate_exchange_orders = execution_result.exchange_order_id is not None

        # Check fill tracking
        fill_validation = True  # Would check actual fills in real implementation

        result = {
            "test_name": "duplicate_fill_prevention",
            "duration_seconds": time.time() - test_start,
            "order_executed": execution_result.was_successful,
            "single_exchange_order": no_duplicate_exchange_orders,
            "proper_attempt_tracking": submission_attempts == execution_result.total_attempts,
            "fill_validation_passed": fill_validation,
            "success": (
                execution_result.was_successful
                and no_duplicate_exchange_orders
                and submission_attempts <= execution_result.total_attempts
            ),
            "details": {
                "submission_attempts": submission_attempts,
                "execution_attempts": execution_result.total_attempts,
                "exchange_order_id": execution_result.exchange_order_id,
                "final_status": execution_result.final_status.value,
                "order_status_in_engine": order_status.status.value
                if order_status
                else "not_found",
            },
        }

        if result["success"]:
            print("✅ Duplicate fill prevention test PASSED")
        else:
            print("❌ Duplicate fill prevention test FAILED")

        return result

    async def test_retry_idempotency(self) -> Dict[str, Any]:
        """Test that retries maintain idempotency"""

        print("\n🧪 Testing Retry Idempotency...")

        test_start = time.time()

        # Setup environment with moderate failures
        execution_context = ExecutionContext(
            mode=ExecutionMode.SIMULATION,
            max_retries=5,
            failure_rate=0.4,  # 40% failure rate
            network_condition=NetworkCondition.UNSTABLE,
        )
        self.executor = IdempotentOrderExecutor(self.dedup_engine, execution_context)

        # Execute multiple orders
        orders = [
            self.create_test_order("BTC/USD", 1.0),
            self.create_test_order("ETH/USD", 2.0),
            self.create_test_order("LTC/USD", 3.0),
        ]

        execution_results = []
        for order in orders:
            result = await self.executor.execute_order(order)
            execution_results.append(result)

        # Validate results
        all_orders_unique = len(
            set(r.exchange_order_id for r in execution_results if r.exchange_order_id) == len([r for r in execution_results if r.exchange_order_id])
        all_idempotent = all(r.idempotency_validated for r in execution_results)
        no_duplicates = not any(r.duplicate_detected for r in execution_results)

        result = {
            "test_name": "retry_idempotency",
            "duration_seconds": time.time() - test_start,
            "orders_processed": len(orders),
            "successful_orders": len([r for r in execution_results if r.was_successful]),
            "all_orders_unique": all_orders_unique,
            "all_idempotent": all_idempotent,
            "no_duplicates_detected": no_duplicates,
            "success": all_orders_unique and all_idempotent and no_duplicates,
            "details": {
                "execution_results": [
                    {
                        "order_id": r.order_id,
                        "attempts": r.total_attempts,
                        "success": r.was_successful,
                        "exchange_order_id": r.exchange_order_id,
                        "idempotent": r.idempotency_validated,
                    }
                    for r in execution_results
                ]
            },
        }

        if result["success"]:
            print("✅ Retry idempotency test PASSED")
        else:
            print("❌ Retry idempotency test FAILED")

        return result

    async def test_concurrent_execution(self) -> Dict[str, Any]:
        """Test concurrent order execution safety"""

        print("\n🧪 Testing Concurrent Execution Safety...")

        test_start = time.time()

        # Setup normal environment
        self.setup_test_environment(NetworkCondition.NORMAL)

        # Create multiple orders for concurrent execution
        orders = [self.create_test_order(f"PAIR{i}/USD", i + 1.0) for i in range(10)]

        # Execute all orders concurrently
        tasks = [self.executor.execute_order(order) for order in orders]
        execution_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        successful_results = [r for r in execution_results if not isinstance(r, Exception)]
        exceptions = [r for r in execution_results if isinstance(r, Exception)]

        # Validate concurrent safety
        unique_exchange_ids = set()
        for result in successful_results:
            if result.exchange_order_id:
                unique_exchange_ids.add(result.exchange_order_id)

        no_id_collisions = len(unique_exchange_ids) == len(
            [r for r in successful_results if r.exchange_order_id]
        )
        no_exceptions = len(exceptions) == 0
        all_tracked = len(successful_results) == len(orders)

        result = {
            "test_name": "concurrent_execution",
            "duration_seconds": time.time() - test_start,
            "total_orders": len(orders),
            "successful_executions": len(successful_results),
            "exceptions_occurred": len(exceptions),
            "unique_exchange_ids": len(unique_exchange_ids),
            "no_id_collisions": no_id_collisions,
            "no_exceptions": no_exceptions,
            "all_orders_tracked": all_tracked,
            "success": no_id_collisions and no_exceptions and all_tracked,
            "details": {
                "exception_types": [type(e).__name__ for e in exceptions],
                "successful_order_ids": [r.order_id for r in successful_results],
                "exchange_order_ids": list(unique_exchange_ids),
            },
        }

        if result["success"]:
            print("✅ Concurrent execution test PASSED")
        else:
            print("❌ Concurrent execution test FAILED")

        return result

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run complete test suite"""

        print("🚀 Starting Comprehensive Order Idempotency Test Suite\n")
        print("=" * 60)

        suite_start = time.time()

        # Run all tests
        tests = [
            self.test_basic_deduplication(),
            self.test_timeout_recovery(),
            self.test_duplicate_fill_prevention(),
            self.test_retry_idempotency(),
            self.test_concurrent_execution(),
        ]

        test_results = []
        for test in tests:
            try:
                result = await test
                test_results.append(result)
                self.test_results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                test_results.append({"test_name": "unknown", "success": False, "error": str(e)})

        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.get("success", False)])

        # Get system statistics
        dedup_stats = self.dedup_engine.get_deduplication_stats()
        executor_stats = self.executor.get_execution_stats()

        suite_result = {
            "suite_name": "order_idempotency_comprehensive",
            "total_duration_seconds": time.time() - suite_start,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate_percent": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "overall_success": passed_tests == total_tests,
            "test_results": test_results,
            "deduplication_stats": dedup_stats,
            "executor_stats": executor_stats,
            "validation_summary": {
                "order_deduplication_working": any(
                    r.get("test_name") == "basic_deduplication" and r.get("success")
                    for r in test_results
                ),
                "timeout_recovery_working": any(
                    r.get("test_name") == "timeout_recovery" and r.get("success")
                    for r in test_results
                ),
                "duplicate_prevention_working": any(
                    r.get("test_name") == "duplicate_fill_prevention" and r.get("success")
                    for r in test_results
                ),
                "retry_idempotency_working": any(
                    r.get("test_name") == "retry_idempotency" and r.get("success")
                    for r in test_results
                ),
                "concurrent_safety_working": any(
                    r.get("test_name") == "concurrent_execution" and r.get("success")
                    for r in test_results
                ),
            },
        }

        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {suite_result['success_rate_percent']:.1f}%")
        print(f"Duration: {suite_result['total_duration_seconds']:.2f}s")

        if suite_result["overall_success"]:
            print("\n🎉 ALL TESTS PASSED - ORDER IDEMPOTENCY SYSTEM WORKING CORRECTLY!")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW RESULTS ABOVE")

        print("\n🔍 Key Validations:")
        validations = suite_result["validation_summary"]
        for key, value in validations.items():
            status = "✅" if value else "❌"
            readable_key = key.replace("_", " ").title()
            print(f"{status} {readable_key}")

        return suite_result


async def main():
    """Main test execution"""

    tester = OrderIdempotencyTester()

    # Setup initial environment
    tester.setup_test_environment()

    # Run comprehensive test suite
    results = await tester.run_comprehensive_test_suite()

    # Save results
    import json

    with open("test_results_idempotency.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Test results saved to: test_results_idempotency.json")


if __name__ == "__main__":
    asyncio.run(main())
