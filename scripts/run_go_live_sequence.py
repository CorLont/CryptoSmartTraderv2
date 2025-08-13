#!/usr/bin/env python3
"""
Go-Live Sequence Runner
Execute the complete staging → production deployment with monitoring and validation.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptosmarttrader.deployment.go_live_system import GoLiveManager, SLOMonitor, ChaosTestRunner


async def run_slo_validation():
    """Run SLO validation check."""
    print("🎯 SLO Validation Check")
    print("-" * 30)

    slo_monitor = SLOMonitor()

    for environment in ["staging", "production"]:
        print(f"\n📊 Checking {environment.title()} SLOs:")

        results = await slo_monitor.check_slo_compliance(environment)

        overall_emoji = "✅" if results["overall_compliance"] else "❌"
        print(f"{overall_emoji} Overall Compliance: {results['overall_compliance']}")

        for slo_name, slo_result in results["slo_results"].items():
            status_emoji = "✅" if slo_result["status"] == "compliant" else "❌"
            percentage = slo_result.get("percentage", 0)
            measurement = slo_result.get("measurement", "N/A")
            print(f"  {status_emoji} {slo_name}: {percentage:.1f}% ({measurement})")


async def run_chaos_tests():
    """Run chaos engineering tests."""
    print("\n🔥 Chaos Engineering Tests")
    print("-" * 30)

    chaos_tester = ChaosTestRunner()
    results = await chaos_tester.run_chaos_tests()

    success_emoji = "✅" if results["overall_success_rate"] >= 0.8 else "❌"
    print(f"{success_emoji} Overall Success Rate: {results['overall_success_rate']:.1%}")
    print(f"   Passed: {results['passed_tests']}/{results['total_tests']} tests")

    for test_name, test_result in results["test_results"].items():
        status_emoji = "✅" if test_result["status"] == "passed" else "❌"
        duration = test_result.get("duration", 0)
        print(
            f"  {status_emoji} {test_name.replace('_', ' ').title()}: {test_result['status']} ({duration:.2f}s)"
        )

        if test_result["status"] != "passed":
            reason = test_result.get("reason", "Unknown failure")
            print(f"      Reason: {reason}")


async def run_full_go_live():
    """Run the complete go-live sequence."""
    print("\n🚀 Full Go-Live Sequence")
    print("=" * 50)

    go_live_manager = GoLiveManager()

    # Show current status
    status = go_live_manager.get_deployment_status()
    print(f"Current Stage: {status['current_stage']}")
    print(f"Deployment History: {status['deployment_history_count']} previous deployments")

    # Execute sequence
    print("\n⏳ Executing go-live sequence...")
    result = await go_live_manager.execute_go_live_sequence()

    print(f"\n📋 Go-Live Results")
    print(f"Sequence ID: {result['sequence_id']}")
    print(f"Start Time: {result['start_time']}")
    print(f"End Time: {result.get('end_time', 'In Progress')}")

    # Result summary
    result_emoji = {
        "success": "🎉",
        "failed_staging": "🚫",
        "failed_canary_rolled_back": "🔄",
        "failed_production_rolled_back": "🚨",
        "sequence_failed": "💥",
    }.get(result["result"], "❓")

    print(f"{result_emoji} Result: {result['result']}")
    print(f"📝 Recommendation: {result['recommendation']}")

    # Stage details
    print(f"\n📊 Stage Results:")
    for stage_name, stage_result in result["stages"].items():
        success = stage_result.get("success", False)
        stage_emoji = "✅" if success else "❌"
        print(f"  {stage_emoji} {stage_name.title()}: {'Passed' if success else 'Failed'}")

        # Show issues if any
        if "issues" in stage_result and stage_result["issues"]:
            for issue in stage_result["issues"]:
                print(f"      ⚠️ {issue}")

        # Show metrics summary
        if "metrics" in stage_result:
            metrics = stage_result["metrics"]
            if "slo_compliance" in metrics:
                slo_compliance = metrics["slo_compliance"]["overall_compliance"]
                print(f"      📊 SLO Compliance: {'✅' if slo_compliance else '❌'}")

            if "chaos_tests" in metrics:
                chaos_success = metrics["chaos_tests"]["overall_success_rate"]
                print(f"      🔥 Chaos Tests: {chaos_success:.1%} success rate")

    # Save results
    results_file = Path(f"go_live_results_{result['sequence_id']}.json")
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {results_file}")

    return result


async def show_deployment_history():
    """Show deployment history."""
    print("\n📚 Deployment History")
    print("-" * 30)

    history_file = Path("deployment_history.json")

    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

        if history:
            print(f"Found {len(history)} previous deployments:")

            for i, deployment in enumerate(history[-5:], 1):  # Show last 5
                stage = deployment["stage"]
                start_time = deployment["start_time"]
                success_rate = deployment.get("success_rate", 0)
                rollback = deployment.get("rollback_triggered", False)

                success_emoji = "✅" if success_rate > 0.9 and not rollback else "❌"
                rollback_text = " (ROLLED BACK)" if rollback else ""

                print(
                    f"  {success_emoji} {i}. {stage} - {start_time} - {success_rate:.1%} success{rollback_text}"
                )
        else:
            print("No deployment history found.")
    else:
        print("No deployment history file found.")


async def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("🎯 CryptoSmartTrader V2 - Go-Live System")
    print("=" * 50)
    print("Enterprise staging → production deployment system")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "slo":
            await run_slo_validation()
        elif command == "chaos":
            await run_chaos_tests()
        elif command == "deploy":
            await run_full_go_live()
        elif command == "history":
            await show_deployment_history()
        elif command == "status":
            go_live_manager = GoLiveManager()
            status = go_live_manager.get_deployment_status()
            print(json.dumps(status, indent=2, default=str))
        else:
            print(f"❌ Unknown command: {command}")
            print_usage()
    else:
        # Interactive menu
        await interactive_menu()


async def interactive_menu():
    """Interactive menu for go-live operations."""
    while True:
        print("\n🎯 Go-Live Operations Menu")
        print("1. SLO Validation Check")
        print("2. Chaos Engineering Tests")
        print("3. Full Go-Live Sequence")
        print("4. Deployment History")
        print("5. Current Status")
        print("0. Exit")

        try:
            choice = input("\nSelect option (0-5): ").strip()

            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                await run_slo_validation()
            elif choice == "2":
                await run_chaos_tests()
            elif choice == "3":
                confirm = input(
                    "\n⚠️ Are you sure you want to start the go-live sequence? (yes/no): "
                )
                if confirm.lower() in ["yes", "y"]:
                    await run_full_go_live()
                else:
                    print("❌ Go-live sequence cancelled.")
            elif choice == "4":
                await show_deployment_history()
            elif choice == "5":
                go_live_manager = GoLiveManager()
                status = go_live_manager.get_deployment_status()
                print(json.dumps(status, indent=2, default=str))
            else:
                print("❌ Invalid option. Please select 0-5.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print("  python run_go_live_sequence.py                 # Interactive menu")
    print("  python run_go_live_sequence.py slo             # Run SLO validation")
    print("  python run_go_live_sequence.py chaos           # Run chaos tests")
    print("  python run_go_live_sequence.py deploy          # Execute go-live sequence")
    print("  python run_go_live_sequence.py history         # Show deployment history")
    print("  python run_go_live_sequence.py status          # Show current status")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user.")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        sys.exit(1)
