#!/usr/bin/env python3
"""
Dashboard Gate Checker - Pre-Display Validation
Implements strict gates before showing any recommendations to users
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.logging_manager import get_logger

class DashboardGateChecker:
    """Implements strict gates for dashboard display logic"""
    
    def __init__(self):
        self.logger = get_logger()
        self.gate_id = f"gate_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def check_display_gates(
        self,
        predictions_file: str = "data/batch_output/predictions.csv",
        health_file: str = "logs/system/latest_health_assessment.json"
    ) -> Dict[str, Any]:
        """
        Check all gates before allowing dashboard display
        
        Returns:
            Dictionary with gate status and recommendations to display
        """
        
        self.logger.info(f"Starting dashboard gate check: {self.gate_id}")
        
        gate_results = {
            "gate_id": self.gate_id,
            "timestamp": datetime.now().isoformat(),
            "gates_checked": [],
            "gates_passed": [],
            "gates_failed": [],
            "final_decision": "blocked",
            "recommendations_to_display": [],
            "empty_state_reason": "unknown",
            "health_status": "unknown"
        }
        
        try:
            # Gate 1: Health Score Check
            health_gate_result = self._check_health_gate(health_file)
            gate_results.update(health_gate_result)
            
            # Gate 2: Strict Confidence Filter
            confidence_gate_result = self._check_confidence_gate(predictions_file)
            gate_results.update(confidence_gate_result)
            
            # Gate 3: Feature Completeness Check
            completeness_gate_result = self._check_completeness_gate(predictions_file)
            gate_results.update(completeness_gate_result)
            
            # Final decision logic
            self._make_final_gate_decision(gate_results)
            
            self.logger.info(
                f"Dashboard gate check completed: {self.gate_id} - {gate_results['final_decision']}"
            )
            
            return gate_results
            
        except Exception as e:
            self.logger.error(f"Dashboard gate check failed: {e}")
            gate_results["final_decision"] = "error"
            gate_results["empty_state_reason"] = f"Gate check error: {e}"
            return gate_results
    
    def _check_health_gate(self, health_file: str) -> Dict[str, Any]:
        """Check system health gate"""
        
        gate_name = "health_gate"
        gate_result = {
            "gates_checked": [gate_name],
            "gates_passed": [],
            "gates_failed": []
        }
        
        try:
            health_path = Path(health_file)
            
            if not health_path.exists():
                # No health assessment available
                gate_result["gates_failed"].append(gate_name)
                gate_result["health_status"] = "unknown"
                gate_result["empty_state_reason"] = "No system health assessment available"
                return gate_result
            
            # Load health assessment
            with open(health_path, 'r') as f:
                health_data = json.load(f)
            
            health_score = health_data.get("health_score", 0)
            decision = health_data.get("decision", "NO-GO")
            
            gate_result["health_status"] = decision
            gate_result["health_score"] = health_score
            
            # Health gate logic: require score ≥ 60 for ANY display
            if health_score >= 60:
                gate_result["gates_passed"].append(gate_name)
                self.logger.info(f"Health gate passed: score={health_score}, status={decision}")
            else:
                gate_result["gates_failed"].append(gate_name)
                gate_result["empty_state_reason"] = f"System health too low: {health_score:.1f} < 60.0"
                self.logger.warning(f"Health gate failed: score={health_score}")
            
        except Exception as e:
            gate_result["gates_failed"].append(gate_name)
            gate_result["health_status"] = "error"
            gate_result["empty_state_reason"] = f"Health check error: {e}"
            self.logger.error(f"Health gate check failed: {e}")
        
        return gate_result
    
    def _check_confidence_gate(self, predictions_file: str) -> Dict[str, Any]:
        """Check strict confidence threshold gate"""
        
        gate_name = "confidence_gate"
        gate_result = {
            "gates_checked": [gate_name],
            "gates_passed": [],
            "gates_failed": [],
            "high_confidence_count": 0,
            "total_predictions": 0
        }
        
        try:
            predictions_path = Path(predictions_file)
            
            if not predictions_path.exists():
                gate_result["gates_failed"].append(gate_name)
                gate_result["empty_state_reason"] = "No predictions available"
                return gate_result
            
            # Load predictions and apply strict filter
            import pandas as pd
            
            df = pd.read_csv(predictions_path)
            gate_result["total_predictions"] = len(df)
            
            if df.empty:
                gate_result["gates_failed"].append(gate_name)
                gate_result["empty_state_reason"] = "No predictions in file"
                return gate_result
            
            # Apply 80% confidence threshold across all horizons
            high_confidence_recommendations = []
            
            # Check each horizon
            horizons = [
                ("pred_1h", "conf_1h"),
                ("pred_4h", "conf_4h"), 
                ("pred_24h", "conf_24h"),
                ("pred_7d", "conf_7d"),
                ("pred_30d", "conf_30d")
            ]
            
            for pred_col, conf_col in horizons:
                if pred_col in df.columns and conf_col in df.columns:
                    # Filter for high confidence (≥80%) and positive predictions
                    high_conf = df[
                        (df[conf_col] >= 0.80) & 
                        (df[pred_col] > 0) &
                        (df[pred_col].notna()) &
                        (df[conf_col].notna())
                    ].copy()
                    
                    if not high_conf.empty:
                        # Sort by predicted return
                        high_conf = high_conf.sort_values(pred_col, ascending=False)
                        
                        # Add to recommendations
                        for _, row in high_conf.head(10).iterrows():  # Top 10 per horizon
                            high_confidence_recommendations.append({
                                "coin": row.get("coin", "Unknown"),
                                "horizon": pred_col.replace("pred_", ""),
                                "predicted_return": float(row[pred_col]),
                                "confidence": float(row[conf_col]),
                                "timestamp": row.get("timestamp", "")
                            })
            
            gate_result["high_confidence_count"] = len(high_confidence_recommendations)
            gate_result["recommendations_to_display"] = high_confidence_recommendations
            
            # Confidence gate logic: require at least 1 high-confidence recommendation
            if len(high_confidence_recommendations) >= 1:
                gate_result["gates_passed"].append(gate_name)
                self.logger.info(f"Confidence gate passed: {len(high_confidence_recommendations)} high-confidence recommendations")
            else:
                gate_result["gates_failed"].append(gate_name)
                gate_result["empty_state_reason"] = "No coins meet 80% confidence threshold"
                self.logger.warning("Confidence gate failed: no high-confidence recommendations")
        
        except Exception as e:
            gate_result["gates_failed"].append(gate_name)
            gate_result["empty_state_reason"] = f"Confidence check error: {e}"
            self.logger.error(f"Confidence gate check failed: {e}")
        
        return gate_result
    
    def _check_completeness_gate(self, predictions_file: str) -> Dict[str, Any]:
        """Check feature completeness gate"""
        
        gate_name = "completeness_gate"
        gate_result = {
            "gates_checked": [gate_name],
            "gates_passed": [],
            "gates_failed": [],
            "completeness_score": 0.0
        }
        
        try:
            predictions_path = Path(predictions_file)
            
            if not predictions_path.exists():
                gate_result["gates_failed"].append(gate_name)
                return gate_result
            
            # Load and analyze completeness
            import pandas as pd
            
            df = pd.read_csv(predictions_path)
            
            if df.empty:
                gate_result["gates_failed"].append(gate_name)
                return gate_result
            
            # Calculate completeness across required columns
            required_columns = [
                "coin", "timestamp",
                "pred_1h", "conf_1h",
                "pred_24h", "conf_24h",
                "pred_7d", "conf_7d",
                "pred_30d", "conf_30d"
            ]
            
            # Check which required columns exist and have data
            completeness_scores = []
            
            for col in required_columns:
                if col in df.columns:
                    non_null_ratio = df[col].notna().mean()
                    completeness_scores.append(non_null_ratio)
                else:
                    completeness_scores.append(0.0)
            
            overall_completeness = sum(completeness_scores) / len(completeness_scores)
            gate_result["completeness_score"] = overall_completeness
            
            # Completeness gate logic: require ≥90% feature completeness
            if overall_completeness >= 0.90:
                gate_result["gates_passed"].append(gate_name)
                self.logger.info(f"Completeness gate passed: {overall_completeness:.1%}")
            else:
                gate_result["gates_failed"].append(gate_name)
                gate_result["empty_state_reason"] = f"Incomplete features: {overall_completeness:.1%} < 90%"
                self.logger.warning(f"Completeness gate failed: {overall_completeness:.1%}")
        
        except Exception as e:
            gate_result["gates_failed"].append(gate_name)
            gate_result["empty_state_reason"] = f"Completeness check error: {e}"
            self.logger.error(f"Completeness gate check failed: {e}")
        
        return gate_result
    
    def _make_final_gate_decision(self, gate_results: Dict[str, Any]) -> None:
        """Make final decision on whether to display recommendations"""
        
        total_gates = len(gate_results["gates_checked"])
        passed_gates = len(gate_results["gates_passed"])
        failed_gates = len(gate_results["gates_failed"])
        
        # Strict gate logic: ALL gates must pass
        if failed_gates == 0 and passed_gates == total_gates:
            gate_results["final_decision"] = "display"
            self.logger.info(f"All {total_gates} gates passed - recommendations will be displayed")
        else:
            gate_results["final_decision"] = "blocked"
            gate_results["recommendations_to_display"] = []  # Clear recommendations
            
            if not gate_results.get("empty_state_reason"):
                gate_results["empty_state_reason"] = f"{failed_gates}/{total_gates} gates failed"
            
            self.logger.warning(f"Gates failed: {failed_gates}/{total_gates} - no recommendations will be displayed")

def print_gate_status(gate_results: Dict[str, Any]) -> None:
    """Print human-readable gate status"""
    
    decision = gate_results["final_decision"]
    decision_icon = {
        "display": "✅",
        "blocked": "❌", 
        "error": "🚨"
    }.get(decision, "❓")
    
    print(f"🚪 DASHBOARD GATE CHECK RESULTS")
    print(f"📅 Timestamp: {gate_results['timestamp']}")
    print("=" * 50)
    
    print(f"🚦 FINAL DECISION: {decision_icon} {decision.upper()}")
    
    if decision == "display":
        recommendations = gate_results.get("recommendations_to_display", [])
        print(f"📊 Recommendations to display: {len(recommendations)}")
    else:
        reason = gate_results.get("empty_state_reason", "Unknown")
        print(f"🚫 Empty state reason: {reason}")
    
    print()
    
    # Gate breakdown
    print(f"🔍 GATE BREAKDOWN:")
    
    for gate in gate_results["gates_checked"]:
        if gate in gate_results["gates_passed"]:
            print(f"   ✅ {gate.replace('_', ' ').title()}")
        else:
            print(f"   ❌ {gate.replace('_', ' ').title()}")
    
    print()
    
    # Health status
    if "health_status" in gate_results:
        health_icon = {"GO": "🟢", "WARNING": "🟡", "NO-GO": "🔴"}.get(gate_results["health_status"], "❓")
        print(f"🏥 System Health: {health_icon} {gate_results['health_status']}")
        
        if "health_score" in gate_results:
            print(f"   Health Score: {gate_results['health_score']:.1f}/100")
    
    # Confidence metrics
    if "high_confidence_count" in gate_results:
        print(f"🎯 High Confidence: {gate_results['high_confidence_count']}/{gate_results.get('total_predictions', 0)} predictions")
    
    # Completeness metrics
    if "completeness_score" in gate_results:
        print(f"📋 Feature Completeness: {gate_results['completeness_score']:.1%}")

def main():
    """Main entry point for dashboard gate checker"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Dashboard Gate Checker - Pre-Display Validation"
    )
    
    parser.add_argument(
        '--predictions',
        default='data/batch_output/predictions.csv',
        help='Path to predictions CSV file'
    )
    
    parser.add_argument(
        '--health',
        default='logs/system/latest_health_assessment.json',
        help='Path to health assessment file'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"🚪 DASHBOARD GATE CHECK STARTING")
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        
        # Initialize gate checker
        checker = DashboardGateChecker()
        
        # Run gate checks
        results = checker.check_display_gates(args.predictions, args.health)
        
        # Display results
        print_gate_status(results)
        
        # Save results
        gate_results_dir = Path("logs/dashboard_gates")
        gate_results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = gate_results_dir / f"gate_check_{results['gate_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest
        latest_file = gate_results_dir / "latest_gate_check.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📝 Gate check results saved to: {results_file}")
        
        # Return appropriate exit code
        decision = results["final_decision"]
        
        if decision == "display":
            print("✅ GATES PASSED: Dashboard may display recommendations")
            return 0
        elif decision == "blocked":
            print("❌ GATES FAILED: Dashboard should show empty state")
            return 1
        else:
            print("🚨 GATE ERROR: Dashboard should show error state")
            return 2
            
    except Exception as e:
        print(f"❌ Gate check error: {e}")
        return 3

if __name__ == "__main__":
    sys.exit(main())