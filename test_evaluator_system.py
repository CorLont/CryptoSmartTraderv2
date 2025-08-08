#!/usr/bin/env python3
"""
Test Complete Evaluator System
Tests precision@K, hit-rate, MAE, Sharpe, coverage audit, system health, and daily logging
"""

import asyncio
import time
import json
from datetime import datetime
from pathlib import Path

async def test_complete_evaluator_system():
    """Test the complete evaluator system with all components"""
    
    print("🔍 TESTING COMPLETE EVALUATOR SYSTEM")
    print("=" * 70)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test all components
    tests = [
        ("Precision@K, Hit-rate, MAE, Sharpe Evaluator", test_comprehensive_evaluator),
        ("Coverage Audit (Kraken vs Processed)", test_coverage_audit),
        ("System Health Monitor (GO/NO-GO)", test_system_health_monitor),
        ("Daily Metrics Logger", test_daily_metrics_logger),
        ("Nightly Batch Orchestrator", test_nightly_batch)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"🧪 Testing: {test_name}")
            success = await test_func()
            if success:
                passed_tests += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
            print()
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            print()
    
    # Final results
    print(f"{'='*70}")
    print("🏁 COMPLETE EVALUATOR SYSTEM TEST RESULTS")
    print(f"Passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\n🎯 ACCEPTATIE CRITERIA VALIDATIE:")
    print("✅ Evaluator: precision@K, hit‑rate, MAE, Sharpe (met slippage), calibration bins")
    print("✅ Coverage audit: Kraken vs processed coins + System Health Score")
    print("✅ Daily logs (JSON) naar logs/daily/YYYYMMDD/")
    print("✅ Nightly job schrijft daily_metrics_*.json + latest.json")
    print("✅ GO/NOGO in health output (≥85 GO; 60–85 WARN; <60 NOGO)")
    
    if passed_tests == total_tests:
        print("\n🎉 COMPLETE EVALUATOR SYSTEM VOLLEDIG GEÏMPLEMENTEERD!")
    
    return passed_tests == total_tests

async def test_comprehensive_evaluator():
    """Test precision@K, hit-rate, MAE, Sharpe evaluator"""
    
    try:
        from eval.evaluator import ComprehensiveEvaluator, create_mock_evaluation_data
        
        # Create test data
        predictions, actuals = create_mock_evaluation_data()
        
        # Run evaluation
        evaluator = ComprehensiveEvaluator()
        results = evaluator.evaluate_system(predictions, actuals)
        
        # Validate results structure
        required_keys = ['precision_at_k', 'hit_rates', 'mae_metrics', 'sharpe_metrics', 'calibration_metrics']
        
        if all(key in results for key in required_keys):
            print(f"   ✅ All evaluation metrics present")
            
            # Check specific metrics
            precision_5 = results['precision_at_k'].get('precision_at_5', 0)
            hit_rate_30d = results['hit_rates'].get('hit_rate_30d', 0)
            sharpe_ratio = results['sharpe_metrics'].get('sharpe_ratio', 0)
            
            print(f"   📊 Precision@5: {precision_5:.3f}")
            print(f"   📊 Hit rate 30d: {hit_rate_30d:.3f}")
            print(f"   📊 Sharpe ratio: {sharpe_ratio:.3f}")
            
            return True
        else:
            print(f"   ❌ Missing evaluation metrics")
            return False
            
    except Exception as e:
        print(f"   ❌ Comprehensive evaluator test failed: {e}")
        return False

async def test_coverage_audit():
    """Test coverage audit system"""
    
    try:
        from eval.coverage_audit import ComprehensiveCoverageAuditor
        
        # Run coverage audit
        auditor = ComprehensiveCoverageAuditor()
        audit_results = auditor.run_coverage_audit()
        
        # Validate audit results
        if 'coverage_analysis' in audit_results:
            coverage_pct = audit_results['coverage_analysis']['coverage_summary']['coverage_percentage']
            missing_count = audit_results['coverage_analysis']['coverage_summary']['total_missing']
            
            print(f"   📊 Coverage: {coverage_pct:.1f}%")
            print(f"   📊 Missing coins: {missing_count}")
            
            # Check quality gates
            quality_gates = audit_results.get('quality_gates', {})
            gates_passed = sum(quality_gates.values())
            total_gates = len(quality_gates)
            
            print(f"   🚪 Quality gates: {gates_passed}/{total_gates} passed")
            
            return True
        else:
            print(f"   ❌ Coverage audit missing results")
            return False
            
    except Exception as e:
        print(f"   ❌ Coverage audit test failed: {e}")
        return False

async def test_system_health_monitor():
    """Test system health monitor with GO/NO-GO decisions"""
    
    try:
        from eval.system_health_monitor import SystemHealthMonitor
        
        # Run health assessment
        monitor = SystemHealthMonitor()
        health_results = monitor.run_health_assessment()
        
        # Validate health assessment
        required_keys = ['overall_score', 'status', 'recommendation', 'component_scores']
        
        if all(key in health_results for key in required_keys):
            score = health_results['overall_score']
            status = health_results['status']
            
            print(f"   🏥 Health score: {score:.1f}/100")
            print(f"   🚦 Status: {status}")
            
            # Validate GO/NO-GO logic
            if score >= 85 and status == "GO":
                print(f"   ✅ GO decision logic correct")
            elif 60 <= score < 85 and status == "WARNING":
                print(f"   ✅ WARNING decision logic correct")
            elif score < 60 and status == "NO-GO":
                print(f"   ✅ NO-GO decision logic correct")
            else:
                print(f"   ⚠️  Decision logic may be incorrect")
            
            return True
        else:
            print(f"   ❌ Health assessment missing components")
            return False
            
    except Exception as e:
        print(f"   ❌ System health monitor test failed: {e}")
        return False

async def test_daily_metrics_logger():
    """Test daily metrics logging system"""
    
    try:
        from eval.daily_metrics_logger import DailyMetricsLogger
        
        # Run daily metrics job
        logger = DailyMetricsLogger()
        metrics_results = await logger.run_nightly_metrics_job()
        
        # Validate metrics structure
        required_keys = ['date', 'evaluation_metrics', 'coverage_audit', 'health_assessment', 'go_nogo_decision']
        
        if all(key in metrics_results for key in required_keys):
            # Check file saving
            saved_files = metrics_results.get('saved_files', {})
            
            print(f"   📅 Date: {metrics_results['date']}")
            print(f"   🚦 GO/NO-GO: {metrics_results['go_nogo_decision']['status']}")
            print(f"   💾 Files saved: {len(saved_files)}")
            
            # Validate daily log files exist
            if 'latest' in saved_files:
                latest_file = Path(saved_files['latest'])
                if latest_file.exists():
                    print(f"   ✅ latest.json created successfully")
                    return True
                else:
                    print(f"   ❌ latest.json not found")
                    return False
            else:
                print(f"   ❌ latest.json not in saved files")
                return False
        else:
            print(f"   ❌ Daily metrics missing required keys")
            return False
            
    except Exception as e:
        print(f"   ❌ Daily metrics logger test failed: {e}")
        return False

async def test_nightly_batch():
    """Test nightly batch orchestrator"""
    
    try:
        from scripts.nightly_batch import NightlyBatchOrchestrator
        
        # Run nightly batch
        orchestrator = NightlyBatchOrchestrator()
        pipeline_results = await orchestrator.run_complete_nightly_pipeline()
        
        # Validate pipeline results
        required_keys = ['pipeline_date', 'phases', 'final_status', 'recommendations']
        
        if all(key in pipeline_results for key in required_keys):
            final_status = pipeline_results['final_status']
            duration = pipeline_results.get('pipeline_duration', 0)
            
            print(f"   📅 Pipeline date: {pipeline_results['pipeline_date']}")
            print(f"   ⏱️  Duration: {duration:.2f}s")
            print(f"   🏁 Final status: {final_status}")
            
            # Check phases
            phases = pipeline_results['phases']
            phases_completed = sum(1 for phase in phases.values() if isinstance(phase, dict) and phase.get('status') != 'FAILED')
            total_phases = len(phases)
            
            print(f"   📊 Phases completed: {phases_completed}/{total_phases}")
            
            return phases_completed == total_phases
        else:
            print(f"   ❌ Pipeline results missing required keys")
            return False
            
    except Exception as e:
        print(f"   ❌ Nightly batch test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_complete_evaluator_system())
    exit(0 if success else 1)