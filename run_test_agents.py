#!/usr/bin/env python3
"""
Quick test run of agents to generate sample data for UI testing
"""
import asyncio
import sys
from pathlib import Path

async def run_single_cycle():
    """Run one cycle of each agent to generate test data"""
    
    print("🧪 Running single cycle of all agents for testing...")
    
    # Create directories
    Path("logs/daily").mkdir(parents=True, exist_ok=True)
    Path("exports/production").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    try:
        # 1. Data Collector
        print("📊 Running Data Collector...")
        from agents.data_collector import DataCollectorAgent
        data_agent = DataCollectorAgent()
        await data_agent.collect_market_data()
        
        # Save features for other agents
        import pandas as pd
        import json
        from datetime import datetime
        
        # Create sample feature data
        sample_data = [
            {
                'coin': 'BTC',
                'timestamp': datetime.utcnow(),
                'feat_rsi_14': 65.5,
                'feat_macd': 0.02,
                'feat_vol_24h': 1500000000,
                'feat_price_change_1h': 0.012,
                'feat_price_change_24h': 0.045,
                'feat_sent_score': 0.75,
                'feat_whale_score': 0.85,
                'target_1h': 0.015,
                'target_24h': 0.035,  
                'target_168h': 0.085,
                'target_720h': 0.125
            },
            {
                'coin': 'ETH',
                'timestamp': datetime.utcnow(),
                'feat_rsi_14': 58.3,
                'feat_macd': -0.01,
                'feat_vol_24h': 800000000,
                'feat_price_change_1h': -0.005,
                'feat_price_change_24h': 0.025,
                'feat_sent_score': 0.68,
                'feat_whale_score': 0.72,
                'target_1h': 0.008,
                'target_24h': 0.025,
                'target_168h': 0.065,
                'target_720h': 0.095
            },
            {
                'coin': 'ADA',
                'timestamp': datetime.utcnow(),
                'feat_rsi_14': 72.1,
                'feat_macd': 0.08,
                'feat_vol_24h': 200000000,
                'feat_price_change_1h': 0.025,
                'feat_price_change_24h': 0.089,
                'feat_sent_score': 0.82,
                'feat_whale_score': 0.91,
                'target_1h': 0.028,
                'target_24h': 0.065,
                'target_168h': 0.145,
                'target_720h': 0.225
            }
        ]
        
        df = pd.DataFrame(sample_data)
        df.to_parquet("exports/features.parquet")
        print("✅ Features saved")
        
        # 2. Health Monitor
        print("🏥 Running Health Monitor...")
        from agents.health_monitor import HealthMonitorAgent
        health_agent = HealthMonitorAgent()
        health_score, scores = await health_agent.calculate_health_score()
        go_nogo, message = await health_agent.determine_go_nogo(health_score)
        
        health_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'health_score': round(health_score, 1),
            'go_nogo': go_nogo,
            'message': message,
            'component_scores': scores
        }
        
        with open('logs/daily/latest.json', 'w') as f:
            json.dump(health_report, f, indent=2)
        print(f"✅ Health: {health_score:.1f} ({go_nogo})")
        
        # 3. Coverage metrics
        coverage_metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'coverage_pct': 100.0,
            'coins_collected': 3,
            'target_coins': 3
        }
        
        with open('logs/coverage_metrics.json', 'w') as f:
            json.dump(coverage_metrics, f)
        print("✅ Coverage: 100%")
        
        # 4. ML Predictor
        print("🧠 Running ML Predictor...")
        from agents.ml_predictor import MLPredictorAgent
        ml_agent = MLPredictorAgent()
        predictions = await ml_agent.generate_predictions()
        await ml_agent.save_predictions(predictions)
        print(f"✅ ML Predictions: {len(predictions)} generated")
        
        # 5. Whale Detector
        print("🐋 Running Whale Detector...")
        from agents.whale_detector import WhaleDetectorAgent
        whale_agent = WhaleDetectorAgent()
        whale_data = await whale_agent.detect_whale_activity(sample_data)
        
        with open('logs/whale_activity.json', 'w') as f:
            json.dump(whale_data, f, default=str)
        print(f"✅ Whale Activity: {len(whale_data)} coins analyzed")
        
        # 6. Risk Manager
        print("⚖️ Running Risk Manager...")
        from agents.risk_manager import RiskManagerAgent
        risk_agent = RiskManagerAgent()
        risk_metrics = await risk_agent.analyze_risk_metrics()
        
        if predictions:
            false_positives = await risk_agent.detect_false_positives(predictions)
            await risk_agent.save_risk_report(risk_metrics, false_positives)
        print("✅ Risk Analysis complete")
        
        print("\n🎯 All agents completed successfully!")
        print("Now run the Streamlit app to see live data:")
        print("  streamlit run app_minimal.py --server.port 5000")
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_single_cycle())