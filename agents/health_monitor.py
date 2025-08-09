# agents/health_monitor.py - System health monitoring with GO/NO-GO decisions
import asyncio
import psutil
import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HealthMonitorAgent:
    """Continuous system health monitoring with enterprise-grade scoring"""
    
    def __init__(self):
        self.health_thresholds = {
            'cpu_max': 85.0,
            'memory_max': 90.0,
            'disk_max': 85.0,
            'model_age_hours': 24,
            'data_age_hours': 1,
            'min_coverage': 80.0
        }
        
    async def calculate_health_score(self):
        """Calculate comprehensive system health score"""
        scores = {}
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        scores['cpu'] = max(0, 100 - cpu_percent) if cpu_percent < self.health_thresholds['cpu_max'] else 0
        scores['memory'] = max(0, 100 - memory.percent) if memory.percent < self.health_thresholds['memory_max'] else 0  
        scores['disk'] = max(0, 100 - (disk.used / disk.total * 100)) if (disk.used / disk.total * 100) < self.health_thresholds['disk_max'] else 0
        
        # Model freshness
        model_score = 0
        horizons = ["1h", "24h", "168h", "720h"]
        for h in horizons:
            model_path = Path(f"models/saved/rf_{h}.pkl")
            if model_path.exists():
                age_hours = (datetime.now().timestamp() - model_path.stat().st_mtime) / 3600
                if age_hours < self.health_thresholds['model_age_hours']:
                    model_score += 25  # 25 points per fresh model
        
        scores['models'] = min(100, model_score)
        
        # Data freshness
        features_path = Path("exports/features.parquet")
        if features_path.exists():
            age_hours = (datetime.now().timestamp() - features_path.stat().st_mtime) / 3600
            scores['data'] = 100 if age_hours < self.health_thresholds['data_age_hours'] else 50
        else:
            scores['data'] = 0
            
        # Coverage check
        try:
            with open('logs/coverage_metrics.json', 'r') as f:
                coverage_data = json.load(f)
                coverage_pct = coverage_data.get('coverage_pct', 0)
                scores['coverage'] = coverage_pct if coverage_pct >= self.health_thresholds['min_coverage'] else coverage_pct / 2
        except:
            scores['coverage'] = 0
        
        # Calculate overall score (weighted average)
        weights = {
            'cpu': 0.15,
            'memory': 0.15, 
            'disk': 0.10,
            'models': 0.30,
            'data': 0.20,
            'coverage': 0.10
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        return overall_score, scores
    
    async def determine_go_nogo(self, health_score):
        """Enterprise GO/NO-GO decision logic"""
        if health_score >= 85:
            return "GO", "All systems operational - trading authorized"
        elif health_score >= 70:
            return "CAUTION", "Degraded performance - monitor closely"
        elif health_score >= 50:
            return "NO-GO", "System issues detected - trading suspended"
        else:
            return "CRITICAL", "System failure - immediate intervention required"
    
    async def run_continuous(self):
        """Run health monitoring continuously"""
        while True:
            try:
                health_score, component_scores = await self.calculate_health_score()
                go_nogo, message = await self.determine_go_nogo(health_score)
                
                health_report = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'health_score': round(health_score, 1),
                    'go_nogo': go_nogo,
                    'message': message,
                    'component_scores': component_scores,
                    'system_resources': {
                        'cpu_percent': psutil.cpu_percent(),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_percent': psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
                    }
                }
                
                # Save health report
                Path("logs/daily").mkdir(parents=True, exist_ok=True)
                with open('logs/daily/latest.json', 'w') as f:
                    json.dump(health_report, f, indent=2)
                
                # Log status changes
                if go_nogo in ["NO-GO", "CRITICAL"]:
                    logger.error(f"Health Alert: {go_nogo} - {message} (Score: {health_score:.1f})")
                elif go_nogo == "CAUTION":
                    logger.warning(f"Health Warning: {message} (Score: {health_score:.1f})")
                else:
                    logger.info(f"Health Status: {go_nogo} (Score: {health_score:.1f})")
                
            except Exception as e:
                logger.error(f"Health monitoring cycle failed: {e}")
            
            await asyncio.sleep(60)  # Every minute

if __name__ == "__main__":
    agent = HealthMonitorAgent()
    asyncio.run(agent.run_continuous())