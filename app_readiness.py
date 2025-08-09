# app_readiness.py - Hard readiness gate for the application
from pathlib import Path
import time
import json
import logging

logger = logging.getLogger(__name__)

def _fresh(p, max_age=24*3600):
    """Check if file is fresh (modified within max_age seconds)"""
    try:
        return Path(p).exists() and time.time() - Path(p).stat().st_mtime < max_age
    except:
        return False

def check_readiness():
    """
    Hard readiness check - returns True only when system is actually ready
    This replaces the fake 'System Online' status
    """
    horizons = ["1h", "24h", "168h", "720h"]
    
    # Check if all models exist
    models = all(Path(f"models/saved/rf_{h}.pkl").exists() for h in horizons)
    
    # Check if features are fresh (updated within 24h)
    features_ok = _fresh("exports/features.parquet")
    
    # Check health score from daily logs
    latest = Path("logs/daily/latest.json")
    health = 0
    if latest.exists():
        try:
            health_data = json.loads(latest.read_text())
            health = health_data.get("health_score", 0)
        except:
            health = 0
    
    # System is ready only if ALL components are healthy
    ready = models and features_ok and health >= 85
    
    parts = {
        "models": models,
        "features": features_ok, 
        "health": health,
        "ready": ready
    }
    
    logger.info(f"Readiness check: ready={ready}, models={models}, features={features_ok}, health={health}")
    
    return ready, parts

def get_system_status():
    """Get detailed system status for UI display"""
    ready, parts = check_readiness()
    
    if ready:
        return "🟢 System Ready", "All components operational"
    elif not parts["models"]:
        return "🔴 Models Missing", "Train models first: python ml/train_baseline.py"
    elif not parts["features"]:
        return "🟡 Features Stale", "Features older than 24h - refresh data"
    elif parts["health"] < 85:
        return "🟡 Health Low", f"Health score: {parts['health']}/100"
    else:
        return "🔴 System Down", "Multiple components failing"

def enforce_readiness_gate(tab_name="Unknown"):
    """
    Hard gate that stops execution if system not ready
    Use this at the start of every tab that requires trained models
    """
    import streamlit as st
    
    ready, parts = check_readiness()
    
    if not ready:
        if not parts["models"]:
            st.error("⚠️ GEEN GETRAINDE MODELLEN - Tab uitgeschakeld")
            st.info("**Oplossing**: `python ml/train_baseline.py`")
            st.stop()
        
        if not parts["features"]:
            st.error("⚠️ FEATURES VEROUDERD - Tab uitgeschakeld") 
            st.info("**Oplossing**: Refresh data pipeline")
            st.stop()
            
        if parts["health"] < 85:
            st.error(f"⚠️ SYSTEM HEALTH LAAG ({parts['health']}/100) - Tab uitgeschakeld")
            st.info("**Oplossing**: Run system health check")
            st.stop()
    
    return True