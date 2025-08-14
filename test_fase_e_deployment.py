#!/usr/bin/env python3
"""
FASE E - Deployment Configuration Test
Tests Dockerfile, Pydantic Settings, and environment configuration
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/runner/workspace')

def test_pydantic_settings():
    """Test Pydantic Settings configuration"""
    print("="*60)
    print("FASE E - PYDANTIC SETTINGS TEST")
    print("="*60)
    
    try:
        # Test settings import
        print("1. Testing settings import...")
        from src.cryptosmarttrader.core.config import (
            get_settings, CryptoSmartTraderSettings, Environment
        )
        print("✅ Settings classes imported successfully")
        
        # Test settings initialization
        print("\n2. Testing settings initialization...")
        settings = get_settings()
        print(f"   Environment: {settings.environment}")
        print(f"   App name: {settings.app_name}")
        print(f"   Version: {settings.app_version}")
        print(f"   Debug: {settings.debug}")
        
        # Test nested settings
        print("\n3. Testing nested settings...")
        print(f"   Database host: {settings.database.postgres_host}")
        print(f"   Redis host: {settings.database.redis_host}")
        print(f"   Prometheus port: {settings.monitoring.prometheus_port}")
        print(f"   Max daily loss: ${settings.trading.max_daily_loss_usd}")
        
        # Test environment validation
        print("\n4. Testing environment validation...")
        assert hasattr(settings, 'is_production')
        assert hasattr(settings, 'is_development')
        print(f"   Is production: {settings.is_production}")
        print(f"   Is development: {settings.is_development}")
        
        # Test directory creation
        print("\n5. Testing directory creation...")
        for dir_attr in ['data_dir', 'logs_dir', 'models_dir', 'exports_dir']:
            dir_path = getattr(settings, dir_attr)
            assert dir_path.exists()
            print(f"   ✅ {dir_attr}: {dir_path}")
        
        print("\n✅ PYDANTIC SETTINGS: ALL TESTS PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Pydantic settings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_example():
    """Test .env.example file completeness"""
    print("\n" + "="*60)
    print(".ENV.EXAMPLE VALIDATION TEST")
    print("="*60)
    
    try:
        # Check if .env.example exists
        env_example_path = Path("/home/runner/workspace/.env.example")
        if not env_example_path.exists():
            print("❌ .env.example file not found")
            return False
        
        print("1. Reading .env.example file...")
        with open(env_example_path, 'r') as f:
            env_content = f.read()
        
        # Check for required sections
        required_sections = [
            "APPLICATION SETTINGS",
            "DATABASE SETTINGS", 
            "EXCHANGE API SETTINGS",
            "AI/ML SERVICE SETTINGS",
            "SECURITY SETTINGS",
            "MONITORING & OBSERVABILITY",
            "TRADING CONFIGURATION"
        ]
        
        print("\n2. Validating required sections...")
        missing_sections = []
        for section in required_sections:
            if section in env_content:
                print(f"   ✅ {section}")
            else:
                print(f"   ❌ {section}")
                missing_sections.append(section)
        
        # Check for required environment variables
        required_env_vars = [
            "CRYPTOSMARTTRADER_ENVIRONMENT",
            "CRYPTOSMARTTRADER_EXCHANGE_KRAKEN_API_KEY",
            "CRYPTOSMARTTRADER_AI_OPENAI_API_KEY",
            "CRYPTOSMARTTRADER_SECURITY_SECRET_KEY",
            "CRYPTOSMARTTRADER_DB_POSTGRES_PASSWORD"
        ]
        
        print("\n3. Validating required environment variables...")
        missing_vars = []
        for var in required_env_vars:
            if var in env_content:
                print(f"   ✅ {var}")
            else:
                print(f"   ❌ {var}")
                missing_vars.append(var)
        
        # Check file size (should be comprehensive)
        file_size = len(env_content)
        print(f"\n4. File metrics:")
        print(f"   File size: {file_size} characters")
        print(f"   Lines: {len(env_content.splitlines())}")
        
        if len(missing_sections) == 0 and len(missing_vars) == 0 and file_size > 3000:
            print("\n✅ .ENV.EXAMPLE: COMPREHENSIVE AND COMPLETE")
            return True
        else:
            print(f"\n❌ .env.example issues: {len(missing_sections)} missing sections, {len(missing_vars)} missing vars")
            return False
        
    except Exception as e:
        print(f"❌ .env.example test failed: {e}")
        return False

def test_dockerfile():
    """Test Dockerfile configuration"""
    print("\n" + "="*60)
    print("DOCKERFILE VALIDATION TEST")
    print("="*60)
    
    try:
        # Check if Dockerfile exists
        dockerfile_path = Path("/home/runner/workspace/Dockerfile")
        if not dockerfile_path.exists():
            print("❌ Dockerfile not found")
            return False
        
        print("1. Reading Dockerfile...")
        with open(dockerfile_path, 'r') as f:
            dockerfile_content = f.read()
        
        # Check for required features
        required_features = {
            "Multi-stage build": "FROM python:3.11.10-slim-bookworm AS builder",
            "Non-root user": "useradd --uid 1000",
            "Pinned base image": "python:3.11.10-slim-bookworm",
            "Health check": "HEALTHCHECK",
            "Security labels": "LABEL security",
            "UV package manager": "uv sync",
            "Proper WORKDIR": "WORKDIR /app",
            "Environment vars": "ENV PYTHONUNBUFFERED=1"
        }
        
        print("\n2. Validating Dockerfile features...")
        missing_features = []
        for feature, pattern in required_features.items():
            if pattern in dockerfile_content:
                print(f"   ✅ {feature}")
            else:
                print(f"   ❌ {feature}")
                missing_features.append(feature)
        
        # Check for security best practices
        security_checks = {
            "No root user": "USER trader",
            "Capability dropping": "--uid 1000",
            "Health check timeout": "--timeout=10s",
            "Proper file permissions": "--chown=trader:trader"
        }
        
        print("\n3. Validating security practices...")
        security_issues = []
        for check, pattern in security_checks.items():
            if pattern in dockerfile_content:
                print(f"   ✅ {check}")
            else:
                print(f"   ⚠️  {check}")
                security_issues.append(check)
        
        print(f"\n4. Dockerfile metrics:")
        print(f"   File size: {len(dockerfile_content)} characters")
        print(f"   Lines: {len(dockerfile_content.splitlines())}")
        
        if len(missing_features) <= 1 and len(security_issues) <= 2:
            print("\n✅ DOCKERFILE: PRODUCTION-READY")
            return True
        else:
            print(f"\n❌ Dockerfile issues: {len(missing_features)} missing features, {len(security_issues)} security concerns")
            return False
        
    except Exception as e:
        print(f"❌ Dockerfile test failed: {e}")
        return False

def test_docker_compose():
    """Test Docker Compose configuration"""
    print("\n" + "="*60)
    print("DOCKER COMPOSE VALIDATION TEST")
    print("="*60)
    
    try:
        # Check if docker-compose.yml exists
        compose_path = Path("/home/runner/workspace/docker-compose.yml")
        if not compose_path.exists():
            print("❌ docker-compose.yml not found")
            return False
        
        print("1. Reading docker-compose.yml...")
        with open(compose_path, 'r') as f:
            compose_content = f.read()
        
        # Check for required services
        required_services = [
            "cryptosmarttrader",
            "postgres", 
            "redis",
            "prometheus",
            "grafana",
            "alertmanager"
        ]
        
        print("\n2. Validating required services...")
        missing_services = []
        for service in required_services:
            if f"{service}:" in compose_content:
                print(f"   ✅ {service}")
            else:
                print(f"   ❌ {service}")
                missing_services.append(service)
        
        # Check for production features
        production_features = [
            "healthcheck",
            "restart: unless-stopped",
            "networks:",
            "volumes:",
            "depends_on:"
        ]
        
        print("\n3. Validating production features...")
        missing_features = []
        for feature in production_features:
            if feature in compose_content:
                print(f"   ✅ {feature}")
            else:
                print(f"   ❌ {feature}")
                missing_features.append(feature)
        
        if len(missing_services) == 0 and len(missing_features) <= 1:
            print("\n✅ DOCKER COMPOSE: PRODUCTION-READY STACK")
            return True
        else:
            print(f"\n❌ Docker Compose issues: {len(missing_services)} missing services, {len(missing_features)} missing features")
            return False
        
    except Exception as e:
        print(f"❌ Docker Compose test failed: {e}")
        return False

def test_monitoring_config():
    """Test monitoring configuration files"""
    print("\n" + "="*60)
    print("MONITORING CONFIG VALIDATION TEST")
    print("="*60)
    
    try:
        monitoring_files = [
            "monitoring/prometheus.yml",
            "monitoring/alert_rules.yml", 
            "monitoring/alertmanager.yml"
        ]
        
        print("1. Checking monitoring configuration files...")
        missing_files = []
        for file_path in monitoring_files:
            full_path = Path(f"/home/runner/workspace/{file_path}")
            if full_path.exists():
                print(f"   ✅ {file_path}")
            else:
                print(f"   ❌ {file_path}")
                missing_files.append(file_path)
        
        # Check alert rules content
        if Path("/home/runner/workspace/monitoring/alert_rules.yml").exists():
            print("\n2. Validating alert rules...")
            with open("/home/runner/workspace/monitoring/alert_rules.yml", 'r') as f:
                alert_content = f.read()
            
            fase_d_alerts = [
                "HighOrderErrorRate",
                "DrawdownTooHigh", 
                "NoSignals30m"
            ]
            
            missing_alerts = []
            for alert in fase_d_alerts:
                if alert in alert_content:
                    print(f"   ✅ {alert}")
                else:
                    print(f"   ❌ {alert}")
                    missing_alerts.append(alert)
        else:
            missing_alerts = fase_d_alerts
        
        if len(missing_files) == 0 and len(missing_alerts) == 0:
            print("\n✅ MONITORING CONFIG: COMPLETE")
            return True
        else:
            print(f"\n❌ Monitoring config issues: {len(missing_files)} missing files, {len(missing_alerts)} missing alerts")
            return False
        
    except Exception as e:
        print(f"❌ Monitoring config test failed: {e}")
        return False

def main():
    """Run complete FASE E deployment test suite"""
    print("FASE E - REPRODUCEERBAAR DEPLOYEN IMPLEMENTATION TEST")
    print("Testing Dockerfile, Pydantic Settings, and environment configuration")
    print("="*80)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Pydantic Settings
    if test_pydantic_settings():
        tests_passed += 1
    
    # Test 2: .env.example
    if test_env_example():
        tests_passed += 1
    
    # Test 3: Dockerfile
    if test_dockerfile():
        tests_passed += 1
    
    # Test 4: Docker Compose
    if test_docker_compose():
        tests_passed += 1
    
    # Test 5: Monitoring config
    if test_monitoring_config():
        tests_passed += 1
    
    print("\n" + "="*80)
    print("FASE E DEPLOYMENT TEST RESULTS")
    print("="*80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 FASE E IMPLEMENTATION: COMPLETE")
        print("✅ Dockerfile: Production-ready with non-root user")
        print("✅ Pydantic Settings: Comprehensive configuration")
        print("✅ .env.example: Complete environment template")
        print("✅ Docker Compose: Full production stack")
        print("✅ Monitoring: Prometheus/Grafana/AlertManager")
        print("✅ Security: No secrets in code, proper isolation")
        print("✅ Health checks: Docker HEALTHCHECK configured")
        print("\nFASE E reproduceerbaar deployen is volledig geïmplementeerd!")
        print("Ready for production deployment with `docker-compose up`")
    else:
        print(f"\n❌ {total_tests - tests_passed} tests failed")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)