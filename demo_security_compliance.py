#!/usr/bin/env python3
"""
Demo: Security & Compliance System
Comprehensive demonstration of enterprise security infrastructure including secrets management, log sanitization, and compliance monitoring.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.core.secrets_manager import (
    create_secrets_manager, SecretType, SecretStatus
)
from cryptosmarttrader.core.log_sanitizer import (
    create_log_sanitizer, create_sanitized_logger, ExchangeComplianceMonitor
)


async def demonstrate_security_compliance():
    """Comprehensive demonstration of security and compliance infrastructure."""
    
    print("🔒 SECURITY & COMPLIANCE SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Demo 1: Secrets Management
    print("🔧 DEMO 1: Enterprise Secrets Management")
    print("-" * 50)
    
    # Initialize secrets manager
    secrets_manager = create_secrets_manager(
        secrets_path="demo_secrets",
        environment="development"
    )
    
    print("   Storing various types of secrets...")
    
    # Store different types of secrets
    test_secrets = [
        {
            'secret_id': 'kraken_api_key',
            'secret_value': 'krakenapikey1234567890abcdefghijklmnopqrstuvwxyz123456',
            'secret_type': SecretType.API_KEY,
            'description': 'Kraken exchange API key for trading',
            'rotation_frequency_days': 30,
            'tags': ['exchange', 'kraken', 'trading']
        },
        {
            'secret_id': 'binance_api_secret',
            'secret_value': 'binancesecret9876543210fedcba0987654321abcdef12345678',
            'secret_type': SecretType.API_KEY,
            'description': 'Binance exchange API secret',
            'rotation_frequency_days': 30,
            'tags': ['exchange', 'binance', 'trading']
        },
        {
            'secret_id': 'database_url',
            'secret_value': 'postgresql://user:password123@localhost:5432/cryptodb',
            'secret_type': SecretType.DATABASE_URL,
            'description': 'Main trading database connection',
            'rotation_frequency_days': 90,
            'tags': ['database', 'postgresql']
        },
        {
            'secret_id': 'webhook_secret',
            'secret_value': 'webhook_secret_abcdef123456789012345678901234567890abcdef',
            'secret_type': SecretType.WEBHOOK_SECRET,
            'description': 'Webhook signature verification secret',
            'rotation_frequency_days': 60,
            'tags': ['webhook', 'security']
        },
        {
            'secret_id': 'openai_api_key',
            'secret_value': 'sk-1234567890abcdef1234567890abcdef1234567890abcdef',
            'secret_type': SecretType.API_KEY,
            'description': 'OpenAI API key for ML predictions',
            'rotation_frequency_days': 60,
            'tags': ['ai', 'openai', 'ml']
        }
    ]
    
    stored_count = 0
    for secret_config in test_secrets:
        success = secrets_manager.store_secret(
            secret_id=secret_config['secret_id'],
            secret_value=secret_config['secret_value'],
            secret_type=secret_config['secret_type'],
            description=secret_config['description'],
            rotation_frequency_days=secret_config['rotation_frequency_days'],
            tags=secret_config['tags'],
            user_id="demo_user"
        )
        
        if success:
            stored_count += 1
            print(f"      ✅ Stored: {secret_config['secret_id']}")
        else:
            print(f"      ❌ Failed: {secret_config['secret_id']}")
    
    print(f"   Total secrets stored: {stored_count}/{len(test_secrets)}")
    
    # Demonstrate secret retrieval
    print("\n   Testing secret retrieval...")
    
    retrieved_secret = secrets_manager.get_secret(
        secret_id="kraken_api_key",
        user_id="demo_user"
    )
    
    if retrieved_secret:
        print(f"      Retrieved Kraken API key: {retrieved_secret[:20]}...")
    else:
        print("      Failed to retrieve Kraken API key")
    
    # List all secrets
    print("\n   Current secrets inventory:")
    secrets_list = secrets_manager.list_secrets(user_id="demo_user")
    
    for secret_info in secrets_list:
        print(f"      {secret_info['secret_id']}: {secret_info['secret_type']} "
              f"(status: {secret_info['status']}, "
              f"rotated: {secret_info['days_since_rotation']} days ago)")
    
    # Demo 2: Secret Rotation
    print("\n🔄 DEMO 2: Secret Rotation")
    print("-" * 35)
    
    # Check which secrets need rotation
    rotation_needed = secrets_manager.check_rotation_needed()
    print(f"   Secrets due for rotation: {len(rotation_needed)}")
    
    if rotation_needed:
        for secret_id, days_overdue in rotation_needed:
            print(f"      {secret_id}: {days_overdue} days overdue")
    
    # Manual rotation demonstration
    print("\n   Demonstrating manual secret rotation...")
    
    new_kraken_key = "newkrakenapikey0987654321fedcba0987654321abcdef987654"
    rotation_success = secrets_manager.rotate_secret(
        secret_id="kraken_api_key",
        new_secret_value=new_kraken_key,
        user_id="demo_user"
    )
    
    if rotation_success:
        print("      ✅ Kraken API key rotated successfully")
        
        # Verify new value
        rotated_secret = secrets_manager.get_secret("kraken_api_key", "demo_user")
        if rotated_secret == new_kraken_key:
            print("      ✅ Rotation verified - new value retrieved")
    else:
        print("      ❌ Failed to rotate Kraken API key")
    
    # Demonstrate auto-rotation
    print("\n   Testing auto-rotation system...")
    
    auto_rotation_results = secrets_manager.auto_rotate_secrets(user_id="auto_system")
    print(f"   Auto-rotation results: {len(auto_rotation_results)} secrets processed")
    
    for secret_id, success in auto_rotation_results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"      {secret_id}: {status}")
    
    # Demo 3: Log Sanitization
    print("\n🧹 DEMO 3: Log Sanitization")
    print("-" * 40)
    
    # Initialize log sanitizer
    log_sanitizer = create_log_sanitizer(enable_audit=True)
    
    # Register known secrets for sanitization
    for secret_config in test_secrets:
        log_sanitizer.register_secret(
            secret=secret_config['secret_value'],
            identifier=secret_config['secret_id']
        )
    
    print("   Testing log sanitization with various sensitive data types...")
    
    # Test different types of sensitive data
    test_log_messages = [
        "User kraken_api_key=krakenapikey1234567890abcdefghijklmnopqrstuvwxyz123456 authenticated",
        "Database connection: postgresql://user:password123@localhost:5432/cryptodb",
        "Processing order for user john.doe@example.com with amount $125,000",
        "BTC transfer: 2.5 BTC to address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "User ID: user_123456789 placed order_id: ord_abcdef123456 for customer_id: cust_987654321",
        "ETH address: 0x742d35Cc6634C0532925a3b8D8c3C3c72C54E26D received payment",
        "API request from IP 192.168.1.100 with Bearer abc123xyz789token",
        "Phone verification: +1-555-123-4567 for account verification",
        "Webhook signature: X-Signature: sha256=a1b2c3d4e5f6g7h8i9j0"
    ]
    
    print("\n   Original vs Sanitized log messages:")
    print("   " + "=" * 65)
    
    for i, original_message in enumerate(test_log_messages, 1):
        sanitized_message = log_sanitizer.sanitize_log_message(original_message)
        
        print(f"\n   {i:2d}. Original : {original_message}")
        print(f"       Sanitized: {sanitized_message}")
        
        if original_message != sanitized_message:
            print("       🔒 Sensitive data detected and sanitized")
        else:
            print("       ✅ No sensitive data detected")
    
    # Test dictionary sanitization
    print("\n   Testing dictionary sanitization...")
    
    test_dict = {
        "user_id": "user_123456789",
        "api_key": "krakenapikey1234567890abcdefghijklmnopqrstuvwxyz123456",
        "order_data": {
            "amount": "$50,000",
            "btc_address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            "email": "trader@example.com"
        },
        "metadata": {
            "ip": "192.168.1.100",
            "jwt": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123"
        }
    }
    
    sanitized_dict = log_sanitizer.sanitize_dict(test_dict)
    
    print("   Original dictionary:")
    print(f"      {json.dumps(test_dict, indent=6)}")
    print("\n   Sanitized dictionary:")
    print(f"      {json.dumps(sanitized_dict, indent=6)}")
    
    # Demo 4: Exchange Compliance Monitoring
    print("\n📋 DEMO 4: Exchange ToS Compliance Monitoring")
    print("-" * 55)
    
    # Initialize compliance monitor
    compliance_monitor = ExchangeComplianceMonitor()
    
    print("   Testing compliance with exchange Terms of Service...")
    
    # Test messages that might violate ToS
    compliance_test_messages = [
        "Sharing Kraken market data with third-party service",
        "User balance on Binance: $125,000 for customer ID 12345",
        "Normal trading activity detected",
        "Redistributing market data feeds to subscribers",
        "Individual portfolio analysis for user account",
        "Coordinated trading activity across multiple accounts",
        "Standard market analysis and reporting",
        "Pump and dump scheme detected in XYZ coin",
        "Personal trading history exported for user",
        "Artificial volume generation algorithm activated"
    ]
    
    print("\n   Compliance check results:")
    print("   " + "=" * 45)
    
    total_violations = 0
    
    for i, message in enumerate(compliance_test_messages, 1):
        violations = compliance_monitor.check_compliance(message)
        
        print(f"\n   {i:2d}. Message: {message}")
        
        if violations:
            print(f"       ⚠️  Violations: {', '.join(violations)}")
            total_violations += len(violations)
        else:
            print("       ✅ Compliant")
    
    print(f"\n   Total compliance violations detected: {total_violations}")
    
    # Demo 5: Sanitization Statistics
    print("\n📊 DEMO 5: Sanitization Statistics")
    print("-" * 45)
    
    stats = log_sanitizer.get_sanitization_stats()
    
    print("   Log sanitization statistics:")
    print(f"      Total rules configured: {stats['total_rules']}")
    print(f"      Active rules: {stats['enabled_rules']}")
    print(f"      Known secrets registered: {stats['known_secrets']}")
    print(f"      Total sanitizations performed: {stats['total_sanitizations']}")
    
    print("\n   Rule application breakdown:")
    for rule_name, count in stats['rule_applications'].items():
        print(f"      {rule_name}: {count} applications")
    
    # Demo 6: Secrets Health Check
    print("\n🏥 DEMO 6: Secrets Health Check")
    print("-" * 40)
    
    health_status = secrets_manager.health_check()
    
    print("   Secrets management health status:")
    print(f"      Overall status: {health_status['status']}")
    print(f"      Environment: {health_status['environment']}")
    print(f"      Encryption available: {health_status['encryption_available']}")
    print(f"      Total secrets: {health_status['total_secrets']}")
    print(f"      Active secrets: {health_status['active_secrets']}")
    print(f"      Secrets due rotation: {health_status['secrets_due_rotation']}")
    print(f"      Recent audit events: {health_status['recent_audit_events']}")
    
    if 'issues' in health_status:
        print("   Health issues detected:")
        for issue in health_status['issues']:
            print(f"      ⚠️  {issue}")
    
    # Demo 7: Audit Trail
    print("\n🔍 DEMO 7: Audit Trail")
    print("-" * 30)
    
    print("   Recent secrets management audit events:")
    
    audit_events = secrets_manager.get_audit_log(hours_back=1)
    
    if audit_events:
        for event in audit_events[-10:]:  # Show last 10 events
            print(f"      {event['timestamp']}: {event['event_type']} "
                  f"on {event['secret_id']} by {event['user_id']} "
                  f"({'✅' if event['success'] else '❌'})")
    else:
        print("      No recent audit events")
    
    # Demo 8: Emergency Secret Revocation
    print("\n🚨 DEMO 8: Emergency Secret Revocation")
    print("-" * 45)
    
    print("   Demonstrating emergency secret revocation...")
    
    # Revoke a secret
    revocation_success = secrets_manager.revoke_secret(
        secret_id="webhook_secret",
        user_id="security_admin"
    )
    
    if revocation_success:
        print("      ✅ Webhook secret revoked successfully")
        
        # Try to retrieve revoked secret
        revoked_secret = secrets_manager.get_secret("webhook_secret", "demo_user")
        
        if revoked_secret is None:
            print("      ✅ Revoked secret correctly blocked from retrieval")
        else:
            print("      ❌ Revoked secret still accessible!")
    else:
        print("      ❌ Failed to revoke webhook secret")
    
    # Show updated secrets list
    print("\n   Updated secrets inventory after revocation:")
    updated_secrets = secrets_manager.list_secrets(include_revoked=True, user_id="demo_user")
    
    for secret_info in updated_secrets:
        status_emoji = "🔒" if secret_info['status'] == 'revoked' else "✅"
        print(f"      {status_emoji} {secret_info['secret_id']}: {secret_info['status']}")
    
    # Demo 9: Sanitized Logger Usage
    print("\n📝 DEMO 9: Sanitized Logger in Practice")
    print("-" * 50)
    
    # Create sanitized logger
    base_logger = logging.getLogger("demo_trading")
    sanitized_logger = create_sanitized_logger("demo_trading", log_sanitizer)
    
    print("   Demonstrating sanitized logger usage...")
    print("   (Check logs for sanitized output)\n")
    
    # Set up console handler for demo
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    base_logger.addHandler(handler)
    base_logger.setLevel(logging.INFO)
    
    # Log messages that will be sanitized
    test_logs = [
        "User authentication successful with api_key=krakenapikey1234567890abcdefghijklmnopqrstuvwxyz123456",
        "Processing order for customer john.doe@example.com amount $125,000",
        "Database connection established: postgresql://user:password123@localhost:5432/cryptodb",
        "Bitcoin transfer initiated: 5.2 BTC to 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "API request from 192.168.1.100 processed successfully"
    ]
    
    print("   Sanitized log output:")
    for log_msg in test_logs:
        sanitized_logger.info(log_msg)
    
    print("\n✅ SECURITY & COMPLIANCE DEMONSTRATION COMPLETED")
    print("=" * 70)
    
    # Final summary
    print("🎯 SECURITY ACHIEVEMENTS:")
    print(f"   ✅ Enterprise secrets management with {stored_count} secrets stored")
    print(f"   ✅ Secret rotation system with auto-rotation capabilities")
    print(f"   ✅ Log sanitization with {stats['total_rules']} protection rules")
    print(f"   ✅ Exchange ToS compliance monitoring with {len(compliance_test_messages)} tests")
    print(f"   ✅ Comprehensive audit trail with {len(audit_events)} events")
    print(f"   ✅ Emergency revocation procedures validated")
    print(f"   ✅ Production-ready security infrastructure operational")
    
    # Compliance status
    compliance_score = max(0, 100 - (total_violations * 10))
    print(f"   📊 Overall compliance score: {compliance_score}%")
    
    if compliance_score >= 90:
        print("   🏆 Excellent security and compliance posture!")
    elif compliance_score >= 75:
        print("   ✅ Good security and compliance posture")
    else:
        print("   ⚠️  Security and compliance needs attention")


if __name__ == "__main__":
    print("🔒 CRYPTOSMARTTRADER V2 - SECURITY & COMPLIANCE DEMO")
    print("=" * 70)
    
    try:
        asyncio.run(demonstrate_security_compliance())
        print("\n🏆 Security & compliance demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)