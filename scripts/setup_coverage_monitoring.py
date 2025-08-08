#!/usr/bin/env python3
"""
Setup Coverage Monitoring
Sets up automated daily coverage audits and alert systems
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_cron_job():
    """Create cron job for daily coverage audit"""
    
    script_path = project_root / "scripts" / "daily_coverage_audit.py"
    log_path = project_root / "logs" / "coverage_audit.log"
    
    # Create cron job entry
    cron_entry = f"""
# CryptoSmartTrader Daily Coverage Audit
# Runs every day at 6:00 AM to verify 100% exchange coverage
0 6 * * * cd {project_root} && /usr/bin/python3 {script_path} >> {log_path} 2>&1

# Additional audit at 6:00 PM for critical monitoring
0 18 * * * cd {project_root} && /usr/bin/python3 {script_path} >> {log_path} 2>&1
"""
    
    print("🕐 CRON JOB SETUP:")
    print("Add the following to your crontab (run 'crontab -e'):")
    print(cron_entry)
    
    # Save to file for easy reference
    cron_file = project_root / "coverage_monitoring_cron.txt"
    with open(cron_file, 'w') as f:
        f.write(cron_entry)
    
    print(f"✅ Cron job saved to: {cron_file}")

def create_systemd_service():
    """Create systemd service for coverage monitoring"""
    
    service_content = f"""[Unit]
Description=CryptoSmartTrader Coverage Audit
After=network.target

[Service]
Type=oneshot
User=cryptotrader
WorkingDirectory={project_root}
ExecStart=/usr/bin/python3 {project_root}/scripts/daily_coverage_audit.py
StandardOutput=append:{project_root}/logs/coverage_audit.log
StandardError=append:{project_root}/logs/coverage_audit_error.log

[Install]
WantedBy=default.target
"""
    
    timer_content = """[Unit]
Description=Run CryptoSmartTrader Coverage Audit twice daily
Requires=cryptotrader-coverage-audit.service

[Timer]
OnCalendar=06:00
OnCalendar=18:00
Persistent=true

[Install]
WantedBy=timers.target
"""
    
    print("🔧 SYSTEMD SERVICE SETUP:")
    print(f"1. Save the following as /etc/systemd/system/cryptotrader-coverage-audit.service:")
    print(service_content)
    print()
    print(f"2. Save the following as /etc/systemd/system/cryptotrader-coverage-audit.timer:")
    print(timer_content)
    print()
    print("3. Enable and start the timer:")
    print("   sudo systemctl daemon-reload")
    print("   sudo systemctl enable cryptotrader-coverage-audit.timer")
    print("   sudo systemctl start cryptotrader-coverage-audit.timer")
    print()
    
    # Save service files for reference
    service_file = project_root / "systemd_coverage_audit.service"
    timer_file = project_root / "systemd_coverage_audit.timer"
    
    with open(service_file, 'w') as f:
        f.write(service_content)
    
    with open(timer_file, 'w') as f:
        f.write(timer_content)
    
    print(f"✅ Service files saved to: {service_file} and {timer_file}")

def create_alert_config():
    """Create alert configuration"""
    
    alert_config = {
        "coverage_monitoring": {
            "enabled": True,
            "thresholds": {
                "critical_coverage": 0.95,
                "warning_coverage": 0.98,
                "max_missing_coins": 10,
                "high_impact_threshold": 0.3
            },
            "alerts": {
                "email_enabled": False,
                "email_recipients": [],
                "webhook_enabled": False,
                "webhook_url": "",
                "slack_enabled": False,
                "slack_channel": "",
                "log_alerts": True
            },
            "audit_schedule": {
                "frequency_hours": 24,
                "daily_times": ["06:00", "18:00"],
                "enabled": True
            }
        },
        "missing_coin_alerts": {
            "enabled": True,
            "immediate_alert_for_high_volume": True,
            "volume_threshold_usd": 1000000,
            "new_listing_alerts": True,
            "delisting_alerts": True
        }
    }
    
    config_file = project_root / "config" / "coverage_monitoring.json"
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(alert_config, f, indent=2)
    
    print(f"📋 ALERT CONFIGURATION:")
    print(f"Configuration saved to: {config_file}")
    print()
    print("Key settings:")
    print(f"  • Critical coverage threshold: {alert_config['coverage_monitoring']['thresholds']['critical_coverage']:.0%}")
    print(f"  • Warning coverage threshold: {alert_config['coverage_monitoring']['thresholds']['warning_coverage']:.0%}")
    print(f"  • Audit frequency: Every {alert_config['coverage_monitoring']['audit_schedule']['frequency_hours']} hours")
    print(f"  • Daily audit times: {', '.join(alert_config['coverage_monitoring']['audit_schedule']['daily_times'])}")

def create_monitoring_dashboard_config():
    """Create monitoring dashboard configuration"""
    
    dashboard_config = {
        "coverage_dashboard": {
            "refresh_interval_seconds": 300,
            "show_missing_coins": True,
            "show_new_listings": True,
            "show_coverage_trend": True,
            "show_alert_history": True,
            "max_missing_coins_display": 20,
            "max_alerts_display": 10
        },
        "metrics": {
            "track_coverage_percentage": True,
            "track_missing_coin_count": True,
            "track_audit_duration": True,
            "track_new_listings": True,
            "track_alert_frequency": True
        }
    }
    
    dashboard_file = project_root / "config" / "coverage_dashboard.json"
    
    with open(dashboard_file, 'w') as f:
        json.dump(dashboard_config, f, indent=2)
    
    print(f"📊 DASHBOARD CONFIGURATION:")
    print(f"Configuration saved to: {dashboard_file}")

def setup_log_directories():
    """Setup log directories for coverage monitoring"""
    
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create coverage-specific log files
    coverage_log = log_dir / "coverage_audit.log"
    coverage_error_log = log_dir / "coverage_audit_error.log"
    missing_coins_log = log_dir / "missing_coins.log"
    
    # Create initial log files if they don't exist
    for log_file in [coverage_log, coverage_error_log, missing_coins_log]:
        if not log_file.exists():
            with open(log_file, 'w') as f:
                f.write(f"# CryptoSmartTrader Coverage Monitoring Log\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# File: {log_file.name}\n\n")
    
    print(f"📝 LOG SETUP:")
    print(f"Log directory: {log_dir}")
    print(f"Coverage audit log: {coverage_log}")
    print(f"Error log: {coverage_error_log}")
    print(f"Missing coins log: {missing_coins_log}")

def main():
    """Main setup function"""
    
    print("🔍 CRYPTOSMARTTRADER COVERAGE MONITORING SETUP")
    print("=" * 60)
    print()
    
    print("Setting up automated coverage monitoring to ensure 100% exchange coverage...")
    print()
    
    # Setup components
    setup_log_directories()
    print()
    
    create_alert_config()
    print()
    
    create_monitoring_dashboard_config()
    print()
    
    create_cron_job()
    print()
    
    create_systemd_service()
    print()
    
    print("🎯 MANUAL TEST:")
    print("Run a test audit to verify everything works:")
    print(f"  cd {project_root}")
    print("  python scripts/daily_coverage_audit.py --verbose")
    print()
    
    print("📋 NEXT STEPS:")
    print("1. Choose your preferred scheduling method (cron or systemd)")
    print("2. Set up the scheduled jobs as shown above")
    print("3. Configure alert destinations (email/webhook/slack) in coverage_monitoring.json")
    print("4. Run the manual test to verify functionality")
    print("5. Monitor logs to ensure audits are running correctly")
    print()
    
    print("✅ Coverage monitoring setup complete!")
    print(f"📁 All configuration files saved in: {project_root}")

if __name__ == "__main__":
    main()