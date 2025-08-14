#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Terms of Service Compliance Validator
Enterprise-grade TOS enforcement with legal compliance monitoring
"""

import json
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set
import threading
from collections import defaultdict, deque

from core.structured_logger import get_structured_logger


class ComplianceLevel(Enum):
    """Compliance severity levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"
    BANNED = "banned"


class PlatformPolicy(Enum):
    """Platform-specific policy types"""
    RATE_LIMITING = "rate_limiting"
    API_USAGE = "api_usage"
    DATA_COLLECTION = "data_collection"
    USER_PRIVACY = "user_privacy"
    CONTENT_SCRAPING = "content_scraping"
    AUTOMATED_ACCESS = "automated_access"


@dataclass
class TOSRule:
    """Individual Terms of Service rule"""
    platform: str
    policy_type: PlatformPolicy
    rule_id: str
    description: str
    max_requests_per_hour: Optional[int] = None
    max_requests_per_day: Optional[int] = None
    required_headers: List[str] = field(default_factory=list)
    forbidden_endpoints: List[str] = field(default_factory=list)
    required_authentication: bool = True
    commercial_use_allowed: bool = False
    data_retention_limit_days: Optional[int] = None
    user_agent_required: bool = True
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "platform": self.platform,
            "policy_type": self.policy_type.value,
            "rule_id": self.rule_id,
            "description": self.description,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_requests_per_day": self.max_requests_per_day,
            "required_headers": self.required_headers,
            "forbidden_endpoints": self.forbidden_endpoints,
            "required_authentication": self.required_authentication,
            "commercial_use_allowed": self.commercial_use_allowed,
            "data_retention_limit_days": self.data_retention_limit_days,
            "user_agent_required": self.user_agent_required,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class ComplianceViolation:
    """TOS compliance violation record"""
    platform: str
    rule_id: str
    violation_type: PlatformPolicy
    severity: ComplianceLevel
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    request_count: int = 1
    remediation_required: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "platform": self.platform,
            "rule_id": self.rule_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "request_count": self.request_count,
            "remediation_required": self.remediation_required
        }


class TOSComplianceValidator:
    """Enterprise TOS compliance validation and enforcement"""
    
    def __init__(self):
        self.logger = get_structured_logger("TOSComplianceValidator")
        
        # Initialize platform-specific TOS rules
        self.tos_rules = self._initialize_tos_rules()
        
        # Violation tracking
        self.violations = deque(maxlen=10000)
        self.violation_counts = defaultdict(lambda: defaultdict(int))
        
        # Request tracking for rate limiting
        self.request_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Compliance metrics
        self.compliance_metrics = defaultdict(lambda: {
            "total_requests": 0,
            "compliant_requests": 0,
            "violations": 0,
            "last_check": None
        })
        
        self.lock = threading.Lock()
        
        self.logger.info("TOS Compliance Validator initialized")
    
    def _initialize_tos_rules(self) -> Dict[str, List[TOSRule]]:
        """Initialize platform-specific TOS rules"""
        
        rules = defaultdict(list)
        
        # Reddit TOS Rules (as of 2024)
        reddit_rules = [
            TOSRule(
                platform="reddit",
                policy_type=PlatformPolicy.RATE_LIMITING,
                rule_id="reddit_rate_limit",
                description="Reddit API rate limiting: 60 requests per minute",
                max_requests_per_hour=3600,  # 60/min
                max_requests_per_day=86400,
                required_authentication=True,
                user_agent_required=True
            ),
            TOSRule(
                platform="reddit",
                policy_type=PlatformPolicy.API_USAGE,
                rule_id="reddit_oauth_required",
                description="OAuth2 required for API access",
                required_authentication=True,
                required_headers=["Authorization", "User-Agent"]
            ),
            TOSRule(
                platform="reddit",
                policy_type=PlatformPolicy.DATA_COLLECTION,
                rule_id="reddit_no_mass_download",
                description="Prohibited: Mass downloading of user data",
                forbidden_endpoints=["/api/user_data_download"],
                commercial_use_allowed=False
            ),
            TOSRule(
                platform="reddit",
                policy_type=PlatformPolicy.CONTENT_SCRAPING,
                rule_id="reddit_scraping_limits",
                description="Content scraping limited to API endpoints only",
                data_retention_limit_days=30,
                commercial_use_allowed=False
            )
        ]
        
        # Twitter TOS Rules (as of 2024)
        twitter_rules = [
            TOSRule(
                platform="twitter",
                policy_type=PlatformPolicy.RATE_LIMITING,
                rule_id="twitter_rate_limit",
                description="Twitter API v2 rate limiting: 300 requests per 15 minutes",
                max_requests_per_hour=1200,  # 300 per 15min
                max_requests_per_day=28800,
                required_authentication=True
            ),
            TOSRule(
                platform="twitter",
                policy_type=PlatformPolicy.API_USAGE,
                rule_id="twitter_bearer_token",
                description="Bearer Token required for API v2 access",
                required_authentication=True,
                required_headers=["Authorization"]
            ),
            TOSRule(
                platform="twitter",
                policy_type=PlatformPolicy.DATA_COLLECTION,
                rule_id="twitter_data_limits",
                description="Data collection limits per Twitter Developer Policy",
                data_retention_limit_days=90,
                commercial_use_allowed=True  # With proper license
            ),
            TOSRule(
                platform="twitter",
                policy_type=PlatformPolicy.AUTOMATED_ACCESS,
                rule_id="twitter_automation_rules",
                description="Automated access restrictions",
                forbidden_endpoints=["/1.1/statuses/update", "/1.1/friendships/create"]
            )
        ]
        
        # Telegram TOS Rules
        telegram_rules = [
            TOSRule(
                platform="telegram",
                policy_type=PlatformPolicy.RATE_LIMITING,
                rule_id="telegram_rate_limit",
                description="Telegram Bot API: 30 requests per second",
                max_requests_per_hour=108000,
                max_requests_per_day=2592000,
                required_authentication=True
            ),
            TOSRule(
                platform="telegram",
                policy_type=PlatformPolicy.DATA_COLLECTION,
                rule_id="telegram_public_only",
                description="Only public channels and groups allowed",
                commercial_use_allowed=True,
                data_retention_limit_days=365
            )
        ]
        
        rules["reddit"] = reddit_rules
        rules["twitter"] = twitter_rules
        rules["telegram"] = telegram_rules
        
        return dict(rules)
    
    def validate_request(self, 
                        platform: str,
                        endpoint: str,
                        headers: Dict[str, str],
                        request_type: str = "GET") -> Dict[str, Any]:
        """Validate request against TOS rules"""
        
        with self.lock:
            validation_result = {
                "compliant": True,
                "violations": [],
                "warnings": [],
                "compliance_level": ComplianceLevel.COMPLIANT,
                "timestamp": datetime.now().isoformat()
            }
            
            if platform not in self.tos_rules:
                validation_result["warnings"].append(f"No TOS rules defined for platform: {platform}")
                return validation_result
            
            platform_rules = self.tos_rules[platform]
            
            # Check each rule
            for rule in platform_rules:
                violation = self._check_rule_compliance(
                    rule, platform, endpoint, headers, request_type
                )
                
                if violation:
                    validation_result["violations"].append(violation.to_dict())
                    
                    if violation.severity in [ComplianceLevel.VIOLATION, ComplianceLevel.CRITICAL, ComplianceLevel.BANNED]:
                        validation_result["compliant"] = False
                    
                    # Track the most severe violation
                    if (validation_result["compliance_level"] == ComplianceLevel.COMPLIANT or
                        violation.severity.value > validation_result["compliance_level"].value):
                        validation_result["compliance_level"] = violation.severity
            
            # Record validation metrics
            self._record_validation_metrics(platform, validation_result["compliant"])
            
            return validation_result
    
    def _check_rule_compliance(self,
                              rule: TOSRule,
                              platform: str,
                              endpoint: str,
                              headers: Dict[str, str],
                              request_type: str) -> Optional[ComplianceViolation]:
        """Check compliance against specific rule"""
        
        # Rate limiting check
        if rule.policy_type == PlatformPolicy.RATE_LIMITING:
            rate_violation = self._check_rate_limits(rule, platform)
            if rate_violation:
                return rate_violation
        
        # Authentication check
        if rule.policy_type == PlatformPolicy.API_USAGE:
            auth_violation = self._check_authentication(rule, headers)
            if auth_violation:
                return auth_violation
        
        # Endpoint restrictions
        if rule.policy_type == PlatformPolicy.AUTOMATED_ACCESS:
            endpoint_violation = self._check_forbidden_endpoints(rule, endpoint)
            if endpoint_violation:
                return endpoint_violation
        
        # Header requirements
        if rule.required_headers:
            header_violation = self._check_required_headers(rule, headers)
            if header_violation:
                return header_violation
        
        # User-Agent requirement
        if rule.user_agent_required:
            ua_violation = self._check_user_agent(rule, headers)
            if ua_violation:
                return ua_violation
        
        return None
    
    def _check_rate_limits(self, rule: TOSRule, platform: str) -> Optional[ComplianceViolation]:
        """Check rate limiting compliance"""
        
        now = datetime.now()
        platform_history = self.request_history[platform]
        
        # Count requests in last hour
        hour_ago = now - timedelta(hours=1)
        hourly_requests = sum(1 for req_time in platform_history 
                            if req_time > hour_ago)
        
        # Count requests in last day
        day_ago = now - timedelta(days=1)
        daily_requests = sum(1 for req_time in platform_history 
                           if req_time > day_ago)
        
        # Check hourly limit
        if rule.max_requests_per_hour and hourly_requests >= rule.max_requests_per_hour:
            return ComplianceViolation(
                platform=platform,
                rule_id=rule.rule_id,
                violation_type=PlatformPolicy.RATE_LIMITING,
                severity=ComplianceLevel.VIOLATION,
                description=f"Hourly rate limit exceeded: {hourly_requests}/{rule.max_requests_per_hour}",
                request_count=hourly_requests
            )
        
        # Check daily limit
        if rule.max_requests_per_day and daily_requests >= rule.max_requests_per_day:
            return ComplianceViolation(
                platform=platform,
                rule_id=rule.rule_id,
                violation_type=PlatformPolicy.RATE_LIMITING,
                severity=ComplianceLevel.CRITICAL,
                description=f"Daily rate limit exceeded: {daily_requests}/{rule.max_requests_per_day}",
                request_count=daily_requests
            )
        
        return None
    
    def _check_authentication(self, rule: TOSRule, headers: Dict[str, str]) -> Optional[ComplianceViolation]:
        """Check authentication requirements"""
        
        if rule.required_authentication:
            auth_header = headers.get("Authorization")
            if not auth_header:
                return ComplianceViolation(
                    platform=rule.platform,
                    rule_id=rule.rule_id,
                    violation_type=PlatformPolicy.API_USAGE,
                    severity=ComplianceLevel.VIOLATION,
                    description="Required authentication header missing"
                )
        
        return None
    
    def _check_forbidden_endpoints(self, rule: TOSRule, endpoint: str) -> Optional[ComplianceViolation]:
        """Check forbidden endpoint access"""
        
        for forbidden in rule.forbidden_endpoints:
            if forbidden in endpoint:
                return ComplianceViolation(
                    platform=rule.platform,
                    rule_id=rule.rule_id,
                    violation_type=PlatformPolicy.AUTOMATED_ACCESS,
                    severity=ComplianceLevel.CRITICAL,
                    description=f"Access to forbidden endpoint: {endpoint}",
                    endpoint=endpoint
                )
        
        return None
    
    def _check_required_headers(self, rule: TOSRule, headers: Dict[str, str]) -> Optional[ComplianceViolation]:
        """Check required headers"""
        
        missing_headers = []
        for required_header in rule.required_headers:
            if required_header not in headers:
                missing_headers.append(required_header)
        
        if missing_headers:
            return ComplianceViolation(
                platform=rule.platform,
                rule_id=rule.rule_id,
                violation_type=PlatformPolicy.API_USAGE,
                severity=ComplianceLevel.WARNING,
                description=f"Missing required headers: {missing_headers}"
            )
        
        return None
    
    def _check_user_agent(self, rule: TOSRule, headers: Dict[str, str]) -> Optional[ComplianceViolation]:
        """Check User-Agent header requirement"""
        
        user_agent = headers.get("User-Agent")
        if not user_agent or user_agent.lower() in ["python-requests", "curl", "wget"]:
            return ComplianceViolation(
                platform=rule.platform,
                rule_id=rule.rule_id,
                violation_type=PlatformPolicy.API_USAGE,
                severity=ComplianceLevel.WARNING,
                description="Invalid or missing User-Agent header",
                user_agent=user_agent
            )
        
        return None
    
    def record_request(self, platform: str):
        """Record request for rate limiting tracking"""
        with self.lock:
            self.request_history[platform].append(datetime.now())
    
    def _record_validation_metrics(self, platform: str, compliant: bool):
        """Record validation metrics"""
        metrics = self.compliance_metrics[platform]
        metrics["total_requests"] += 1
        if compliant:
            metrics["compliant_requests"] += 1
        else:
            metrics["violations"] += 1
        metrics["last_check"] = datetime.now().isoformat()
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        
        with self.lock:
            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_compliance": {},
                "platform_metrics": {},
                "recent_violations": [],
                "risk_assessment": {}
            }
            
            # Calculate overall compliance
            total_requests = sum(m["total_requests"] for m in self.compliance_metrics.values())
            total_compliant = sum(m["compliant_requests"] for m in self.compliance_metrics.values())
            total_violations = sum(m["violations"] for m in self.compliance_metrics.values())
            
            report["overall_compliance"] = {
                "total_requests": total_requests,
                "compliance_rate": total_compliant / max(1, total_requests),
                "violation_rate": total_violations / max(1, total_requests),
                "total_violations": total_violations
            }
            
            # Platform-specific metrics
            for platform, metrics in self.compliance_metrics.items():
                report["platform_metrics"][platform] = {
                    "total_requests": metrics["total_requests"],
                    "compliance_rate": metrics["compliant_requests"] / max(1, metrics["total_requests"]),
                    "violation_count": metrics["violations"],
                    "last_check": metrics["last_check"]
                }
            
            # Recent violations (last 24 hours)
            day_ago = datetime.now() - timedelta(days=1)
            recent_violations = [v.to_dict() for v in self.violations 
                               if v.timestamp > day_ago]
            
            report["recent_violations"] = recent_violations[-50:]  # Last 50 violations
            
            # Risk assessment
            report["risk_assessment"] = self._assess_compliance_risk()
            
            return report
    
    def _assess_compliance_risk(self) -> Dict[str, str]:
        """Assess overall compliance risk level"""
        
        # Calculate risk factors
        total_requests = sum(m["total_requests"] for m in self.compliance_metrics.values())
        total_violations = sum(m["violations"] for m in self.compliance_metrics.values())
        
        if total_requests == 0:
            return {"level": "unknown", "reason": "No requests recorded"}
        
        violation_rate = total_violations / total_requests
        
        # Recent violation trend
        recent_violations = [v for v in self.violations 
                           if (datetime.now() - v.timestamp).total_seconds() < 3600]
        
        critical_violations = [v for v in recent_violations 
                             if v.severity in [ComplianceLevel.CRITICAL, ComplianceLevel.BANNED]]
        
        # Risk assessment logic
        if critical_violations:
            return {"level": "critical", "reason": f"{len(critical_violations)} critical violations in last hour"}
        elif violation_rate > 0.3:
            return {"level": "high", "reason": f"High violation rate: {violation_rate:.1%}"}
        elif violation_rate > 0.1:
            return {"level": "medium", "reason": f"Moderate violation rate: {violation_rate:.1%}"}
        elif violation_rate > 0.05:
            return {"level": "low", "reason": f"Low violation rate: {violation_rate:.1%}"}
        else:
            return {"level": "minimal", "reason": "Excellent compliance record"}
    
    def update_tos_rules(self, platform: str, rules: List[TOSRule]):
        """Update TOS rules for platform"""
        with self.lock:
            self.tos_rules[platform] = rules
            self.logger.info(f"Updated TOS rules for {platform}: {len(rules)} rules")
    
    def export_compliance_data(self, file_path: str):
        """Export compliance data for audit purposes"""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "tos_rules": {platform: [rule.to_dict() for rule in rules] 
                         for platform, rules in self.tos_rules.items()},
            "violations": [v.to_dict() for v in self.violations],
            "compliance_metrics": dict(self.compliance_metrics),
            "compliance_report": self.get_compliance_report()
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Compliance data exported to {file_path}")


# Global singleton
_compliance_validator = None

def get_tos_validator() -> TOSComplianceValidator:
    """Get singleton TOS compliance validator"""
    global _compliance_validator
    if _compliance_validator is None:
        _compliance_validator = TOSComplianceValidator()
    return _compliance_validator


if __name__ == "__main__":
    # Basic validation test
    validator = get_tos_validator()
    
    # Test Reddit request validation
    reddit_headers = {
        "Authorization": "Bearer test_token",
        "User-Agent": "CryptoSmartTrader/2.0"
    }
    
    result = validator.validate_request(
        platform="reddit",
        endpoint="/r/cryptocurrency/hot",
        headers=reddit_headers
    )
    
    print(f"Reddit validation result: {result}")
    
    # Generate compliance report
    report = validator.get_compliance_report()
    print(f"Compliance report: {report}")
    
    print("TOS Compliance Validator: Operational")