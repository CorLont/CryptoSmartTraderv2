#!/usr/bin/env python3
"""
Core CryptoSmartTrader components with mandatory risk enforcement
"""

from .mandatory_execution_gateway import MandatoryExecutionGateway, UniversalOrderRequest, GatewayResult
from .mandatory_risk_enforcement import (
    mandatory_risk_enforcement,
    enforce_order_risk_check,
    mandatory_risk_check,
    require_risk_approval
)
from .centralized_risk_integration import (
    apply_system_wide_risk_integration,
    get_risk_integration_status, 
    generate_risk_integration_report
)

# Auto-initialize risk integration when core is imported
try:
    integration_result = apply_system_wide_risk_integration()
    print(f"🛡️ Risk integration auto-initialized: {integration_result['integrated_modules']} modules")
except Exception as e:
    print(f"Warning: Risk integration initialization failed: {str(e)}")

__all__ = [
    # Gateway components
    'MandatoryExecutionGateway',
    'UniversalOrderRequest', 
    'GatewayResult',
    
    # Risk enforcement
    'mandatory_risk_enforcement',
    'enforce_order_risk_check',
    'mandatory_risk_check',
    'require_risk_approval',
    
    # Integration management
    'apply_system_wide_risk_integration',
    'get_risk_integration_status',
    'generate_risk_integration_report'
]