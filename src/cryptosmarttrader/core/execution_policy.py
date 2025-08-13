"""
Execution Policy - Alias to canonical implementation
This file redirects to the single source of truth in execution module
"""

# Import from canonical source
from ..execution.execution_policy import *

# Maintain backward compatibility
from ..execution.execution_policy import ExecutionPolicy as ExecutionPolicySystem
