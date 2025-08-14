#!/usr/bin/env python3
"""
Critical Syntax Error Fixer
Fast bulk fix for all syntax errors in the codebase
"""

import os
import ast
import re
from pathlib import Path

def fix_syntax_errors():
    """Fix all critical syntax errors in bulk"""
    
    fixes_applied = 0
    
    # Pattern-based fixes
    pattern_fixes = [
        # Fix malformed regex patterns
        (r'\(r\'.*["\'].*\)', lambda m: m.group(0).replace('["\']', '["\'"]')),
        
        # Fix missing method definitions  
        (r'def # REMOVED:.*\n', 'def simulate_method('),
        
        # Fix broken numpy calls
        (r'np\.random\.normal\([^)]*\) // 2,', 'np.random.choice(10,'),
        
        # Fix incomplete mock data patterns
        (r'# REMOVED: Mock data pattern.*', ''),
        
        # Fix broken function calls
        (r'self\._# REMOVED:.*\(', 'self._calculate_value('),
        
        # Fix incomplete field definitions
        (r'default=False,\s*env="[^"]*",\s*description="[^"]*"\s*\)', 
         'default=False, env="MOCK_EXCHANGES", description="Mock data"'),
    ]
    
    # Files to fix
    files_to_fix = [
        'src/cryptosmarttrader/core/improved_logging_manager.py',
        'src/cryptosmarttrader/core/pydantic_settings.py', 
        'src/cryptosmarttrader/core/black_swan_simulation_engine.py',
        'src/cryptosmarttrader/core/continual_learning_engine.py',
        'src/cryptosmarttrader/core/execution_simulator.py',
    ]
    
    for filepath in files_to_fix:
        if not os.path.exists(filepath):
            continue
            
        try:
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Apply pattern fixes
            original_content = content
            for pattern, replacement in pattern_fixes:
                if callable(replacement):
                    content = re.sub(pattern, replacement, content)
                else:
                    content = re.sub(pattern, replacement, content)
            
            # Specific file fixes
            if 'improved_logging_manager.py' in filepath:
                content = content.replace(
                    r'(token["\'"]?\s*[:=]\s*["\'"]?)([^"\'"\s]+)',
                    r'(token["\']?\s*[:=]\s*["\']?)([^"\'\s]+)'
                )
                
            elif 'pydantic_settings.py' in filepath:
                # Fix the broken field definition
                content = re.sub(
                    r'# === DEVELOPMENT SETTINGS ===\s*# REMOVED:.*?\)',
                    '''# === DEVELOPMENT SETTINGS ===
    mock_exchanges: bool = Field(
        default=False,
        env="MOCK_EXCHANGES",
        description="Use mock exchange data for development"
    )''',
                    content,
                    flags=re.DOTALL
                )
                
            elif 'black_swan_simulation_engine.py' in filepath:
                # Fix method definitions
                content = re.sub(
                    r'def # REMOVED:.*?self, event: BlackSwanEvent,',
                    'def simulate_stress_test(self, event: BlackSwanEvent,',
                    content
                )
                content = re.sub(
                    r'max_drawdown = self\._# REMOVED:.*?event, market_condition\)',
                    'max_drawdown = self._calculate_max_drawdown(event, market_condition)',
                    content
                )
                content = re.sub(
                    r'model_degradation = self\._# REMOVED:.*?event\)',
                    'model_degradation = self._calculate_model_degradation(event)',
                    content
                )
                
            elif 'continual_learning_engine.py' in filepath:
                # Fix numpy choice call
                content = re.sub(
                    r'indices_to_remove = np\.random\.normal.*?replace=False\s*\)',
                    '''indices_to_remove = np.random.choice(
                len(buffer) // 2,
                size=remove_count,
                replace=False
            )''',
                    content,
                    flags=re.DOTALL
                )
                
            elif 'execution_simulator.py' in filepath:
                # Fix async method definition
                content = re.sub(
                    r'async def # REMOVED:.*?\n',
                    'async def simulate_order_execution(\n',
                    content
                )
                
            # Check if content changed and syntax is valid
            if content != original_content:
                try:
                    ast.parse(content)
                    with open(filepath, 'w') as f:
                        f.write(content)
                    fixes_applied += 1
                    print(f"✅ Fixed: {filepath}")
                except SyntaxError as e:
                    print(f"❌ Still has syntax error: {filepath} - {e}")
            else:
                print(f"⚠️ No changes: {filepath}")
                
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
    
    print(f"\n🎯 Total fixes applied: {fixes_applied}")
    
    # Validate all Python files
    total_files = 0
    valid_files = 0
    
    for root, dirs, files in os.walk('src/'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                total_files += 1
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    valid_files += 1
                except:
                    pass
    
    print(f"📊 Validation: {valid_files}/{total_files} files have valid syntax")
    return fixes_applied

if __name__ == "__main__":
    fix_syntax_errors()