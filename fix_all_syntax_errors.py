#!/usr/bin/env python3
"""
Definitieve syntax error fix script voor CryptoSmartTrader V2
"""
import os
import re
import sys

def fix_file_content(content):
    """Fix all known syntax error patterns"""
    
    # Fix broken function definitions
    content = re.sub(r'def _generate_mock_data(self', 'def _generate_mock_data(self', content)
    content = re.sub(r'def _generate_mock_data([^(]*)', r'def _generate_mock_data\1', content)
    content = re.sub(r'def _add_generate_sample_data(self', 'def _add_generate_sample_data(self', content)
    content = re.sub(r'def _audit_no_generate_sample_data_self\)', 'def _audit_no_generate_sample_data(self):', content)
    content = re.sub(r'def _generate_sample_data_self\)', 'def _generate_sample_data(self):', content)
    content = re.sub(r'async def mock_task([^(]*)', r'async def mock_task\1', content)
    
    # Fix broken function calls
    content = re.sub(r'return self\._generate_# REMOVED: Mock data pattern not allowed in production([^)]*)\)', r'return self._generate_mock_data\1)', content)
    
    # Fix broken parameters and types
    content = re.sub(r'market_condition: MarketCondition = MarketCondition\.BULL_MARKET\) -> StressTestResult:', 'market_condition: MarketCondition = MarketCondition.BULL_MARKET) -> StressTestResult:', content)
    content = re.sub(r'market_data: Dict\[str, Any\]\) -> ExecutionResult:', 'market_data: dict[str, Any]) -> ExecutionResult:', content)
    content = re.sub(r'market_data: Dict\[str, Any\]\) -> Dict\[str, Any\]:', 'market_data: dict[str, Any]) -> dict[str, Any]:', content)
    
    # Fix numpy random issues
    content = re.sub(r'np\.random\.normal\(0, 1\)\), replace=False\)', 'np.random.choice(range(100), size=10, replace=False)', content)
    content = re.sub(r'np\.random\.normal\(0, 1\) - recovery_hours - 10\)', 'random.randint(1, 100) - recovery_hours - 10', content)
    content = re.sub(r'np\.random\.normal\(0, 1\), batch_size, replace=False\)', 'np.random.choice(range(1000), batch_size, replace=False)', content)
    content = re.sub(r'np\.random\.normal\(0, 1\), size=sample_size, replace=False\)', 'np.random.choice(range(1000), size=sample_size, replace=False)', content)
    
    # Fix parenthesis issues
    content = re.sub(r'\)\s*\n\s*\) -> ExecutionResult:', ') -> ExecutionResult:', content)
    content = re.sub(r'\)\s*\n\s*\)', ')', content)
    
    # Fix specific regex pattern
    content = re.sub(r"\(r'api_key\[\"'\?\]\\s\*\[:=\]\\s\*\[\"'\?\]\?\', r'\\1\*\*\*MASKED\*\*\*'\),", '(r"api_key=.+", r"api_key=***MASKED***"),', content)
    
    # Fix annotation issues
    content = re.sub(r'# REMOVED: Mock data pattern not allowed in production\{', 'mock_posts.append({', content)
    
    return content

def fix_all_files():
    """Fix all Python files with syntax errors"""
    fixed_count = 0
    
    # Get all Python files
    for root, dirs, files in os.walk('.'):
        # Skip cache directories
        if any(skip in root for skip in ['__pycache__', '.git', '.cache', 'venv']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()
                    
                    fixed_content = fix_file_content(original_content)
                    
                    if fixed_content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        print(f"Fixed: {file_path}")
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"Error fixing {file_path}: {e}")
    
    print(f"\n🎯 Fixed {fixed_count} files with syntax errors")
    return fixed_count > 0

if __name__ == "__main__":
    success = fix_all_files()
    sys.exit(0 if success else 1)
