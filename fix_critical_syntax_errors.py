#!/usr/bin/env python3
"""Fix critical syntax errors that prevent compilation."""

import re
from pathlib import Path


def fix_critical_syntax_errors():
    """Fix the most critical syntax errors."""
    
    fixes = [
        # Fix broken function definitions
        ('src/cryptosmarttrader/agents/ml_predictor_agent.py', 
         r'def _generate_generate_sample_data_self, symbol: str\) -> pd\.DataFrame:',
         'def _generate_sample_data(self, symbol: str) -> pd.DataFrame:'),
         
        ('src/cryptosmarttrader/agents/whale_detector.py',
         r'def _generate_sample_data_self\):',
         'def _generate_sample_data(self):'),
         
        ('src/cryptosmarttrader/core/batch_inference_engine.py',
         r'def _create_generate_sample_data_self\):',
         'def _create_sample_data(self):'),
         
        ('src/cryptosmarttrader/regime/regime_detector.py',
         r'def _get_generate_sample_data_self\) -> Dict\[str, pd\.DataFrame\]:',
         'def _get_sample_data(self) -> Dict[str, pd.DataFrame]:'),
         
        # Fix broken function calls with extra parentheses
        ('src/cryptosmarttrader/core/crypto_ai_system.py',
         r"feature_df\[f'target_{horizon}'\] = np\.random\.normal\(0, 1\)\)",
         "feature_df[f'target_{horizon}'] = np.random.normal(0, 1)"),
         
        ('src/cryptosmarttrader/core/explainability_engine.py',
         r'importance_values = np\.random\.normal\(0, 1\)\)',
         'importance_values = np.random.normal(0, 1)'),
         
        ('src/cryptosmarttrader/core/explainable_ai.py',
         r'background = np\.random\.normal\(0, 1\)\)',
         'background = np.random.normal(0, 1)'),
         
        # Fix broken function names with mock pattern remnants
        ('src/cryptosmarttrader/core/enterprise_integrator.py',
         r'def # REMOVED: Mock data pattern not allowed in productionself, trading_opportunities: List\[Dict\]\) -> Dict\[str, Any\]:',
         'def process_trading_opportunities(self, trading_opportunities: List[Dict]) -> Dict[str, Any]:'),
         
        ('src/cryptosmarttrader/risk/risk_limits.py',
         r'def # REMOVED: Mock data pattern not allowed in productionself, duration_seconds: float\):',
         'def simulate_stress_test(self, duration_seconds: float):'),
         
        # Fix f-string issues  
        ('src/cryptosmarttrader/agents/enhanced_whale_agent.py',
         r"'hash': f\"0x\{''.join\(# REMOVED: Mock data pattern not allowed in productions\('0123456789abcdef', k=64\)\)\}\",",
         "'hash': f\"0x{''.join(random.choices('0123456789abcdef', k=64))}\","),
         
        # Fix dictionary annotation syntax
        ('src/cryptosmarttrader/agents/scraping_core/data_sources.py',
         r'"id": f"tweet_\{i\}_\{hash\(query\) % 10000\}",',
         '"id": f"tweet_{i}_{hash(query) % 10000}",'),
    ]
    
    for filepath, pattern, replacement in fixes:
        file_path = Path(filepath)
        if file_path.exists():
            try:
                content = file_path.read_text()
                new_content = re.sub(pattern, replacement, content)
                
                if new_content != content:
                    file_path.write_text(new_content)
                    print(f"✅ Fixed syntax in {filepath}")
                    
            except Exception as e:
                print(f"❌ Error fixing {filepath}: {e}")


def fix_indentation_errors():
    """Fix indentation errors."""
    files_to_fix = [
        'src/cryptosmarttrader/core/continual_learning_engine.py',
        'src/cryptosmarttrader/core/fine_tune_scheduler.py'
    ]
    
    for filepath in files_to_fix:
        file_path = Path(filepath)
        if file_path.exists():
            try:
                lines = file_path.read_text().splitlines()
                fixed_lines = []
                
                for i, line in enumerate(lines):
                    # Fix common indentation issues
                    if line.strip().startswith('size=') and i > 0:
                        # Align with previous line's indentation
                        prev_line = lines[i-1]
                        indent = len(prev_line) - len(prev_line.lstrip())
                        line = ' ' * (indent + 4) + line.strip()
                    
                    fixed_lines.append(line)
                
                file_path.write_text('\n'.join(fixed_lines) + '\n')
                print(f"✅ Fixed indentation in {filepath}")
                
            except Exception as e:
                print(f"❌ Error fixing indentation in {filepath}: {e}")


if __name__ == "__main__":
    print("🔧 Fixing Critical Syntax Errors")
    print("=" * 40)
    
    fix_critical_syntax_errors()
    fix_indentation_errors()
    
    print("\n✨ Critical syntax fixes complete")