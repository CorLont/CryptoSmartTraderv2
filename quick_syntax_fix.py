#!/usr/bin/env python3
import os
import re

# Quick fixes for all remaining syntax errors
files_to_fix = [
    ('src/cryptosmarttrader/core/process_isolation.py', [
        (r'async def # REMOVED:.*?\) -> None:', 'async def monitor_process(self, stop_event) -> None:')
    ]),
    ('src/cryptosmarttrader/core/shap_regime_analyzer.py', [
        (r'sample_indices = np\.random\.normal\(0, 1\), size=sample_size, replace=False\)', 
         'sample_indices = np.random.choice(len(X), size=sample_size, replace=False)')
    ]),
    ('src/cryptosmarttrader/core/synthetic_data_augmentation.py', [
        (r'crash_start = np\.random\.normal\(0, 1\) - recovery_hours - 10\)',
         'crash_start = max(0, len(returns) - recovery_hours - 10)')
    ]),
    ('src/cryptosmarttrader/ml/continual_learning/drift_detection_ewc.py', [
        (r'indices = np\.random\.normal\(0, 1\), batch_size, replace=False\)',
         'indices = np.random.choice(len(data), batch_size, replace=False)')
    ])
]

fixed_count = 0
for filepath, fixes in files_to_fix:
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)
            
            with open(filepath, 'w') as f:
                f.write(content)
            fixed_count += 1
            print(f"Fixed: {filepath}")
        except Exception as e:
            print(f"Error: {filepath} - {e}")

print(f"Fixed {fixed_count} files")