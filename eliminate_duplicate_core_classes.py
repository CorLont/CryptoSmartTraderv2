#!/usr/bin/env python3
"""
Eliminate Duplicate Core Classes
Ensures single canonical implementation for RiskGuard and ExecutionPolicy
"""

import os
import shutil
from pathlib import Path

def eliminate_duplicates():
    """Remove duplicate core classes and establish canonical paths"""
    
    # Define canonical paths (these are the authoritative implementations)
    canonical_classes = {
        'RiskGuard': 'src/cryptosmarttrader/risk/risk_guard.py',
        'ExecutionPolicy': 'src/cryptosmarttrader/execution/execution_policy.py'
    }
    
    # Identify duplicate files to remove
    duplicate_files = [
        'src/cryptosmarttrader/core/risk_guard.py',
        'src/cryptosmarttrader/core/execution_policy.py'
    ]
    
    removed_count = 0
    
    print("🔧 Eliminating Duplicate Core Classes")
    print("=" * 40)
    
    # Remove duplicates
    for duplicate in duplicate_files:
        if os.path.exists(duplicate):
            # Move to backup before removal
            backup_dir = "backups/eliminated_duplicates"
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = f"{backup_dir}/{os.path.basename(duplicate)}"
            
            shutil.move(duplicate, backup_path)
            removed_count += 1
            print(f"✅ Removed duplicate: {duplicate}")
            print(f"   Backed up to: {backup_path}")
    
    # Create alias files in core/ to redirect imports
    for class_name, canonical_path in canonical_classes.items():
        if class_name == 'RiskGuard':
            alias_path = 'src/cryptosmarttrader/core/risk_guard.py'
            relative_import = '..risk.risk_guard'
        elif class_name == 'ExecutionPolicy':
            alias_path = 'src/cryptosmarttrader/core/execution_policy.py'
            relative_import = '..execution.execution_policy'
        
        # Create alias file
        alias_content = f'''"""
Alias for {class_name}
Redirects to canonical implementation at {canonical_path}
"""

# Import canonical implementation
from {relative_import} import {class_name}

# Re-export for backward compatibility
__all__ = ['{class_name}']
'''
        
        with open(alias_path, 'w') as f:
            f.write(alias_content)
        print(f"✅ Created alias: {alias_path} → {canonical_path}")
    
    # Update imports in critical files
    critical_files = [
        'src/cryptosmarttrader/core/order_pipeline.py',
        'src/cryptosmarttrader/core/integrated_trading_engine.py',
        'src/cryptosmarttrader/core/data_flow_orchestrator.py'
    ]
    
    import_fixes = 0
    for filepath in critical_files:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                
                original_content = content
                
                # Fix imports to use canonical paths
                content = content.replace(
                    'from .risk_guard import RiskGuard',
                    'from ..risk.risk_guard import RiskGuard'
                )
                content = content.replace(
                    'from .execution_policy import ExecutionPolicy',
                    'from ..execution.execution_policy import ExecutionPolicy'
                )
                content = content.replace(
                    'from cryptosmarttrader.core.risk_guard import RiskGuard',
                    'from cryptosmarttrader.risk.risk_guard import RiskGuard'
                )
                content = content.replace(
                    'from cryptosmarttrader.core.execution_policy import ExecutionPolicy',
                    'from cryptosmarttrader.execution.execution_policy import ExecutionPolicy'
                )
                
                if content != original_content:
                    with open(filepath, 'w') as f:
                        f.write(content)
                    import_fixes += 1
                    print(f"✅ Fixed imports: {filepath}")
                    
            except Exception as e:
                print(f"❌ Error fixing imports in {filepath}: {e}")
    
    # Verify canonical implementations exist and are valid
    for class_name, canonical_path in canonical_classes.items():
        if os.path.exists(canonical_path):
            try:
                with open(canonical_path, 'r') as f:
                    content = f.read()
                if f'class {class_name}' in content:
                    print(f"✅ Verified canonical: {canonical_path}")
                else:
                    print(f"⚠️  Warning: {class_name} not found in {canonical_path}")
            except Exception as e:
                print(f"❌ Error verifying {canonical_path}: {e}")
        else:
            print(f"❌ Missing canonical: {canonical_path}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Duplicates removed: {removed_count}")
    print(f"✅ Aliases created: {len(canonical_classes)}")
    print(f"✅ Import fixes: {import_fixes}")
    print(f"\n🎯 Canonical paths established:")
    for class_name, path in canonical_classes.items():
        print(f"  {class_name}: {path}")

if __name__ == "__main__":
    eliminate_duplicates()