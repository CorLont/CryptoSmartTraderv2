#!/usr/bin/env python3
"""
Fix Temporal Violations Script
Scans and fixes look-ahead bias across the entire ML pipeline
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.temporal_integrity_validator import (
    validate_ml_dataset,
    TemporalIntegrityValidator,
    TemporalDataBuilder,
)


class MLPipelineTemporalAuditor:
    """Audits entire ML pipeline for temporal violations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent.parent
        self.violations_found = []
        self.files_scanned = 0

    def audit_entire_pipeline(self) -> dict:
        """Comprehensive audit of entire ML pipeline"""

        print("🔍 COMPREHENSIVE TEMPORAL INTEGRITY AUDIT")
        print("=" * 60)

        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "files_scanned": 0,
            "violations_found": [],
            "critical_files": [],
            "fixed_files": [],
            "recommendations": [],
        }

        # Scan ML modules
        ml_files = list((self.project_root / "ml").glob("*.py"))
        audit_results.update(self._scan_ml_modules(ml_files))

        # Scan data files
        data_files = []
        for data_dir in ["data", "exports", "cache"]:
            data_path = self.project_root / data_dir
            if data_path.exists():
                data_files.extend(list(data_path.glob("*.csv")))
                data_files.extend(list(data_path.glob("*.json")))

        if data_files:
            audit_results.update(self._scan_data_files(data_files))

        # Generate final report
        audit_results["files_scanned"] = self.files_scanned
        audit_results["total_violations"] = len(self.violations_found)
        audit_results["critical_violations"] = len(
            [v for v in self.violations_found if v.get("severity") == "critical"]
        )

        # Generate recommendations
        recommendations = self._generate_pipeline_recommendations()
        audit_results["recommendations"] = recommendations

        return audit_results

    def _scan_ml_modules(self, ml_files: list) -> dict:
        """Scan ML Python files for temporal violations"""

        print(f"📁 Scanning {len(ml_files)} ML modules...")

        critical_patterns = [
            # Target calculation patterns that indicate look-ahead bias
            r"\.shift\(-\d+\)",  # Negative shifts
            r"\.shift\(\s*-\s*\d+\s*\)",  # Negative shifts with spaces
            r"target.*=.*\[\s*\+\d+\s*\]",  # Forward indexing for targets
            r"target.*=.*\.iloc\[\s*\+\d+\s*\]",  # Forward iloc for targets
            # Feature calculation patterns
            r"feature.*\.shift\(-\d+\)",  # Features using future data
            r"forward_fill\(\)",  # Forward filling (suspicious)
            r'fillna\(method=["\']forward["\']',  # Forward fill method
            # Rolling calculation issues
            r"\.rolling\(\d+\)\.(?!.*min_periods)",  # Rolling without min_periods
            r"\.expanding\(\)\.(?!.*min_periods)",  # Expanding without min_periods
        ]

        violations = []

        for ml_file in ml_files:
            self.files_scanned += 1

            try:
                content = ml_file.read_text()

                for pattern in critical_patterns:
                    import re

                    matches = re.findall(pattern, content, re.IGNORECASE)

                    if matches:
                        violation = {
                            "file": str(ml_file.relative_to(self.project_root)),
                            "pattern": pattern,
                            "matches": len(matches),
                            "severity": "critical",
                            "type": "code_pattern",
                            "description": f"Found {len(matches)} potential temporal violations",
                        }
                        violations.append(violation)
                        self.violations_found.append(violation)

                        print(f"⚠️  {ml_file.name}: {len(matches)} violations ({pattern[:20]}...)")

            except Exception as e:
                print(f"❌ Error scanning {ml_file.name}: {e}")

        return {"ml_violations": violations}

    def _scan_data_files(self, data_files: list) -> dict:
        """Scan data files for temporal integrity issues"""

        print(f"📊 Scanning {len(data_files)} data files...")

        data_violations = []

        for data_file in data_files:
            self.files_scanned += 1

            try:
                if data_file.suffix == ".csv":
                    df = pd.read_csv(data_file, nrows=100)  # Sample first 100 rows

                    # Check for timestamp column
                    timestamp_cols = [
                        col
                        for col in df.columns
                        if "timestamp" in col.lower() or "time" in col.lower()
                    ]

                    if timestamp_cols:
                        # Basic temporal validation
                        validation_result = validate_ml_dataset(
                            df, timestamp_col=timestamp_cols[0], fix_violations=False
                        )

                        if not validation_result["is_valid"]:
                            violation = {
                                "file": str(data_file.relative_to(self.project_root)),
                                "critical_violations": validation_result["critical_violations"],
                                "warning_violations": validation_result["warning_violations"],
                                "severity": "critical"
                                if validation_result["critical_violations"] > 0
                                else "warning",
                                "type": "data_integrity",
                                "description": f"Data file has {validation_result['critical_violations']} critical temporal violations",
                            }
                            data_violations.append(violation)
                            self.violations_found.append(violation)

                            print(
                                f"⚠️  {data_file.name}: {validation_result['critical_violations']} critical violations"
                            )

            except Exception as e:
                print(f"❌ Error scanning {data_file.name}: {e}")

        return {"data_violations": data_violations}

    def _generate_pipeline_recommendations(self) -> list:
        """Generate comprehensive recommendations for fixing temporal violations"""

        recommendations = []

        critical_count = len([v for v in self.violations_found if v.get("severity") == "critical"])

        if critical_count > 0:
            recommendations.extend(
                [
                    "🚨 CRITICAL: Fix all temporal violations before training any models",
                    "📋 Target Calculation Rules:",
                    "   • Use target = price.shift(-horizon) for future targets",
                    "   • Calculate returns as: (future_price - current_price) / current_price",
                    "   • Always shift targets to future, never features to future",
                    "",
                    "⚙️ Feature Engineering Rules:",
                    "   • Use .rolling(window, min_periods=window) for all rolling calculations",
                    "   • Lag all features by at least 1 period: feature.shift(1)",
                    "   • Never use forward fill (fillna method='forward')",
                    "   • Avoid negative shifts for features: NO .shift(-N)",
                    "",
                    "🔧 Immediate Actions:",
                    "   1. Run temporal validator on all datasets",
                    "   2. Fix target calculations using TemporalDataBuilder",
                    "   3. Implement walk-forward validation",
                    "   4. Add temporal integrity checks to ML pipeline",
                ]
            )

        # Pattern-specific recommendations
        code_violations = [v for v in self.violations_found if v.get("type") == "code_pattern"]
        if code_violations:
            recommendations.extend(
                [
                    "",
                    "📝 Code Pattern Fixes Required:",
                ]
            )

            for violation in code_violations[:5]:  # Top 5
                recommendations.append(f"   • {violation['file']}: Fix {violation['pattern']}")

        # Data-specific recommendations
        data_violations = [v for v in self.violations_found if v.get("type") == "data_integrity"]
        if data_violations:
            recommendations.extend(
                [
                    "",
                    "📊 Data File Fixes Required:",
                ]
            )

            for violation in data_violations[:5]:  # Top 5
                recommendations.append(f"   • {violation['file']}: {violation['description']}")

        return recommendations

    def fix_common_violations(self) -> dict:
        """Attempt to fix common temporal violations automatically"""

        print("🔧 ATTEMPTING AUTOMATIC FIXES...")

        fixes_applied = []

        # Fix common patterns in ML files
        ml_files = list((self.project_root / "ml").glob("*.py"))

        for ml_file in ml_files:
            try:
                content = ml_file.read_text()
                original_content = content

                # Fix common negative shift patterns
                import re

                # Fix target calculations: .shift(-N) -> .shift(-N) with proper comment
                content = re.sub(
                    r"target.*=.*\.shift\(-(\d+)\)",
                    r"# WARNING: Verify this is intentional future shift for target calculation\n        target = price.shift(-\1)  # Future price for target calculation",
                    content,
                )

                # Fix rolling without min_periods
                content = re.sub(
                    r"\.rolling\((\d+)\)\.(?!.*min_periods)",
                    r".rolling(\1, min_periods=\1).",
                    content,
                )

                # Fix expanding without min_periods
                content = re.sub(
                    r"\.expanding\(\)\.(?!.*min_periods)", r".expanding(min_periods=1).", content
                )

                if content != original_content:
                    # Create backup
                    backup_file = ml_file.with_suffix(".py.backup")
                    backup_file.write_text(original_content)

                    # Write fixed content
                    ml_file.write_text(content)

                    fixes_applied.append(
                        {
                            "file": str(ml_file.relative_to(self.project_root)),
                            "backup_created": str(backup_file.relative_to(self.project_root)),
                            "fixes": [
                                "negative_shift_warning",
                                "rolling_min_periods",
                                "expanding_min_periods",
                            ],
                        }
                    )

                    print(f"✅ Fixed {ml_file.name} (backup created)")

            except Exception as e:
                print(f"❌ Error fixing {ml_file.name}: {e}")

        return {"fixes_applied": fixes_applied}


def main():
    """Run comprehensive temporal integrity audit and fixes"""

    auditor = MLPipelineTemporalAuditor()

    # Run comprehensive audit
    audit_results = auditor.audit_entire_pipeline()

    print("\\n" + "=" * 60)
    print("📋 AUDIT SUMMARY")
    print("=" * 60)
    print(f"Files Scanned: {audit_results['files_scanned']}")
    print(f"Total Violations: {audit_results['total_violations']}")
    print(f"Critical Violations: {audit_results['critical_violations']}")

    if audit_results["critical_violations"] > 0:
        print("\\n🚨 CRITICAL ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
    else:
        print("\\n✅ NO CRITICAL TEMPORAL VIOLATIONS DETECTED")

    print("\\n📋 RECOMMENDATIONS:")
    for rec in audit_results["recommendations"]:
        print(rec)

    # Save audit report
    report_path = Path("logs/audit/temporal_integrity_audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"\\n📄 Full audit report saved: {report_path}")

    # Ask for fixes
    if audit_results["critical_violations"] > 0:
        print("\\n🔧 APPLYING AUTOMATIC FIXES...")
        fix_results = auditor.fix_common_violations()

        if fix_results["fixes_applied"]:
            print(f"✅ Applied fixes to {len(fix_results['fixes_applied'])} files")
            print("⚠️  Please review all changes and test thoroughly")
        else:
            print("ℹ️  No automatic fixes could be applied - manual review required")

    return audit_results["critical_violations"] == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
