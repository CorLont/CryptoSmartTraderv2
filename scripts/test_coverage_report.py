#!/usr/bin/env python3
"""
Generate comprehensive test coverage reports with analysis
"""

import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


class CoverageAnalyzer:
    """Analyze and report on test coverage"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.coverage_threshold = 80.0
        self.critical_files = [
            "core/secrets_manager.py",
            "core/async_data_manager.py",
            "core/logging_manager.py",
            "agents/data_collector.py",
            "config/settings.py",
        ]

    def run_coverage_analysis(self) -> Dict:
        """Run coverage analysis and return results"""
        print("🔍 RUNNING COMPREHENSIVE COVERAGE ANALYSIS")
        print("=" * 60)

        # Run pytest with coverage
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=core",
            "--cov=agents",
            "--cov=config",
            "--cov-report=json:detailed_coverage.json",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "tests/unit/",
            "-v",
        ]

        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Coverage analysis failed:")
            print(result.stderr)
            return None

        # Load coverage data
        coverage_file = self.project_root / "detailed_coverage.json"
        if not coverage_file.exists():
            print("❌ Coverage report not generated")
            return None

        with open(coverage_file) as f:
            coverage_data = json.load(f)

        return coverage_data

    def analyze_file_coverage(self, coverage_data: Dict) -> List[Tuple[str, float, str]]:
        """Analyze coverage by file"""
        files_analysis = []

        for filename, file_data in coverage_data["files"].items():
            coverage_percent = file_data["summary"]["percent_covered"]

            # Determine status
            if coverage_percent >= self.coverage_threshold:
                status = "✅ GOOD"
            elif coverage_percent >= 60:
                status = "⚠️ NEEDS IMPROVEMENT"
            else:
                status = "❌ CRITICAL"

            files_analysis.append((filename, coverage_percent, status))

        return sorted(files_analysis, key=lambda x: x[1])

    def analyze_critical_files(self, coverage_data: Dict) -> List[Tuple[str, float, List[int]]]:
        """Analyze coverage of critical system files"""
        critical_analysis = []

        for critical_file in self.critical_files:
            if critical_file in coverage_data["files"]:
                file_data = coverage_data["files"][critical_file]
                coverage_percent = file_data["summary"]["percent_covered"]
                missing_lines = file_data["missing_lines"]

                critical_analysis.append((critical_file, coverage_percent, missing_lines))
            else:
                critical_analysis.append((critical_file, 0.0, []))

        return critical_analysis

    def find_untested_functions(self, coverage_data: Dict) -> Dict[str, List[int]]:
        """Find untested functions and methods"""
        untested_functions = {}

        for filename, file_data in coverage_data["files"].items():
            missing_lines = file_data["missing_lines"]

            if missing_lines:
                # Read the file to identify function definitions on missing lines
                try:
                    file_path = self.project_root / filename
                    if file_path.exists():
                        with open(file_path) as f:
                            lines = f.readlines()

                        untested_funcs = []
                        for line_num in missing_lines:
                            if line_num <= len(lines):
                                line = lines[line_num - 1].strip()
                                if line.startswith("def ") or line.startswith("async def "):
                                    func_name = (
                                        line.split("(")[0]
                                        .replace("def ", "")
                                        .replace("async ", "")
                                        .strip()
                                    untested_funcs.append(line_num)

                        if untested_funcs:
                            untested_functions[filename] = untested_funcs

                except Exception as e:
                    print(f"Warning: Could not analyze {filename}: {e}")

        return untested_functions

    def generate_recommendations(self, coverage_data: Dict) -> List[str]:
        """Generate actionable recommendations for improving coverage"""
        recommendations = []

        total_coverage = coverage_data["totals"]["percent_covered"]

        if total_coverage < self.coverage_threshold:
            gap = self.coverage_threshold - total_coverage
            recommendations.append(
                f"🎯 Need {gap:.1f}% more coverage to meet {self.coverage_threshold}% threshold"
            )

        # Analyze files with low coverage
        low_coverage_files = [
            (filename, data["summary"]["percent_covered"])
            for filename, data in coverage_data["files"].items()
            if data["summary"]["percent_covered"] < 60
        ]

        if low_coverage_files:
            recommendations.append("📝 Priority files for testing:")
            for filename, coverage in sorted(low_coverage_files, key=lambda x: x[1])[:5]:
                recommendations.append(f"   • {filename}: {coverage:.1f}% coverage")

        # Check critical files
        critical_low = [
            filename
            for filename in self.critical_files
            if filename in coverage_data["files"]
            and coverage_data["files"][filename]["summary"]["percent_covered"]
            < self.coverage_threshold
        ]

        if critical_low:
            recommendations.append("🔥 Critical files needing immediate attention:")
            for filename in critical_low:
                coverage = coverage_data["files"][filename]["summary"]["percent_covered"]
                recommendations.append(f"   • {filename}: {coverage:.1f}% coverage")

        # Test type recommendations
        if total_coverage < 90:
            recommendations.append("🧪 Consider adding these test types:")
            recommendations.append("   • Edge case testing for error conditions")
            recommendations.append("   • Integration tests for component interactions")
            recommendations.append("   • Contract tests for external API assumptions")

        return recommendations

    def print_coverage_report(self, coverage_data: Dict):
        """Print comprehensive coverage report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST COVERAGE REPORT")
        print("=" * 80)

        # Overall summary
        total_coverage = coverage_data["totals"]["percent_covered"]
        total_lines = coverage_data["totals"]["num_statements"]
        covered_lines = coverage_data["totals"]["covered_lines"]
        missing_lines = coverage_data["totals"]["missing_lines"]

        print(f"\n📈 OVERALL COVERAGE:")
        print(f"   Coverage: {total_coverage:.2f}%")
        print(f"   Total Lines: {total_lines}")
        print(f"   Covered Lines: {covered_lines}")
        print(f"   Missing Lines: {missing_lines}")

        if total_coverage >= self.coverage_threshold:
            print(f"   Status: ✅ MEETS THRESHOLD ({self.coverage_threshold}%)")
        else:
            print(f"   Status: ❌ BELOW THRESHOLD ({self.coverage_threshold}%)")

        # File-by-file analysis
        print(f"\n📁 FILE COVERAGE ANALYSIS:")
        files_analysis = self.analyze_file_coverage(coverage_data)

        for filename, coverage, status in files_analysis:
            print(f"   {status} {filename}: {coverage:.1f}%")

        # Critical files analysis
        print(f"\n🔥 CRITICAL FILES ANALYSIS:")
        critical_analysis = self.analyze_critical_files(coverage_data)

        for filename, coverage, missing_lines in critical_analysis:
            if coverage >= self.coverage_threshold:
                status = "✅"
            elif coverage >= 60:
                status = "⚠️"
            else:
                status = "❌"

            print(f"   {status} {filename}: {coverage:.1f}%")
            if missing_lines and len(missing_lines) <= 10:
                print(f"      Missing lines: {missing_lines}")
            elif missing_lines:
                print(f"      Missing lines: {len(missing_lines)} lines total")

        # Untested functions
        print(f"\n🔍 UNTESTED FUNCTIONS:")
        untested = self.find_untested_functions(coverage_data)

        if untested:
            for filename, line_numbers in list(untested.items())[:5]:  # Top 5 files
                print(f"   📄 {filename}:")
                for line_num in line_numbers[:3]:  # Top 3 functions per file
                    print(f"      Line {line_num}")
        else:
            print("   ✅ All major functions appear to have some test coverage")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = self.generate_recommendations(coverage_data)

        for rec in recommendations:
            print(f"   {rec}")

        # Coverage gate status
        print(f"\n🚪 COVERAGE GATE:")
        if total_coverage >= self.coverage_threshold:
            print(
                f"   ✅ PASSED - Coverage {total_coverage:.2f}% meets threshold {self.coverage_threshold}%"
            )
        else:
            print(
                f"   ❌ FAILED - Coverage {total_coverage:.2f}% below threshold {self.coverage_threshold}%"
            )
            print(f"   Need {self.coverage_threshold - total_coverage:.2f}% more coverage")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Coverage Analysis Tool")
    parser.add_argument(
        "--threshold", type=float, default=80.0, help="Coverage threshold percentage"
    )

    args = parser.parse_args()

    analyzer = CoverageAnalyzer()
    analyzer.coverage_threshold = args.threshold

    # Run analysis
    coverage_data = analyzer.run_coverage_analysis()

    if coverage_data:
        analyzer.print_coverage_report(coverage_data)

        # Exit with appropriate code
        total_coverage = coverage_data["totals"]["percent_covered"]
        sys.exit(0 if total_coverage >= analyzer.coverage_threshold else 1)
    else:
        print("❌ Coverage analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
