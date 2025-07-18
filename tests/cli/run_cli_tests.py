#!/usr/bin/env python3
"""Run all CLI tests with detailed reporting."""

import subprocess
import sys
from pathlib import Path
import time
from typing import List, Dict, Tuple
import json


class CLITestRunner:
    """Runner for CLI unit and integration tests."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent
        self.results = {}
        
    def run_test_file(self, test_file: Path) -> Tuple[bool, str, float]:
        """Run a single test file and return results."""
        print(f"\n🧪 Running {test_file.name}...")
        
        start_time = time.time()
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--no-header"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.project_root)
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        
        return success, result.stdout + result.stderr, duration
    
    def parse_test_output(self, output: str) -> Dict[str, any]:
        """Parse pytest output for test statistics."""
        stats = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'total': 0
        }
        
        # Look for pytest summary line
        for line in output.split('\n'):
            if 'passed' in line or 'failed' in line:
                if 'passed' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'passed' in part and i > 0:
                            try:
                                stats['passed'] = int(parts[i-1])
                            except ValueError:
                                pass
                if 'failed' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'failed' in part and i > 0:
                            try:
                                stats['failed'] = int(parts[i-1])
                            except ValueError:
                                pass
        
        stats['total'] = stats['passed'] + stats['failed'] + stats['skipped']
        return stats
    
    def run_all_tests(self) -> bool:
        """Run all CLI tests and generate report."""
        print("🚀 MLX BERT CLI Test Suite")
        print("=" * 50)
        
        # Find all test files
        test_files = sorted(self.test_dir.glob("test_*.py"))
        if not test_files:
            print("❌ No test files found!")
            return False
        
        print(f"Found {len(test_files)} test files")
        
        all_passed = True
        total_duration = 0
        
        # Run each test file
        for test_file in test_files:
            success, output, duration = self.run_test_file(test_file)
            total_duration += duration
            
            stats = self.parse_test_output(output)
            self.results[test_file.name] = {
                'success': success,
                'duration': duration,
                'stats': stats,
                'output': output if not success else None
            }
            
            if success:
                print(f"✅ {test_file.name} - {stats['passed']} tests passed ({duration:.2f}s)")
            else:
                print(f"❌ {test_file.name} - {stats['failed']} failed, {stats['passed']} passed ({duration:.2f}s)")
                all_passed = False
                
                # Show failures
                if stats['failed'] > 0:
                    print("\n  Failed tests:")
                    for line in output.split('\n'):
                        if 'FAILED' in line:
                            print(f"    - {line.strip()}")
        
        # Generate summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        
        total_tests = sum(r['stats']['total'] for r in self.results.values())
        total_passed = sum(r['stats']['passed'] for r in self.results.values())
        total_failed = sum(r['stats']['failed'] for r in self.results.values())
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed} ✅")
        print(f"Failed: {total_failed} ❌")
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Show failed test files
        if not all_passed:
            print("\n❌ Failed Test Files:")
            for file_name, result in self.results.items():
                if not result['success']:
                    print(f"  - {file_name}")
        
        # Save detailed report
        self.save_report()
        
        return all_passed
    
    def save_report(self):
        """Save detailed test report."""
        report_path = self.test_dir / "test_report.json"
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_files': len(self.results),
                'passed_files': sum(1 for r in self.results.values() if r['success']),
                'total_tests': sum(r['stats']['total'] for r in self.results.values()),
                'passed_tests': sum(r['stats']['passed'] for r in self.results.values()),
                'failed_tests': sum(r['stats']['failed'] for r in self.results.values())
            },
            'details': self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_path}")
    
    def run_specific_tests(self, pattern: str) -> bool:
        """Run tests matching a specific pattern."""
        print(f"🔍 Running tests matching pattern: {pattern}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "-k", pattern,
            "-v"
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        return result.returncode == 0
    
    def run_with_coverage(self) -> bool:
        """Run tests with coverage reporting."""
        print("📊 Running tests with coverage...")
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov=cli",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ]
        
        result = subprocess.run(cmd, cwd=str(self.project_root))
        
        if result.returncode == 0:
            print("\n✅ Coverage report generated in htmlcov/")
        
        return result.returncode == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CLI tests")
    parser.add_argument(
        "--pattern", "-k",
        help="Run only tests matching this pattern"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--file", "-f",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    runner = CLITestRunner()
    
    if args.pattern:
        success = runner.run_specific_tests(args.pattern)
    elif args.coverage:
        success = runner.run_with_coverage()
    elif args.file:
        test_file = Path(args.file)
        if not test_file.exists():
            test_file = runner.test_dir / args.file
        success, output, duration = runner.run_test_file(test_file)
        print(output)
    else:
        success = runner.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()