"""Comprehensive test runner for MLflow validation.

This script runs all MLflow-related tests and provides a detailed report
of the MLflow integration status.
"""

import sys
import unittest
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.mlflow_health import MLflowHealthChecker
from utils.mlflow_central import MLflowCentral


class MLflowTestRunner:
    """Comprehensive MLflow test runner."""
    
    def __init__(self):
        """Initialize test runner."""
        self.console = Console()
        self.results = {}
        self.temp_dir = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all MLflow tests and return results."""
        self.console.print("\n[bold blue]MLflow Comprehensive Test Suite[/bold blue]")
        self.console.print("=" * 60)
        
        # Setup temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        try:
            # Run different test categories
            test_categories = [
                ("Health Check", self._run_health_check),
                ("Unit Tests", self._run_unit_tests),
                ("Integration Tests", self._run_integration_tests),
                ("Performance Tests", self._run_performance_tests),
                ("Configuration Tests", self._run_configuration_tests),
            ]
            
            for category_name, test_func in test_categories:
                with self.console.status(f"Running {category_name}..."):
                    self.results[category_name] = test_func()
            
            # Generate summary report
            self._generate_summary_report()
            
            return self.results
            
        finally:
            # Cleanup
            if self.temp_dir:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _run_health_check(self) -> Dict[str, Any]:
        """Run MLflow health check."""
        try:
            health_checker = MLflowHealthChecker()
            health_results = health_checker.run_full_check()
            
            passed = sum(1 for r in health_results.values() if r["status"] == "PASS")
            total = len(health_results)
            
            return {
                "status": "PASS" if passed == total else "FAIL",
                "passed": passed,
                "total": total,
                "details": health_results,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time()
            }
    
    def _run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests."""
        try:
            # Import test modules
            from tests.unit.test_mlflow_integration import (
                TestMLflowCentral,
                TestMLflowHealthChecker,
                TestMLflowTracker,
                TestTrainingMetrics
            )
            
            # Create test suite
            suite = unittest.TestSuite()
            
            # Add test classes
            test_classes = [
                TestMLflowCentral,
                TestMLflowHealthChecker,
                TestMLflowTracker,
                TestTrainingMetrics
            ]
            
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
            result = runner.run(suite)
            
            return {
                "status": "PASS" if result.wasSuccessful() else "FAIL",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time()
            }
    
    def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        try:
            # Import integration test modules
            from tests.integration.test_mlflow_training import (
                TestMLflowTrainingIntegration,
                TestMLflowConfigurationIntegration
            )
            
            # Create test suite
            suite = unittest.TestSuite()
            
            # Add test classes
            test_classes = [
                TestMLflowTrainingIntegration,
                TestMLflowConfigurationIntegration
            ]
            
            for test_class in test_classes:
                tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
                suite.addTests(tests)
            
            # Run tests
            runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
            result = runner.run(suite)
            
            return {
                "status": "PASS" if result.wasSuccessful() else "FAIL",
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time()
            }
    
    def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        try:
            # Reset singleton
            MLflowCentral._instance = None
            
            # Performance test 1: MLflow initialization time
            start_time = time.time()
            central = MLflowCentral()
            central.initialize(
                tracking_uri=f"sqlite:///{self.temp_dir}/perf_test.db",
                artifact_root=f"{self.temp_dir}/artifacts"
            )
            init_time = time.time() - start_time
            
            # Performance test 2: Metric logging throughput
            import mlflow
            
            start_time = time.time()
            with mlflow.start_run():
                for i in range(100):
                    mlflow.log_metric("test_metric", i * 0.01, step=i)
            logging_time = time.time() - start_time
            
            # Performance test 3: Experiment creation time
            start_time = time.time()
            for i in range(10):
                exp_id = central.get_experiment_id(f"perf_test_{i}")
            experiment_time = time.time() - start_time
            
            # Evaluate performance
            performance_issues = []
            
            if init_time > 5.0:
                performance_issues.append(f"Initialization too slow: {init_time:.2f}s")
            
            if logging_time > 10.0:
                performance_issues.append(f"Metric logging too slow: {logging_time:.2f}s")
            
            if experiment_time > 5.0:
                performance_issues.append(f"Experiment creation too slow: {experiment_time:.2f}s")
            
            return {
                "status": "PASS" if not performance_issues else "FAIL",
                "initialization_time": init_time,
                "logging_time": logging_time,
                "experiment_time": experiment_time,
                "issues": performance_issues,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time()
            }
    
    def _run_configuration_tests(self) -> Dict[str, Any]:
        """Run configuration validation tests."""
        try:
            # Reset singleton
            MLflowCentral._instance = None
            
            test_results = {}
            
            # Test 1: Valid configuration
            try:
                central = MLflowCentral()
                central.initialize(
                    tracking_uri=f"sqlite:///{self.temp_dir}/config_test.db",
                    artifact_root=f"{self.temp_dir}/artifacts"
                )
                test_results["valid_config"] = "PASS"
            except Exception as e:
                test_results["valid_config"] = f"FAIL: {str(e)}"
            
            # Test 2: Invalid tracking URI
            try:
                MLflowCentral._instance = None
                central = MLflowCentral()
                central.initialize(tracking_uri="")
                test_results["invalid_uri"] = "FAIL: Should have raised error"
            except Exception:
                test_results["invalid_uri"] = "PASS"
            
            # Test 3: Invalid artifact root
            try:
                MLflowCentral._instance = None
                central = MLflowCentral()
                central.initialize(
                    tracking_uri=f"sqlite:///{self.temp_dir}/test.db",
                    artifact_root=""
                )
                test_results["invalid_artifact_root"] = "FAIL: Should have raised error"
            except Exception:
                test_results["invalid_artifact_root"] = "PASS"
            
            # Test 4: Connection validation
            try:
                MLflowCentral._instance = None
                central = MLflowCentral()
                central.initialize(
                    tracking_uri=f"sqlite:///{self.temp_dir}/validation_test.db",
                    artifact_root=f"{self.temp_dir}/artifacts"
                )
                status = central.validate_connection()
                test_results["connection_validation"] = (
                    "PASS" if status["status"] == "CONNECTED" else f"FAIL: {status['message']}"
                )
            except Exception as e:
                test_results["connection_validation"] = f"FAIL: {str(e)}"
            
            passed = sum(1 for result in test_results.values() if result == "PASS")
            total = len(test_results)
            
            return {
                "status": "PASS" if passed == total else "FAIL",
                "passed": passed,
                "total": total,
                "test_results": test_results,
                "execution_time": time.time()
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "execution_time": time.time()
            }
    
    def _generate_summary_report(self) -> None:
        """Generate comprehensive summary report."""
        self.console.print("\n[bold green]Test Results Summary[/bold green]")
        self.console.print("=" * 60)
        
        # Create summary table
        table = Table(title="MLflow Test Results")
        table.add_column("Test Category", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details", style="dim")
        table.add_column("Issues", style="red")
        
        overall_status = "PASS"
        total_issues = []
        
        for category, result in self.results.items():
            status = result.get("status", "UNKNOWN")
            
            if status == "PASS":
                status_display = "[green]✓ PASS[/green]"
            elif status == "FAIL":
                status_display = "[red]✗ FAIL[/red]"
                overall_status = "FAIL"
            else:
                status_display = "[yellow]! ERROR[/yellow]"
                overall_status = "ERROR"
            
            # Format details
            details = []
            if "passed" in result and "total" in result:
                details.append(f"{result['passed']}/{result['total']} passed")
            if "tests_run" in result:
                details.append(f"{result['tests_run']} tests run")
            if "failures" in result and result["failures"] > 0:
                details.append(f"{result['failures']} failures")
            if "errors" in result and result["errors"] > 0:
                details.append(f"{result['errors']} errors")
            
            details_str = ", ".join(details) if details else "N/A"
            
            # Format issues
            issues = []
            if "error" in result:
                issues.append(result["error"])
            if "issues" in result:
                issues.extend(result["issues"])
            
            issues_str = "; ".join(issues) if issues else "None"
            if issues:
                total_issues.extend(issues)
            
            table.add_row(category, status_display, details_str, issues_str)
        
        self.console.print(table)
        
        # Overall status
        if overall_status == "PASS":
            self.console.print(f"\n[bold green]✓ Overall Status: PASS[/bold green]")
            self.console.print("[green]All MLflow tests passed successfully![/green]")
        else:
            self.console.print(f"\n[bold red]✗ Overall Status: {overall_status}[/bold red]")
            if total_issues:
                self.console.print("\n[bold red]Issues found:[/bold red]")
                for i, issue in enumerate(total_issues, 1):
                    self.console.print(f"  {i}. {issue}")
        
        # Recommendations
        self._generate_recommendations()
    
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on test results."""
        self.console.print("\n[bold yellow]Recommendations[/bold yellow]")
        self.console.print("-" * 60)
        
        recommendations = []
        
        # Check health check results
        if "Health Check" in self.results:
            health_result = self.results["Health Check"]
            if health_result.get("status") == "FAIL":
                recommendations.append(
                    "Run 'uv run python mlx_bert_cli.py mlflow-health' to diagnose MLflow issues"
                )
        
        # Check unit test results
        if "Unit Tests" in self.results:
            unit_result = self.results["Unit Tests"]
            if unit_result.get("failures", 0) > 0 or unit_result.get("errors", 0) > 0:
                recommendations.append(
                    "Fix unit test failures before running training"
                )
        
        # Check integration test results
        if "Integration Tests" in self.results:
            integration_result = self.results["Integration Tests"]
            if integration_result.get("failures", 0) > 0 or integration_result.get("errors", 0) > 0:
                recommendations.append(
                    "Fix integration test failures to ensure training pipeline works"
                )
        
        # Check performance results
        if "Performance Tests" in self.results:
            perf_result = self.results["Performance Tests"]
            if perf_result.get("issues"):
                recommendations.append(
                    "Address performance issues to improve training efficiency"
                )
        
        # Check configuration results
        if "Configuration Tests" in self.results:
            config_result = self.results["Configuration Tests"]
            if config_result.get("status") == "FAIL":
                recommendations.append(
                    "Fix configuration issues before running training"
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests passed! MLflow is ready for training.")
        else:
            recommendations.append(
                "After fixing issues, run this test suite again to verify fixes"
            )
        
        for i, rec in enumerate(recommendations, 1):
            self.console.print(f"  {i}. {rec}")
    
    def save_report(self, output_path: str) -> None:
        """Save detailed test report to file."""
        import json
        
        # Prepare report data
        report_data = {
            "timestamp": time.time(),
            "overall_status": "PASS" if all(
                r.get("status") == "PASS" for r in self.results.values()
            ) else "FAIL",
            "test_results": self.results,
            "summary": {
                "total_categories": len(self.results),
                "passed_categories": sum(
                    1 for r in self.results.values() if r.get("status") == "PASS"
                ),
                "failed_categories": sum(
                    1 for r in self.results.values() if r.get("status") != "PASS"
                )
            }
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.console.print(f"\n[green]Detailed report saved to: {output_path}[/green]")


def main():
    """Main function to run MLflow test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive MLflow test suite"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for detailed report",
        default="mlflow_test_report.json"
    )
    parser.add_argument(
        "--category",
        "-c",
        help="Run specific test category",
        choices=["health", "unit", "integration", "performance", "configuration"]
    )
    
    args = parser.parse_args()
    
    # Run tests
    runner = MLflowTestRunner()
    
    if args.category:
        # Run specific category
        category_map = {
            "health": runner._run_health_check,
            "unit": runner._run_unit_tests,
            "integration": runner._run_integration_tests,
            "performance": runner._run_performance_tests,
            "configuration": runner._run_configuration_tests
        }
        
        if args.category in category_map:
            runner.console.print(f"\n[bold blue]Running {args.category} tests only[/bold blue]")
            result = category_map[args.category]()
            runner.results[args.category] = result
            runner._generate_summary_report()
        else:
            runner.console.print(f"[red]Unknown category: {args.category}[/red]")
            return 1
    else:
        # Run all tests
        runner.run_all_tests()
    
    # Save report
    runner.save_report(args.output)
    
    # Return appropriate exit code
    overall_status = all(r.get("status") == "PASS" for r in runner.results.values())
    return 0 if overall_status else 1


if __name__ == "__main__":
    sys.exit(main())