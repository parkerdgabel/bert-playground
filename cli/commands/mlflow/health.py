"""MLflow health check command."""

import typer
from pathlib import Path
import sys
from typing import Optional

from ...utils import (
    get_console, print_success, print_error, print_warning, print_info,
    handle_errors
)
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
def health_command(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed health information"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix common issues"),
    export: Optional[Path] = typer.Option(None, "--export", "-e", help="Export report to file"),
):
    """Check MLflow health and configuration.
    
    Performs comprehensive health checks on your MLflow setup including:
    - Server connectivity
    - Database integrity
    - Artifact storage
    - Configuration validation
    - Environment compatibility
    
    Examples:
        # Basic health check
        bert mlflow health
        
        # Detailed check with fix attempts
        bert mlflow health --detailed --fix
        
        # Export health report
        bert mlflow health --export health_report.json
    """
    console = get_console()
    
    console.print("\n[bold blue]MLflow Health Check[/bold blue]")
    console.print("=" * 60)
    
    try:
        from utils.mlflow_health import MLflowHealthChecker
    except ImportError:
        print_error(
            "Failed to import MLflow health checker. Make sure mlflow is installed:\n"
            "pip install mlflow",
            title="Import Error"
        )
        raise typer.Exit(1)
    
    try:
        health_checker = MLflowHealthChecker()
        
        # Run health checks
        with console.status("[yellow]Running health checks...[/yellow]"):
            results = health_checker.run_full_check()  # run_full_check doesn't accept parameters
        
        # Display results
        _display_health_results(console, results)
        
        # Attempt fixes if requested
        if fix and results.get("issues"):
            console.print("\n[yellow]Attempting to fix issues...[/yellow]")
            
            for issue in results.get("issues", []):
                if issue.get("fixable"):
                    fix_result = health_checker.fix_issue(issue["id"])
                    if fix_result["success"]:
                        print_success(f"Fixed: {issue['description']}")
                    else:
                        print_warning(f"Could not fix: {issue['description']} - {fix_result.get('error', 'Unknown error')}")
        
        # Export if requested
        if export:
            import json
            with open(export, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print_info(f"Health report exported to: {export}")
        
        # Calculate overall status based on check results
        passed_checks = sum(1 for r in results.values() if r.get("status") == "PASS")
        failed_checks = sum(1 for r in results.values() if r.get("status") == "FAIL")
        total_checks = len(results)
        
        console.print(f"\n[bold]Summary:[/bold] {passed_checks}/{total_checks} checks passed")
        
        if passed_checks == total_checks:
            print_success("MLflow is healthy and ready to use!")
        elif failed_checks > 0:
            critical_checks = ["database_connectivity", "configuration_validity"]
            has_critical_failure = any(
                results.get(check, {}).get("status") == "FAIL" 
                for check in critical_checks
            )
            
            if has_critical_failure:
                print_error("MLflow has critical issues that need attention.")
                raise typer.Exit(1)
            else:
                print_warning("MLflow has some issues but is functional.")
        else:
            print_info("MLflow health check completed.")
        
    except Exception as e:
        print_error(f"Health check failed: {str(e)}", title="Health Check Error")
        raise typer.Exit(1)


def _display_health_results(console, results):
    """Display health check results in a formatted way."""
    
    # Basic info
    info_table = create_table("MLflow Health Check Results", ["Check", "Status", "Details"])
    
    # Map check names to user-friendly names
    check_name_map = {
        "database_connectivity": "Database Connectivity",
        "directory_permissions": "Directory Permissions", 
        "configuration_validity": "Configuration Validity",
        "experiment_creation": "Experiment Creation",
        "metric_logging": "Metric Logging",
        "artifact_logging": "Artifact Logging",
        "run_management": "Run Management",
        "performance": "Performance",
        "cleanup": "Cleanup Capabilities"
    }
    
    # Display each check result
    for check_name, result in results.items():
        friendly_name = check_name_map.get(check_name, check_name.replace("_", " ").title())
        status = result.get("status", "UNKNOWN")
        message = result.get("message", "No details available")
        
        info_table.add_row(
            friendly_name,
            _format_check_status(status),
            message[:80] + "..." if len(message) > 80 else message
        )
    
    console.print(info_table)
    
    # Show failed checks with suggestions
    failed_checks = [(name, result) for name, result in results.items() 
                     if result.get("status") == "FAIL" and result.get("suggestions")]
    
    if failed_checks:
        console.print(f"\n[yellow]Found {len(failed_checks)} failed checks with suggestions:[/yellow]")
        
        for check_name, result in failed_checks:
            friendly_name = check_name_map.get(check_name, check_name.replace("_", " ").title())
            console.print(f"\n[bold red]{friendly_name}:[/bold red]")
            
            suggestions = result.get("suggestions", [])
            for suggestion in suggestions:
                console.print(f"  • {suggestion}")


def _format_status(status):
    """Format status with color."""
    if status == "healthy":
        return "[green]✓ Healthy[/green]"
    elif status == "warning":
        return "[yellow]⚠ Warning[/yellow]"
    elif status == "error":
        return "[red]✗ Error[/red]"
    else:
        return status or "Unknown"


def _format_check_status(status):
    """Format check status (PASS/FAIL) with color."""
    if status == "PASS":
        return "[green]✓ PASS[/green]"
    elif status == "FAIL":
        return "[red]✗ FAIL[/red]"
    else:
        return f"[yellow]? {status}[/yellow]"