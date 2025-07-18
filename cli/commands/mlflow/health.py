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
            results = health_checker.run_full_check(detailed=detailed)
        
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
        
        # Overall status
        overall_status = results.get("overall_status", "unknown")
        if overall_status == "healthy":
            print_success("MLflow is healthy and ready to use!")
        elif overall_status == "warning":
            print_warning("MLflow has some issues but is functional.")
        else:
            print_error("MLflow has critical issues that need attention.")
            raise typer.Exit(1)
        
    except Exception as e:
        print_error(f"Health check failed: {str(e)}", title="Health Check Error")
        raise typer.Exit(1)


def _display_health_results(console, results):
    """Display health check results in a formatted way."""
    
    # Basic info
    info_table = create_table("MLflow Configuration", ["Component", "Status", "Details"])
    
    # Server status
    server_status = results.get("server", {})
    info_table.add_row(
        "Tracking Server",
        _format_status(server_status.get("status")),
        server_status.get("uri", "Not configured")
    )
    
    # Database status
    db_status = results.get("database", {})
    info_table.add_row(
        "Backend Store",
        _format_status(db_status.get("status")),
        db_status.get("type", "Unknown")
    )
    
    # Artifact store
    artifact_status = results.get("artifacts", {})
    info_table.add_row(
        "Artifact Store",
        _format_status(artifact_status.get("status")),
        artifact_status.get("uri", "Not configured")
    )
    
    # Environment
    env_status = results.get("environment", {})
    info_table.add_row(
        "Environment",
        _format_status(env_status.get("status")),
        f"Python {env_status.get('python_version', 'Unknown')}, MLflow {env_status.get('mlflow_version', 'Unknown')}"
    )
    
    console.print(info_table)
    
    # Issues if any
    issues = results.get("issues", [])
    if issues:
        console.print(f"\n[yellow]Found {len(issues)} issues:[/yellow]")
        
        issues_table = create_table("Issues", ["Severity", "Component", "Description", "Fixable"])
        
        for issue in issues:
            severity = issue.get("severity", "unknown")
            if severity == "critical":
                severity_str = f"[red]{severity}[/red]"
            elif severity == "warning":
                severity_str = f"[yellow]{severity}[/yellow]"
            else:
                severity_str = severity
            
            issues_table.add_row(
                severity_str,
                issue.get("component", "Unknown"),
                issue.get("description", "No description"),
                "Yes" if issue.get("fixable") else "No"
            )
        
        console.print(issues_table)
    
    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")


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