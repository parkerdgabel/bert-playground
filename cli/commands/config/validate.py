"""Validate configuration command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from ...config import (
    ConfigManager,
    validate_config,
    validate_competition_config,
    ConfigValidationError,
    COMPETITION_PROFILES,
)
from ...utils import handle_errors


console = Console()


@handle_errors
def validate_command(
    config_file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Specific configuration file to validate",
    ),
    competition: Optional[str] = typer.Option(
        None,
        "--competition",
        "-c",
        help="Validate for specific competition",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix common issues",
    ),
):
    """Validate configuration.
    
    This command validates the current configuration, checking for:
    - Valid paths and directories
    - Correct value types and ranges
    - Required fields
    - Competition-specific requirements
    
    Examples:
        # Validate current configuration
        k-bert config validate
        
        # Validate specific file
        k-bert config validate --file k-bert.yaml
        
        # Validate for specific competition
        k-bert config validate --competition titanic
        
        # Attempt to fix issues
        k-bert config validate --fix
    """
    manager = ConfigManager()
    
    try:
        # Load configuration
        if config_file:
            # Validate specific file
            console.print(f"Validating configuration file: [cyan]{config_file}[/cyan]")
            
            if not config_file.exists():
                console.print(f"[red]Configuration file not found: {config_file}[/red]")
                raise typer.Exit(1)
            
            # Load and validate
            from ...config.schemas import KBertConfig, ProjectConfig
            import yaml
            import json
            
            with open(config_file) as f:
                if config_file.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    data = json.load(f)
                else:
                    console.print("[red]Unsupported file format. Use .yaml, .yml, or .json[/red]")
                    raise typer.Exit(1)
            
            # Try to parse as project or user config
            if 'name' in data:
                config = ProjectConfig(**data)
                config_type = "project"
            else:
                config = KBertConfig.from_dict(data)
                config_type = "user"
        else:
            # Validate merged configuration
            config = manager.get_merged_config(validate=False)
            config_type = "merged"
        
        # Run validation
        validate_config(config)
        
        # Additional competition validation if specified
        if competition:
            if competition not in COMPETITION_PROFILES:
                console.print(
                    f"[yellow]Unknown competition: {competition}[/yellow]\n"
                    f"Known competitions: {', '.join(COMPETITION_PROFILES.keys())}"
                )
            else:
                comp_config = COMPETITION_PROFILES[competition]
                errors = validate_competition_config(comp_config)
                if errors:
                    raise ConfigValidationError(errors)
        
        # Success!
        console.print(
            Panel(
                f"[green]✓ Configuration is valid![/green]\n\n"
                f"Type: {config_type} configuration",
                title="Validation Passed",
                style="green",
            )
        )
        
        # Show warnings if any
        warnings = _check_warnings(config)
        if warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  [dim]•[/dim] {warning}")
    
    except ConfigValidationError as e:
        console.print(
            Panel(
                "[red]✗ Configuration validation failed![/red]",
                title="Validation Failed",
                style="red",
            )
        )
        console.print("\n[red]Errors found:[/red]")
        for error in e.errors:
            console.print(f"  [red]•[/red] {error}")
        
        if fix:
            console.print("\n[yellow]Attempting to fix issues...[/yellow]")
            fixed_count = _attempt_fixes(config, e.errors)
            
            if fixed_count > 0:
                console.print(f"\n[green]Fixed {fixed_count} issue(s).[/green]")
                console.print("Please run validation again to check remaining issues.")
            else:
                console.print("\n[red]No issues could be automatically fixed.[/red]")
        else:
            console.print("\n[dim]Tip: Use --fix to attempt automatic fixes[/dim]")
        
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


def _check_warnings(config) -> list[str]:
    """Check for configuration warnings (non-fatal issues)."""
    warnings = []
    
    # Check if Kaggle credentials are set
    if hasattr(config, 'kaggle'):
        if not config.kaggle.username or not config.kaggle.key:
            warnings.append(
                "Kaggle credentials not set. You won't be able to download "
                "competition data or submit predictions."
            )
    
    # Check if MLflow is disabled
    if hasattr(config, 'mlflow') and not config.mlflow.auto_log:
        warnings.append(
            "MLflow tracking is disabled. Consider enabling it for "
            "experiment tracking."
        )
    
    # Check batch size for MLX performance
    if hasattr(config, 'training') and config.training.default_batch_size < 16:
        warnings.append(
            f"Batch size {config.training.default_batch_size} is small. "
            "MLX performs better with batch sizes >= 16."
        )
    
    # Check if compilation is disabled
    if hasattr(config, 'training') and not config.training.use_compilation:
        warnings.append(
            "MLX compilation is disabled. This may result in slower training."
        )
    
    return warnings


def _attempt_fixes(config, errors: list[str]) -> int:
    """Attempt to fix common configuration issues."""
    fixed_count = 0
    
    for error in errors:
        # Create missing directories
        if "directory does not exist" in error or "is not a directory" in error:
            # Extract path from error message
            import re
            match = re.search(r'(?:directory does not exist|is not a directory): (.+)', error)
            if match:
                path = Path(match.group(1))
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    console.print(f"  [green]✓[/green] Created directory: {path}")
                    fixed_count += 1
                except Exception as e:
                    console.print(f"  [red]✗[/red] Failed to create {path}: {e}")
        
        # Fix invalid numeric values
        elif "must be positive" in error or "must be at least" in error:
            # Would need to know which field to fix - skip for now
            pass
    
    return fixed_count