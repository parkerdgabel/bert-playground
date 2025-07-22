"""Template management command for k-bert projects."""

from pathlib import Path
from typing import Optional
import shutil

import typer
from rich.console import Console
from rich.table import Table

from ...utils import handle_errors


console = Console()


@handle_errors
def template_command(
    action: str = typer.Argument(
        ...,
        help="Action to perform (list, create, info)",
    ),
    name: Optional[str] = typer.Argument(
        None,
        help="Template name (for create/info)",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        "-s",
        help="Source directory for create action",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description",
        "-d",
        help="Template description",
    ),
):
    """Manage k-bert project templates.
    
    This command allows you to list available templates, get information
    about templates, and create custom templates from existing projects.
    
    Actions:
    - list: Show all available templates
    - info: Show detailed information about a template
    - create: Create a new template from existing project
    
    Examples:
        # List all templates
        k-bert project template list
        
        # Get template info
        k-bert project template info advanced
        
        # Create custom template
        k-bert project template create my-template --source ./my-project
    """
    if action == "list":
        _list_templates()
    
    elif action == "info":
        if not name:
            console.print("[red]Template name required for info action.[/red]")
            raise typer.Exit(1)
        _show_template_info(name)
    
    elif action == "create":
        if not name:
            console.print("[red]Template name required for create action.[/red]")
            raise typer.Exit(1)
        if not source:
            console.print("[red]Source directory required for create action.[/red]")
            raise typer.Exit(1)
        _create_template(name, source, description)
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: list, info, create")
        raise typer.Exit(1)


def _list_templates() -> None:
    """List all available templates."""
    templates_dir = Path(__file__).parent.parent.parent / "templates"
    
    if not templates_dir.exists():
        console.print("[red]No templates directory found.[/red]")
        return
    
    # Built-in templates info
    template_info = {
        "base": {
            "description": "Basic project with example custom components",
            "features": ["Custom heads", "Augmenters", "Feature extractors"],
            "use_case": "General purpose, learning k-bert",
        },
        "advanced": {
            "description": "Advanced project with optimizations and best practices",
            "features": ["All base features", "Custom metrics", "Advanced pipelines"],
            "use_case": "Production projects, complex competitions",
        },
        "kaggle_starter": {
            "description": "Optimized starter for Kaggle competitions",
            "features": ["Competition utilities", "Submission helpers", "CV strategies"],
            "use_case": "Kaggle competitions",
        },
    }
    
    # Find all templates
    templates = []
    for template_dir in templates_dir.iterdir():
        if template_dir.is_dir() and not template_dir.name.startswith('.'):
            info = template_info.get(template_dir.name, {})
            templates.append({
                "name": template_dir.name,
                "path": template_dir,
                "description": info.get("description", "Custom template"),
                "features": info.get("features", []),
                "use_case": info.get("use_case", ""),
            })
    
    if not templates:
        console.print("[yellow]No templates found.[/yellow]")
        return
    
    # Create table
    table = Table(title="Available K-BERT Templates", show_header=True)
    table.add_column("Template", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Use Case", style="yellow")
    
    for template in sorted(templates, key=lambda x: x["name"]):
        table.add_row(
            template["name"],
            template["description"],
            template["use_case"],
        )
    
    console.print(table)
    
    console.print(
        "\n[dim]Use 'k-bert project template info <name>' for more details.[/dim]"
    )


def _show_template_info(template_name: str) -> None:
    """Show detailed information about a template."""
    templates_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = templates_dir / template_name
    
    if not template_path.exists():
        console.print(f"[red]Template '{template_name}' not found.[/red]")
        _list_templates()
        return
    
    # Template details
    details = {
        "base": {
            "description": "Basic k-bert project template with example custom components",
            "features": [
                "Custom BERT heads (binary and multiclass)",
                "Data augmentation strategies",
                "Feature extraction pipelines",
                "Example project configuration",
                "Plugin auto-discovery",
            ],
            "structure": [
                "src/heads/ - Custom BERT head implementations",
                "src/augmenters/ - Data augmentation strategies",
                "src/features/ - Feature extraction components",
                "src/models/ - Custom model architectures",
                "src/metrics/ - Custom evaluation metrics",
                "configs/ - Configuration files",
                "notebooks/ - Jupyter notebooks",
                "data/ - Data directories",
            ],
            "components": [
                "CustomBinaryHead - Binary classification with hidden layer",
                "CustomMulticlassHead - Multiclass with label smoothing",
                "DomainSpecificAugmenter - Text augmentation techniques",
                "TabularDataAugmenter - Structured data augmentation",
                "TextStatisticsExtractor - Extract text features",
                "TemporalFeatureExtractor - Time-based features",
            ],
        },
        "advanced": {
            "description": "Advanced template with production-ready features",
            "features": [
                "All base template features",
                "Advanced model architectures",
                "Custom training loops",
                "Ensemble strategies",
                "Hyperparameter optimization",
                "Model serving utilities",
            ],
            "structure": [
                "All base directories plus:",
                "src/ensemble/ - Model ensemble strategies",
                "src/optimization/ - Hyperparameter tuning",
                "src/serving/ - Model deployment utilities",
                "tests/ - Unit and integration tests",
            ],
            "components": [
                "All base components plus:",
                "EnsembleHead - Multi-model ensemble",
                "BayesianOptimizer - Hyperparameter search",
                "ModelServer - REST API for predictions",
            ],
        },
        "kaggle_starter": {
            "description": "Kaggle competition optimized starter template",
            "features": [
                "Competition-specific utilities",
                "Cross-validation strategies",
                "Submission generation",
                "Feature engineering helpers",
                "Stacking and blending",
            ],
            "structure": [
                "src/cv/ - Cross-validation strategies",
                "src/stacking/ - Model stacking utilities",
                "src/submission/ - Submission helpers",
                "scripts/ - Competition scripts",
            ],
            "components": [
                "KFoldTrainer - K-fold cross-validation",
                "SubmissionGenerator - Format predictions",
                "FeatureSelector - Automatic feature selection",
            ],
        },
    }
    
    info = details.get(template_name, {
        "description": "Custom template",
        "features": [],
        "structure": [],
        "components": [],
    })
    
    # Display info
    console.print(f"\n[bold cyan]{template_name.upper()} Template[/bold cyan]")
    console.print(f"\n{info['description']}\n")
    
    if info["features"]:
        console.print("[bold]Features:[/bold]")
        for feature in info["features"]:
            console.print(f"  • {feature}")
    
    if info["structure"]:
        console.print("\n[bold]Directory Structure:[/bold]")
        for item in info["structure"]:
            console.print(f"  • {item}")
    
    if info["components"]:
        console.print("\n[bold]Included Components:[/bold]")
        for component in info["components"]:
            console.print(f"  • {component}")
    
    # Show files
    console.print("\n[bold]Template Files:[/bold]")
    _show_template_files(template_path)
    
    console.print(
        f"\n[dim]Create project with: k-bert project init my-project --template {template_name}[/dim]"
    )


def _show_template_files(template_path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    """Show template file structure."""
    if current_depth >= max_depth:
        return
    
    items = sorted(template_path.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for item in items[:10]:  # Limit display
        if item.name.startswith('.'):
            continue
        
        if item.is_file():
            console.print(f"{prefix}├── {item.name}")
        else:
            console.print(f"{prefix}├── [bold]{item.name}/[/bold]")
            if current_depth < max_depth - 1:
                _show_template_files(item, prefix + "│   ", max_depth, current_depth + 1)


def _create_template(name: str, source: Path, description: Optional[str]) -> None:
    """Create a new template from existing project."""
    templates_dir = Path(__file__).parent.parent.parent / "templates"
    template_path = templates_dir / name
    
    if template_path.exists():
        console.print(f"[red]Template '{name}' already exists.[/red]")
        raise typer.Exit(1)
    
    if not source.exists():
        console.print(f"[red]Source directory not found: {source}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Creating template '{name}' from {source}...[/cyan]")
    
    # Create template directory
    template_path.mkdir(parents=True)
    
    # Copy project structure
    ignore_patterns = [
        "__pycache__",
        "*.pyc",
        ".git",
        ".venv",
        "venv",
        "outputs",
        "mlruns",
        "*.log",
        ".DS_Store",
        "*.egg-info",
        "dist",
        "build",
    ]
    
    def should_ignore(path: Path) -> bool:
        """Check if path should be ignored."""
        for pattern in ignore_patterns:
            if pattern.startswith("*"):
                if path.name.endswith(pattern[1:]):
                    return True
            elif pattern in path.parts:
                return True
        return False
    
    # Copy files
    for item in source.rglob("*"):
        if should_ignore(item):
            continue
        
        relative = item.relative_to(source)
        dest = template_path / relative
        
        if item.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
    
    # Create template metadata
    metadata = {
        "name": name,
        "description": description or f"Custom template created from {source.name}",
        "created_from": str(source),
        "files": len(list(template_path.rglob("*"))),
    }
    
    import json
    with open(template_path / ".template.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    console.print(f"[green]✓ Successfully created template '{name}'[/green]")
    console.print(f"\nTemplate location: [cyan]{template_path}[/cyan]")
    console.print(
        f"\n[dim]Use with: k-bert project init my-project --template {name}[/dim]"
    )