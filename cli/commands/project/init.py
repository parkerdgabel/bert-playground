"""Initialize k-bert project command."""

from pathlib import Path
from typing import Optional
import shutil

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ...utils import handle_errors


console = Console()


@handle_errors
def init_command(
    name: str = typer.Argument(
        "my-k-bert-project",
        help="Project name",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Project directory (defaults to ./{name})",
    ),
    template: str = typer.Option(
        "base",
        "--template",
        "-t",
        help="Project template (base, advanced, kaggle_starter)",
    ),
    competition: Optional[str] = typer.Option(
        None,
        "--competition",
        "-c",
        help="Initialize for specific competition",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing project",
    ),
):
    """Initialize a new k-bert project with custom components.
    
    This command creates a project structure that seamlessly integrates
    with k-bert's models, data loaders, and trainers. The project includes
    example custom components that you can modify for your use case.
    
    Templates:
    - base: Basic project with example components
    - advanced: Advanced features and optimizations
    - kaggle_starter: Optimized for Kaggle competitions
    
    Examples:
        # Create basic project
        k-bert project init my-project
        
        # Create project with specific template
        k-bert project init my-nlp --template advanced
        
        # Create project for competition
        k-bert project init titanic --competition titanic
    """
    # Determine project path
    if path is None:
        path = Path(f"./{name}")
    
    # Check if exists
    if path.exists() and not force:
        if path.is_file():
            console.print(f"[red]Path exists and is a file: {path}[/red]")
            raise typer.Exit(1)
        
        if any(path.iterdir()):
            console.print(f"[yellow]Directory {path} already exists and is not empty.[/yellow]")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(0)
    
    # Get template path
    template_path = Path(__file__).parent.parent.parent / "templates" / template
    
    if not template_path.exists():
        console.print(f"[red]Template '{template}' not found.[/red]")
        console.print("Available templates: base, advanced, kaggle_starter")
        raise typer.Exit(1)
    
    # Create project directory
    path.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Creating k-bert project '{name}' from template '{template}'...[/cyan]")
    
    # Copy template files
    for item in template_path.iterdir():
        if item.name.startswith('.'):
            continue
        
        dest = path / item.name
        
        if item.is_file():
            shutil.copy2(item, dest)
        else:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
    
    # Customize for competition if specified
    if competition:
        _customize_for_competition(path, competition)
    
    # Update project configuration
    config_path = path / "k-bert.yaml"
    if config_path.exists():
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Update project name and competition
        config["name"] = name
        if competition:
            config["competition"] = competition
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Create additional directories
    additional_dirs = [
        "data/raw",
        "data/processed", 
        "data/submissions",
        "notebooks",
        "outputs",
        "configs/models",
        "configs/training",
        "configs/experiments",
    ]
    
    for dir_name in additional_dirs:
        (path / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Show summary
    console.print(f"\n[green]✓ Successfully created k-bert project![/green]")
    console.print(f"\nProject location: [cyan]{path.absolute()}[/cyan]")
    
    # Show project structure
    console.print("\n[bold]Project structure:[/bold]")
    _show_tree(path, max_depth=3)
    
    # Next steps
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"1. Navigate to project: [cyan]cd {path}[/cyan]")
    
    if competition:
        console.print(f"2. Download data: [cyan]k-bert competition download {competition} --path data/raw[/cyan]")
    
    console.print("3. Customize components in src/")
    console.print("4. Configure training in k-bert.yaml")
    console.print("5. Run training: [cyan]k-bert run[/cyan]")
    
    # Show component examples
    console.print("\n[bold]Custom components included:[/bold]")
    console.print("  • Custom BERT heads (src/heads/)")
    console.print("  • Data augmenters (src/augmenters/)")  
    console.print("  • Feature extractors (src/features/)")
    
    console.print("\n[dim]See README.md for detailed documentation.[/dim]")


def _customize_for_competition(project_path: Path, competition: str) -> None:
    """Customize project for specific competition."""
    from ...config import COMPETITION_PROFILES
    
    if competition not in COMPETITION_PROFILES:
        return
    
    profile = COMPETITION_PROFILES[competition]
    
    # Update configuration based on profile
    config_path = project_path / "k-bert.yaml"
    if config_path.exists():
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Update model settings
        if "models" not in config:
            config["models"] = {}
        
        if profile.recommended_models:
            config["models"]["default_model"] = profile.recommended_models[0]
        
        # Update data settings
        if "data" not in config:
            config["data"] = {}
        
        config["data"]["max_length"] = profile.recommended_max_length or 256
        
        # Update training settings
        if "training" not in config:
            config["training"] = {}
        
        config["training"]["default_batch_size"] = profile.recommended_batch_size or 32
        
        # Add competition info
        config["competition_info"] = {
            "name": competition,
            "type": profile.type,
            "metrics": profile.metrics,
            "target_column": profile.target_column,
            "id_column": profile.id_column,
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _show_tree(path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> None:
    """Show directory tree."""
    if current_depth >= max_depth:
        return
    
    items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        if item.name.startswith('.'):
            continue
        
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        next_prefix = "    " if is_last else "│   "
        
        if item.is_file():
            console.print(f"{prefix}{current_prefix}[cyan]{item.name}[/cyan]")
        else:
            console.print(f"{prefix}{current_prefix}[bold blue]{item.name}/[/bold blue]")
            
            if current_depth < max_depth - 1:
                _show_tree(item, prefix + next_prefix, max_depth, current_depth + 1)