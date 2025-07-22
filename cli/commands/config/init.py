"""Initialize configuration command."""

from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ...config import ConfigManager
from ...config.defaults import get_competition_defaults
from ...utils import handle_errors, print_success, print_error


console = Console()


# Competition presets for quick project setup
COMPETITION_PRESETS = {
    "titanic": {
        "name": "titanic-bert",
        "competition": "titanic",
        "description": "Titanic survival prediction with BERT",
        "data": {
            "train_path": "data/titanic/train.csv",
            "val_path": "data/titanic/val.csv",
            "test_path": "data/titanic/test.csv",
            "max_length": 256,
            "batch_size": 32,
        },
        "training": {
            "epochs": 5,
            "learning_rate": 2e-5,
        },
        "models": {
            "default_model": "answerdotai/ModernBERT-base",
            "head": {
                "type": "binary_classification"
            }
        }
    },
    "disaster-tweets": {
        "name": "disaster-tweets-bert",
        "competition": "nlp-getting-started",
        "description": "Real or Not? NLP with Disaster Tweets",
        "data": {
            "train_path": "data/disaster-tweets/train.csv",
            "val_path": "data/disaster-tweets/val.csv",
            "test_path": "data/disaster-tweets/test.csv",
            "max_length": 128,
            "batch_size": 64,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 3e-5,
        },
        "models": {
            "default_model": "answerdotai/ModernBERT-base",
            "head": {
                "type": "binary_classification"
            }
        }
    }
}


@handle_errors
def init_command(
    project: bool = typer.Option(
        False,
        "--project",
        "-p",
        help="Initialize project configuration (k-bert.yaml) instead of user config",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for project configuration (default: k-bert.yaml)",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        help="Use a competition preset (e.g., titanic, disaster-tweets)",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run in interactive mode",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing configuration",
    ),
):
    """Initialize k-bert configuration.
    
    This command can create either:
    1. User configuration at ~/.k-bert/config.yaml (default)
    2. Project configuration at k-bert.yaml (with --project)
    
    Examples:
        # Initialize user config (interactive)
        k-bert config init
        
        # Initialize project config
        k-bert config init --project
        
        # Use competition preset
        k-bert config init --project --preset titanic
        
        # Non-interactive with defaults
        k-bert config init --no-interactive
        
        # Custom output path
        k-bert config init --project --output configs/my-config.yaml
    """
    manager = ConfigManager()
    
    if project:
        # Initialize project configuration
        _init_project_config(manager, output, preset, interactive, force)
    else:
        # Initialize user configuration
        _init_user_config(manager, interactive, force)


def _init_user_config(manager: ConfigManager, interactive: bool, force: bool):
    """Initialize user configuration."""
    # Check if config already exists
    if manager.user_config_path.exists() and not force:
        console.print(
            f"[yellow]Configuration already exists at {manager.user_config_path}[/yellow]"
        )
        
        if interactive:
            if not Confirm.ask("Overwrite existing configuration?", default=False):
                console.print("[red]Configuration initialization cancelled.[/red]")
                raise typer.Exit(0)
        else:
            console.print(
                "[red]Use --force to overwrite existing configuration.[/red]"
            )
            raise typer.Exit(1)
    
    # Initialize configuration
    try:
        config = manager.init_user_config(interactive=interactive)
        
        if not interactive:
            console.print(
                f"[green]Created default configuration at {manager.user_config_path}[/green]"
            )
        
        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Set your Kaggle credentials:")
        console.print("   [cyan]k-bert config set kaggle.username YOUR_USERNAME[/cyan]")
        console.print("   [cyan]k-bert config set kaggle.key YOUR_API_KEY[/cyan]")
        console.print("\n2. Initialize a project configuration:")
        console.print("   [cyan]k-bert config init --project[/cyan]")
        console.print("\n3. Start training:")
        console.print("   [cyan]k-bert train[/cyan]")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize configuration: {e}[/red]")
        raise typer.Exit(1)


def _init_project_config(
    manager: ConfigManager,
    output: Optional[Path],
    preset: Optional[str],
    interactive: bool,
    force: bool
):
    """Initialize project configuration."""
    # Determine output path
    if output is None:
        output = Path("k-bert.yaml")
    
    # Check if file exists
    if output.exists() and not force:
        if interactive:
            if not Confirm.ask(f"[yellow]{output} already exists. Overwrite?[/yellow]"):
                console.print("[red]Configuration initialization cancelled.[/red]")
                raise typer.Exit(0)
        else:
            print_error(
                f"Configuration file already exists: {output}\n"
                "Use --force to overwrite.",
                title="File Exists"
            )
            raise typer.Exit(1)
    
    # Start with defaults or preset
    if preset:
        if preset not in COMPETITION_PRESETS:
            available = ", ".join(COMPETITION_PRESETS.keys())
            print_error(
                f"Unknown preset: {preset}\n"
                f"Available presets: {available}",
                title="Invalid Preset"
            )
            raise typer.Exit(1)
        
        config_dict = COMPETITION_PRESETS[preset].copy()
        console.print(f"[cyan]Using {preset} competition preset[/cyan]")
    else:
        # Start with minimal defaults
        config_dict = {
            "name": "my-bert-project",
            "description": "BERT project for Kaggle competitions",
            "version": "1.0"
        }
    
    # Interactive configuration
    if interactive and not preset:
        console.print("\n[bold]Project Configuration Setup[/bold]\n")
        
        # Project info
        config_dict["name"] = Prompt.ask(
            "Project name",
            default=config_dict.get("name", "my-bert-project")
        )
        
        config_dict["description"] = Prompt.ask(
            "Project description",
            default=config_dict.get("description", "")
        )
        
        # Competition
        if Confirm.ask("Are you working on a specific Kaggle competition?"):
            comp_name = Prompt.ask("Competition name (e.g., titanic)")
            config_dict["competition"] = comp_name
            
            # Apply competition defaults if available
            comp_defaults = get_competition_defaults(comp_name)
            if comp_defaults:
                console.print(f"[green]Applied defaults for {comp_name} competition[/green]")
                # Merge competition defaults
                for key, value in comp_defaults.items():
                    if key not in config_dict:
                        config_dict[key] = value
        
        # Model configuration
        console.print("\n[bold]Model Configuration[/bold]")
        model_choice = Prompt.ask(
            "Default model",
            default="answerdotai/ModernBERT-base",
            choices=[
                "answerdotai/ModernBERT-base",
                "answerdotai/ModernBERT-large",
                "bert-base-uncased",
                "custom"
            ]
        )
        
        if model_choice == "custom":
            model_choice = Prompt.ask("Enter model name/path")
        
        config_dict.setdefault("models", {})["default_model"] = model_choice
        
        if Confirm.ask("Use LoRA for efficient fine-tuning?", default=False):
            config_dict["models"]["use_lora"] = True
            config_dict["models"]["lora_preset"] = Prompt.ask(
                "LoRA preset",
                default="balanced",
                choices=["minimal", "balanced", "aggressive"]
            )
        
        # Data configuration
        console.print("\n[bold]Data Configuration[/bold]")
        config_dict.setdefault("data", {})
        
        config_dict["data"]["train_path"] = Prompt.ask(
            "Training data path",
            default="data/train.csv"
        )
        
        if Confirm.ask("Do you have validation data?", default=True):
            config_dict["data"]["val_path"] = Prompt.ask(
                "Validation data path",
                default="data/val.csv"
            )
        
        config_dict["data"]["test_path"] = Prompt.ask(
            "Test data path",
            default="data/test.csv"
        )
        
        config_dict["data"]["batch_size"] = int(Prompt.ask(
            "Batch size",
            default="32"
        ))
        
        config_dict["data"]["max_length"] = int(Prompt.ask(
            "Maximum sequence length",
            default="256"
        ))
        
        # Training configuration
        console.print("\n[bold]Training Configuration[/bold]")
        config_dict.setdefault("training", {})
        
        config_dict["training"]["epochs"] = int(Prompt.ask(
            "Number of epochs",
            default="5"
        ))
        
        config_dict["training"]["learning_rate"] = float(Prompt.ask(
            "Learning rate",
            default="2e-5"
        ))
        
        config_dict["training"]["output_dir"] = Prompt.ask(
            "Output directory",
            default="./outputs"
        )
        
        # MLflow
        if Confirm.ask("Enable MLflow tracking?", default=True):
            config_dict.setdefault("mlflow", {})["auto_log"] = True
            config_dict["mlflow"]["experiment_name"] = Prompt.ask(
                "MLflow experiment name",
                default=config_dict["name"]
            )
        
        # Experiments
        if Confirm.ask("Would you like to define experiments?", default=True):
            experiments = []
            
            # Quick test experiment
            experiments.append({
                "name": "quick_test",
                "description": "Quick test with 1 epoch",
                "config": {
                    "training": {
                        "epochs": 1
                    }
                }
            })
            
            # Full training experiment
            experiments.append({
                "name": "full_training",
                "description": "Full training with best settings",
                "config": {
                    "training": {
                        "epochs": config_dict["training"]["epochs"] * 2,
                        "learning_rate": float(config_dict["training"]["learning_rate"]) / 2
                    },
                    "data": {
                        "batch_size": config_dict["data"]["batch_size"] * 2
                    }
                }
            })
            
            config_dict["experiments"] = experiments
            console.print("[green]Added 2 default experiments: quick_test, full_training[/green]")
    
    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print_success(
        f"Project configuration saved to: {output}\n"
        f"You can now run: k-bert train",
        title="Configuration Created"
    )
    
    # Show configuration preview
    if interactive:
        if Confirm.ask("\nWould you like to see the configuration?"):
            console.print("\n[bold]Configuration Preview:[/bold]")
            with open(output, "r") as f:
                console.print(f.read())