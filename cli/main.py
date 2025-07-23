"""Main entry point for the K-BERT CLI.

This module creates the main Typer application and registers all commands.
It acts as a thin layer that delegates to individual command modules.
"""

import typer
from rich.console import Console

from cli import __version__
from cli.commands.train import train
from cli.commands.evaluate import evaluate
from cli.commands.predict import predict
from cli.commands.info import info


# Create the main Typer app
app = typer.Typer(
    name="k-bert",
    help="K-BERT: MLX-based ModernBERT for Kaggle competitions",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)

# Create console for output
console = Console()


# Register commands
app.command(name="train", help="Train a BERT model")(train)
app.command(name="evaluate", help="Evaluate a trained model")(evaluate)
app.command(name="predict", help="Generate predictions")(predict)
app.command(name="info", help="Display system and configuration information")(info)


# Version command
@app.command()
def version():
    """Display K-BERT version information."""
    console.print(f"[bold blue]K-BERT[/bold blue] version {__version__}")


# Config command group
config_app = typer.Typer(help="Configuration management commands")


@config_app.command()
def init(
    user: bool = typer.Option(
        False,
        "--user",
        help="Initialize user configuration (~/.k-bert/config.yaml)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing configuration",
    ),
):
    """Initialize K-BERT configuration.
    
    Creates a configuration file with default settings.
    By default, creates k-bert.yaml in the current directory.
    Use --user to create user configuration instead.
    """
    from pathlib import Path
    import yaml
    
    # Determine target path
    if user:
        config_dir = Path.home() / ".k-bert"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_type = "user"
    else:
        config_path = Path.cwd() / "k-bert.yaml"
        config_type = "project"
    
    # Check if exists
    if config_path.exists() and not force:
        console.print(f"[yellow]Configuration already exists at {config_path}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    # Create default configuration
    default_config = {
        "models": {
            "type": "modernbert_with_head",
            "default_model": "answerdotai/ModernBERT-base",
            "head_type": "binary_classification",
            "num_labels": 2,
        },
        "data": {
            "batch_size": 32,
            "max_length": 512,
            "num_workers": 0,
        },
        "training": {
            "epochs": 3,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "optimizer": "adamw",
            "scheduler": "warmup_linear",
            "output_dir": "output",
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_steps": 100,
            "load_best_model_at_end": True,
        },
        "mlflow": {
            "enabled": True,
            "tracking_uri": None,
            "experiment_name": "k-bert-experiments",
        },
        "logging": {
            "level": "INFO",
        }
    }
    
    # Write configuration
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    console.print(f"[green]Created {config_type} configuration at {config_path}[/green]")
    console.print("\nNext steps:")
    console.print("  1. Edit the configuration file to match your needs")
    console.print("  2. Set data paths (data.train_path, data.val_path)")
    console.print("  3. Run training: [cyan]k-bert train[/cyan]")


@config_app.command()
def show(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format",
    ),
):
    """Show current configuration.
    
    Displays the merged configuration from all sources.
    """
    from cli.config.loader import ConfigurationLoader
    from pathlib import Path
    import yaml
    import json
    
    loader = ConfigurationLoader()
    configs = []
    
    # Load configs in order
    sources = []
    
    # User config
    if user_path := loader.find_user_config():
        configs.append(loader.load_yaml_config(user_path))
        sources.append(f"User: {user_path}")
    
    # Project config
    if project_path := loader.find_project_config():
        configs.append(loader.load_yaml_config(project_path))
        sources.append(f"Project: {project_path}")
    
    if not configs:
        console.print("[yellow]No configuration files found[/yellow]")
        console.print("Run [cyan]k-bert config init[/cyan] to create one")
        raise typer.Exit(1)
    
    # Merge configs
    merged = loader.merge_configs(configs)
    
    # Display
    if json_output:
        console.print_json(data=merged)
    else:
        console.print("[bold blue]Current Configuration[/bold blue]")
        console.print("\nSources (in order of precedence):")
        for source in sources:
            console.print(f"  • {source}")
        
        console.print("\n[bold]Merged Configuration:[/bold]")
        yaml_output = yaml.dump(merged, default_flow_style=False, sort_keys=False)
        
        from rich.syntax import Syntax
        syntax = Syntax(yaml_output, "yaml", theme="monokai", line_numbers=False)
        console.print(syntax)


@config_app.command()
def validate(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Configuration file to validate",
    ),
):
    """Validate configuration file.
    
    Checks for required fields and valid values.
    """
    from cli.config.loader import ConfigurationLoader
    from pathlib import Path
    
    loader = ConfigurationLoader()
    
    # Determine which config to validate
    if config:
        config_path = config
    elif project_path := loader.find_project_config():
        config_path = project_path
    else:
        console.print("[red]No configuration file found[/red]")
        raise typer.Exit(1)
    
    console.print(f"Validating: [cyan]{config_path}[/cyan]\n")
    
    try:
        # Load config
        config_data = loader.load_yaml_config(config_path)
        
        # Check for each command type
        commands = ["train", "evaluate", "predict"]
        all_valid = True
        
        for command in commands:
            errors = loader.validate_config(config_data, command)
            if errors:
                all_valid = False
                console.print(f"[red]✗[/red] {command.title()} command:")
                for error in errors:
                    console.print(f"    • {error}")
            else:
                console.print(f"[green]✓[/green] {command.title()} command: Valid")
        
        if all_valid:
            console.print("\n[green]Configuration is valid![/green]")
        else:
            console.print("\n[yellow]Configuration has errors. Please fix them before running.[/yellow]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)


# Register config subcommands
app.add_typer(config_app, name="config")


# Project command group  
project_app = typer.Typer(help="Project management commands")


@project_app.command()
def init(
    name: str = typer.Argument(
        "my-bert-project",
        help="Project name",
    ),
    template: str = typer.Option(
        "basic",
        "--template", "-t",
        help="Project template (basic/kaggle/advanced)",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing files",
    ),
):
    """Initialize a new K-BERT project.
    
    Creates a project structure with configuration and example files.
    """
    from pathlib import Path
    
    project_dir = Path.cwd() / name
    
    # Check if exists
    if project_dir.exists() and not force:
        console.print(f"[yellow]Directory {name} already exists[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    # Create project structure
    console.print(f"Creating K-BERT project: [cyan]{name}[/cyan]")
    
    project_dir.mkdir(exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)
    (project_dir / "notebooks").mkdir(exist_ok=True)
    (project_dir / "src").mkdir(exist_ok=True)
    
    # Create .gitignore
    gitignore_content = """# K-BERT
output/
*.safetensors
*.ckpt
*.pkl
__pycache__/
*.pyc
.DS_Store
mlruns/
.tokenizer_cache/
*.log
"""
    (project_dir / ".gitignore").write_text(gitignore_content)
    
    # Create README
    readme_content = f"""# {name}

A K-BERT project for training BERT models with MLX.

## Setup

1. Install K-BERT: `pip install k-bert`
2. Configure your data paths in `k-bert.yaml`
3. Train your model: `k-bert train`

## Project Structure

- `data/`: Training and evaluation data
- `models/`: Saved models and checkpoints
- `output/`: Training outputs and logs
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Custom code and plugins
"""
    (project_dir / "README.md").write_text(readme_content)
    
    # Create config based on template
    if template == "kaggle":
        # Kaggle competition template
        config_content = """# K-BERT Configuration for Kaggle Competition

models:
  type: modernbert_with_head
  default_model: answerdotai/ModernBERT-base
  head_type: binary_classification
  num_labels: 2

data:
  train_path: data/train.csv
  val_path: data/val.csv  # Create this from train.csv
  test_path: data/test.csv
  batch_size: 32
  max_length: 512
  
training:
  epochs: 5
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  optimizer: adamw
  scheduler: warmup_linear
  output_dir: output
  eval_strategy: epoch
  save_strategy: epoch
  early_stopping_patience: 3
  metric_for_best_model: eval_f1
  greater_is_better: true
  
prediction:
  output_path: submissions/submission.csv
  include_probabilities: false
  
mlflow:
  enabled: true
  experiment_name: kaggle-competition
"""
    else:
        # Basic template
        config_content = """# K-BERT Configuration

models:
  type: modernbert_with_head
  default_model: answerdotai/ModernBERT-base
  head_type: binary_classification
  num_labels: 2

data:
  train_path: data/train.csv
  val_path: data/val.csv
  test_path: data/test.csv
  batch_size: 32
  max_length: 512
  
training:
  epochs: 3
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  optimizer: adamw
  scheduler: warmup_linear
  output_dir: output
  
mlflow:
  enabled: true
  experiment_name: my-bert-experiments
"""
    
    (project_dir / "k-bert.yaml").write_text(config_content)
    
    # Create example notebook
    notebook_content = """# K-BERT Analysis Notebook

## Load libraries
```python
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
```

## Load results
```python
# Load training history
# history = pd.read_json('output/run_*/training_history.json')
```

## Visualize training
```python
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Val Loss')
# plt.legend()
# plt.show()
```
"""
    (project_dir / "notebooks" / "analysis.md").write_text(notebook_content)
    
    console.print(f"\n[green]Project created successfully![/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. cd {name}")
    console.print(f"  2. Add your data to the data/ directory")
    console.print(f"  3. Edit k-bert.yaml to configure your model")
    console.print(f"  4. Run: k-bert train")


# Register project subcommands
app.add_typer(project_app, name="project")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()