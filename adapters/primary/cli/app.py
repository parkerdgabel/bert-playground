"""Main CLI application adapter.

This module creates the main Typer application and registers all command adapters.
It acts as the entry point for the CLI, delegating to individual command adapters.
"""

import sys
import typer
from rich.console import Console
from pathlib import Path
from typing import Optional

from adapters.primary.cli.train_adapter import train
from adapters.primary.cli.predict_adapter import predict
from adapters.primary.cli.evaluate_adapter import evaluate
from adapters.primary.cli.benchmark_adapter import benchmark
from adapters.primary.cli.info_adapter import info
from adapters.primary.cli.config_adapter import config_app


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


# Register commands with callback to get context
@app.callback()
def callback(ctx: typer.Context):
    """
    Callback to ensure container is available in context.
    The container should be set by the main entry point.
    """
    # The container will be passed as ctx.obj from the main entry point
    pass


# Register commands
app.command(name="train", help="Train a BERT model")(train)
app.command(name="evaluate", help="Evaluate a trained model")(evaluate)
app.command(name="predict", help="Generate predictions")(predict)
app.command(name="benchmark", help="Run performance benchmarks")(benchmark)
app.command(name="info", help="Display system and configuration information")(info)

# Register subcommand apps
app.add_typer(config_app, name="config")


# Version command
@app.command()
def version(ctx: typer.Context):
    """Display K-BERT version information."""
    __version__ = "0.1.0"  # TODO: Import from package metadata
    console.print(f"[bold blue]K-BERT[/bold blue] version {__version__}")


# Project command group
project_app = typer.Typer(help="Project management commands")


@project_app.command()
def init(
    ctx: typer.Context,
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
    """Main entry point for the CLI with DI support."""
    # Check if we're being run with a container from bootstrap
    if hasattr(sys.modules.get('__main__'), 'container'):
        # Get container from main module
        container = sys.modules['__main__'].container
        app(obj=container)
    else:
        # Try to initialize container if not provided
        try:
            from infrastructure.bootstrap import initialize_application
            container = initialize_application()
            app(obj=container)
        except ImportError:
            # Run without DI container
            app()


def main_with_di(container=None):
    """Main entry point with explicit DI container."""
    app(obj=container)


if __name__ == "__main__":
    main()