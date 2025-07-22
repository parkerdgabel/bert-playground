"""Initialize competition project command."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from ...config import (
    ConfigManager,
    ProjectConfig,
    COMPETITION_PROFILES,
    get_competition_defaults,
)
from ...utils import handle_errors


console = Console()


@handle_errors
def init_command(
    competition: str = typer.Argument(
        ...,
        help="Competition name (e.g., titanic)",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--path",
        "-p",
        help="Project directory (defaults to ./{competition})",
    ),
    template: str = typer.Option(
        "modernbert",
        "--template",
        "-t",
        help="Project template to use",
    ),
    download: bool = typer.Option(
        True,
        "--download/--no-download",
        help="Download competition data",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing project",
    ),
):
    """Initialize a new competition project.
    
    This command creates a project structure optimized for a specific
    Kaggle competition, including configuration, directories, and
    optionally downloading the competition data.
    
    Examples:
        # Initialize Titanic project
        k-bert competition init titanic
        
        # Initialize in specific directory
        k-bert competition init house-prices --path ./projects/house-prices
        
        # Initialize without downloading data
        k-bert competition init titanic --no-download
    """
    # Determine project path
    if path is None:
        path = Path(f"./{competition}")
    
    # Check if directory exists
    if path.exists() and not force:
        if path.is_file():
            console.print(f"[red]Path exists and is a file: {path}[/red]")
            raise typer.Exit(1)
        
        if any(path.iterdir()):
            console.print(f"[yellow]Directory {path} already exists and is not empty.[/yellow]")
            if not Confirm.ask("Continue anyway?", default=False):
                raise typer.Exit(0)
    
    # Create project structure
    console.print(f"[cyan]Creating project structure in {path}...[/cyan]")
    
    # Create directories
    dirs = [
        "data/raw",
        "data/processed",
        "data/submissions",
        "configs/models",
        "configs/training",
        "configs/experiments",
        "notebooks",
        "src/features",
        "src/models",
        "src/utils",
        "outputs",
    ]
    
    for dir_path in dirs:
        (path / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create .gitignore
    gitignore_content = """# Data files
data/raw/
data/processed/
*.csv
*.parquet
*.feather

# Model files
*.pth
*.ckpt
*.safetensors

# Outputs
outputs/
mlruns/
*.log

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Environment
.env
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
"""
    
    with open(path / ".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Create README
    readme_content = f"""# {competition.title()} Competition

This project was created with k-bert for the Kaggle {competition} competition.

## Project Structure

```
{competition}/
├── data/
│   ├── raw/          # Original competition data
│   ├── processed/    # Preprocessed data
│   └── submissions/  # Generated submissions
├── configs/
│   ├── models/       # Model configurations
│   ├── training/     # Training configurations
│   └── experiments/  # Experiment configs
├── notebooks/        # Jupyter notebooks
├── src/
│   ├── features/     # Feature engineering
│   ├── models/       # Custom models
│   └── utils/        # Utilities
├── outputs/          # Training outputs
└── k-bert.yaml       # Project configuration
```

## Quick Start

1. **Download data** (if not already done):
   ```bash
   k-bert competition download {competition} --path data/raw
   ```

2. **Train model**:
   ```bash
   k-bert run
   ```

3. **Generate predictions**:
   ```bash
   k-bert predict --test data/raw/test.csv --output data/submissions/submission.csv
   ```

4. **Submit to Kaggle**:
   ```bash
   k-bert competition submit {competition} data/submissions/submission.csv
   ```

## Configuration

Edit `k-bert.yaml` to customize training parameters, model architecture, and data processing.

## Experiments

Track experiments with MLflow:
```bash
k-bert mlflow ui
```
"""
    
    with open(path / "README.md", "w") as f:
        f.write(readme_content)
    
    # Get competition profile if available
    profile = COMPETITION_PROFILES.get(competition)
    comp_defaults = get_competition_defaults(competition) if profile else {}
    
    # Create project configuration
    project_config = {
        "name": competition,
        "competition": competition,
        "description": f"K-BERT project for {competition} competition",
        "version": "1.0",
    }
    
    # Add competition-specific defaults
    if comp_defaults:
        project_config.update(comp_defaults)
    
    # Add template-specific configuration
    if template == "modernbert":
        project_config["models"] = {
            "default_model": "answerdotai/ModernBERT-base",
            "use_mlx_embeddings": True,
            "default_architecture": "modernbert",
        }
        project_config["training"] = {
            "default_epochs": 5,
            "default_batch_size": comp_defaults.get("training", {}).get("default_batch_size", 32),
            "default_learning_rate": 2e-5,
            "save_best_only": True,
        }
    
    # Add data paths if we know the structure
    if profile:
        project_config["data"] = {
            "train_path": f"data/raw/{profile.train_file}",
            "test_path": f"data/raw/{profile.test_file}",
        }
        
        if profile.target_column:
            project_config["data"]["target_column"] = profile.target_column
        
        if profile.id_column:
            project_config["data"]["id_column"] = profile.id_column
    
    # Save project configuration
    config_manager = ConfigManager()
    config_manager.save_project_config(project_config, path / "k-bert.yaml")
    
    console.print(f"[green]✓[/green] Created project configuration")
    
    # Create example training config
    train_config = f"""# Training configuration for {competition}
model:
  name: answerdotai/ModernBERT-base
  architecture: modernbert
  head_type: {"binary_classification" if profile and profile.type == "binary_classification" else "multiclass_classification"}
  num_labels: {profile.recommended_models[0] if profile and profile.recommended_models else 2}

training:
  epochs: 5
  batch_size: {comp_defaults.get("training", {}).get("default_batch_size", 32)}
  learning_rate: 2e-5
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1

data:
  max_length: {comp_defaults.get("data", {}).get("max_length", 256)}
  augmentation_mode: moderate
  use_pretokenized: true

optimizer:
  type: adamw
  weight_decay: 0.01
  
scheduler:
  type: cosine
  num_warmup_steps: 500
"""
    
    with open(path / "configs/training/default.yaml", "w") as f:
        f.write(train_config)
    
    # Download data if requested
    if download:
        console.print(f"\n[cyan]Downloading competition data...[/cyan]")
        
        from .download import download_command
        try:
            download_command(
                competition=competition,
                path=path / "data" / "raw",
                force=False,
                unzip=True,
            )
        except typer.Exit:
            console.print(
                "[yellow]Data download failed. You can download manually with:[/yellow]\n"
                f"  [cyan]k-bert competition download {competition} --path {path}/data/raw[/cyan]"
            )
    
    # Create example notebook
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {competition.title()} Competition\\n\\n",
                    "This notebook provides a starting point for exploring the competition data."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\\n",
                    "import numpy as np\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "import seaborn as sns\\n",
                    "\\n",
                    "# Set style\\n",
                    "plt.style.use('ggplot')\\n",
                    "sns.set_palette('husl')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load data\\n",
                    f"train_df = pd.read_csv('../data/raw/{profile.train_file if profile else 'train.csv'}')\\n",
                    f"test_df = pd.read_csv('../data/raw/{profile.test_file if profile else 'test.csv'}')\\n",
                    "\\n",
                    "print(f'Training samples: {len(train_df)}')\\n",
                    "print(f'Test samples: {len(test_df)}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    import json
    with open(path / "notebooks/exploration.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    # Show summary
    console.print(f"\n[green]✓ Successfully initialized {competition} project![/green]")
    console.print(f"\nProject created at: [cyan]{path.absolute()}[/cyan]")
    
    console.print("\n[bold]Next steps:[/bold]")
    console.print(f"1. Navigate to project: [cyan]cd {path}[/cyan]")
    
    if not download:
        console.print(f"2. Download data: [cyan]k-bert competition download {competition} --path data/raw[/cyan]")
    
    console.print("3. Explore data: [cyan]jupyter notebook notebooks/exploration.ipynb[/cyan]")
    console.print("4. Configure training: [cyan]edit k-bert.yaml[/cyan]")
    console.print("5. Start training: [cyan]k-bert run[/cyan]")
    
    # Show tips for known competitions
    if profile:
        console.print(f"\n[dim]Tip: This competition uses {profile.type.replace('_', ' ')} "
                     f"with {', '.join(profile.metrics)} as evaluation metrics.[/dim]")