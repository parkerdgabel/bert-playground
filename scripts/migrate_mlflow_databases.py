#!/usr/bin/env python3
"""Script to migrate existing MLflow databases to the central location.

This script finds all MLflow databases in the project and migrates them
to the central tracking location at mlruns/mlflow.db.
"""

import os
import shutil
import sqlite3
from pathlib import Path
from typing import List, Tuple

import click
import mlflow
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.mlflow_central import mlflow_central


console = Console()


def find_mlflow_databases(root_dir: Path) -> List[Tuple[Path, int]]:
    """Find all MLflow databases in the project.
    
    Returns:
        List of (database_path, experiment_count) tuples
    """
    databases = []
    
    # Common patterns for MLflow databases
    patterns = [
        "**/mlruns/meta.db",
        "**/mlruns/mlflow.db",
        "**/mlflow.db",
        "**/meta.db"
    ]
    
    for pattern in patterns:
        for db_path in root_dir.glob(pattern):
            # Skip the central database
            if "mlruns/mlflow.db" in str(db_path):
                continue
            
            try:
                # Check if it's a valid SQLite database
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Count experiments
                cursor.execute("SELECT COUNT(*) FROM experiments")
                exp_count = cursor.fetchone()[0]
                
                conn.close()
                
                if exp_count > 0:
                    databases.append((db_path, exp_count))
                    
            except Exception as e:
                logger.debug(f"Skipping {db_path}: {e}")
    
    return databases


def get_experiment_info(db_path: Path) -> List[dict]:
    """Get experiment information from a database."""
    experiments = []
    
    try:
        # Set tracking URI to the database
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        
        # List all experiments
        for exp in mlflow.list_experiments():
            run_count = len(mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                output_format="list"
            ))
            
            experiments.append({
                "id": exp.experiment_id,
                "name": exp.name,
                "run_count": run_count,
                "artifact_location": exp.artifact_location
            })
    
    except Exception as e:
        logger.error(f"Error reading experiments from {db_path}: {e}")
    
    return experiments


def migrate_database(
    source_db: Path,
    experiment_mapping: dict,
    dry_run: bool = False
) -> Tuple[int, int]:
    """Migrate a database to the central location.
    
    Returns:
        (experiments_migrated, runs_migrated)
    """
    experiments_migrated = 0
    runs_migrated = 0
    
    # Initialize central MLflow
    mlflow_central.initialize()
    
    # Get experiments from source
    experiments = get_experiment_info(source_db)
    
    for exp in experiments:
        source_name = exp["name"]
        target_name = experiment_mapping.get(source_name, source_name)
        
        if dry_run:
            console.print(
                f"Would migrate: {source_name} -> {target_name} "
                f"({exp['run_count']} runs)"
            )
            experiments_migrated += 1
            runs_migrated += exp['run_count']
        else:
            try:
                # Use the migration utility
                mlflow_central.migrate_experiment(
                    source_uri=f"sqlite:///{source_db}",
                    experiment_name=source_name,
                    target_experiment_name=target_name
                )
                
                experiments_migrated += 1
                runs_migrated += exp['run_count']
                
                console.print(
                    f"✓ Migrated: {source_name} -> {target_name} "
                    f"({exp['run_count']} runs)",
                    style="green"
                )
                
                # Migrate artifacts if they exist
                if exp['artifact_location'] and Path(exp['artifact_location']).exists():
                    target_artifacts = Path(mlflow_central.artifact_root) / target_name
                    if not dry_run:
                        shutil.copytree(
                            exp['artifact_location'],
                            target_artifacts,
                            dirs_exist_ok=True
                        )
                        console.print(f"  ✓ Copied artifacts", style="dim green")
                        
            except Exception as e:
                console.print(
                    f"✗ Failed to migrate {source_name}: {e}",
                    style="red"
                )
    
    return experiments_migrated, runs_migrated


@click.command()
@click.option(
    "--root-dir",
    type=click.Path(exists=True, path_type=Path),
    default=".",
    help="Root directory to search for MLflow databases"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be migrated without actually migrating"
)
@click.option(
    "--interactive",
    is_flag=True,
    help="Interactively choose experiments to migrate"
)
@click.option(
    "--backup",
    is_flag=True,
    default=True,
    help="Backup existing central database before migration"
)
def main(root_dir: Path, dry_run: bool, interactive: bool, backup: bool):
    """Migrate all MLflow databases to the central location."""
    console.print("\n[bold]MLflow Database Migration Tool[/bold]\n")
    
    # Find all databases
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Searching for MLflow databases...", total=None)
        databases = find_mlflow_databases(root_dir)
        progress.update(task, completed=True)
    
    if not databases:
        console.print("No MLflow databases found.", style="yellow")
        return
    
    # Display found databases
    table = Table(title="Found MLflow Databases")
    table.add_column("Path", style="cyan")
    table.add_column("Experiments", justify="right")
    
    total_experiments = 0
    for db_path, exp_count in databases:
        table.add_row(
            str(db_path.relative_to(root_dir)),
            str(exp_count)
        )
        total_experiments += exp_count
    
    console.print(table)
    console.print(f"\nTotal: {len(databases)} databases, {total_experiments} experiments\n")
    
    # Backup central database if it exists
    central_db = Path("mlruns/mlflow.db")
    if backup and central_db.exists() and not dry_run:
        backup_path = central_db.with_suffix(".db.backup")
        shutil.copy2(central_db, backup_path)
        console.print(f"✓ Backed up central database to {backup_path}", style="green")
    
    # Collect all experiments for mapping
    all_experiments = {}
    experiment_mapping = {}
    
    for db_path, _ in databases:
        experiments = get_experiment_info(db_path)
        for exp in experiments:
            key = f"{db_path}:{exp['name']}"
            all_experiments[key] = exp
            
            # Default mapping (may have conflicts)
            if exp['name'] in experiment_mapping:
                # Add suffix for conflicts
                suffix = 1
                new_name = f"{exp['name']}_{suffix}"
                while new_name in experiment_mapping.values():
                    suffix += 1
                    new_name = f"{exp['name']}_{suffix}"
                experiment_mapping[exp['name']] = new_name
            else:
                experiment_mapping[exp['name']] = exp['name']
    
    # Interactive mode
    if interactive and not dry_run:
        console.print("\n[bold]Experiment Name Mapping[/bold]")
        console.print("You can rename experiments during migration:\n")
        
        for source_name, target_name in experiment_mapping.items():
            new_name = click.prompt(
                f"  {source_name}",
                default=target_name,
                show_default=True
            )
            experiment_mapping[source_name] = new_name
    
    # Confirm migration
    if not dry_run:
        if not click.confirm("\nProceed with migration?"):
            console.print("Migration cancelled.", style="yellow")
            return
    
    # Perform migration
    console.print("\n[bold]Starting Migration[/bold]\n")
    
    total_exp_migrated = 0
    total_runs_migrated = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        
        task = progress.add_task(
            "Migrating databases...",
            total=len(databases)
        )
        
        for db_path, _ in databases:
            progress.update(
                task,
                description=f"Migrating {db_path.name}..."
            )
            
            exp_migrated, runs_migrated = migrate_database(
                db_path,
                experiment_mapping,
                dry_run
            )
            
            total_exp_migrated += exp_migrated
            total_runs_migrated += runs_migrated
            
            progress.advance(task)
    
    # Summary
    console.print(f"\n[bold]Migration {'Preview' if dry_run else 'Complete'}[/bold]")
    console.print(f"  Experiments: {total_exp_migrated}")
    console.print(f"  Runs: {total_runs_migrated}")
    
    if not dry_run:
        console.print(f"\n✓ All experiments migrated to: mlruns/mlflow.db", style="green")
        console.print("\nYou can now view all experiments with: mlflow ui")
    else:
        console.print("\nRun without --dry-run to perform the migration.", style="yellow")


if __name__ == "__main__":
    main()