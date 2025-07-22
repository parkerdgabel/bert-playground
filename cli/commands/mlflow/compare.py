"""MLflow experiment comparison command."""

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ...config import get_config
from ...utils import handle_errors


console = Console()


@handle_errors
def compare_command(
    experiments: str = typer.Option(
        ..., 
        "--experiments", 
        "-e", 
        help="Comma-separated list of experiment names or IDs"
    ),
    metrics: Optional[str] = typer.Option(
        None,
        "--metrics",
        "-m", 
        help="Comma-separated list of metrics to compare"
    ),
    params: Optional[str] = typer.Option(
        None,
        "--params",
        "-p",
        help="Comma-separated list of parameters to compare"
    ),
    runs: Optional[int] = typer.Option(
        None,
        "--runs",
        "-r",
        help="Maximum number of runs per experiment"
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save comparison to file (CSV or JSON)"
    ),
):
    """Compare MLflow experiments side by side.
    
    Shows a comparison table of experiments with their best performing runs,
    metrics, and parameters.
    
    Examples:
        # Compare two experiments
        k-bert mlflow compare --experiments exp1,exp2
        
        # Compare specific metrics
        k-bert mlflow compare --experiments exp1,exp2 --metrics accuracy,loss
        
        # Save comparison to file
        k-bert mlflow compare --experiments exp1,exp2 --output comparison.csv
    """
    # Get configuration
    config = get_config()
    
    try:
        import mlflow
        import pandas as pd
        
        # Set tracking URI
        if config.mlflow and config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        console.print("[dim]Loading experiment data...[/dim]")
        
        # Parse experiment list
        exp_list = [exp.strip() for exp in experiments.split(",")]
        
        # Parse metric and param lists
        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",")]
        
        param_list = None
        if params:
            param_list = [p.strip() for p in params.split(",")]
        
        comparison_data = []
        
        for exp_name in exp_list:
            try:
                # Try to get experiment by name first, then by ID
                try:
                    exp = mlflow.get_experiment_by_name(exp_name)
                    if exp is None:
                        exp = mlflow.get_experiment(exp_name)
                except Exception:
                    exp = mlflow.get_experiment(exp_name)
                
                if exp is None:
                    console.print(f"[yellow]Warning: Experiment '{exp_name}' not found[/yellow]")
                    continue
                
                # Get runs for this experiment
                runs_df = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    max_results=runs or 10,
                    order_by=["metrics.accuracy DESC", "start_time DESC"]
                )
                
                if runs_df.empty:
                    console.print(f"[yellow]Warning: No runs found in experiment '{exp_name}'[/yellow]")
                    continue
                
                # Get the best run (first one after sorting)
                best_run = runs_df.iloc[0]
                
                exp_data = {
                    "experiment": exp.name,
                    "experiment_id": exp.experiment_id,
                    "total_runs": len(runs_df),
                    "best_run_id": best_run["run_id"],
                    "status": best_run["status"],
                    "start_time": best_run["start_time"],
                    "duration": (best_run["end_time"] - best_run["start_time"]).total_seconds() if best_run["end_time"] else None,
                }
                
                # Add metrics
                if metric_list:
                    for metric in metric_list:
                        metric_col = f"metrics.{metric}"
                        if metric_col in runs_df.columns:
                            exp_data[f"best_{metric}"] = best_run.get(metric_col)
                        else:
                            exp_data[f"best_{metric}"] = None
                else:
                    # Add all available metrics
                    for col in runs_df.columns:
                        if col.startswith("metrics."):
                            metric_name = col.replace("metrics.", "")
                            exp_data[f"best_{metric_name}"] = best_run.get(col)
                
                # Add parameters
                if param_list:
                    for param in param_list:
                        param_col = f"params.{param}"
                        if param_col in runs_df.columns:
                            exp_data[param] = best_run.get(param_col)
                        else:
                            exp_data[param] = None
                else:
                    # Add key parameters
                    for col in runs_df.columns:
                        if col.startswith("params.") and col != "params.mlflow.source.git.commit":
                            param_name = col.replace("params.", "")
                            exp_data[param_name] = best_run.get(col)
                
                comparison_data.append(exp_data)
                
            except Exception as e:
                console.print(f"[red]Error loading experiment '{exp_name}': {e}[/red]")
                continue
        
        if not comparison_data:
            console.print("[red]No valid experiments found for comparison.[/red]")
            raise typer.Exit(1)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save to file if requested
        if output:
            output_path = Path(output)
            if output_path.suffix.lower() == '.csv':
                comparison_df.to_csv(output_path, index=False)
                console.print(f"[green]Comparison saved to {output_path}[/green]")
            elif output_path.suffix.lower() == '.json':
                comparison_df.to_json(output_path, orient='records', indent=2)
                console.print(f"[green]Comparison saved to {output_path}[/green]")
            else:
                console.print("[yellow]Unknown output format. Supported: .csv, .json[/yellow]")
        
        # Display comparison table
        table = Table(title="Experiment Comparison", show_header=True)
        
        # Add columns dynamically based on available data
        columns_to_show = ["experiment", "total_runs", "status"]
        
        # Add metric columns
        metric_cols = [col for col in comparison_df.columns if col.startswith("best_")]
        columns_to_show.extend(sorted(metric_cols))
        
        # Add parameter columns (limit to avoid too wide table)
        param_cols = [col for col in comparison_df.columns 
                      if col not in columns_to_show and col not in ["experiment_id", "best_run_id", "start_time", "duration"]]
        columns_to_show.extend(sorted(param_cols)[:5])  # Limit to 5 params
        
        # Create table columns
        for col in columns_to_show:
            style = "cyan" if col == "experiment" else None
            if col.startswith("best_"):
                style = "green"
            table.add_column(col.replace("_", " ").title(), style=style)
        
        # Add rows
        for _, row in comparison_df.iterrows():
            row_data = []
            for col in columns_to_show:
                value = row.get(col)
                if value is None:
                    row_data.append("N/A")
                elif isinstance(value, float):
                    row_data.append(f"{value:.4f}")
                else:
                    row_data.append(str(value))
            table.add_row(*row_data)
        
        console.print(table)
        
        # Show summary statistics
        if len(comparison_data) > 1:
            console.print("\n[bold]Summary:[/bold]")
            
            # Find best performing experiment for each metric
            for col in metric_cols:
                values = comparison_df[col].dropna()
                if not values.empty:
                    if col in ["best_loss", "best_error"]:  # Lower is better
                        best_idx = values.idxmin()
                        best_value = values.min()
                    else:  # Higher is better
                        best_idx = values.idxmax()
                        best_value = values.max()
                    
                    best_exp = comparison_df.loc[best_idx, "experiment"]
                    metric_name = col.replace("best_", "").title()
                    console.print(f"  • Best {metric_name}: [green]{best_exp}[/green] ({best_value:.4f})")
        
    except ImportError:
        console.print(
            "[red]MLflow not installed.[/red]\n"
            "Install with: [cyan]uv add mlflow pandas[/cyan]"
        )
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to compare experiments: {e}[/red]")
        raise typer.Exit(1)