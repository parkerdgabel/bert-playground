"""
Example workflow demonstrating comprehensive Kaggle integration.
Shows how to use the new Kaggle commands and MLflow tracking together.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


def run_command(cmd: str, description: str):
    """Run a command and display the output."""
    console.print(f"\n[bold cyan]Running: {description}[/bold cyan]")
    console.print(Panel(Syntax(cmd, "bash", theme="monokai"), title="Command"))
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            console.print(f"[green]✓ Success[/green]")
            if result.stdout:
                console.print(result.stdout)
        else:
            console.print(f"[red]✗ Failed[/red]")
            if result.stderr:
                console.print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return False


def main():
    """Demonstrate the complete Kaggle workflow."""
    
    console.print(Panel.fit(
        "[bold]Kaggle Integration Workflow Example[/bold]\n"
        "This example demonstrates the complete workflow for:\n"
        "• Listing and downloading competitions\n"
        "• Training models with MLflow tracking\n"
        "• Submitting to Kaggle\n"
        "• Viewing leaderboards and history",
        title="MLX-BERT Kaggle Integration"
    ))
    
    # Step 1: List available competitions
    console.print("\n[bold yellow]Step 1: Browse Kaggle Competitions[/bold yellow]")
    run_command(
        "uv run python bert_cli.py kaggle-competitions --category tabular --limit 10",
        "List tabular competitions"
    )
    
    # Step 2: Download competition data
    console.print("\n[bold yellow]Step 2: Download Competition Data[/bold yellow]")
    if run_command(
        "uv run python bert_cli.py kaggle-download titanic --output data/titanic_demo",
        "Download Titanic competition data"
    ):
        console.print("[green]Data downloaded successfully![/green]")
    
    # Step 3: Quick training run
    console.print("\n[bold yellow]Step 3: Train Model with MLflow Tracking[/bold yellow]")
    run_command(
        """uv run python bert_cli.py train \\
        --train data/titanic_demo/train.csv \\
        --val data/titanic_demo/train.csv \\
        --epochs 1 \\
        --batch-size 32 \\
        --experiment kaggle_titanic_demo""",
        "Quick training run"
    )
    
    # Step 4: Generate predictions
    console.print("\n[bold yellow]Step 4: Generate Predictions[/bold yellow]")
    
    # Find the latest checkpoint
    output_dir = Path("output")
    if output_dir.exists():
        runs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
        if runs:
            latest_run = runs[-1]
            checkpoint = latest_run / "best_model_accuracy"
            
            if checkpoint.exists():
                run_command(
                    f"""uv run python bert_cli.py predict \\
                    --test data/titanic_demo/test.csv \\
                    --checkpoint {checkpoint} \\
                    --output submissions/titanic_demo.csv""",
                    "Generate predictions"
                )
    
    # Step 5: Submit to Kaggle (demonstration only)
    console.print("\n[bold yellow]Step 5: Submit to Kaggle[/bold yellow]")
    console.print(Panel(
        "[yellow]Note: Actual submission requires accepted competition rules[/yellow]\n"
        "Example command:\n"
        "[cyan]uv run python bert_cli.py kaggle-submit titanic submissions/titanic_demo.csv \\[/cyan]\n"
        "[cyan]    --message \"MLX-BERT baseline model\" \\[/cyan]\n"
        "[cyan]    --checkpoint output/run_001/best_model_accuracy[/cyan]",
        title="Submission Example"
    ))
    
    # Step 6: View leaderboard
    console.print("\n[bold yellow]Step 6: View Competition Leaderboard[/bold yellow]")
    run_command(
        "uv run python bert_cli.py kaggle-leaderboard titanic --top 20",
        "View top 20 leaderboard entries"
    )
    
    # Step 7: Check submission history
    console.print("\n[bold yellow]Step 7: View Submission History[/bold yellow]")
    run_command(
        "uv run python bert_cli.py kaggle-history titanic --limit 5",
        "View recent submissions"
    )
    
    # Step 8: Search for datasets
    console.print("\n[bold yellow]Step 8: Explore Kaggle Datasets[/bold yellow]")
    run_command(
        "uv run python bert_cli.py kaggle-datasets --search \"text classification\" --limit 5",
        "Search for text classification datasets"
    )
    
    # Summary
    console.print("\n" + "="*60 + "\n")
    console.print(Panel.fit(
        "[bold green]Workflow Complete![/bold green]\n\n"
        "You've seen how to:\n"
        "✓ Browse and download competitions\n"
        "✓ Train models with experiment tracking\n"
        "✓ Generate and submit predictions\n"
        "✓ Monitor leaderboard and submissions\n"
        "✓ Discover new datasets\n\n"
        "[cyan]Next steps:[/cyan]\n"
        "• Use [cyan]kaggle-auto-submit[/cyan] for automated submissions\n"
        "• Create competition configs in [cyan]configs/competitions/[/cyan]\n"
        "• View MLflow UI: [cyan]mlflow ui --backend-store-uri ./output/mlruns[/cyan]",
        title="Summary"
    ))


if __name__ == "__main__":
    main()