#!/usr/bin/env python3
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

console = Console()


def demo_mlflow_features():
    """Demonstrate the MLflow and logging features."""

    console.print(
        Panel.fit(
            "[bold blue]MLX ModernBERT with MLflow & Enhanced Logging[/bold blue]",
            border_style="blue",
        )
    )

    # Feature overview
    console.print("\n[bold]✨ New Features:[/bold]\n")

    features = Tree("🚀 Enhanced Training Pipeline")

    mlflow_node = features.add("📊 MLflow Integration")
    mlflow_node.add("• Automatic experiment tracking")
    mlflow_node.add("• Model versioning and registry")
    mlflow_node.add("• Metrics visualization")
    mlflow_node.add("• Artifact management")

    logging_node = features.add("📝 Comprehensive Logging")
    logging_node.add("• Structured logs with Loguru")
    logging_node.add("• Rich console output")
    logging_node.add("• Detailed error tracking")
    logging_node.add("• Performance metrics")

    viz_node = features.add("📈 Visualization Tools")
    viz_node.add("• Training curves")
    viz_node.add("• Confusion matrices")
    viz_node.add("• ROC curves")
    viz_node.add("• Experiment comparison")

    console.print(features)

    # Command examples
    console.print("\n[bold]🎯 Usage Examples:[/bold]\n")

    # Basic training with MLflow
    console.print(
        Panel(
            Syntax(
                "# Train with MLflow tracking\n"
                "uv run python train_titanic_v2.py \\\n"
                "  --do_train \\\n"
                "  --num_epochs 3 \\\n"
                "  --experiment_name 'titanic_experiment' \\\n"
                "  --run_name 'baseline_run'",
                "bash",
                theme="monokai",
                line_numbers=False,
            ),
            title="Basic Training",
            border_style="green",
        )
    )

    # Advanced training with all features
    console.print(
        Panel(
            Syntax(
                "# Full pipeline with visualization\n"
                "uv run python train_titanic_v2.py \\\n"
                "  --do_train \\\n"
                "  --do_predict \\\n"
                "  --do_visualize \\\n"
                "  --launch_mlflow \\\n"
                "  --num_epochs 5 \\\n"
                "  --batch_size 32 \\\n"
                "  --learning_rate 3e-5 \\\n"
                "  --log_level DEBUG",
                "bash",
                theme="monokai",
                line_numbers=False,
            ),
            title="Advanced Training",
            border_style="yellow",
        )
    )

    # MLflow UI
    console.print(
        Panel(
            Syntax(
                "# Launch MLflow UI separately\n"
                "mlflow ui --backend-store-uri ./output/mlruns --port 5000\n"
                "\n"
                "# Then navigate to http://localhost:5000",
                "bash",
                theme="monokai",
                line_numbers=False,
            ),
            title="MLflow Dashboard",
            border_style="blue",
        )
    )

    # Log structure
    console.print("\n[bold]📁 Output Structure:[/bold]\n")

    structure = Tree("output/")
    structure.add("mlruns/           # MLflow tracking data")
    structure.add("logs/             # Detailed logs")
    structure.add("best_model_*/     # Model checkpoints")
    structure.add("training_history.json")
    structure.add("config.json")
    structure.add("experiment_report.md")
    structure.add("*.png             # Visualizations")

    console.print(structure)

    # Key benefits
    console.print("\n[bold]💡 Key Benefits:[/bold]\n")

    benefits_table = Table(show_header=True, header_style="bold magenta")
    benefits_table.add_column("Feature", style="cyan", no_wrap=True)
    benefits_table.add_column("Benefit", style="white")

    benefits = [
        (
            "MLflow Tracking",
            "Compare experiments, track metrics over time, reproduce results",
        ),
        (
            "Enhanced Logging",
            "Debug issues quickly, monitor training progress, analyze performance",
        ),
        (
            "Visualizations",
            "Understand model behavior, identify problems, present results",
        ),
        (
            "Model Registry",
            "Version control models, deploy best performers, collaborate with team",
        ),
        (
            "Rich CLI",
            "Better user experience, clear progress indication, formatted output",
        ),
    ]

    for feature, benefit in benefits:
        benefits_table.add_row(feature, benefit)

    console.print(benefits_table)

    # Tips
    console.print("\n[bold]💡 Pro Tips:[/bold]\n")
    console.print("• Use [cyan]--log_level DEBUG[/cyan] for detailed debugging")
    console.print("• Check [cyan]./output/logs/[/cyan] for comprehensive logs")
    console.print("• Compare runs in MLflow UI for hyperparameter tuning")
    console.print("• Use [cyan]--disable_mlflow[/cyan] for quick experiments")
    console.print(
        "• Generate reports with [cyan]--do_visualize[/cyan] for presentations"
    )

    # Environment variables
    console.print("\n[bold]🔧 Environment Variables:[/bold]\n")
    console.print("Create a [cyan].env[/cyan] file with:")
    console.print(
        Panel(
            Syntax(
                "# MLflow settings\n"
                "MLFLOW_TRACKING_URI=./output/mlruns\n"
                "MLFLOW_EXPERIMENT_NAME=titanic_experiments\n"
                "\n"
                "# Logging settings\n"
                "LOG_LEVEL=INFO\n"
                "LOG_TO_FILE=true",
                "bash",
                theme="monokai",
                line_numbers=False,
            ),
            border_style="dim",
        )
    )


if __name__ == "__main__":
    demo_mlflow_features()
