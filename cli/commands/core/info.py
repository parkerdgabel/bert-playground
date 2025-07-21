"""System information command."""

import platform
import sys
from pathlib import Path

import mlx.core as mx
import typer

from ...utils import get_console, handle_errors
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@handle_errors
def info_command(
    mlx: bool = typer.Option(False, "--mlx", help="Show MLX information"),
    models: bool = typer.Option(False, "--models", help="List available models"),
    datasets: bool = typer.Option(False, "--datasets", help="List configured datasets"),
    mlflow: bool = typer.Option(False, "--mlflow", help="Show MLflow configuration"),
    embeddings: bool = typer.Option(
        False, "--embeddings", help="Show embeddings information"
    ),
    all: bool = typer.Option(False, "--all", "-a", help="Show all information"),
):
    """Display system and project information.

    This command shows various system information including hardware capabilities,
    installed packages, and project configuration.

    Examples:
        # Basic system info
        bert info

        # MLX-specific information
        bert info --mlx

        # Show everything
        bert info --all
    """
    console = get_console()

    console.print("\n[bold blue]System Information[/bold blue]")
    console.print("=" * 60)

    # Always show basic system info
    _show_system_info(console)

    # Show specific sections based on flags
    if mlx or all:
        _show_mlx_info(console)

    if models or all:
        _show_models_info(console)

    if datasets or all:
        _show_datasets_info(console)

    if mlflow or all:
        _show_mlflow_info(console)

    if embeddings or all:
        _show_embeddings_info(console)

    # Show project info if none selected
    if not any([mlx, models, datasets, mlflow, embeddings, all]):
        _show_project_info(console)


def _show_system_info(console):
    """Show basic system information."""
    info_table = create_table("System Information", ["Property", "Value"])

    # Platform info
    info_table.add_row("Platform", platform.platform())
    info_table.add_row("Architecture", platform.machine())
    info_table.add_row("Processor", platform.processor() or "Unknown")
    info_table.add_row("Python", platform.python_version())

    # Memory info
    try:
        import psutil

        mem = psutil.virtual_memory()
        info_table.add_row("System Memory", f"{mem.total / 1e9:.1f} GB")
        info_table.add_row(
            "Available Memory",
            f"{mem.available / 1e9:.1f} GB ({mem.percent:.1f}% used)",
        )

        # CPU info
        info_table.add_row("CPU Count", str(psutil.cpu_count(logical=False)))
        info_table.add_row("CPU Threads", str(psutil.cpu_count(logical=True)))
    except ImportError:
        info_table.add_row("Memory", "Install psutil for memory info")

    console.print(info_table)


def _show_mlx_info(console):
    """Show MLX-specific information."""
    console.print("\n[bold blue]MLX Information[/bold blue]")
    console.print("=" * 60)

    mlx_table = create_table("MLX Configuration", ["Property", "Value"])

    # MLX version
    try:
        import mlx

        mlx_table.add_row("MLX Version", mlx.__version__)
    except:
        mlx_table.add_row("MLX Version", "Not installed")

    # MLX device
    mlx_table.add_row("Default Device", str(mx.default_device()))

    # MLX settings
    try:
        mlx_table.add_row(
            "Metal Device", "Available" if mx.metal.is_available() else "Not available"
        )
    except:
        mlx_table.add_row("Metal Device", "Unknown")

    # Memory limits
    try:
        mlx_table.add_row("Memory Limit", f"{mx.metal.get_memory_limit() / 1e9:.1f} GB")
        mlx_table.add_row("Cache Limit", f"{mx.metal.get_cache_limit() / 1e9:.1f} GB")
    except:
        pass

    console.print(mlx_table)

    # Check MLX embeddings
    try:
        import mlx_embeddings

        console.print("\n[green]✓ MLX Embeddings is installed[/green]")
    except ImportError:
        console.print("\n[yellow]⚠ MLX Embeddings not installed[/yellow]")
        console.print("  Install with: pip install mlx-embeddings")


def _show_models_info(console):
    """Show available models information."""
    console.print("\n[bold blue]Available Models[/bold blue]")
    console.print("=" * 60)

    try:
        from models.factory import list_available_models
        from models.heads import list_available_heads

        # Show BERT models
        models_table = create_table("BERT Models", ["Model Type", "Description"])
        models_table.add_row("base", "Standard ModernBERT model")
        models_table.add_row("cnn_hybrid", "CNN-enhanced ModernBERT")
        models_table.add_row("mlx_embeddings", "MLX embeddings optimized model")
        console.print(models_table)

        # Show available heads
        console.print("\n[bold]Classification Heads:[/bold]")
        heads = list_available_heads()
        heads_table = create_table(
            "Available Heads", ["Head Type", "Competition Types", "Description"]
        )

        for head_name, specs in heads.items():
            for spec in specs:
                comp_types = ", ".join(spec.competition_types)
                heads_table.add_row(head_name, comp_types, spec.description)

        console.print(heads_table)

    except ImportError as e:
        console.print(f"[yellow]Could not load model information: {e}[/yellow]")


def _show_datasets_info(console):
    """Show configured datasets information."""
    console.print("\n[bold blue]Configured Datasets[/bold blue]")
    console.print("=" * 60)

    try:
        from data.datasets import DatasetRegistry

        registry = DatasetRegistry()
        datasets = registry.list_datasets()

        if datasets:
            datasets_table = create_table(
                "Available Datasets", ["Name", "Task Type", "Features", "Samples"]
            )

            for name, spec in datasets.items():
                datasets_table.add_row(
                    name,
                    spec.task_type,
                    str(len(spec.feature_columns)),
                    str(spec.num_samples) if spec.num_samples else "Unknown",
                )

            console.print(datasets_table)
        else:
            console.print("[yellow]No datasets configured[/yellow]")
            console.print("Add datasets to data/datasets.py or use universal loader")

    except Exception as e:
        console.print(f"[yellow]Could not load dataset information: {e}[/yellow]")


def _show_mlflow_info(console):
    """Show MLflow configuration."""
    console.print("\n[bold blue]MLflow Configuration[/bold blue]")
    console.print("=" * 60)

    try:
        from utils.mlflow_central import mlflow_central

        mlflow_central.initialize()

        mlflow_table = create_table("MLflow Settings", ["Property", "Value"])
        mlflow_table.add_row("Tracking URI", mlflow_central.tracking_uri)
        mlflow_table.add_row("Artifact Root", mlflow_central.artifact_root)
        mlflow_table.add_row("Default Experiment", mlflow_central.DEFAULT_EXPERIMENT)

        # Check if MLflow server is running
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_central.tracking_uri)
            experiments = mlflow.search_experiments(max_results=1)
            mlflow_table.add_row("Server Status", "[green]Connected[/green]")
            mlflow_table.add_row("Experiments", str(len(mlflow.search_experiments())))
        except:
            mlflow_table.add_row("Server Status", "[red]Not connected[/red]")
            mlflow_table.add_row("Start Server", "bert mlflow server start")

        console.print(mlflow_table)

    except Exception as e:
        console.print(f"[yellow]Could not load MLflow information: {e}[/yellow]")


def _show_embeddings_info(console):
    """Show embeddings information."""
    console.print("\n[bold blue]Embeddings Configuration[/bold blue]")
    console.print("=" * 60)

    try:
        from embeddings.config import MODEL_MAPPINGS

        embeddings_table = create_table(
            "MLX Embeddings Models", ["HuggingFace Model", "MLX Model"]
        )

        for hf_model, mlx_model in MODEL_MAPPINGS.items():
            embeddings_table.add_row(hf_model, mlx_model)

        console.print(embeddings_table)

        # Check tokenizer backends
        console.print("\n[bold]Tokenizer Backends:[/bold]")
        console.print("  • auto - Automatically select best backend")
        console.print("  • mlx - Use MLX tokenizer (requires mlx-embeddings)")
        console.print("  • huggingface - Use HuggingFace tokenizer")

    except Exception as e:
        console.print(f"[yellow]Could not load embeddings information: {e}[/yellow]")


def _show_project_info(console):
    """Show project-specific information."""
    console.print("\n[bold blue]Project Information[/bold blue]")
    console.print("=" * 60)

    # Find project root
    project_root = Path.cwd()
    while project_root != project_root.parent:
        if (project_root / "pyproject.toml").exists() or (
            project_root / ".git"
        ).exists():
            break
        project_root = project_root.parent

    project_table = create_table("Project Details", ["Property", "Value"])

    # Project location
    project_table.add_row("Project Root", str(project_root))

    # Check for key files
    key_files = {
        "Configuration": ["bert.yaml", "bert.yml", ".bertrc"],
        "Data": ["data/", "datasets/"],
        "Models": ["output/", "checkpoints/"],
        "Configs": ["configs/"],
    }

    for category, paths in key_files.items():
        found = []
        for path in paths:
            full_path = project_root / path
            if full_path.exists():
                found.append(path)

        if found:
            project_table.add_row(category, ", ".join(found))
        else:
            project_table.add_row(category, "[yellow]Not found[/yellow]")

    console.print(project_table)

    # Show quick tips
    console.print("\n[bold]Quick Tips:[/bold]")
    console.print("  • Use [cyan]bert info --all[/cyan] to see all information")
    console.print("  • Use [cyan]bert init[/cyan] to create a new project")
    console.print("  • Use [cyan]bert --help[/cyan] for available commands")
