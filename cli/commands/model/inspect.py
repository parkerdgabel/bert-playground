"""Model inspection command - inspect model architecture and parameters."""

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import typer

from ...utils import (
    create_table,
    format_bytes,
    get_console,
    handle_errors,
    print_error,
    print_success,
    print_warning,
    requires_project,
    validate_path,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = get_console()


@handle_errors
@requires_project()
def inspect_command(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    component: str | None = typer.Option(
        None, "--component", "-C", help="Specific component to inspect"
    ),
    layer: int | None = typer.Option(
        None, "--layer", "-l", help="Specific layer number to inspect"
    ),
    show_weights: bool = typer.Option(
        False, "--weights", "-w", help="Show weight statistics"
    ),
    show_architecture: bool = typer.Option(
        True, "--arch/--no-arch", help="Show model architecture"
    ),
    show_config: bool = typer.Option(
        True, "--config/--no-config", help="Show model configuration"
    ),
    show_memory: bool = typer.Option(
        False, "--memory", "-m", help="Show memory usage estimates"
    ),
    show_computation: bool = typer.Option(
        False, "--computation", help="Show computational complexity"
    ),
    export_json: Path | None = typer.Option(
        None, "--export", "-e", help="Export inspection to JSON"
    ),
    format: str = typer.Option(
        "tree", "--format", "-f", help="Display format: tree, table, json"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
):
    """Inspect model architecture, parameters, and configuration.

    This command provides detailed information about a trained model,
    including architecture, parameter counts, and memory usage.

    Examples:
        # Basic inspection
        bert model inspect -c output/run_001/best_model

        # Inspect specific layer
        bert model inspect -c output/run_001/best_model --layer 5

        # Show weight statistics
        bert model inspect -c output/run_001/best_model --weights

        # Show memory and computation analysis
        bert model inspect -c output/run_001/best_model --memory --computation

        # Export detailed inspection
        bert model inspect -c output/run_001/best_model --export analysis.json
    """
    console.print("\n[bold blue]Model Inspection[/bold blue]")
    console.print(f"Checkpoint: {checkpoint}")
    console.print()

    try:
        # Load model information
        model_info = _load_model_info(checkpoint, verbose)

        # Display configuration
        if show_config:
            _display_config(model_info["config"])

        # Display architecture
        if show_architecture:
            _display_architecture(model_info, component, layer, format)

        # Display weight statistics
        if show_weights:
            _display_weights(model_info, component, layer, verbose)

        # Display memory usage
        if show_memory:
            _display_memory(model_info)

        # Display computational complexity
        if show_computation:
            _display_computation(model_info)

        # Export if requested
        if export_json:
            _export_inspection(model_info, export_json)
            print_success(f"Inspection exported to: {export_json}")

    except Exception as e:
        print_error(f"Inspection failed: {e}")
        raise typer.Exit(1)


def _load_model_info(checkpoint: Path, verbose: bool) -> dict[str, Any]:
    """Load comprehensive model information."""
    import mlx.core as mx

    info = {
        "checkpoint": str(checkpoint),
        "config": {},
        "architecture": {},
        "weights": {},
        "metadata": {},
    }

    # Load config
    config_path = checkpoint / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            info["config"] = json.load(f)
    else:
        raise FileNotFoundError(f"Config not found at {config_path}")

    # Load weights for inspection
    weights_path = checkpoint / "model.safetensors"
    if weights_path.exists():
        from safetensors import safe_open

        weights = {}
        with safe_open(weights_path, framework="mlx") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                weights[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size": tensor.nbytes,
                    "stats": _compute_tensor_stats(tensor) if verbose else {},
                }
        info["weights"] = weights
    else:
        # Try npz format
        weights_path = checkpoint / "model.npz"
        if weights_path.exists():
            weights_data = mx.load(str(weights_path))
            weights = {}
            for key, tensor in weights_data.items():
                weights[key] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "size": tensor.nbytes,
                    "stats": _compute_tensor_stats(tensor) if verbose else {},
                }
            info["weights"] = weights

    # Build architecture info
    info["architecture"] = _analyze_architecture(info["config"], info["weights"])

    # Load metadata if available
    metadata_paths = [
        checkpoint / "training_info.json",
        checkpoint / "metrics.json",
        checkpoint / "export_metadata.json",
    ]

    for path in metadata_paths:
        if path.exists():
            with open(path) as f:
                info["metadata"][path.name] = json.load(f)

    return info


def _compute_tensor_stats(tensor) -> dict[str, float]:
    """Compute statistics for a tensor."""
    import mlx.core as mx

    return {
        "mean": float(mx.mean(tensor).item()),
        "std": float(mx.std(tensor).item()),
        "min": float(mx.min(tensor).item()),
        "max": float(mx.max(tensor).item()),
        "norm": float(mx.linalg.norm(tensor).item()),
        "sparsity": float(mx.sum(tensor == 0).item() / tensor.size),
    }


def _analyze_architecture(config: dict, weights: dict) -> dict:
    """Analyze model architecture from config and weights."""
    arch = {
        "model_type": config.get("model_type", "bert"),
        "total_parameters": 0,
        "trainable_parameters": 0,
        "layers": OrderedDict(),
        "components": OrderedDict(),
    }

    # Group weights by component
    for key, info in weights.items():
        parts = key.split(".")
        component = parts[0] if parts else "unknown"

        if component not in arch["components"]:
            arch["components"][component] = {"parameters": 0, "layers": OrderedDict()}

        # Calculate parameters
        param_count = 1
        for dim in info["shape"]:
            param_count *= dim

        arch["total_parameters"] += param_count
        arch["components"][component]["parameters"] += param_count

        # Organize by layer if applicable
        if "layer" in key or "layers" in key:
            # Extract layer number
            layer_num = None
            for part in parts:
                if part.isdigit():
                    layer_num = int(part)
                    break

            if layer_num is not None:
                layer_key = f"layer_{layer_num}"
                if layer_key not in arch["layers"]:
                    arch["layers"][layer_key] = {"parameters": 0, "weights": []}
                arch["layers"][layer_key]["parameters"] += param_count
                arch["layers"][layer_key]["weights"].append(key)

    arch["trainable_parameters"] = arch["total_parameters"]  # Assume all trainable

    return arch


def _display_config(config: dict):
    """Display model configuration."""
    console.print("\n[bold green]Model Configuration[/bold green]")

    table = create_table("Configuration Parameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="yellow")

    # Key parameters to display
    key_params = [
        ("model_type", "Model Type"),
        ("model_name", "Model Name"),
        ("hidden_size", "Hidden Size"),
        ("num_hidden_layers", "Layers"),
        ("num_attention_heads", "Attention Heads"),
        ("intermediate_size", "Intermediate Size"),
        ("max_position_embeddings", "Max Position"),
        ("vocab_size", "Vocabulary Size"),
        ("type_vocab_size", "Type Vocab Size"),
        ("hidden_dropout_prob", "Dropout"),
        ("attention_probs_dropout_prob", "Attention Dropout"),
        ("classifier_dropout", "Classifier Dropout"),
        ("num_labels", "Number of Labels"),
    ]

    for key, display_name in key_params:
        value = config.get(key, "N/A")
        if isinstance(value, float):
            value = f"{value:.3f}"
        table.add_row(display_name, str(value))

    console.print(table)


def _display_architecture(
    info: dict, component: str | None, layer: int | None, format: str
):
    """Display model architecture."""
    console.print("\n[bold green]Model Architecture[/bold green]")

    arch = info["architecture"]

    # Summary
    console.print(f"Total parameters: [yellow]{arch['total_parameters']:,}[/yellow]")
    console.print(
        f"Trainable parameters: [green]{arch['trainable_parameters']:,}[/green]"
    )
    console.print()

    if format == "tree":
        _display_architecture_tree(arch, component, layer)
    elif format == "table":
        _display_architecture_table(arch, component, layer)
    elif format == "json":
        console.print_json(data=arch)


def _display_architecture_tree(arch: dict, component: str | None, layer: int | None):
    """Display architecture as tree."""
    from rich.tree import Tree

    tree = Tree("[bold]Model Architecture[/bold]")

    # Add components
    for comp_name, comp_info in arch["components"].items():
        if component and comp_name != component:
            continue

        comp_node = tree.add(
            f"[cyan]{comp_name}[/cyan] "
            f"([yellow]{comp_info['parameters']:,}[/yellow] params)"
        )

        # Add layers if any
        if comp_name in ["encoder", "bert", "roberta"]:
            for layer_name, layer_info in arch["layers"].items():
                layer_num = int(layer_name.split("_")[1])
                if layer is not None and layer_num != layer:
                    continue

                layer_node = comp_node.add(
                    f"[green]{layer_name}[/green] "
                    f"([yellow]{layer_info['parameters']:,}[/yellow] params)"
                )

                # Add weight details
                for weight_name in layer_info["weights"][:5]:  # Show first 5
                    weight_info = info["weights"].get(weight_name, {})
                    shape_str = "x".join(map(str, weight_info.get("shape", [])))
                    layer_node.add(f"[dim]{weight_name}: {shape_str}[/dim]")

                if len(layer_info["weights"]) > 5:
                    layer_node.add(
                        f"[dim]... and {len(layer_info['weights']) - 5} more[/dim]"
                    )

    console.print(tree)


def _display_architecture_table(arch: dict, component: str | None, layer: int | None):
    """Display architecture as table."""
    table = create_table("Model Components")
    table.add_column("Component", style="cyan")
    table.add_column("Parameters", style="yellow")
    table.add_column("Percentage", style="green")

    total_params = arch["total_parameters"]

    for comp_name, comp_info in arch["components"].items():
        if component and comp_name != component:
            continue

        percentage = (comp_info["parameters"] / total_params) * 100
        table.add_row(comp_name, f"{comp_info['parameters']:,}", f"{percentage:.1f}%")

    console.print(table)


def _display_weights(
    info: dict, component: str | None, layer: int | None, verbose: bool
):
    """Display weight statistics."""
    console.print("\n[bold green]Weight Statistics[/bold green]")

    weights_to_analyze = []

    for key, weight_info in info["weights"].items():
        # Filter by component
        if component and not key.startswith(component):
            continue

        # Filter by layer
        if layer is not None:
            if f"layers.{layer}." not in key and f"layer.{layer}." not in key:
                continue

        weights_to_analyze.append((key, weight_info))

    if not weights_to_analyze:
        print_warning("No weights match the specified filters")
        return

    table = create_table(f"Weight Statistics ({len(weights_to_analyze)} tensors)")
    table.add_column("Weight", style="cyan", overflow="fold")
    table.add_column("Shape", style="yellow")
    table.add_column("Size", style="blue")

    if verbose and weights_to_analyze[0][1].get("stats"):
        table.add_column("Mean", style="green")
        table.add_column("Std", style="green")
        table.add_column("Sparsity", style="magenta")

    total_size = 0

    for key, weight_info in weights_to_analyze[:20]:  # Show first 20
        shape_str = "x".join(map(str, weight_info["shape"]))
        size = weight_info["size"]
        total_size += size

        row = [key, shape_str, format_bytes(size)]

        if verbose and weight_info.get("stats"):
            stats = weight_info["stats"]
            row.extend(
                [
                    f"{stats['mean']:.3f}",
                    f"{stats['std']:.3f}",
                    f"{stats['sparsity']:.1%}",
                ]
            )

        table.add_row(*row)

    if len(weights_to_analyze) > 20:
        table.add_row("[dim]...[/dim]", "[dim]...[/dim]", "[dim]...[/dim]")

    console.print(table)
    console.print(f"\nTotal size: [yellow]{format_bytes(total_size)}[/yellow]")


def _display_memory(info: dict):
    """Display memory usage estimates."""
    console.print("\n[bold green]Memory Usage Estimates[/bold green]")

    config = info["config"]

    # Calculate memory requirements
    batch_sizes = [1, 8, 16, 32, 64]
    seq_lengths = [128, 256, 512]

    table = create_table("Memory Requirements")
    table.add_column("Batch Size", style="cyan")
    table.add_column("Seq Length", style="yellow")
    table.add_column("Activations", style="blue")
    table.add_column("Total (Est.)", style="green")

    # Model size
    model_size = sum(w["size"] for w in info["weights"].values())

    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            # Estimate activation memory
            hidden_size = config.get("hidden_size", 768)
            num_layers = config.get("num_hidden_layers", 12)

            # Rough estimation of activation memory
            activation_size = (
                batch_size
                * seq_length
                * hidden_size
                * 4  # embeddings
                * (2 + num_layers * 4)  # attention + FFN per layer
            )

            total_size = model_size + activation_size

            table.add_row(
                str(batch_size),
                str(seq_length),
                format_bytes(activation_size),
                format_bytes(total_size),
            )

    console.print(table)
    console.print(f"\nModel weights: [yellow]{format_bytes(model_size)}[/yellow]")


def _display_computation(info: dict):
    """Display computational complexity estimates."""
    console.print("\n[bold green]Computational Complexity[/bold green]")

    config = info["config"]

    # Extract model dimensions
    hidden_size = config.get("hidden_size", 768)
    num_layers = config.get("num_hidden_layers", 12)
    num_heads = config.get("num_attention_heads", 12)
    intermediate_size = config.get("intermediate_size", 3072)
    vocab_size = config.get("vocab_size", 30522)

    # Calculate FLOPs for different operations
    table = create_table("Computational Complexity (per token)")
    table.add_column("Operation", style="cyan")
    table.add_column("FLOPs", style="yellow")
    table.add_column("Percentage", style="green")

    operations = {
        "Embeddings": 2 * vocab_size * hidden_size,
        "Attention (per layer)": 4 * hidden_size * hidden_size
        + 2 * hidden_size * hidden_size,
        "FFN (per layer)": 2 * hidden_size * intermediate_size
        + 2 * intermediate_size * hidden_size,
        "LayerNorm (per layer)": 5 * hidden_size,
    }

    total_flops = operations["Embeddings"]
    for _ in range(num_layers):
        total_flops += operations["Attention (per layer)"]
        total_flops += operations["FFN (per layer)"]
        total_flops += operations["LayerNorm (per layer)"] * 2

    for op, flops in operations.items():
        if "per layer" in op:
            flops *= num_layers
        percentage = (flops / total_flops) * 100
        table.add_row(op, f"{flops:,}", f"{percentage:.1f}%")

    table.add_row(
        "[bold]Total[/bold]", f"[bold]{total_flops:,}[/bold]", "[bold]100.0%[/bold]"
    )

    console.print(table)

    # Show estimates for different sequence lengths
    console.print("\n[bold]Inference Time Estimates[/bold] (rough approximation)")
    console.print("[dim]Assuming 100 GFLOPS on Apple Silicon[/dim]")

    for seq_len in [128, 256, 512]:
        time_ms = (total_flops * seq_len) / (100e9) * 1000
        console.print(f"Sequence length {seq_len}: ~{time_ms:.1f}ms")


def _export_inspection(info: dict, output_path: Path):
    """Export inspection results to JSON."""
    # Convert non-serializable objects
    export_data = {
        "checkpoint": info["checkpoint"],
        "config": info["config"],
        "architecture": info["architecture"],
        "metadata": info["metadata"],
        "weight_summary": {
            "total_weights": len(info["weights"]),
            "total_size": sum(w["size"] for w in info["weights"].values()),
            "weights": {
                k: {"shape": v["shape"], "dtype": v["dtype"], "size": v["size"]}
                for k, v in info["weights"].items()
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
