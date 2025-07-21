"""Model export command - export models to different formats."""

import json
import shutil
import sys
from pathlib import Path

import typer

from ...utils import (
    get_console,
    handle_errors,
    print_error,
    print_info,
    print_success,
    print_warning,
    requires_project,
    track_time,
    validate_path,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

console = get_console()


@handle_errors
@requires_project()
@track_time("Exporting model")
def export_command(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Path to model checkpoint",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output path for exported model"
    ),
    format: str = typer.Option(
        "mlx", "--format", "-f", help="Export format: mlx, onnx, coreml, safetensors"
    ),
    model_name: str | None = typer.Option(
        None, "--name", "-n", help="Name for the exported model"
    ),
    quantize: bool = typer.Option(
        False, "--quantize", "-q", help="Quantize model (format-specific)"
    ),
    bits: int = typer.Option(8, "--bits", "-b", help="Quantization bits (4, 8, or 16)"),
    optimize: bool = typer.Option(
        True, "--optimize/--no-optimize", help="Apply format-specific optimizations"
    ),
    include_tokenizer: bool = typer.Option(
        True, "--tokenizer/--no-tokenizer", help="Include tokenizer with export"
    ),
    metadata: Path | None = typer.Option(
        None, "--metadata", "-m", help="JSON file with model metadata"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed export information"
    ),
):
    """Export trained models to different formats for deployment.
    
    This command supports exporting models to various formats:
    
    • MLX: Native MLX format (default)
    • ONNX: For cross-platform deployment
    • CoreML: For iOS/macOS deployment
    • SafeTensors: Efficient tensor serialization
    
    Examples:
        # Export to MLX format
        bert model export -c output/run_001/best_model -o exports/model.mlx
        
        # Export to ONNX with quantization
        bert model export -c output/run_001/best_model -o exports/model.onnx \\
            --format onnx --quantize --bits 8
        
        # Export to CoreML for iOS
        bert model export -c output/run_001/best_model -o exports/model.mlmodel \\
            --format coreml --optimize
        
        # Export with metadata
        bert model export -c output/run_001/best_model -o exports/model.mlx \\
            --metadata model_info.json --name "TitanicClassifier-v1.0"
    """
    # Validate format
    supported_formats = ["mlx", "onnx", "coreml", "safetensors"]
    if format.lower() not in supported_formats:
        print_error(
            f"Unsupported format: {format}. Supported: {', '.join(supported_formats)}"
        )
        raise typer.Exit(1)

    # Check if output exists
    if output_path.exists() and not force:
        print_error(
            f"Output path already exists: {output_path}\nUse --force to overwrite"
        )
        raise typer.Exit(1)

    # Load metadata if provided
    export_metadata = {}
    if metadata:
        try:
            with open(metadata) as f:
                export_metadata = json.load(f)
        except Exception as e:
            print_error(f"Failed to load metadata: {e}")
            raise typer.Exit(1)

    # Add export info to metadata
    export_metadata.update(
        {
            "export_format": format,
            "source_checkpoint": str(checkpoint),
            "model_name": model_name or checkpoint.name,
            "quantized": quantize,
            "quantization_bits": bits if quantize else None,
            "optimized": optimize,
        }
    )

    console.print("\n[bold blue]Model Export Configuration[/bold blue]")
    console.print(f"Source checkpoint: {checkpoint}")
    console.print(f"Output path: {output_path}")
    console.print(f"Format: {format.upper()}")
    if quantize:
        console.print(f"Quantization: {bits}-bit")
    console.print()

    # Export based on format
    try:
        if format == "mlx":
            _export_mlx(
                checkpoint, output_path, export_metadata, include_tokenizer, verbose
            )

        elif format == "onnx":
            _export_onnx(
                checkpoint,
                output_path,
                export_metadata,
                quantize,
                bits,
                optimize,
                verbose,
            )

        elif format == "coreml":
            _export_coreml(
                checkpoint, output_path, export_metadata, quantize, optimize, verbose
            )

        elif format == "safetensors":
            _export_safetensors(checkpoint, output_path, export_metadata, verbose)

        print_success(f"Model exported successfully to: {output_path}")

        # Save metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(export_metadata, f, indent=2)

        if verbose:
            console.print(f"\n[dim]Metadata saved to: {metadata_path}[/dim]")

    except Exception as e:
        print_error(f"Export failed: {e}")
        raise typer.Exit(1)


def _export_mlx(
    checkpoint: Path,
    output_path: Path,
    metadata: dict,
    include_tokenizer: bool,
    verbose: bool,
):
    """Export to native MLX format."""

    if verbose:
        console.print("[dim]Loading MLX checkpoint...[/dim]")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy model files
    model_files = ["model.safetensors", "config.json"]
    for file in model_files:
        src = checkpoint / file
        if src.exists():
            dst = output_path / file
            shutil.copy2(src, dst)
            if verbose:
                console.print(f"[dim]Copied {file}[/dim]")

    # Copy tokenizer if requested
    if include_tokenizer:
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
        ]
        for file in tokenizer_files:
            src = checkpoint / file
            if src.exists():
                dst = output_path / file
                shutil.copy2(src, dst)
                if verbose:
                    console.print(f"[dim]Copied {file}[/dim]")

    # Save export metadata
    metadata_path = output_path / "export_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def _export_onnx(
    checkpoint: Path,
    output_path: Path,
    metadata: dict,
    quantize: bool,
    bits: int,
    optimize: bool,
    verbose: bool,
):
    """Export to ONNX format."""
    print_warning("ONNX export is not yet implemented for MLX models")
    print_info("Consider using MLX format for deployment on Apple Silicon")

    # TODO: Implement ONNX export
    # This would require:
    # 1. Loading the MLX model
    # 2. Converting to PyTorch
    # 3. Exporting to ONNX
    # 4. Optionally quantizing
    raise typer.Exit(1)


def _export_coreml(
    checkpoint: Path,
    output_path: Path,
    metadata: dict,
    quantize: bool,
    optimize: bool,
    verbose: bool,
):
    """Export to CoreML format."""
    print_warning("CoreML export is not yet implemented")
    print_info("This feature is planned for a future release")

    # TODO: Implement CoreML export
    # This would require:
    # 1. Loading the MLX model
    # 2. Converting to CoreML using coremltools
    # 3. Applying optimizations
    raise typer.Exit(1)


def _export_safetensors(
    checkpoint: Path, output_path: Path, metadata: dict, verbose: bool
):
    """Export to SafeTensors format."""
    try:
        import mlx.core as mx
        import safetensors

        from models.modernbert_optimized import ModernBertModel

        if verbose:
            console.print("[dim]Loading model weights...[/dim]")

        # Load model weights
        weights_path = checkpoint / "model.safetensors"
        if not weights_path.exists():
            # Try npz format
            weights_path = checkpoint / "model.npz"
            if weights_path.exists():
                weights = mx.load(str(weights_path))
            else:
                raise FileNotFoundError("No model weights found")
        else:
            # Load safetensors
            from safetensors import safe_open

            weights = {}
            with safe_open(weights_path, framework="mlx") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)

        # Create output directory if exporting to directory
        if output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
            save_path = output_path / "model.safetensors"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_path = output_path

        # Save weights
        from safetensors.mlx import save_file

        save_file(weights, str(save_path), metadata=metadata)

        if verbose:
            console.print(f"[dim]Saved SafeTensors to: {save_path}[/dim]")

        # Copy config
        config_src = checkpoint / "config.json"
        if config_src.exists():
            config_dst = save_path.parent / "config.json"
            shutil.copy2(config_src, config_dst)

    except ImportError:
        print_error("SafeTensors not installed. Run: uv pip install safetensors")
        raise typer.Exit(1)
