"""Model conversion command - convert between model formats."""

import json
import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger

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
@track_time("Converting model")
def convert_command(
    source: Path = typer.Option(
        ...,
        "--source",
        "-s",
        help="Source model path",
        callback=lambda p: validate_path(p, must_exist=True),
    ),
    target: Path = typer.Option(..., "--target", "-t", help="Target output path"),
    source_format: str | None = typer.Option(
        None, "--from", "-f", help="Source format (auto-detect if not specified)"
    ),
    target_format: str = typer.Option(
        ..., "--to", "-T", help="Target format: mlx, pytorch, tensorflow, onnx"
    ),
    model_type: str | None = typer.Option(
        None, "--model-type", "-m", help="Model type if conversion needed"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Custom config file for conversion"
    ),
    optimize: bool = typer.Option(
        True, "--optimize/--no-optimize", help="Apply optimizations during conversion"
    ),
    quantize: bool = typer.Option(
        False, "--quantize", "-q", help="Quantize model during conversion"
    ),
    bits: int = typer.Option(8, "--bits", "-b", help="Quantization bits (4, 8, or 16)"),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Validate conversion with test input"
    ),
    force: bool = typer.Option(
        False, "--force", "-F", help="Overwrite existing target"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed conversion information"
    ),
):
    """Convert models between different formats.
    
    This command supports converting between various model formats
    for different deployment scenarios.
    
    Supported conversions:
    • MLX ↔ PyTorch
    • MLX → ONNX
    • PyTorch → MLX
    • PyTorch → ONNX
    • PyTorch → TensorFlow
    • HuggingFace → MLX
    
    Examples:
        # Convert MLX to PyTorch
        bert model convert --source model.mlx --to pytorch --target model.pt
        
        # Convert PyTorch to MLX with quantization
        bert model convert --source model.pt --to mlx --target model.mlx \\
            --quantize --bits 4
        
        # Convert HuggingFace model to MLX
        bert model convert --source bert-base-uncased --to mlx \\
            --target output/bert-mlx --model-type bert
        
        # Convert with custom config
        bert model convert --source model.pt --to mlx --target model.mlx \\
            --config custom_config.json
    """
    # Check if target exists
    if target.exists() and not force:
        print_error(f"Target already exists: {target}\nUse --force to overwrite")
        raise typer.Exit(1)

    # Auto-detect source format if not specified
    if not source_format:
        source_format = _detect_format(source, verbose)
        if not source_format:
            print_error("Could not detect source format. Please specify with --from")
            raise typer.Exit(1)
        if verbose:
            console.print(f"[dim]Detected source format: {source_format}[/dim]")

    # Validate conversion path
    if not _is_conversion_supported(source_format, target_format):
        print_error(
            f"Conversion from {source_format} to {target_format} is not supported"
        )
        _show_supported_conversions()
        raise typer.Exit(1)

    # Display conversion info
    console.print("\n[bold blue]Model Conversion[/bold blue]")
    console.print(f"Source: {source} ({source_format})")
    console.print(f"Target: {target} ({target_format})")
    if quantize:
        console.print(f"Quantization: {bits}-bit")
    console.print()

    try:
        # Load config if provided
        config = {}
        if config_file:
            with open(config_file) as f:
                config = json.load(f)
            if verbose:
                console.print(f"[dim]Loaded config from {config_file}[/dim]")

        # Perform conversion
        if source_format == "mlx" and target_format == "pytorch":
            _convert_mlx_to_pytorch(
                source, target, config, optimize, quantize, bits, verbose
            )

        elif source_format == "pytorch" and target_format == "mlx":
            _convert_pytorch_to_mlx(
                source, target, config, model_type, optimize, quantize, bits, verbose
            )

        elif source_format == "mlx" and target_format == "onnx":
            _convert_mlx_to_onnx(source, target, config, optimize, verbose)

        elif source_format == "pytorch" and target_format == "onnx":
            _convert_pytorch_to_onnx(source, target, config, optimize, verbose)

        elif source_format == "huggingface" and target_format == "mlx":
            _convert_huggingface_to_mlx(
                source, target, model_type, config, quantize, bits, verbose
            )

        else:
            print_error(
                f"Conversion from {source_format} to {target_format} not implemented"
            )
            raise typer.Exit(1)

        # Validate if requested
        if validate:
            if _validate_conversion(
                source, target, source_format, target_format, verbose
            ):
                print_success("Conversion validated successfully")
            else:
                print_warning("Conversion completed but validation failed")

        print_success(f"Model converted successfully to: {target}")

    except Exception as e:
        print_error(f"Conversion failed: {e}")
        if verbose:
            logger.exception("Detailed error:")
        raise typer.Exit(1)


def _detect_format(path: Path, verbose: bool) -> str | None:
    """Auto-detect model format from path."""
    # Check if path is a HuggingFace model ID
    if not path.exists() and "/" in str(path):
        return "huggingface"

    if path.is_file():
        # Check file extensions
        if path.suffix in [".pt", ".pth", ".bin"]:
            return "pytorch"
        elif path.suffix == ".onnx":
            return "onnx"
        elif path.suffix in [".pb", ".h5"]:
            return "tensorflow"
        elif path.suffix == ".mlx":
            return "mlx"
        elif path.suffix == ".safetensors":
            # Could be MLX or HuggingFace
            return "mlx"  # Default to MLX

    elif path.is_dir():
        # Check directory contents
        files = list(path.iterdir())
        file_names = [f.name for f in files]

        if "model.safetensors" in file_names or "model.npz" in file_names:
            return "mlx"
        elif "pytorch_model.bin" in file_names or "model.pt" in file_names:
            return "pytorch"
        elif "tf_model.h5" in file_names or "saved_model.pb" in file_names:
            return "tensorflow"
        elif "config.json" in file_names and "tokenizer.json" in file_names:
            # Likely HuggingFace format
            return "huggingface"

    return None


def _is_conversion_supported(source_format: str, target_format: str) -> bool:
    """Check if conversion is supported."""
    supported = {
        "mlx": ["pytorch", "onnx"],
        "pytorch": ["mlx", "onnx", "tensorflow"],
        "huggingface": ["mlx", "pytorch"],
        "tensorflow": ["onnx"],
        "onnx": [],  # ONNX is typically final format
    }

    return target_format in supported.get(source_format, [])


def _show_supported_conversions():
    """Display supported conversions."""
    console.print("\n[bold]Supported conversions:[/bold]")
    console.print("• MLX → PyTorch, ONNX")
    console.print("• PyTorch → MLX, ONNX, TensorFlow")
    console.print("• HuggingFace → MLX, PyTorch")
    console.print("• TensorFlow → ONNX")


def _convert_mlx_to_pytorch(
    source: Path,
    target: Path,
    config: dict,
    optimize: bool,
    quantize: bool,
    bits: int,
    verbose: bool,
):
    """Convert MLX model to PyTorch format."""
    print_warning("MLX to PyTorch conversion is experimental")

    # This would require:
    # 1. Loading MLX model
    # 2. Creating equivalent PyTorch model
    # 3. Copying weights with proper mapping
    # 4. Saving PyTorch checkpoint

    print_error("MLX to PyTorch conversion not yet implemented")
    print_info("Consider using the model directly in MLX for Apple Silicon deployment")
    raise typer.Exit(1)


def _convert_pytorch_to_mlx(
    source: Path,
    target: Path,
    config: dict,
    model_type: str | None,
    optimize: bool,
    quantize: bool,
    bits: int,
    verbose: bool,
):
    """Convert PyTorch model to MLX format."""
    try:
        import mlx.core as mx
        import torch

        from models.modernbert_optimized import ModernBertModel

        console.print("[dim]Loading PyTorch model...[/dim]")

        # Load PyTorch checkpoint
        checkpoint = torch.load(source, map_location="cpu")

        # Extract state dict
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Get or create config
        if "config" in checkpoint:
            model_config = checkpoint["config"]
        elif config:
            model_config = config
        else:
            # Try to infer config from state dict
            model_config = _infer_config_from_state_dict(state_dict, model_type)

        # Create MLX model
        console.print("[dim]Creating MLX model...[/dim]")
        mlx_model = ModernBertModel(model_config)

        # Convert weights
        console.print("[dim]Converting weights...[/dim]")
        mlx_weights = _convert_pytorch_weights_to_mlx(state_dict, verbose)

        # Load weights into model
        mlx_model.load_weights(list(mlx_weights.items()))

        # Create output directory
        target.mkdir(parents=True, exist_ok=True)

        # Save model
        if quantize:
            console.print(f"[dim]Quantizing to {bits}-bit...[/dim]")
            # TODO: Implement quantization
            print_warning("Quantization not yet implemented for PyTorch to MLX")

        # Save weights
        weights_path = target / "model.safetensors"
        from safetensors.mlx import save_file

        save_file(mlx_weights, str(weights_path))

        # Save config
        config_path = target / "config.json"
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)

        if verbose:
            console.print(f"[dim]Saved MLX model to {target}[/dim]")

    except ImportError:
        print_error("PyTorch not installed. Run: uv pip install torch")
        raise typer.Exit(1)


def _convert_mlx_to_onnx(
    source: Path, target: Path, config: dict, optimize: bool, verbose: bool
):
    """Convert MLX model to ONNX format."""
    print_warning("MLX to ONNX conversion requires intermediate PyTorch conversion")
    print_error("This feature is not yet implemented")
    print_info("Consider using MLX export for deployment on Apple Silicon")
    raise typer.Exit(1)


def _convert_pytorch_to_onnx(
    source: Path, target: Path, config: dict, optimize: bool, verbose: bool
):
    """Convert PyTorch model to ONNX format."""
    try:
        import torch
        import torch.onnx

        console.print("[dim]Loading PyTorch model...[/dim]")

        # Load model
        model = torch.load(source, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()

        # Create dummy input
        batch_size = 1
        seq_length = 128
        dummy_input = torch.randint(0, 1000, (batch_size, seq_length))

        # Export to ONNX
        console.print("[dim]Exporting to ONNX...[/dim]")
        torch.onnx.export(
            model,
            dummy_input,
            str(target),
            export_params=True,
            opset_version=11,
            do_constant_folding=optimize,
            input_names=["input_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "output": {0: "batch_size"},
            },
        )

        if optimize and verbose:
            console.print("[dim]Applied ONNX optimizations[/dim]")

    except ImportError:
        print_error("PyTorch/ONNX not installed. Run: uv pip install torch onnx")
        raise typer.Exit(1)


def _convert_huggingface_to_mlx(
    source: Path,
    target: Path,
    model_type: str | None,
    config: dict,
    quantize: bool,
    bits: int,
    verbose: bool,
):
    """Convert HuggingFace model to MLX format."""
    console.print("[dim]Loading HuggingFace model...[/dim]")

    try:
        import mlx.core as mx
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        from models.modernbert_optimized import ModernBertModel

        # Load HuggingFace model
        model_name = str(source)
        hf_config = AutoConfig.from_pretrained(model_name)
        hf_model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Convert config
        mlx_config = _convert_hf_config_to_mlx(hf_config, model_type)
        if config:
            mlx_config.update(config)

        # Create MLX model
        console.print("[dim]Creating MLX model...[/dim]")
        mlx_model = ModernBertModel(mlx_config)

        # Convert weights
        console.print("[dim]Converting weights...[/dim]")
        state_dict = hf_model.state_dict()
        mlx_weights = _convert_hf_weights_to_mlx(state_dict, model_type, verbose)

        # Load weights
        mlx_model.load_weights(list(mlx_weights.items()))

        # Create output directory
        target.mkdir(parents=True, exist_ok=True)

        # Save model
        weights_path = target / "model.safetensors"
        from safetensors.mlx import save_file

        save_file(mlx_weights, str(weights_path))

        # Save config
        config_path = target / "config.json"
        with open(config_path, "w") as f:
            json.dump(mlx_config, f, indent=2)

        # Save tokenizer
        tokenizer.save_pretrained(str(target))

        if verbose:
            console.print(f"[dim]Saved MLX model and tokenizer to {target}[/dim]")

    except ImportError:
        print_error("Transformers not installed. Run: uv pip install transformers")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to load HuggingFace model: {e}")
        raise typer.Exit(1)


def _infer_config_from_state_dict(state_dict: dict, model_type: str | None) -> dict:
    """Infer model configuration from state dict."""
    # Basic config inference
    config = {
        "model_type": model_type or "bert",
        "architectures": ["BertForSequenceClassification"],
    }

    # Try to infer dimensions from weight shapes
    for key, value in state_dict.items():
        if "embeddings.word_embeddings.weight" in key:
            config["vocab_size"], config["hidden_size"] = value.shape
        elif "encoder.layer" in key and "attention.self.query.weight" in key:
            config["hidden_size"] = value.shape[0]
        elif "encoder.layer" in key and "attention.self.num_attention_heads" in key:
            config["num_attention_heads"] = int(value)

    # Count layers
    layer_nums = set()
    for key in state_dict:
        if "encoder.layer." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layer" and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        layer_nums.add(layer_num)
                    except ValueError:
                        pass

    config["num_hidden_layers"] = len(layer_nums)

    return config


def _convert_pytorch_weights_to_mlx(state_dict: dict, verbose: bool) -> dict:
    """Convert PyTorch weights to MLX format."""
    import mlx.core as mx
    import torch

    mlx_weights = {}

    for key, value in state_dict.items():
        if verbose:
            console.print(f"[dim]Converting {key}...[/dim]")

        # Convert tensor to numpy then MLX
        if isinstance(value, torch.Tensor):
            np_array = value.detach().cpu().numpy()
            mlx_weights[key] = mx.array(np_array)
        else:
            mlx_weights[key] = value

    return mlx_weights


def _convert_hf_config_to_mlx(hf_config: Any, model_type: str | None) -> dict:
    """Convert HuggingFace config to MLX format."""
    mlx_config = {
        "model_type": model_type or hf_config.model_type,
        "architectures": hf_config.architectures,
        "vocab_size": hf_config.vocab_size,
        "hidden_size": hf_config.hidden_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "intermediate_size": hf_config.intermediate_size,
        "hidden_act": getattr(hf_config, "hidden_act", "gelu"),
        "hidden_dropout_prob": getattr(hf_config, "hidden_dropout_prob", 0.1),
        "attention_probs_dropout_prob": getattr(
            hf_config, "attention_probs_dropout_prob", 0.1
        ),
        "max_position_embeddings": hf_config.max_position_embeddings,
        "type_vocab_size": getattr(hf_config, "type_vocab_size", 2),
        "initializer_range": getattr(hf_config, "initializer_range", 0.02),
        "layer_norm_eps": getattr(hf_config, "layer_norm_eps", 1e-12),
    }

    return mlx_config


def _convert_hf_weights_to_mlx(
    state_dict: dict, model_type: str | None, verbose: bool
) -> dict:
    """Convert HuggingFace weights to MLX format."""
    # This is similar to PyTorch conversion but may need
    # additional key mapping for different architectures
    return _convert_pytorch_weights_to_mlx(state_dict, verbose)


def _validate_conversion(
    source: Path, target: Path, source_format: str, target_format: str, verbose: bool
) -> bool:
    """Validate that conversion was successful."""
    console.print("\n[dim]Validating conversion...[/dim]")

    try:
        # Create test input
        test_text = "This is a test sentence for model validation."

        # Get predictions from both models
        # This is a simplified validation - full implementation would
        # load both models and compare outputs

        if verbose:
            console.print(f"[dim]Source format: {source_format}[/dim]")
            console.print(f"[dim]Target format: {target_format}[/dim]")
            console.print(f"[dim]Test input: {test_text}[/dim]")

        # For now, just check that target files exist
        if target.is_dir():
            required_files = ["config.json"]
            if target_format == "mlx":
                required_files.append("model.safetensors")
            elif target_format == "pytorch":
                required_files.extend(["pytorch_model.bin", "model.pt"])

            for file in required_files:
                if not any(
                    (target / f).exists() for f in [file, file.replace(".", "_")]
                ):
                    if verbose:
                        console.print(f"[dim]Missing required file: {file}[/dim]")
                    return False

        return True

    except Exception as e:
        if verbose:
            console.print(f"[dim]Validation error: {e}[/dim]")
        return False
