"""Utility fixtures for model testing."""

from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import time
import json
import tempfile


def compare_model_outputs(
    output1: Dict[str, mx.array],
    output2: Dict[str, mx.array],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """Compare two model outputs for equality."""
    if set(output1.keys()) != set(output2.keys()):
        return False, {"error": "Different output keys"}
    
    differences = {}
    all_close = True
    
    for key in output1:
        if not mx.allclose(output1[key], output2[key], rtol=rtol, atol=atol):
            all_close = False
            # Calculate max difference
            diff = mx.abs(output1[key] - output2[key])
            differences[key] = float(mx.max(diff))
    
    return all_close, differences


def compare_model_parameters(
    model1: nn.Module,
    model2: nn.Module,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[bool, Dict[str, float]]:
    """Compare parameters of two models."""
    params1 = model1.parameters()
    params2 = model2.parameters()
    
    # Flatten parameters
    flat1 = dict(mx.tree_flatten(params1))
    flat2 = dict(mx.tree_flatten(params2))
    
    if set(flat1.keys()) != set(flat2.keys()):
        return False, {"error": "Different parameter names"}
    
    differences = {}
    all_close = True
    
    for name in flat1:
        param1 = flat1[name]
        param2 = flat2[name]
        
        if param1.shape != param2.shape:
            all_close = False
            differences[name] = f"Shape mismatch: {param1.shape} vs {param2.shape}"
        elif not mx.allclose(param1, param2, rtol=rtol, atol=atol):
            all_close = False
            diff = mx.abs(param1 - param2)
            differences[name] = float(mx.max(diff))
    
    return all_close, differences


def initialize_model_weights(
    model: nn.Module,
    init_type: str = "normal",
    init_range: float = 0.02,
    seed: Optional[int] = 42,
) -> nn.Module:
    """Initialize model weights with specific strategy."""
    if seed is not None:
        mx.random.seed(seed)
    
    def init_fn(module, name, param):
        if "weight" in name:
            if init_type == "normal":
                return mx.random.normal(param.shape) * init_range
            elif init_type == "uniform":
                return mx.random.uniform(-init_range, init_range, param.shape)
            elif init_type == "xavier":
                # Xavier/Glorot initialization
                fan_in = param.shape[0] if len(param.shape) > 1 else 1
                fan_out = param.shape[1] if len(param.shape) > 1 else 1
                scale = mx.sqrt(2.0 / (fan_in + fan_out))
                return mx.random.normal(param.shape) * scale
            elif init_type == "zeros":
                return mx.zeros(param.shape)
            elif init_type == "ones":
                return mx.ones(param.shape)
        elif "bias" in name:
            return mx.zeros(param.shape)
        return param
    
    model.apply(init_fn)
    return model


def check_gradient_flow(
    model: nn.Module,
    loss_fn: Callable,
    input_data: Dict[str, mx.array],
    check_nan: bool = True,
    check_zero: bool = True,
) -> Dict[str, Any]:
    """Check gradient flow through model."""
    # Forward and backward pass
    loss, grads = mx.value_and_grad(loss_fn)(model, input_data)
    
    results = {
        "loss": float(loss),
        "loss_is_finite": bool(mx.isfinite(loss)),
        "parameters": {},
    }
    
    # Check each parameter
    flat_params = dict(mx.tree_flatten(model.parameters()))
    flat_grads = dict(mx.tree_flatten(grads))
    
    for name in flat_params:
        param = flat_params[name]
        grad = flat_grads.get(name)
        
        param_info = {
            "shape": param.shape,
            "has_gradient": grad is not None,
        }
        
        if grad is not None:
            param_info.update({
                "grad_norm": float(mx.norm(grad)),
                "grad_max": float(mx.max(mx.abs(grad))),
                "grad_min": float(mx.min(mx.abs(grad))),
            })
            
            if check_nan:
                param_info["has_nan"] = bool(mx.any(mx.isnan(grad)))
            
            if check_zero:
                param_info["all_zero"] = bool(mx.all(grad == 0))
        
        results["parameters"][name] = param_info
    
    return results


def save_and_load_model(
    model: nn.Module,
    save_format: str = "safetensors",
    tmp_dir: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Test saving and loading a model."""
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Save model
        if save_format == "safetensors":
            save_path = tmp_dir / "model.safetensors"
            model.save_safetensors(str(save_path))
        elif save_format == "npz":
            save_path = tmp_dir / "model.npz"
            model.save_weights(str(save_path))
        else:
            return False, f"Unknown format: {save_format}"
        
        # Create new model instance
        new_model = type(model)(model.config if hasattr(model, "config") else None)
        
        # Load weights
        if save_format == "safetensors":
            new_model.load_safetensors(str(save_path))
        else:
            new_model.load_weights(str(save_path))
        
        # Compare parameters
        all_close, differences = compare_model_parameters(model, new_model)
        
        if not all_close:
            return False, f"Parameter mismatch: {differences}"
        
        return True, None
        
    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        if tmp_dir.exists():
            import shutil
            shutil.rmtree(tmp_dir)


def benchmark_model_forward(
    model: nn.Module,
    input_batch: Dict[str, mx.array],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """Benchmark model forward pass performance."""
    # Warmup
    for _ in range(warmup_iterations):
        _ = model(**input_batch)
        mx.eval(model.parameters())
    
    # Timing
    times = []
    for _ in range(num_iterations):
        start = time.time()
        output = model(**input_batch)
        mx.eval(output)  # Force evaluation
        end = time.time()
        times.append(end - start)
    
    # Calculate statistics
    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # Calculate throughput
    batch_size = input_batch["input_ids"].shape[0] if "input_ids" in input_batch else 1
    samples_per_second = batch_size / mean_time
    
    return {
        "mean_time": mean_time,
        "min_time": min_time,
        "max_time": max_time,
        "std_time": mx.std(mx.array(times)).item(),
        "samples_per_second": samples_per_second,
        "iterations": num_iterations,
    }


def profile_model_memory(
    model: nn.Module,
    input_batch: Dict[str, mx.array],
    include_gradients: bool = True,
) -> Dict[str, Any]:
    """Profile model memory usage."""
    # Get initial memory
    mx.metal.clear_cache()
    initial_memory = mx.metal.get_active_memory()
    
    # Forward pass
    output = model(**input_batch)
    mx.eval(output)
    forward_memory = mx.metal.get_active_memory()
    
    results = {
        "initial_memory": initial_memory,
        "forward_memory": forward_memory,
        "forward_memory_used": forward_memory - initial_memory,
    }
    
    if include_gradients:
        # Backward pass
        def loss_fn(model, batch):
            output = model(**batch)
            if isinstance(output, dict):
                output = output.get("logits", output.get("last_hidden_state"))
            return mx.mean(output)
        
        loss, grads = mx.value_and_grad(loss_fn)(model, input_batch)
        mx.eval(grads)
        backward_memory = mx.metal.get_active_memory()
        
        results.update({
            "backward_memory": backward_memory,
            "backward_memory_used": backward_memory - forward_memory,
            "total_memory_used": backward_memory - initial_memory,
        })
    
    # Count parameters
    param_count = sum(p.size for _, p in mx.tree_flatten(model.parameters()))
    param_memory = param_count * 4  # Assuming float32
    
    results.update({
        "parameter_count": param_count,
        "parameter_memory": param_memory,
    })
    
    mx.metal.clear_cache()
    return results


def create_test_checkpoint(
    model: nn.Module,
    checkpoint_dir: Path,
    epoch: int = 1,
    step: int = 100,
    metrics: Optional[Dict[str, float]] = None,
) -> Path:
    """Create a test checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = checkpoint_dir / "model.safetensors"
    model.save_safetensors(str(model_path))
    
    # Save metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "model_type": type(model).__name__,
        "timestamp": time.time(),
    }
    
    if metrics:
        metadata["metrics"] = metrics
    
    if hasattr(model, "config"):
        metadata["config"] = model.config.to_dict() if hasattr(model.config, "to_dict") else {}
    
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return checkpoint_dir


def verify_model_outputs(
    model: nn.Module,
    test_cases: List[Dict[str, mx.array]],
    expected_keys: List[str],
    check_shapes: bool = True,
    check_values: bool = True,
) -> Tuple[bool, List[str]]:
    """Verify model outputs against expectations."""
    errors = []
    
    for i, test_case in enumerate(test_cases):
        try:
            output = model(**test_case)
            
            # Check output type
            if not isinstance(output, (dict, mx.array)):
                errors.append(f"Test case {i}: Invalid output type {type(output)}")
                continue
            
            # Convert to dict if needed
            if isinstance(output, mx.array):
                output = {"output": output}
            
            # Check expected keys
            for key in expected_keys:
                if key not in output:
                    errors.append(f"Test case {i}: Missing expected key '{key}'")
            
            # Check shapes
            if check_shapes:
                for key, value in output.items():
                    if not isinstance(value, mx.array):
                        errors.append(f"Test case {i}: Output '{key}' is not an array")
                    elif len(value.shape) == 0:
                        errors.append(f"Test case {i}: Output '{key}' is scalar")
            
            # Check values
            if check_values:
                for key, value in output.items():
                    if isinstance(value, mx.array):
                        if mx.any(mx.isnan(value)):
                            errors.append(f"Test case {i}: Output '{key}' contains NaN")
                        if mx.any(mx.isinf(value)):
                            errors.append(f"Test case {i}: Output '{key}' contains Inf")
            
        except Exception as e:
            errors.append(f"Test case {i}: Exception {type(e).__name__}: {str(e)}")
    
    return len(errors) == 0, errors


def create_model_comparison_report(
    model1: nn.Module,
    model2: nn.Module,
    test_batch: Dict[str, mx.array],
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> Dict[str, Any]:
    """Create detailed comparison report between two models."""
    report = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "comparison_time": time.time(),
    }
    
    # Compare architectures
    params1 = dict(mx.tree_flatten(model1.parameters()))
    params2 = dict(mx.tree_flatten(model2.parameters()))
    
    report["architecture"] = {
        "same_param_names": set(params1.keys()) == set(params2.keys()),
        "model1_params": len(params1),
        "model2_params": len(params2),
        "model1_total_size": sum(p.size for p in params1.values()),
        "model2_total_size": sum(p.size for p in params2.values()),
    }
    
    # Compare outputs
    output1 = model1(**test_batch)
    output2 = model2(**test_batch)
    
    if isinstance(output1, dict) and isinstance(output2, dict):
        outputs_match, output_diffs = compare_model_outputs(output1, output2)
        report["outputs"] = {
            "match": outputs_match,
            "differences": output_diffs,
        }
    
    # Compare performance
    perf1 = benchmark_model_forward(model1, test_batch, num_iterations=10)
    perf2 = benchmark_model_forward(model2, test_batch, num_iterations=10)
    
    report["performance"] = {
        f"{model1_name}_mean_time": perf1["mean_time"],
        f"{model2_name}_mean_time": perf2["mean_time"],
        "speedup": perf1["mean_time"] / perf2["mean_time"],
    }
    
    return report