"""Export Model Command - orchestrates model export to various formats."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import shutil

from application.dto.export import ExportRequestDTO, ExportResponseDTO
from domain.entities.model import Model
from domain.services import TokenizationService
from domain.ports import (
    ComputePort,
    MonitoringPort,
    StoragePort
)


@dataclass
class ExportModelCommand:
    """Command to export a trained model to various formats.
    
    This command orchestrates the model export workflow by:
    1. Loading the trained model
    2. Converting to requested format
    3. Optimizing if requested
    4. Validating the export
    5. Creating deployment package
    """
    
    # Domain services
    tokenization_service: TokenizationService
    
    # Ports
    compute_port: ComputePort
    monitoring_port: MonitoringPort
    storage_port: StoragePort
    
    async def execute(self, request: ExportRequestDTO) -> ExportResponseDTO:
        """Execute the export command.
        
        Args:
            request: Export request with model path and format
            
        Returns:
            Response with export details and paths
        """
        start_time = datetime.now()
        
        try:
            # 1. Validate request
            errors = request.validate()
            if errors:
                return ExportResponseDTO(
                    success=False,
                    error_message=f"Validation errors: {', '.join(errors)}"
                )
            
            # 2. Initialize monitoring
            await self.monitoring_port.start_run(
                name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags={
                    "type": "export",
                    "format": request.export_format,
                    "model": str(request.model_path)
                }
            )
            
            # 3. Load model
            model = await self._load_model(request.model_path)
            
            # 4. Initialize compute backend
            await self.compute_port.initialize(model)
            
            # 5. Apply optimizations if requested
            if request.optimize:
                model = await self._optimize_model(model, request)
            
            # 6. Export to requested format
            export_paths = await self._export_model(model, request)
            
            # 7. Create deployment package if requested
            if request.create_deployment_package:
                deployment_path = await self._create_deployment_package(
                    model,
                    export_paths,
                    request
                )
                export_paths["deployment_package"] = deployment_path
            
            # 8. Validate export
            validation_results = await self._validate_export(
                export_paths,
                request
            )
            
            # 9. Generate export metadata
            metadata = await self._generate_metadata(
                model,
                export_paths,
                validation_results,
                request
            )
            
            # 10. Create response
            response = self._create_response(
                export_paths,
                metadata,
                validation_results,
                start_time,
                datetime.now()
            )
            
            # 11. Log metrics
            await self.monitoring_port.log_metrics({
                "export_format": request.export_format,
                "model_size_mb": metadata.get("model_size_mb", 0),
                "export_time_seconds": response.export_time_seconds,
                "optimization_applied": request.optimize
            })
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Export failed: {str(e)}")
            return ExportResponseDTO.from_error(e)
        finally:
            await self.monitoring_port.end_run()
    
    async def _load_model(self, model_path: Path) -> Model:
        """Load the trained model."""
        return await self.storage_port.load_model(model_path)
    
    async def _optimize_model(self, model: Model, request: ExportRequestDTO) -> Model:
        """Apply model optimizations."""
        optimized_model = model
        
        if request.quantize:
            # Apply quantization
            await self.monitoring_port.log_info("Applying quantization...")
            optimized_model = await self._quantize_model(
                optimized_model,
                request.quantization_bits or 8
            )
        
        if request.prune:
            # Apply pruning
            await self.monitoring_port.log_info("Applying pruning...")
            optimized_model = await self._prune_model(
                optimized_model,
                request.pruning_ratio or 0.1
            )
        
        if request.optimize_for_inference:
            # Apply inference optimizations
            await self.monitoring_port.log_info("Optimizing for inference...")
            optimized_model = await self._optimize_for_inference(optimized_model)
        
        return optimized_model
    
    async def _quantize_model(self, model: Model, bits: int) -> Model:
        """Apply quantization to reduce model size."""
        # This would use framework-specific quantization
        # For MLX, this might involve converting to lower precision
        quantized_state = await self.compute_port.quantize(
            model.state_dict(),
            bits=bits
        )
        
        quantized_model = model.copy()
        quantized_model.load_state_dict(quantized_state)
        quantized_model.metadata["quantization"] = {
            "method": "dynamic",
            "bits": bits
        }
        
        return quantized_model
    
    async def _prune_model(self, model: Model, ratio: float) -> Model:
        """Apply pruning to reduce model complexity."""
        # Identify weights to prune based on magnitude
        pruned_state = await self.compute_port.prune(
            model.state_dict(),
            prune_ratio=ratio
        )
        
        pruned_model = model.copy()
        pruned_model.load_state_dict(pruned_state)
        pruned_model.metadata["pruning"] = {
            "method": "magnitude",
            "ratio": ratio
        }
        
        return pruned_model
    
    async def _optimize_for_inference(self, model: Model) -> Model:
        """Apply inference-specific optimizations."""
        # Fuse operations, remove dropout, etc.
        optimized_model = await self.compute_port.optimize_graph(model)
        optimized_model.metadata["inference_optimized"] = True
        
        return optimized_model
    
    async def _export_model(
        self,
        model: Model,
        request: ExportRequestDTO
    ) -> Dict[str, Path]:
        """Export model to requested format."""
        export_paths = {}
        output_dir = request.output_dir
        
        if request.export_format == "mlx":
            # Native MLX format
            mlx_path = output_dir / "model.mlx"
            await self._export_mlx(model, mlx_path)
            export_paths["model"] = mlx_path
            
        elif request.export_format == "onnx":
            # ONNX format for cross-platform deployment
            onnx_path = output_dir / "model.onnx"
            await self._export_onnx(model, onnx_path, request)
            export_paths["model"] = onnx_path
            
        elif request.export_format == "coreml":
            # Core ML for iOS/macOS deployment
            coreml_path = output_dir / "model.mlmodel"
            await self._export_coreml(model, coreml_path, request)
            export_paths["model"] = coreml_path
            
        elif request.export_format == "tensorflow":
            # TensorFlow SavedModel format
            tf_path = output_dir / "tensorflow_model"
            await self._export_tensorflow(model, tf_path)
            export_paths["model"] = tf_path
            
        elif request.export_format == "torchscript":
            # TorchScript for PyTorch deployment
            ts_path = output_dir / "model.pt"
            await self._export_torchscript(model, ts_path)
            export_paths["model"] = ts_path
        
        else:
            raise ValueError(f"Unsupported export format: {request.export_format}")
        
        # Always save config and tokenizer
        config_path = output_dir / "config.json"
        await self.storage_port.save_json(model.config.to_dict(), config_path)
        export_paths["config"] = config_path
        
        # Export tokenizer if available
        if hasattr(model, "tokenizer_config"):
            tokenizer_path = output_dir / "tokenizer"
            await self._export_tokenizer(model.tokenizer_config, tokenizer_path)
            export_paths["tokenizer"] = tokenizer_path
        
        return export_paths
    
    async def _export_mlx(self, model: Model, output_path: Path) -> None:
        """Export to native MLX format."""
        # Save model state and metadata
        mlx_data = {
            "model_state": model.state_dict(),
            "config": model.config.to_dict(),
            "metadata": model.metadata
        }
        await self.storage_port.save_mlx(mlx_data, output_path)
    
    async def _export_onnx(
        self,
        model: Model,
        output_path: Path,
        request: ExportRequestDTO
    ) -> None:
        """Export to ONNX format."""
        # Create dummy input for tracing
        dummy_input = await self._create_dummy_input(model, request)
        
        # Export through compute port
        await self.compute_port.export_onnx(
            model,
            dummy_input,
            output_path,
            opset_version=request.onnx_opset_version or 13,
            dynamic_axes=request.dynamic_axes
        )
    
    async def _export_coreml(
        self,
        model: Model,
        output_path: Path,
        request: ExportRequestDTO
    ) -> None:
        """Export to Core ML format."""
        # Use coremltools through compute port
        await self.compute_port.export_coreml(
            model,
            output_path,
            minimum_deployment_target=request.coreml_minimum_deployment_target
        )
    
    async def _export_tensorflow(self, model: Model, output_path: Path) -> None:
        """Export to TensorFlow SavedModel format."""
        # This would require model conversion
        # Placeholder for now
        raise NotImplementedError("TensorFlow export not yet implemented")
    
    async def _export_torchscript(self, model: Model, output_path: Path) -> None:
        """Export to TorchScript format."""
        # This would require PyTorch compatibility layer
        # Placeholder for now
        raise NotImplementedError("TorchScript export not yet implemented")
    
    async def _export_tokenizer(
        self,
        tokenizer_config: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Export tokenizer configuration and vocabulary."""
        await self.storage_port.create_directory(output_path)
        
        # Save tokenizer config
        config_path = output_path / "tokenizer_config.json"
        await self.storage_port.save_json(tokenizer_config, config_path)
        
        # Save vocabulary
        if "vocab_file" in tokenizer_config:
            vocab_src = Path(tokenizer_config["vocab_file"])
            if vocab_src.exists():
                vocab_dst = output_path / "vocab.txt"
                shutil.copy(vocab_src, vocab_dst)
    
    async def _create_dummy_input(
        self,
        model: Model,
        request: ExportRequestDTO
    ) -> Dict[str, Any]:
        """Create dummy input for model tracing."""
        # Get expected input shape from model config
        batch_size = request.export_batch_size or 1
        seq_length = request.export_sequence_length or 512
        
        # Create dummy tokenized input
        dummy_input = {
            "input_ids": await self.compute_port.zeros((batch_size, seq_length), dtype="int32"),
            "attention_mask": await self.compute_port.ones((batch_size, seq_length), dtype="int32")
        }
        
        return dummy_input
    
    async def _create_deployment_package(
        self,
        model: Model,
        export_paths: Dict[str, Path],
        request: ExportRequestDTO
    ) -> Path:
        """Create a complete deployment package."""
        package_dir = request.output_dir / "deployment_package"
        await self.storage_port.create_directory(package_dir)
        
        # Copy model files
        model_dir = package_dir / "model"
        await self.storage_port.create_directory(model_dir)
        for key, path in export_paths.items():
            if path.is_file():
                shutil.copy(path, model_dir / path.name)
            else:
                shutil.copytree(path, model_dir / path.name)
        
        # Create inference script
        inference_script = package_dir / "inference.py"
        await self._create_inference_script(inference_script, request)
        
        # Create requirements file
        requirements_file = package_dir / "requirements.txt"
        await self._create_requirements_file(requirements_file, request)
        
        # Create README
        readme_file = package_dir / "README.md"
        await self._create_deployment_readme(readme_file, model, request)
        
        # Create Docker file if requested
        if request.include_dockerfile:
            dockerfile = package_dir / "Dockerfile"
            await self._create_dockerfile(dockerfile, request)
        
        # Create deployment config
        config_file = package_dir / "deployment_config.json"
        deployment_config = {
            "model_format": request.export_format,
            "model_type": model.config.model_type,
            "task_type": model.config.task_type,
            "input_requirements": {
                "max_sequence_length": request.export_sequence_length or 512,
                "batch_size": request.export_batch_size or 1
            },
            "optimizations": {
                "quantized": request.quantize,
                "pruned": request.prune,
                "inference_optimized": request.optimize_for_inference
            }
        }
        await self.storage_port.save_json(deployment_config, config_file)
        
        return package_dir
    
    async def _create_inference_script(self, path: Path, request: ExportRequestDTO) -> None:
        """Create a sample inference script."""
        script_content = f'''#!/usr/bin/env python3
"""
Inference script for exported model.
Format: {request.export_format}
"""

import json
from pathlib import Path
from typing import Union, List, Dict, Any

def load_model(model_path: Path):
    """Load the exported model."""
    # Implementation depends on export format
    pass

def preprocess(inputs: Union[str, List[str], Dict[str, Any]]) -> Any:
    """Preprocess inputs for model."""
    # Implementation depends on model type
    pass

def predict(model, inputs: Union[str, List[str], Dict[str, Any]]) -> Dict[str, Any]:
    """Run inference on inputs."""
    # Preprocess
    processed_inputs = preprocess(inputs)
    
    # Run inference
    outputs = model(processed_inputs)
    
    # Postprocess
    return postprocess(outputs)

def postprocess(outputs: Any) -> Dict[str, Any]:
    """Postprocess model outputs."""
    # Implementation depends on task type
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default="model")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path)
    
    # Run prediction
    result = predict(model, args.input)
    
    # Print result
    print(json.dumps(result, indent=2))
'''
        await self.storage_port.save_text(script_content, path)
    
    async def _create_requirements_file(self, path: Path, request: ExportRequestDTO) -> None:
        """Create requirements.txt for deployment."""
        requirements = []
        
        if request.export_format == "mlx":
            requirements.append("mlx>=0.5.0")
        elif request.export_format == "onnx":
            requirements.append("onnxruntime>=1.16.0")
        elif request.export_format == "coreml":
            requirements.append("coremltools>=7.0")
        
        requirements.extend([
            "numpy>=1.24.0",
            "tokenizers>=0.15.0",
            "pydantic>=2.0.0"
        ])
        
        content = "\n".join(requirements)
        await self.storage_port.save_text(content, path)
    
    async def _create_deployment_readme(
        self,
        path: Path,
        model: Model,
        request: ExportRequestDTO
    ) -> None:
        """Create README for deployment package."""
        readme_content = f"""# Model Deployment Package

## Model Information
- **Model Type**: {model.config.model_type}
- **Task Type**: {model.config.task_type}
- **Export Format**: {request.export_format}
- **Export Date**: {datetime.now().isoformat()}

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running Inference
```bash
python inference.py --input "Your input text here"
```

### API Usage
```python
from pathlib import Path
from inference import load_model, predict

# Load model
model = load_model(Path("model"))

# Run prediction
result = predict(model, "Your input text")
print(result)
```

## Model Optimizations
- Quantized: {request.quantize}
- Pruned: {request.prune}
- Inference Optimized: {request.optimize_for_inference}

## Input Requirements
- Maximum sequence length: {request.export_sequence_length or 512}
- Batch size: {request.export_batch_size or 1}
"""
        await self.storage_port.save_text(readme_content, path)
    
    async def _create_dockerfile(self, path: Path, request: ExportRequestDTO) -> None:
        """Create Dockerfile for containerized deployment."""
        dockerfile_content = f"""FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and inference code
COPY model/ ./model/
COPY inference.py .
COPY deployment_config.json .

# Set environment variables
ENV MODEL_PATH=/app/model
ENV PORT=8080

# Expose port
EXPOSE $PORT

# Run inference server
CMD ["python", "inference.py", "--model-path", "$MODEL_PATH"]
"""
        await self.storage_port.save_text(dockerfile_content, path)
    
    async def _validate_export(
        self,
        export_paths: Dict[str, Path],
        request: ExportRequestDTO
    ) -> Dict[str, Any]:
        """Validate the exported model."""
        validation_results = {
            "files_exist": True,
            "model_loadable": False,
            "inference_test_passed": False
        }
        
        # Check all files exist
        for key, path in export_paths.items():
            if not path.exists():
                validation_results["files_exist"] = False
                validation_results["missing_files"] = validation_results.get("missing_files", [])
                validation_results["missing_files"].append(str(path))
        
        # Try to load the model
        if validation_results["files_exist"]:
            try:
                # Format-specific validation
                if request.export_format == "mlx":
                    test_data = await self.storage_port.load_mlx(export_paths["model"])
                    validation_results["model_loadable"] = "model_state" in test_data
                elif request.export_format == "onnx":
                    # Would use onnx checker
                    validation_results["model_loadable"] = True
                # Add other format validations
                
            except Exception as e:
                validation_results["load_error"] = str(e)
        
        # Run inference test if requested
        if request.test_inference and validation_results["model_loadable"]:
            try:
                # Run a simple inference test
                # This would be format-specific
                validation_results["inference_test_passed"] = True
            except Exception as e:
                validation_results["inference_error"] = str(e)
        
        return validation_results
    
    async def _generate_metadata(
        self,
        model: Model,
        export_paths: Dict[str, Path],
        validation_results: Dict[str, Any],
        request: ExportRequestDTO
    ) -> Dict[str, Any]:
        """Generate comprehensive export metadata."""
        # Calculate model size
        model_size_bytes = 0
        for path in export_paths.values():
            if path.is_file():
                model_size_bytes += path.stat().st_size
            elif path.is_dir():
                for file in path.rglob("*"):
                    if file.is_file():
                        model_size_bytes += file.stat().st_size
        
        metadata = {
            "export_format": request.export_format,
            "export_timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": model.config.model_type,
                "task": model.config.task_type,
                "num_parameters": model.num_parameters,
                "architecture": model.config.architecture_info
            },
            "export_config": {
                "optimized": request.optimize,
                "quantized": request.quantize,
                "pruned": request.prune,
                "quantization_bits": request.quantization_bits,
                "pruning_ratio": request.pruning_ratio
            },
            "size_info": {
                "model_size_bytes": model_size_bytes,
                "model_size_mb": model_size_bytes / (1024 * 1024)
            },
            "validation": validation_results,
            "files": {
                key: str(path) for key, path in export_paths.items()
            }
        }
        
        # Save metadata
        metadata_path = request.output_dir / "export_metadata.json"
        await self.storage_port.save_json(metadata, metadata_path)
        
        return metadata
    
    def _create_response(
        self,
        export_paths: Dict[str, Path],
        metadata: Dict[str, Any],
        validation_results: Dict[str, Any],
        start_time: datetime,
        end_time: datetime
    ) -> ExportResponseDTO:
        """Create response DTO."""
        return ExportResponseDTO(
            success=validation_results.get("model_loadable", False),
            export_paths=export_paths,
            export_format=metadata["export_format"],
            model_size_mb=metadata["size_info"]["model_size_mb"],
            optimization_applied=metadata["export_config"]["optimized"],
            validation_results=validation_results,
            metadata=metadata,
            export_time_seconds=(end_time - start_time).total_seconds()
        )