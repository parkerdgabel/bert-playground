"""Get Model Info Query - retrieves model information and metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from domain.ports import StoragePort, MonitoringPort


@dataclass
class ModelInfoRequest:
    """Request for model information."""
    model_path: Path
    include_architecture: bool = True
    include_training_history: bool = False
    include_performance_metrics: bool = False
    include_file_info: bool = True


@dataclass
class ModelInfoResponse:
    """Response containing model information."""
    success: bool
    error_message: Optional[str] = None
    
    # Basic info
    model_type: Optional[str] = None
    task_type: Optional[str] = None
    num_parameters: Optional[int] = None
    num_labels: Optional[int] = None
    label_names: Optional[List[str]] = None
    
    # Architecture details
    architecture: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    
    # Training info
    training_history: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    training_config: Optional[Dict[str, Any]] = None
    
    # Performance info
    inference_speed: Optional[Dict[str, float]] = None
    memory_usage: Optional[Dict[str, float]] = None
    
    # File info
    file_size_mb: Optional[float] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    checkpoint_info: Optional[Dict[str, Any]] = None
    
    # Export info
    export_formats_available: Optional[List[str]] = None
    optimization_applied: Optional[Dict[str, bool]] = None


@dataclass
class GetModelInfoQuery:
    """Query to retrieve model information.
    
    This query handles read-only operations to get model metadata,
    architecture details, and performance characteristics.
    """
    
    storage_port: StoragePort
    monitoring_port: MonitoringPort
    
    async def execute(self, request: ModelInfoRequest) -> ModelInfoResponse:
        """Execute the query to get model information.
        
        Args:
            request: Request specifying what information to retrieve
            
        Returns:
            Response with model information
        """
        try:
            # Check if model exists
            if not request.model_path.exists():
                return ModelInfoResponse(
                    success=False,
                    error_message=f"Model not found at {request.model_path}"
                )
            
            response = ModelInfoResponse(success=True)
            
            # Load model metadata
            metadata = await self._load_model_metadata(request.model_path)
            
            # Extract basic information
            response.model_type = metadata.get("model_type")
            response.task_type = metadata.get("task_type")
            response.num_parameters = metadata.get("num_parameters")
            response.num_labels = metadata.get("num_labels")
            response.label_names = metadata.get("label_names")
            
            # Get architecture details if requested
            if request.include_architecture:
                response.architecture = await self._get_architecture_info(
                    request.model_path,
                    metadata
                )
                response.config = metadata.get("config")
            
            # Get training history if requested
            if request.include_training_history:
                training_info = await self._get_training_history(request.model_path)
                response.training_history = training_info.get("history")
                response.best_metrics = training_info.get("best_metrics")
                response.training_config = training_info.get("config")
            
            # Get performance metrics if requested
            if request.include_performance_metrics:
                perf_info = await self._get_performance_metrics(
                    request.model_path,
                    metadata
                )
                response.inference_speed = perf_info.get("inference_speed")
                response.memory_usage = perf_info.get("memory_usage")
            
            # Get file information if requested
            if request.include_file_info:
                file_info = await self._get_file_info(request.model_path)
                response.file_size_mb = file_info.get("size_mb")
                response.created_date = file_info.get("created")
                response.modified_date = file_info.get("modified")
                response.checkpoint_info = file_info.get("checkpoint_info")
            
            # Check available export formats
            response.export_formats_available = self._get_available_export_formats(metadata)
            response.optimization_applied = metadata.get("optimizations", {})
            
            return response
            
        except Exception as e:
            await self.monitoring_port.log_error(f"Failed to get model info: {str(e)}")
            return ModelInfoResponse(
                success=False,
                error_message=str(e)
            )
    
    async def _load_model_metadata(self, model_path: Path) -> Dict[str, Any]:
        """Load model metadata from various sources."""
        metadata = {}
        
        # Try to load from metadata.json
        metadata_file = model_path / "metadata.json"
        if metadata_file.exists():
            metadata.update(await self.storage_port.load_json(metadata_file))
        
        # Try to load from config.json
        config_file = model_path / "config.json"
        if config_file.exists():
            config = await self.storage_port.load_json(config_file)
            metadata["config"] = config
            metadata["model_type"] = config.get("model_type")
            metadata["task_type"] = config.get("task_type")
            metadata["num_labels"] = config.get("num_labels")
            
            # Extract label names if available
            if "id2label" in config:
                metadata["label_names"] = list(config["id2label"].values())
        
        # Try to load from checkpoint
        checkpoint_file = model_path / "checkpoint.json"
        if checkpoint_file.exists():
            checkpoint = await self.storage_port.load_json(checkpoint_file)
            metadata.update(checkpoint.get("metadata", {}))
        
        return metadata
    
    async def _get_architecture_info(
        self,
        model_path: Path,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract architecture information."""
        arch_info = {}
        
        config = metadata.get("config", {})
        
        # BERT-specific architecture details
        if "bert" in metadata.get("model_type", "").lower():
            arch_info["model_family"] = "BERT"
            arch_info["hidden_size"] = config.get("hidden_size")
            arch_info["num_hidden_layers"] = config.get("num_hidden_layers")
            arch_info["num_attention_heads"] = config.get("num_attention_heads")
            arch_info["intermediate_size"] = config.get("intermediate_size")
            arch_info["max_position_embeddings"] = config.get("max_position_embeddings")
            arch_info["vocab_size"] = config.get("vocab_size")
            arch_info["type_vocab_size"] = config.get("type_vocab_size")
            
            # ModernBERT specific
            if "modern" in metadata.get("model_type", "").lower():
                arch_info["uses_rope"] = config.get("position_embedding_type") == "rope"
                arch_info["uses_geglu"] = config.get("hidden_act") == "geglu"
                arch_info["attention_type"] = config.get("attention_type", "standard")
        
        # Calculate total parameters if not provided
        if "num_parameters" not in metadata:
            arch_info["estimated_parameters"] = self._estimate_parameters(config)
        
        return arch_info
    
    def _estimate_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate number of parameters from config."""
        # Simplified estimation for BERT-like models
        params = 0
        
        hidden_size = config.get("hidden_size", 768)
        num_layers = config.get("num_hidden_layers", 12)
        vocab_size = config.get("vocab_size", 30522)
        intermediate_size = config.get("intermediate_size", 3072)
        
        # Embeddings
        params += vocab_size * hidden_size  # Token embeddings
        params += config.get("max_position_embeddings", 512) * hidden_size  # Position
        params += config.get("type_vocab_size", 2) * hidden_size  # Token type
        
        # Transformer layers
        per_layer = (
            4 * hidden_size * hidden_size +  # Q, K, V, O projections
            2 * hidden_size * intermediate_size +  # FFN
            2 * hidden_size  # LayerNorm
        )
        params += num_layers * per_layer
        
        # Task head
        params += hidden_size * config.get("num_labels", 2)
        
        return params
    
    async def _get_training_history(self, model_path: Path) -> Dict[str, Any]:
        """Load training history and metrics."""
        training_info = {}
        
        # Load training history
        history_file = model_path / "training_history.json"
        if history_file.exists():
            history_data = await self.storage_port.load_json(history_file)
            training_info["history"] = history_data.get("history", {})
            training_info["config"] = history_data.get("config", {})
        
        # Load final metrics
        metrics_file = model_path / "metrics.json"
        if metrics_file.exists():
            metrics = await self.storage_port.load_json(metrics_file)
            training_info["best_metrics"] = metrics
        
        # Try to extract from checkpoint
        checkpoint_file = model_path / "checkpoint.json"
        if checkpoint_file.exists() and not training_info:
            checkpoint = await self.storage_port.load_json(checkpoint_file)
            if "training_state" in checkpoint:
                state = checkpoint["training_state"]
                training_info["best_metrics"] = {
                    "best_loss": state.get("best_metric"),
                    "final_loss": state.get("eval_loss"),
                    "total_steps": state.get("global_step")
                }
        
        return training_info
    
    async def _get_performance_metrics(
        self,
        model_path: Path,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get model performance characteristics."""
        perf_info = {}
        
        # Load benchmark results if available
        benchmark_file = model_path / "benchmark_results.json"
        if benchmark_file.exists():
            benchmarks = await self.storage_port.load_json(benchmark_file)
            perf_info["inference_speed"] = benchmarks.get("inference_speed", {})
            perf_info["memory_usage"] = benchmarks.get("memory_usage", {})
        else:
            # Estimate based on model size
            config = metadata.get("config", {})
            hidden_size = config.get("hidden_size", 768)
            num_layers = config.get("num_hidden_layers", 12)
            
            # Rough estimates
            perf_info["inference_speed"] = {
                "estimated_tokens_per_second": 1000 / (num_layers * 0.1),
                "batch_size_1_latency_ms": num_layers * 2
            }
            
            # Memory estimation (in MB)
            param_memory = metadata.get("num_parameters", 110_000_000) * 4 / 1_000_000
            activation_memory = hidden_size * 512 * 32 * 4 / 1_000_000  # For batch 32
            
            perf_info["memory_usage"] = {
                "model_size_mb": param_memory,
                "activation_memory_mb": activation_memory,
                "total_estimated_mb": param_memory + activation_memory
            }
        
        return perf_info
    
    async def _get_file_info(self, model_path: Path) -> Dict[str, Any]:
        """Get file system information about the model."""
        file_info = {}
        
        # Calculate total size
        total_size = 0
        file_count = 0
        
        if model_path.is_dir():
            for file in model_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
                    file_count += 1
            
            # Get directory stats
            dir_stat = model_path.stat()
            file_info["created"] = datetime.fromtimestamp(dir_stat.st_ctime)
            file_info["modified"] = datetime.fromtimestamp(dir_stat.st_mtime)
        else:
            # Single file
            file_stat = model_path.stat()
            total_size = file_stat.st_size
            file_count = 1
            file_info["created"] = datetime.fromtimestamp(file_stat.st_ctime)
            file_info["modified"] = datetime.fromtimestamp(file_stat.st_mtime)
        
        file_info["size_mb"] = total_size / (1024 * 1024)
        file_info["file_count"] = file_count
        
        # Check for checkpoint info
        checkpoint_file = model_path / "checkpoint.json" if model_path.is_dir() else None
        if checkpoint_file and checkpoint_file.exists():
            checkpoint = await self.storage_port.load_json(checkpoint_file)
            file_info["checkpoint_info"] = {
                "step": checkpoint.get("step"),
                "epoch": checkpoint.get("epoch"),
                "timestamp": checkpoint.get("timestamp")
            }
        
        return file_info
    
    def _get_available_export_formats(self, metadata: Dict[str, Any]) -> List[str]:
        """Determine which export formats are available for this model."""
        formats = ["mlx"]  # Native format always available
        
        model_type = metadata.get("model_type", "").lower()
        
        # ONNX is generally available for most models
        formats.append("onnx")
        
        # CoreML for Apple platforms
        if "bert" in model_type:
            formats.append("coreml")
        
        # TensorFlow/TorchScript depend on original training framework
        # These would be determined by checking model format
        
        return formats