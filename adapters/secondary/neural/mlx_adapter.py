"""MLX Neural Adapter implementation.

This module provides the MLXNeuralAdapter that handles high-level neural operations
using the MLXNeuralBackend for neural operations and MLXComputeAdapter for tensor operations.
This follows the hexagonal architecture pattern where the neural adapter orchestrates
neural workflows while delegating to specialized backends.
"""

from typing import Any, Dict, Optional, Tuple, List

import mlx.core as mx

from domain.entities.model import BertModel
from domain.entities.dataset import DataBatch
from application.ports.secondary.neural import NeuralBackend, LossType, InitializationType
from adapters.secondary.neural.mlx_backend import MLXNeuralBackend
from adapters.secondary.compute.mlx.compute_adapter import MLXComputeAdapter
from adapters.secondary.compute.mlx.model_adapter import MLXModelAdapter
from adapters.secondary.compute.mlx.optimization import MLXOptimizer


class MLXNeuralAdapter:
    """MLX implementation of high-level neural operations.
    
    This adapter handles neural workflows and training operations by coordinating
    between the neural backend (for neural operations) and compute backend 
    (for tensor operations). It provides the bridge between high-level neural
    operations and low-level backends.
    """
    
    def __init__(
        self, 
        neural_backend: Optional[MLXNeuralBackend] = None,
        compute_backend: Optional[MLXComputeAdapter] = None
    ):
        """Initialize MLX neural adapter.
        
        Args:
            neural_backend: Neural backend for neural operations
            compute_backend: Compute backend for tensor operations
        """
        self.neural_backend = neural_backend or MLXNeuralBackend()
        self.compute_backend = compute_backend or MLXComputeAdapter()
        
        # Model management - moved from compute adapter
        self._model_adapters: Dict[int, MLXModelAdapter] = {}
        self._optimizers: Dict[int, MLXOptimizer] = {}
        
    def forward_pass(
        self,
        model: BertModel,
        batch: DataBatch,
        training: bool = False,
    ) -> Dict[str, Any]:
        """Perform forward pass through model.
        
        Args:
            model: The BERT model
            batch: Input data batch
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - 'logits': Model predictions
            - 'loss': Computed loss (if labels provided)
            - 'hidden_states': Optional hidden states
            - 'attentions': Optional attention weights
        """
        # Get or create MLX model adapter
        model_adapter = self._get_or_create_model_adapter(model)
        
        # Convert batch to MLX arrays using compute backend
        mlx_batch = self._convert_batch_to_mlx(batch)
        
        # Set training mode
        model_adapter.train(training)
        
        # Forward pass
        outputs = model_adapter.forward(
            input_ids=mlx_batch["input_ids"],
            attention_mask=mlx_batch.get("attention_mask"),
            token_type_ids=mlx_batch.get("token_type_ids"),
            position_ids=mlx_batch.get("position_ids"),
            labels=mlx_batch.get("labels"),
        )
        
        return outputs
    
    def backward_pass(
        self,
        loss: Any,
        model: BertModel,
        retain_graph: bool = False,
    ) -> Dict[str, Any]:
        """Perform backward pass to compute gradients.
        
        Args:
            loss: Loss value to backpropagate
            model: The model to compute gradients for
            retain_graph: Whether to retain computation graph
            
        Returns:
            Dictionary of gradients
        """
        # MLX computes gradients through mx.grad or mx.value_and_grad
        # The actual backward pass is handled by the optimizer
        # This is a no-op for MLX as gradients are computed in optimize_step
        # Using the neural backend's backward pass implementation
        return self.neural_backend.backward_pass(loss, model, retain_graph)
    
    def optimize_step(
        self,
        model: BertModel,
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """Perform optimization step.
        
        Args:
            model: The model to optimize
            optimizer_state: Current optimizer state
            learning_rate: Current learning rate
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (updated_optimizer_state, gradient_norm)
        """
        model_adapter = self._get_or_create_model_adapter(model)
        
        # Get or create optimizer
        optimizer_id = id(model)
        if optimizer_id not in self._optimizers:
            optimizer_type = optimizer_state.get("type", "adamw")
            self._optimizers[optimizer_id] = self._create_optimizer(
                model_adapter, optimizer_type, learning_rate, **optimizer_state
            )
        
        optimizer = self._optimizers[optimizer_id]
        
        # Update learning rate
        optimizer.learning_rate = learning_rate
        
        # Get gradients and update weights
        grad_norm = optimizer.update(model_adapter, max_grad_norm)
        
        # Update optimizer state
        updated_state = optimizer.get_state()
        updated_state["learning_rate"] = learning_rate
        
        return updated_state, grad_norm
    
    def train_step(
        self,
        model: BertModel,
        batch: DataBatch,
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """Complete training step: forward + backward + optimize.
        
        Args:
            model: The model to train
            batch: Training batch
            optimizer_state: Current optimizer state
            learning_rate: Current learning rate
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (outputs, updated_optimizer_state, gradient_norm)
        """
        # Forward pass
        outputs = self.forward_pass(model, batch, training=True)
        
        # Check if we have a loss to backpropagate
        if "loss" not in outputs:
            raise ValueError("Model outputs must contain 'loss' for training")
        
        # Backward pass
        self.backward_pass(outputs["loss"], model)
        
        # Optimization step
        updated_state, grad_norm = self.optimize_step(
            model, optimizer_state, learning_rate, max_grad_norm
        )
        
        return outputs, updated_state, grad_norm
    
    def eval_step(
        self,
        model: BertModel,
        batch: DataBatch,
    ) -> Dict[str, Any]:
        """Complete evaluation step: forward pass only.
        
        Args:
            model: The model to evaluate
            batch: Evaluation batch
            
        Returns:
            Model outputs
        """
        return self.forward_pass(model, batch, training=False)
    
    def count_parameters(
        self,
        model: BertModel,
        trainable_only: bool = False,
    ) -> int:
        """Count model parameters.
        
        Args:
            model: The model
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        model_adapter = self._get_or_create_model_adapter(model)
        return model_adapter.count_parameters(trainable_only)
    
    def get_model_parameters(
        self,
        model: BertModel,
    ) -> List[Tuple[str, Any]]:
        """Get model parameters.
        
        Args:
            model: The model
            
        Returns:
            List of (name, parameter) tuples
        """
        model_adapter = self._get_or_create_model_adapter(model)
        return list(model_adapter.named_parameters())
    
    def initialize_model_weights(
        self,
        model: BertModel,
        init_type: InitializationType = InitializationType.XAVIER_UNIFORM,
        **kwargs: Any
    ) -> None:
        """Initialize model weights.
        
        Args:
            model: Model to initialize
            init_type: Initialization type
            **kwargs: Additional initialization parameters
        """
        model_adapter = self._get_or_create_model_adapter(model)
        self.neural_backend.initialize_weights(model_adapter, init_type, **kwargs)
    
    def freeze_model_parameters(
        self,
        model: BertModel,
        freeze: bool = True
    ) -> None:
        """Freeze or unfreeze model parameters.
        
        Args:
            model: Model to freeze/unfreeze
            freeze: Whether to freeze parameters
        """
        model_adapter = self._get_or_create_model_adapter(model)
        self.neural_backend.freeze_module(model_adapter, freeze)
    
    def compile_model(
        self,
        model: BertModel,
        backend: Optional[str] = None,
    ) -> BertModel:
        """Compile model for optimized execution.
        
        Args:
            model: Model to compile
            backend: Optional backend specification
            
        Returns:
            Compiled model
        """
        # Delegate to compute backend for compilation
        return self.compute_backend.compile_model(model, backend)
    
    def create_optimizer_state(
        self,
        model: BertModel,
        optimizer_type: str,
        learning_rate: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create optimizer state for a model.
        
        Args:
            model: Model to optimize
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer state dictionary
        """
        state = {
            "type": optimizer_type,
            "learning_rate": learning_rate,
        }
        state.update(kwargs)
        return state
    
    def compute_loss(
        self,
        predictions: Any,
        targets: Any,
        loss_type: LossType = LossType.CROSS_ENTROPY,
        **kwargs: Any,
    ) -> Any:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            loss_type: Type of loss function
            **kwargs: Additional loss-specific parameters
            
        Returns:
            Loss value
        """
        return self.neural_backend.compute_loss(predictions, targets, loss_type, **kwargs)
    
    def synchronize(self) -> None:
        """Synchronize neural operations."""
        # Delegate to compute backend
        self.compute_backend.synchronize()
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device information.
        
        Returns:
            Dictionary with device information
        """
        # Delegate to compute backend
        return self.compute_backend.get_device_info()
    
    # Private helper methods (moved from compute adapter)
    
    def _get_or_create_model_adapter(self, model: BertModel) -> MLXModelAdapter:
        """Get or create MLX model adapter for a BertModel."""
        model_id = id(model)
        if model_id not in self._model_adapters:
            self._model_adapters[model_id] = MLXModelAdapter(model)
        return self._model_adapters[model_id]
    
    def _convert_batch_to_mlx(self, batch: DataBatch) -> Dict[str, mx.array]:
        """Convert DataBatch to MLX arrays.
        
        This delegates the actual tensor conversion to the compute backend's logic.
        """
        # Use the compute backend's conversion logic
        return self.compute_backend._convert_batch_to_mlx(batch)
    
    def _create_optimizer(
        self,
        model_adapter: MLXModelAdapter,
        optimizer_type: str,
        learning_rate: float,
        **kwargs: Any,
    ) -> MLXOptimizer:
        """Create MLX optimizer."""
        return MLXOptimizer(
            model_adapter,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            **kwargs
        )
    
    # Context managers for training workflows
    
    def training_context(self, model: BertModel):
        """Context manager for training mode."""
        model_adapter = self._get_or_create_model_adapter(model)
        original_mode = model_adapter.training
        try:
            model_adapter.train(True)
            yield
        finally:
            model_adapter.train(original_mode)
    
    def evaluation_context(self, model: BertModel):
        """Context manager for evaluation mode."""
        model_adapter = self._get_or_create_model_adapter(model)
        original_mode = model_adapter.training
        try:
            model_adapter.train(False)
            yield
        finally:
            model_adapter.train(original_mode)
    
    def mixed_precision_context(self, enabled: bool = True):
        """Context manager for mixed precision training."""
        # Delegate to compute backend
        return self.compute_backend.mixed_precision_context(enabled)
    
    # High-level workflow methods
    
    def train_epoch(
        self,
        model: BertModel,
        dataloader: Any,  # DataLoader-like object
        optimizer_state: Dict[str, Any],
        learning_rate: float,
        max_grad_norm: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Train for one epoch.
        
        Args:
            model: Model to train
            dataloader: Training data loader
            optimizer_state: Current optimizer state
            learning_rate: Learning rate
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (average_loss, updated_optimizer_state)
        """
        total_loss = 0.0
        num_batches = 0
        
        with self.training_context(model):
            for batch in dataloader:
                outputs, optimizer_state, grad_norm = self.train_step(
                    model, batch, optimizer_state, learning_rate, max_grad_norm
                )
                
                if "loss" in outputs:
                    total_loss += float(outputs["loss"])
                    num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, optimizer_state
    
    def evaluate_epoch(
        self,
        model: BertModel,
        dataloader: Any,  # DataLoader-like object
    ) -> Dict[str, float]:
        """Evaluate for one epoch.
        
        Args:
            model: Model to evaluate
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with self.evaluation_context(model):
            for batch in dataloader:
                outputs = self.eval_step(model, batch)
                
                if "loss" in outputs:
                    total_loss += float(outputs["loss"])
                    num_batches += 1
                
                if "logits" in outputs:
                    all_predictions.extend(outputs["logits"])
                    if hasattr(batch, "labels") and batch.labels is not None:
                        all_targets.extend(batch.labels)
        
        metrics = {}
        if num_batches > 0:
            metrics["loss"] = total_loss / num_batches
        
        # Additional metrics could be computed here based on predictions/targets
        
        return metrics