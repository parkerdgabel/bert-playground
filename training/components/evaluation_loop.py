"""Evaluation loop component for handling validation and evaluation logic.

This component is responsible for:
- Running evaluation on validation datasets
- Computing evaluation metrics
- Managing evaluation state
"""

from typing import Callable, Protocol, Dict, Any, TYPE_CHECKING
from loguru import logger

from core.protocols.training import TrainingState, FrameworkAdapter
from core.protocols.data import DataLoader
from core.protocols.models import Model

if TYPE_CHECKING:
    from core.ports.compute import Array


class EvaluationStepFunction(Protocol):
    """Protocol for evaluation step functions."""
    
    def __call__(self, batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Execute a single evaluation step."""
        ...


class EvaluationLoop:
    """Handles evaluation and validation logic.
    
    This component manages:
    - Model evaluation on datasets
    - Metric computation
    - Loss calculation for validation
    """
    
    def __init__(self, model: Model, framework_adapter: FrameworkAdapter):
        """Initialize the evaluation loop.
        
        Args:
            model: Model to evaluate
            framework_adapter: Adapter for framework-specific operations
        """
        self.model = model
        self.framework = framework_adapter
        self._eval_step = self._create_eval_step()
        self._compiled_eval_step = None
        self._use_compiled = False
        
        logger.debug("Initialized EvaluationLoop")
        
    def _create_eval_step(self) -> EvaluationStepFunction:
        """Create the evaluation step function."""
        
        def eval_step(batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
            """Single evaluation step."""
            # Remove metadata
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k not in ["metadata"] and v is not None
            }
            
            # Forward pass (no gradients)
            try:
                outputs = self.model(**model_inputs)
            except TypeError:
                outputs = self.model(batch)
            
            # Extract or compute loss
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, dict) and "logits" in outputs:
                # Compute loss if we have logits and labels
                logits = outputs["logits"]
                if "labels" in model_inputs:
                    # Use neural ops if available, otherwise fall back to framework adapter
                    if self.framework.neural_ops:
                        loss = self.framework.neural_ops.cross_entropy(
                            logits, model_inputs["labels"], reduction="mean"
                        )
                    else:
                        # Fallback: assume loss is computed by the model
                        loss = self.framework.to_tensor(0.0)
                else:
                    loss = self.framework.to_tensor(0.0)
            else:
                # Handle other output formats
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                elif hasattr(outputs, "logits") and "labels" in model_inputs:
                    if self.framework.neural_ops:
                        loss = self.framework.neural_ops.cross_entropy(
                            outputs.logits, model_inputs["labels"], reduction="mean"
                        )
                    else:
                        loss = self.framework.to_tensor(0.0)
                else:
                    loss = outputs if self._is_tensor(outputs) else self.framework.to_tensor(float(outputs))
            
            # Build metrics
            if isinstance(outputs, dict):
                metrics = {k: v for k, v in outputs.items() if k != "loss"}
            else:
                metrics = {}
                
            return loss, metrics
        
        return eval_step
    
    def _is_tensor(self, obj: Any) -> bool:
        """Check if object is a tensor."""
        # Check for common tensor attributes
        return hasattr(obj, 'shape') and hasattr(obj, 'dtype')
    
    def evaluate_batch(self, batch: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Process a single evaluation batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (loss, metrics)
        """
        if self._use_compiled and self._compiled_eval_step:
            return self._compiled_eval_step(batch)
        else:
            return self._eval_step(batch)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        state: TrainingState | None = None,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, float]:
        """Run evaluation on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
            state: Optional training state
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Set model to eval mode
        self.model.eval()
        
        # Execute callbacks
        if callbacks and state:
            for callback in callbacks:
                if hasattr(callback, "on_evaluate_begin"):
                    callback(state)
        
        # Initialize metrics
        total_loss = 0.0
        total_metrics: dict[str, Any] = {}
        num_batches = 0
        
        # Process batches
        for batch_idx, batch in enumerate(dataloader):
            # Evaluation step
            loss, metrics = self.evaluate_batch(batch)
            
            # Accumulate metrics
            if total_loss == 0.0:
                total_loss = loss
            else:
                total_loss = total_loss + loss
                
            for k, v in metrics.items():
                if v is None or (hasattr(v, "shape") and v.size > 1):
                    continue
                    
                if k not in total_metrics:
                    total_metrics[k] = v
                else:
                    total_metrics[k] = total_metrics[k] + v
                    
            num_batches += 1
        
        # Average metrics
        self.framework.evaluate_tensors(total_loss)
        avg_loss = self.framework.to_python(total_loss) / num_batches
        
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in total_metrics.items():
            self.framework.evaluate_tensors(v)
            avg_metrics[k] = self.framework.to_python(v) / num_batches
        
        # Prefix with eval_
        eval_metrics = {f"eval_{k}": v for k, v in avg_metrics.items()}
        
        # Execute callbacks
        if callbacks and state:
            for callback in callbacks:
                if hasattr(callback, "on_evaluate_end"):
                    callback(state, eval_metrics)
        
        # Set model back to train mode
        self.model.train()
        
        return eval_metrics
    
    def predict(self, dataloader: DataLoader) -> list[Any]:
        """Generate predictions for a dataset.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            List of prediction tensors (one per batch)
        """
        # Set model to eval mode
        self.model.eval()
        
        predictions = []
        
        for batch in dataloader:
            # Remove metadata
            model_inputs = {
                k: v
                for k, v in batch.items()
                if k not in ["metadata"] and v is not None
            }
            
            # Forward pass
            try:
                outputs = self.model(**model_inputs)
            except TypeError:
                outputs = self.model(batch)
            
            # Extract predictions
            if isinstance(outputs, dict):
                if "logits" in outputs:
                    preds = outputs["logits"]
                elif "predictions" in outputs:
                    preds = outputs["predictions"]
                else:
                    raise ValueError("Model must return 'logits' or 'predictions'")
            else:
                # Assume outputs are predictions directly
                preds = outputs
                
            predictions.append(preds)
        
        # Set model back to train mode
        self.model.train()
        
        # Return list of predictions - let calling code handle concatenation
        # This keeps the evaluation loop framework-agnostic
        return predictions
    
    def set_compiled_step(self, compiled_step: EvaluationStepFunction) -> None:
        """Set a compiled evaluation step function.
        
        Args:
            compiled_step: Compiled evaluation step function
        """
        self._compiled_eval_step = compiled_step
        self._use_compiled = True
        logger.info("Using compiled evaluation step")