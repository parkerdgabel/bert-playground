"""Evaluation loop component for handling validation and evaluation logic.

This component is responsible for:
- Running evaluation on validation datasets
- Computing evaluation metrics
- Managing evaluation state
"""

from typing import Callable, Protocol, Dict, Any
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from core.protocols.training import TrainingState
from core.protocols.data import DataLoader
from core.protocols.models import Model


class EvaluationStepFunction(Protocol):
    """Protocol for evaluation step functions."""
    
    def __call__(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict[str, mx.array]]:
        """Execute a single evaluation step."""
        ...


class EvaluationLoop:
    """Handles evaluation and validation logic.
    
    This component manages:
    - Model evaluation on datasets
    - Metric computation
    - Loss calculation for validation
    """
    
    def __init__(self, model: Model):
        """Initialize the evaluation loop.
        
        Args:
            model: Model to evaluate
        """
        self.model = model
        self._eval_step = self._create_eval_step()
        self._compiled_eval_step = None
        self._use_compiled = False
        
        logger.debug("Initialized EvaluationLoop")
        
    def _create_eval_step(self) -> EvaluationStepFunction:
        """Create the evaluation step function."""
        
        def eval_step(batch: dict[str, mx.array]) -> tuple[mx.array, dict[str, mx.array]]:
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
                    loss = nn.losses.cross_entropy(
                        logits, model_inputs["labels"], reduction="mean"
                    )
                else:
                    loss = mx.array(0.0)
            else:
                # Handle other output formats
                if hasattr(outputs, "loss"):
                    loss = outputs.loss
                elif hasattr(outputs, "logits") and "labels" in model_inputs:
                    loss = nn.losses.cross_entropy(
                        outputs.logits, model_inputs["labels"], reduction="mean"
                    )
                else:
                    loss = outputs if isinstance(outputs, mx.array) else mx.array(float(outputs))
            
            # Build metrics
            if isinstance(outputs, dict):
                metrics = {k: v for k, v in outputs.items() if k != "loss"}
            else:
                metrics = {}
                
            return loss, metrics
        
        return eval_step
    
    def evaluate_batch(self, batch: dict[str, mx.array]) -> tuple[mx.array, dict[str, mx.array]]:
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
        total_metrics: dict[str, mx.array] = {}
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
        mx.eval(total_loss)
        avg_loss = float(total_loss.item()) / num_batches if hasattr(total_loss, "item") else float(total_loss) / num_batches
        
        avg_metrics = {"loss": avg_loss}
        
        # Average other metrics
        for k, v in total_metrics.items():
            if hasattr(v, "item"):
                mx.eval(v)
                avg_metrics[k] = float(v.item()) / num_batches
            else:
                avg_metrics[k] = float(v) / num_batches
        
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
    
    def predict(self, dataloader: DataLoader) -> mx.array:
        """Generate predictions for a dataset.
        
        Args:
            dataloader: Data loader for prediction
            
        Returns:
            Predictions as MLX array
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
            if "logits" in outputs:
                preds = outputs["logits"]
            elif "predictions" in outputs:
                preds = outputs["predictions"]
            else:
                raise ValueError("Model must return 'logits' or 'predictions'")
                
            predictions.append(preds)
        
        # Set model back to train mode
        self.model.train()
        
        # Concatenate predictions
        return mx.concatenate(predictions, axis=0)
    
    def set_compiled_step(self, compiled_step: EvaluationStepFunction) -> None:
        """Set a compiled evaluation step function.
        
        Args:
            compiled_step: Compiled evaluation step function
        """
        self._compiled_eval_step = compiled_step
        self._use_compiled = True
        logger.info("Using compiled evaluation step")