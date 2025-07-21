"""
BERT ensemble framework for Kaggle competitions.

This module implements advanced ensemble techniques specifically optimized
for BERT models, including multi-model ensembles, checkpoint averaging,
and layer-wise ensembling.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from loguru import logger

from ..core.protocols import Model, DataLoader
from ...models.factory import create_model, create_model_from_checkpoint


@dataclass
class BERTEnsembleConfig:
    """Configuration for BERT ensemble."""
    
    # Model diversity
    model_types: List[str] = None  # ["bert", "modernbert", "roberta"]
    model_sizes: List[str] = None  # ["base", "large"]
    
    # Training diversity
    random_seeds: List[int] = None  # Different initialization seeds
    max_seq_lengths: List[int] = None  # [256, 384, 512]
    dropout_rates: List[float] = None  # [0.1, 0.2, 0.3]
    
    # LoRA diversity (if using LoRA)
    lora_ranks: List[int] = None  # [8, 16, 32]
    lora_alphas: List[float] = None  # [16, 32, 64]
    
    # Ensemble methods
    ensemble_method: str = "weighted_average"  # weighted_average, voting, stacking
    use_oof_weights: bool = True  # Weight by out-of-fold performance
    temperature_scaling: float = 1.0  # For calibration
    
    # Checkpoint averaging
    checkpoint_average_last_n: int = 3  # Average last N checkpoints
    checkpoint_average_best_n: int = 0  # Average best N checkpoints
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if self.model_types is None:
            self.model_types = ["modernbert"]
        if self.random_seeds is None:
            self.random_seeds = [42, 123, 456]
        if self.max_seq_lengths is None:
            self.max_seq_lengths = [256]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1]


class BERTEnsembleModel:
    """
    Ensemble of BERT models for improved predictions.
    
    This class manages multiple BERT models and combines their
    predictions using various ensemble techniques.
    """
    
    def __init__(self, config: BERTEnsembleConfig):
        """
        Initialize BERT ensemble.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.models = []
        self.model_weights = []
        self.model_configs = []
        
    def add_model(self, model: Model, weight: float = 1.0, 
                  config: Optional[Dict[str, Any]] = None):
        """
        Add a model to the ensemble.
        
        Args:
            model: BERT model to add
            weight: Model weight for ensemble
            config: Model configuration (for tracking)
        """
        self.models.append(model)
        self.model_weights.append(weight)
        self.model_configs.append(config or {})
        logger.info(f"Added model to ensemble (total: {len(self.models)})")
    
    def create_diverse_models(self, base_config: Dict[str, Any],
                            head_type: str = "binary_classification",
                            num_labels: int = 2) -> List[Model]:
        """
        Create diverse BERT models for ensemble.
        
        Args:
            base_config: Base configuration for models
            head_type: Type of prediction head
            num_labels: Number of output labels
            
        Returns:
            List of created models
        """
        created_models = []
        
        # Create models with different architectures
        for model_type in self.config.model_types:
            for seed in self.config.random_seeds:
                for max_len in self.config.max_seq_lengths:
                    for dropout in self.config.dropout_rates:
                        # Set random seed for initialization
                        mx.random.seed(seed)
                        
                        # Create model
                        model_name = f"{model_type}_with_head"
                        model = create_model(
                            model_name,
                            head_type=head_type,
                            num_labels=num_labels,
                            dropout_prob=dropout,
                            max_position_embeddings=max_len
                        )
                        
                        # Add to ensemble
                        model_config = {
                            "type": model_type,
                            "seed": seed,
                            "max_length": max_len,
                            "dropout": dropout
                        }
                        self.add_model(model, weight=1.0, config=model_config)
                        created_models.append(model)
                        
                        logger.info(f"Created {model_type} model: seed={seed}, "
                                  f"max_len={max_len}, dropout={dropout}")
        
        # Normalize weights
        self._normalize_weights()
        
        return created_models
    
    def load_checkpoint_ensemble(self, checkpoint_paths: List[Path]):
        """
        Load models from checkpoints.
        
        Args:
            checkpoint_paths: List of checkpoint paths
        """
        for path in checkpoint_paths:
            try:
                model = create_model_from_checkpoint(path)
                self.add_model(model, weight=1.0, config={"checkpoint": str(path)})
                logger.info(f"Loaded model from checkpoint: {path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {path}: {e}")
    
    def create_checkpoint_average_model(self, checkpoint_dir: Path,
                                      last_n: Optional[int] = None,
                                      best_n: Optional[int] = None) -> Model:
        """
        Create a model by averaging multiple checkpoints.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            last_n: Average last N checkpoints
            best_n: Average best N checkpoints
            
        Returns:
            Averaged model
        """
        # Get checkpoint paths
        checkpoint_paths = self._get_checkpoint_paths(checkpoint_dir, last_n, best_n)
        
        if not checkpoint_paths:
            raise ValueError("No checkpoints found for averaging")
        
        logger.info(f"Averaging {len(checkpoint_paths)} checkpoints")
        
        # Load first model as base
        base_model = create_model_from_checkpoint(checkpoint_paths[0])
        base_weights = dict(base_model.named_parameters())
        
        # Average weights
        for path in checkpoint_paths[1:]:
            model = create_model_from_checkpoint(path)
            for name, param in model.named_parameters():
                if name in base_weights:
                    base_weights[name] = (base_weights[name] + param) / 2
        
        # Update base model with averaged weights
        base_model.load_state_dict(base_weights)
        
        return base_model
    
    def predict(self, dataloader: DataLoader, 
                return_all: bool = False) -> Union[mx.array, List[mx.array]]:
        """
        Generate ensemble predictions.
        
        Args:
            dataloader: Data loader for predictions
            return_all: Whether to return all model predictions
            
        Returns:
            Ensemble predictions or list of all predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        all_predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            logger.info(f"Getting predictions from model {i+1}/{len(self.models)}")
            
            model.eval()
            model_preds = []
            
            for batch in dataloader:
                with mx.no_grad():
                    outputs = model(batch)
                    
                    # Extract logits or predictions
                    if "logits" in outputs:
                        preds = outputs["logits"]
                    else:
                        preds = outputs["predictions"]
                    
                    model_preds.append(preds)
            
            # Concatenate predictions
            model_predictions = mx.concatenate(model_preds, axis=0)
            all_predictions.append(model_predictions)
        
        if return_all:
            return all_predictions
        
        # Combine predictions
        ensemble_preds = self._combine_predictions(all_predictions)
        
        return ensemble_preds
    
    def _combine_predictions(self, predictions: List[mx.array]) -> mx.array:
        """
        Combine predictions from multiple models.
        
        Args:
            predictions: List of prediction arrays
            
        Returns:
            Combined predictions
        """
        if self.config.ensemble_method == "weighted_average":
            # Weighted average of predictions
            weights = mx.array(self.model_weights[:len(predictions)])
            weights = weights / mx.sum(weights)  # Normalize
            
            # Apply temperature scaling for calibration
            if self.config.temperature_scaling != 1.0:
                predictions = [p / self.config.temperature_scaling for p in predictions]
            
            # Stack and average
            stacked = mx.stack(predictions, axis=0)
            ensemble = mx.sum(stacked * weights[:, None, None], axis=0)
            
        elif self.config.ensemble_method == "voting":
            # Majority voting
            # Convert to class predictions
            class_preds = [mx.argmax(p, axis=-1) for p in predictions]
            stacked = mx.stack(class_preds, axis=0)
            
            # Get mode (most common prediction)
            # Note: MLX doesn't have mode, so we'll use a simple approach
            ensemble = []
            for i in range(stacked.shape[1]):
                votes = stacked[:, i]
                # Count votes for each class
                unique_classes = mx.unique(votes)
                counts = mx.array([mx.sum(votes == c) for c in unique_classes])
                most_common = unique_classes[mx.argmax(counts)]
                ensemble.append(most_common)
            
            ensemble = mx.stack(ensemble)
            
        else:  # stacking
            # Simple stacking - concatenate predictions as features
            ensemble = mx.concatenate(predictions, axis=-1)
        
        return ensemble
    
    def _normalize_weights(self):
        """Normalize model weights to sum to 1."""
        total = sum(self.model_weights)
        if total > 0:
            self.model_weights = [w / total for w in self.model_weights]
    
    def _get_checkpoint_paths(self, checkpoint_dir: Path,
                            last_n: Optional[int] = None,
                            best_n: Optional[int] = None) -> List[Path]:
        """
        Get checkpoint paths for averaging.
        
        Args:
            checkpoint_dir: Directory with checkpoints
            last_n: Get last N checkpoints
            best_n: Get best N checkpoints
            
        Returns:
            List of checkpoint paths
        """
        checkpoints = []
        
        # Find all checkpoint directories
        for path in checkpoint_dir.iterdir():
            if path.is_dir() and "checkpoint" in path.name:
                checkpoints.append(path)
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        if last_n:
            return checkpoints[:last_n]
        
        # For best_n, we'd need to read metrics from checkpoints
        # For now, just return newest
        return checkpoints[:best_n] if best_n else checkpoints
    
    def update_weights_from_cv(self, cv_scores: List[float]):
        """
        Update model weights based on CV performance.
        
        Args:
            cv_scores: Cross-validation scores for each model
        """
        if len(cv_scores) != len(self.models):
            raise ValueError("Number of CV scores must match number of models")
        
        # Convert scores to weights (higher score = higher weight)
        # Use softmax for smooth weighting
        scores = mx.array(cv_scores)
        self.model_weights = mx.softmax(scores * 5.0).tolist()  # Temperature = 5
        
        logger.info(f"Updated ensemble weights based on CV: {self.model_weights}")


class BERTEnsembleTrainer:
    """
    Trainer for BERT ensemble models.
    
    Handles training multiple models and combining them into an ensemble.
    """
    
    def __init__(self, ensemble_config: BERTEnsembleConfig):
        """
        Initialize ensemble trainer.
        
        Args:
            ensemble_config: Ensemble configuration
        """
        self.config = ensemble_config
        self.ensemble = BERTEnsembleModel(ensemble_config)
        
    def train_ensemble(self, train_loader: DataLoader,
                      val_loader: Optional[DataLoader] = None,
                      trainer_class=None,
                      trainer_config=None) -> BERTEnsembleModel:
        """
        Train ensemble of models.
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            trainer_class: Trainer class to use
            trainer_config: Trainer configuration
            
        Returns:
            Trained ensemble
        """
        # Create diverse models
        models = self.ensemble.create_diverse_models(
            base_config={},
            head_type=trainer_config.kaggle.competition_type.value,
            num_labels=2  # TODO: Get from config
        )
        
        cv_scores = []
        
        # Train each model
        for i, model in enumerate(models):
            logger.info(f"Training model {i+1}/{len(models)}")
            
            # Create trainer for this model
            trainer = trainer_class(
                model=model,
                config=trainer_config,
                enable_bert_strategies=True
            )
            
            # Train
            result = trainer.train(train_loader, val_loader)
            
            # Track CV score
            best_score = result.best_val_metric or result.best_val_loss
            cv_scores.append(best_score)
            
            logger.info(f"Model {i+1} best score: {best_score}")
        
        # Update ensemble weights based on CV performance
        self.ensemble.update_weights_from_cv(cv_scores)
        
        return self.ensemble