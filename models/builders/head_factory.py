"""Factory for creating task-specific heads.

This module handles the creation of different head types for various tasks,
with support for automatic head selection based on task requirements.
"""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.nn as nn
from loguru import logger

from ..heads import create_head
from ..heads.base import HeadConfig
from .config_resolver import ConfigResolver


@dataclass
class HeadFactory:
    """Factory for creating task-specific heads."""
    
    config_resolver: ConfigResolver
    
    def create_head(
        self,
        head_type: str,
        head_config: Optional[HeadConfig] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """Create a task-specific head.
        
        Args:
            head_type: Type of head to create
            head_config: Optional head configuration
            input_size: Input dimension for the head
            output_size: Output dimension for the head
            **kwargs: Additional head parameters
            
        Returns:
            Initialized head module
        """
        # Resolve configuration if not provided
        if head_config is None:
            if input_size is None or output_size is None:
                raise ValueError(
                    "Either head_config or both input_size and output_size must be provided"
                )
            
            head_config = self.config_resolver.resolve_head_config(
                head_config=None,
                head_type=head_type,
                input_size=input_size,
                output_size=output_size,
            )
        
        # Extract config dict and remove duplicates
        head_config_dict = head_config.__dict__.copy()
        head_type_str = head_config_dict.pop("head_type", head_type)
        
        # Merge with kwargs (kwargs take precedence)
        head_config_dict.update(kwargs)
        
        logger.info(f"Creating head: {head_type_str}")
        
        return create_head(
            head_type=head_type_str,
            **head_config_dict,
        )
    
    def get_head_for_task(self, task_type: str) -> str:
        """Get appropriate head type for a given task.
        
        Args:
            task_type: Type of task (e.g., "binary", "multiclass", "regression")
            
        Returns:
            Head type string
        """
        task_to_head_map = {
            "binary": "binary_classification",
            "multiclass": "multiclass_classification",
            "multilabel": "multilabel_classification",
            "regression": "regression",
            "ordinal": "ordinal_regression",
            "time_series": "time_series",
            "ranking": "ranking",
            "titanic": "binary_classification",
            "hierarchical": "multiclass_classification",
            "ensemble": "multiclass_classification",
            "contrastive": "multiclass_classification",
            "multi_task": "multiclass_classification",
            "metric_learning": "multiclass_classification",
        }
        
        return task_to_head_map.get(task_type, "multiclass_classification")
    
    def get_head_for_competition_type(self, competition_type: Any) -> str:
        """Get appropriate head type for a competition type.
        
        Args:
            competition_type: Competition type enum
            
        Returns:
            Head type string
        """
        # Import here to avoid circular dependencies
        from ..factory import CompetitionType
        
        comp_to_head_map = {
            CompetitionType.BINARY_CLASSIFICATION: "binary_classification",
            CompetitionType.MULTICLASS_CLASSIFICATION: "multiclass_classification",
            CompetitionType.MULTILABEL_CLASSIFICATION: "multilabel_classification",
            CompetitionType.REGRESSION: "regression",
            CompetitionType.ORDINAL_REGRESSION: "ordinal_regression",
            CompetitionType.TIME_SERIES: "time_series",
            CompetitionType.RANKING: "ranking",
        }
        
        return comp_to_head_map.get(competition_type, "binary_classification")
    
    def infer_output_size(
        self,
        head_type: str,
        num_labels: Optional[int] = None,
        num_classes: Optional[int] = None,
    ) -> int:
        """Infer output size for a head type.
        
        Args:
            head_type: Type of head
            num_labels: Number of labels (if provided)
            num_classes: Number of classes (if provided)
            
        Returns:
            Output size for the head
        """
        # Use num_labels if provided, otherwise num_classes
        if num_labels is not None:
            return num_labels
        if num_classes is not None:
            return num_classes
            
        # Default sizes for different head types
        default_sizes = {
            "binary_classification": 2,
            "regression": 1,
            "ordinal_regression": 5,  # Common default
            "time_series": 1,
            "ranking": 1,
        }
        
        return default_sizes.get(head_type, 2)