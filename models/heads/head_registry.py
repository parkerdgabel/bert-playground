"""Head registry system for dynamic head selection and configuration.

This module provides a centralized registry for all head types,
enabling dynamic head selection based on competition characteristics.
"""

from typing import Dict, Type, Optional, List, Callable, Any
from dataclasses import dataclass
from enum import Enum
import inspect

from .base_head import BaseKaggleHead, HeadType, HeadConfig, get_default_config_for_head_type
from loguru import logger


class CompetitionType(Enum):
    """Competition types for automatic head selection."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ORDINAL_REGRESSION = "ordinal_regression"
    TIME_SERIES = "time_series"
    RANKING = "ranking"
    RECOMMENDATION = "recommendation"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    UNKNOWN = "unknown"


@dataclass
class HeadSpec:
    """Specification for a registered head."""
    head_class: Type[BaseKaggleHead]
    head_type: HeadType
    competition_types: List[CompetitionType]
    priority: int = 0  # Higher priority heads are selected first
    requirements: List[str] = None  # Optional requirements (e.g., specific input shapes)
    description: str = ""
    
    def __post_init__(self):
        if self.requirements is None:
            self.requirements = []


class HeadRegistry:
    """Registry for managing all head types and their configurations."""
    
    def __init__(self):
        self._heads: Dict[str, HeadSpec] = {}
        self._competition_mapping: Dict[CompetitionType, List[str]] = {}
        self._head_type_mapping: Dict[HeadType, List[str]] = {}
        
        logger.info("Initialized HeadRegistry")
    
    def register_head(
        self,
        name: str,
        head_class: Type[BaseKaggleHead],
        head_type: HeadType,
        competition_types: List[CompetitionType],
        priority: int = 0,
        requirements: Optional[List[str]] = None,
        description: str = ""
    ) -> None:
        """Register a new head type.
        
        Args:
            name: Unique name for the head
            head_class: Head class to register
            head_type: Type of head
            competition_types: List of compatible competition types
            priority: Priority for automatic selection (higher = preferred)
            requirements: Optional requirements
            description: Description of the head
        """
        if name in self._heads:
            logger.warning(f"Head '{name}' already registered, overwriting")
        
        # Validate head class
        if not issubclass(head_class, BaseKaggleHead):
            raise ValueError(f"Head class {head_class} must inherit from BaseKaggleHead")
        
        spec = HeadSpec(
            head_class=head_class,
            head_type=head_type,
            competition_types=competition_types,
            priority=priority,
            requirements=requirements or [],
            description=description
        )
        
        self._heads[name] = spec
        
        # Update competition mapping
        for comp_type in competition_types:
            if comp_type not in self._competition_mapping:
                self._competition_mapping[comp_type] = []
            self._competition_mapping[comp_type].append(name)
            # Sort by priority (descending)
            self._competition_mapping[comp_type].sort(
                key=lambda x: self._heads[x].priority, 
                reverse=True
            )
        
        # Update head type mapping
        if head_type not in self._head_type_mapping:
            self._head_type_mapping[head_type] = []
        self._head_type_mapping[head_type].append(name)
        self._head_type_mapping[head_type].sort(
            key=lambda x: self._heads[x].priority, 
            reverse=True
        )
        
        logger.info(f"Registered head '{name}' with type {head_type} and priority {priority}")
    
    def get_head_class(self, name: str) -> Type[BaseKaggleHead]:
        """Get head class by name.
        
        Args:
            name: Head name
            
        Returns:
            Head class
            
        Raises:
            KeyError: If head not found
        """
        if name not in self._heads:
            raise KeyError(f"Head '{name}' not found in registry")
        
        return self._heads[name].head_class
    
    def get_head_spec(self, name: str) -> HeadSpec:
        """Get head specification by name.
        
        Args:
            name: Head name
            
        Returns:
            Head specification
            
        Raises:
            KeyError: If head not found
        """
        if name not in self._heads:
            raise KeyError(f"Head '{name}' not found in registry")
        
        return self._heads[name]
    
    def list_heads(self) -> List[str]:
        """List all registered head names.
        
        Returns:
            List of head names
        """
        return list(self._heads.keys())
    
    def list_heads_by_type(self, head_type: HeadType) -> List[str]:
        """List heads by type.
        
        Args:
            head_type: Head type to filter by
            
        Returns:
            List of head names
        """
        return self._head_type_mapping.get(head_type, [])
    
    def list_heads_by_competition(self, competition_type: CompetitionType) -> List[str]:
        """List heads by competition type.
        
        Args:
            competition_type: Competition type to filter by
            
        Returns:
            List of head names sorted by priority
        """
        return self._competition_mapping.get(competition_type, [])
    
    def select_best_head(
        self,
        competition_type: CompetitionType,
        requirements: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select the best head for a given competition type.
        
        Args:
            competition_type: Type of competition
            requirements: Optional requirements to match
            exclude: Optional list of head names to exclude
            
        Returns:
            Best head name or None if no suitable head found
        """
        candidates = self.list_heads_by_competition(competition_type)
        
        if exclude:
            candidates = [name for name in candidates if name not in exclude]
        
        if not candidates:
            logger.warning(f"No heads found for competition type {competition_type}")
            return None
        
        # Filter by requirements if provided
        if requirements:
            filtered_candidates = []
            for name in candidates:
                head_spec = self._heads[name]
                # Check if all requirements are met
                if all(req in head_spec.requirements for req in requirements):
                    filtered_candidates.append(name)
            
            if filtered_candidates:
                candidates = filtered_candidates
            else:
                logger.warning(f"No heads found matching requirements {requirements}")
        
        # Return the highest priority head
        best_head = candidates[0]
        logger.info(f"Selected head '{best_head}' for competition type {competition_type}")
        return best_head
    
    def create_head(
        self,
        name: str,
        config: HeadConfig,
        **kwargs
    ) -> BaseKaggleHead:
        """Create a head instance by name.
        
        Args:
            name: Head name
            config: Head configuration
            **kwargs: Additional arguments passed to head constructor
            
        Returns:
            Head instance
        """
        head_class = self.get_head_class(name)
        
        # Create head instance
        head = head_class(config, **kwargs)
        
        logger.info(f"Created head instance '{name}' with config {config}")
        return head
    
    def create_head_from_competition(
        self,
        competition_type: CompetitionType,
        input_size: int,
        output_size: int,
        head_name: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseKaggleHead:
        """Create a head optimized for a specific competition type.
        
        Args:
            competition_type: Type of competition
            input_size: Input feature size
            output_size: Output size
            head_name: Optional specific head name (auto-select if None)
            config_overrides: Optional configuration overrides
            **kwargs: Additional arguments passed to head constructor
            
        Returns:
            Head instance
        """
        # Select head if not specified
        if head_name is None:
            head_name = self.select_best_head(competition_type)
            if head_name is None:
                raise ValueError(f"No suitable head found for competition type {competition_type}")
        
        # Get head spec
        head_spec = self.get_head_spec(head_name)
        
        # Create default config
        config = get_default_config_for_head_type(
            head_spec.head_type,
            input_size,
            output_size
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown config parameter: {key}")
        
        # Create head
        return self.create_head(head_name, config, **kwargs)
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the registry.
        
        Returns:
            Dictionary with registry information
        """
        return {
            "total_heads": len(self._heads),
            "heads_by_type": {
                head_type.value: len(heads) 
                for head_type, heads in self._head_type_mapping.items()
            },
            "heads_by_competition": {
                comp_type.value: len(heads)
                for comp_type, heads in self._competition_mapping.items()
            },
            "all_heads": {
                name: {
                    "type": spec.head_type.value,
                    "competitions": [ct.value for ct in spec.competition_types],
                    "priority": spec.priority,
                    "description": spec.description
                }
                for name, spec in self._heads.items()
            }
        }


# Global registry instance
_global_registry = HeadRegistry()


def get_head_registry() -> HeadRegistry:
    """Get the global head registry instance.
    
    Returns:
        Global HeadRegistry instance
    """
    return _global_registry


def register_head(*args, **kwargs) -> None:
    """Register a head in the global registry.
    
    Args:
        *args: Arguments passed to HeadRegistry.register_head
        **kwargs: Keyword arguments passed to HeadRegistry.register_head
    """
    _global_registry.register_head(*args, **kwargs)


def create_head_from_competition(*args, **kwargs) -> BaseKaggleHead:
    """Create a head from competition type using global registry.
    
    Args:
        *args: Arguments passed to HeadRegistry.create_head_from_competition
        **kwargs: Keyword arguments passed to HeadRegistry.create_head_from_competition
        
    Returns:
        Head instance
    """
    return _global_registry.create_head_from_competition(*args, **kwargs)


def list_available_heads() -> List[str]:
    """List all available heads in the global registry.
    
    Returns:
        List of head names
    """
    return _global_registry.list_heads()


def get_registry_info() -> Dict[str, Any]:
    """Get information about the global registry.
    
    Returns:
        Registry information
    """
    return _global_registry.get_registry_info()


# Decorator for automatic head registration
def register_head_class(
    name: str,
    head_type: HeadType,
    competition_types: List[CompetitionType],
    priority: int = 0,
    requirements: Optional[List[str]] = None,
    description: str = ""
):
    """Decorator to automatically register a head class.
    
    Args:
        name: Head name
        head_type: Head type
        competition_types: Compatible competition types
        priority: Priority for selection
        requirements: Optional requirements
        description: Description
    """
    def decorator(head_class: Type[BaseKaggleHead]):
        register_head(
            name=name,
            head_class=head_class,
            head_type=head_type,
            competition_types=competition_types,
            priority=priority,
            requirements=requirements,
            description=description
        )
        return head_class
    
    return decorator


# Utility functions for competition type mapping
def infer_competition_type(
    num_classes: int,
    is_regression: bool = False,
    is_multilabel: bool = False,
    is_ranking: bool = False,
    is_time_series: bool = False
) -> CompetitionType:
    """Infer competition type from dataset characteristics.
    
    Args:
        num_classes: Number of classes/targets
        is_regression: Whether it's a regression task
        is_multilabel: Whether it's multilabel classification
        is_ranking: Whether it's a ranking task
        is_time_series: Whether it's time series prediction
        
    Returns:
        Inferred competition type
    """
    if is_time_series:
        return CompetitionType.TIME_SERIES
    elif is_ranking:
        return CompetitionType.RANKING
    elif is_regression:
        return CompetitionType.REGRESSION
    elif is_multilabel:
        return CompetitionType.MULTILABEL_CLASSIFICATION
    elif num_classes == 2:
        return CompetitionType.BINARY_CLASSIFICATION
    elif num_classes > 2:
        return CompetitionType.MULTICLASS_CLASSIFICATION
    else:
        return CompetitionType.UNKNOWN


def get_head_type_from_competition(competition_type: CompetitionType) -> HeadType:
    """Get corresponding head type from competition type.
    
    Args:
        competition_type: Competition type
        
    Returns:
        Corresponding head type
    """
    mapping = {
        CompetitionType.BINARY_CLASSIFICATION: HeadType.BINARY_CLASSIFICATION,
        CompetitionType.MULTICLASS_CLASSIFICATION: HeadType.MULTICLASS_CLASSIFICATION,
        CompetitionType.MULTILABEL_CLASSIFICATION: HeadType.MULTILABEL_CLASSIFICATION,
        CompetitionType.REGRESSION: HeadType.REGRESSION,
        CompetitionType.ORDINAL_REGRESSION: HeadType.ORDINAL_REGRESSION,
        CompetitionType.TIME_SERIES: HeadType.TIME_SERIES,
        CompetitionType.RANKING: HeadType.RANKING,
    }
    
    return mapping.get(competition_type, HeadType.MULTICLASS_CLASSIFICATION)