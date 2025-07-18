"""
Feature-based text converter for structured data.
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig


@dataclass
class FeatureConfig(TextConversionConfig):
    """Configuration for feature-based conversion."""
    
    # Feature organization
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    feature_order: Optional[List[str]] = None
    
    # Feature descriptions
    feature_names: Dict[str, str] = field(default_factory=dict)
    feature_units: Dict[str, str] = field(default_factory=dict)
    
    # Numerical binning
    numerical_bins: Dict[str, List[Tuple[float, float, str]]] = field(default_factory=dict)
    
    # Categorical mappings
    categorical_mappings: Dict[str, Dict[Any, str]] = field(default_factory=dict)
    
    # Formatting options
    separator: str = ", "
    group_separator: str = ". "
    prefix: str = ""
    suffix: str = ""
    include_feature_names: bool = True
    skip_missing: bool = True
    
    # Value formatting
    numerical_precision: int = 2
    boolean_format: Tuple[str, str] = ("yes", "no")
    
    # Advanced features
    feature_interactions: List[Tuple[str, str, Callable]] = field(default_factory=list)
    computed_features: Dict[str, Callable[[Dict[str, Any]], Any]] = field(default_factory=dict)


class FeatureConverter(BaseTextConverter):
    """Convert data by organizing and formatting features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature converter.
        
        Args:
            config: Feature configuration
        """
        if config is None:
            config = FeatureConfig()
        super().__init__(config)
        
        self.config: FeatureConfig = config
    
    def convert(self, data: Dict[str, Any]) -> str:
        """Convert data to text by formatting features."""
        parts = []
        
        # Add prefix
        if self.config.prefix:
            parts.append(self.config.prefix)
        
        # Process features in order
        features_to_process = self._get_ordered_features(data)
        
        # Add computed features
        data_with_computed = self._add_computed_features(data)
        
        # Process feature groups
        if self.config.feature_groups:
            group_texts = []
            processed_features = set()
            
            for group_name, group_features in self.config.feature_groups.items():
                group_text = self._process_feature_group(
                    data_with_computed, group_features, group_name
                )
                if group_text:
                    group_texts.append(group_text)
                processed_features.update(group_features)
            
            # Process remaining features
            remaining_features = [f for f in features_to_process if f not in processed_features]
            if remaining_features:
                remaining_text = self._process_features(data_with_computed, remaining_features)
                if remaining_text:
                    group_texts.append(remaining_text)
            
            parts.extend(group_texts)
        else:
            # Process all features together
            feature_text = self._process_features(data_with_computed, features_to_process)
            if feature_text:
                parts.append(feature_text)
        
        # Add feature interactions
        interaction_text = self._process_feature_interactions(data_with_computed)
        if interaction_text:
            parts.append(interaction_text)
        
        # Add suffix
        if self.config.suffix:
            parts.append(self.config.suffix)
        
        # Join parts
        if self.config.feature_groups:
            return self.config.group_separator.join(parts)
        else:
            return " ".join(parts)
    
    def _get_ordered_features(self, data: Dict[str, Any]) -> List[str]:
        """Get features in specified order."""
        available_features = self._get_fields_to_use(data)
        
        if self.config.feature_order:
            # Use specified order
            ordered = []
            for feature in self.config.feature_order:
                if feature in available_features:
                    ordered.append(feature)
            
            # Add any remaining features
            for feature in available_features:
                if feature not in ordered:
                    ordered.append(feature)
            
            return ordered
        else:
            # Default order
            return sorted(available_features)
    
    def _add_computed_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add computed features to data."""
        if not self.config.computed_features:
            return data
        
        result = data.copy()
        
        for feature_name, compute_func in self.config.computed_features.items():
            try:
                result[feature_name] = compute_func(data)
            except Exception as e:
                logger.warning(f"Error computing feature {feature_name}: {e}")
                result[feature_name] = None
        
        return result
    
    def _process_feature_group(
        self,
        data: Dict[str, Any],
        features: List[str],
        group_name: str
    ) -> str:
        """Process a group of features."""
        group_parts = []
        
        # Optional group header
        if group_name and self.config.augment:
            group_parts.append(f"{group_name}:")
        
        # Process features in group
        feature_text = self._process_features(data, features)
        if feature_text:
            group_parts.append(feature_text)
        
        return " ".join(group_parts) if group_parts else ""
    
    def _process_features(self, data: Dict[str, Any], features: List[str]) -> str:
        """Process a list of features."""
        feature_parts = []
        
        for feature in features:
            value = self.get_field_value(data, feature)
            
            # Skip missing if configured
            if self._should_skip_feature(value):
                continue
            
            # Format feature
            formatted = self._format_feature(feature, value)
            if formatted:
                feature_parts.append(formatted)
        
        return self.config.separator.join(feature_parts)
    
    def _should_skip_feature(self, value: Any) -> bool:
        """Check if feature should be skipped."""
        if not self.config.skip_missing:
            return False
        
        return value is None or (isinstance(value, float) and np.isnan(value))
    
    def _format_feature(self, feature: str, value: Any) -> str:
        """Format a single feature."""
        # Get display name
        display_name = self.config.feature_names.get(feature, feature)
        
        # Format value
        formatted_value = self._format_value(feature, value)
        
        # Add unit if available
        if feature in self.config.feature_units and value is not None:
            formatted_value = f"{formatted_value} {self.config.feature_units[feature]}"
        
        # Combine name and value
        if self.config.include_feature_names:
            return f"{display_name}: {formatted_value}"
        else:
            return formatted_value
    
    def _format_value(self, feature: str, value: Any) -> str:
        """Format a value based on its type and configuration."""
        if value is None:
            return self.config.missing_value_text
        
        # Check categorical mapping
        if feature in self.config.categorical_mappings:
            mapping = self.config.categorical_mappings[feature]
            if value in mapping:
                return mapping[value]
        
        # Check numerical binning
        if feature in self.config.numerical_bins and isinstance(value, (int, float)):
            for min_val, max_val, label in self.config.numerical_bins[feature]:
                if min_val <= value < max_val:
                    return label
        
        # Type-based formatting
        if isinstance(value, bool):
            return self.config.boolean_format[0] if value else self.config.boolean_format[1]
        elif isinstance(value, (int, float)):
            if float(value).is_integer():
                return str(int(value))
            else:
                return f"{value:.{self.config.numerical_precision}f}"
        else:
            return str(value)
    
    def _process_feature_interactions(self, data: Dict[str, Any]) -> str:
        """Process feature interactions."""
        if not self.config.feature_interactions:
            return ""
        
        interaction_parts = []
        
        for feat1, feat2, interaction_func in self.config.feature_interactions:
            if feat1 in data and feat2 in data:
                try:
                    interaction_text = interaction_func(data[feat1], data[feat2])
                    if interaction_text:
                        interaction_parts.append(interaction_text)
                except Exception as e:
                    logger.warning(f"Error processing interaction {feat1} x {feat2}: {e}")
        
        return self.config.separator.join(interaction_parts)
    
    def add_feature_group(self, group_name: str, features: List[str]) -> None:
        """
        Add a feature group.
        
        Args:
            group_name: Name of the group
            features: List of features in the group
        """
        self.config.feature_groups[group_name] = features
    
    def add_numerical_bins(
        self,
        feature: str,
        bins: List[Tuple[float, float, str]]
    ) -> None:
        """
        Add numerical binning for a feature.
        
        Args:
            feature: Feature name
            bins: List of (min, max, label) tuples
        """
        self.config.numerical_bins[feature] = bins
    
    def add_categorical_mapping(
        self,
        feature: str,
        mapping: Dict[Any, str]
    ) -> None:
        """
        Add categorical mapping for a feature.
        
        Args:
            feature: Feature name
            mapping: Value to description mapping
        """
        self.config.categorical_mappings[feature] = mapping
    
    def __repr__(self) -> str:
        """String representation."""
        return f"FeatureConverter(groups={len(self.config.feature_groups)})"