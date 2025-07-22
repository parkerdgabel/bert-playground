"""Advanced builder pattern for dataset creation with fluent API."""

from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import pandas as pd

from loguru import logger

from .core.base import DatasetSpec, CompetitionType, KaggleDataset
from .templates import get_template, TemplateConfig
from .augmentation import get_registry as get_augmentation_registry
from .cache import create_cache, CacheStrategy
from .validation import SchemaValidator, Schema
from .pipeline import PipelineBuilder


class DatasetBuilder:
    """Fluent API builder for creating datasets with advanced features."""
    
    def __init__(self, name: str = "dataset"):
        """Initialize dataset builder.
        
        Args:
            name: Name for the dataset
        """
        self.name = name
        self._data_path: Optional[Path] = None
        self._spec: Optional[DatasetSpec] = None
        self._template = None
        self._augmenter = None
        self._cache_config = None
        self._validator = None
        self._pipeline = None
        self._transformations: List[Callable] = []
        self._metadata: Dict[str, Any] = {}
        
        logger.debug(f"Initialized DatasetBuilder: {name}")
    
    def from_csv(self, path: Union[str, Path]) -> "DatasetBuilder":
        """Set data source as CSV file.
        
        Args:
            path: Path to CSV file
            
        Returns:
            Self for chaining
        """
        self._data_path = Path(path)
        logger.debug(f"Set CSV source: {self._data_path}")
        return self
    
    def with_spec(
        self,
        competition_name: str,
        competition_type: Union[str, CompetitionType],
        text_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        **kwargs
    ) -> "DatasetBuilder":
        """Set dataset specification.
        
        Args:
            competition_name: Name of the competition
            competition_type: Type of competition
            text_columns: List of text column names
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            target_column: Target column name
            **kwargs: Additional spec parameters
            
        Returns:
            Self for chaining
        """
        if isinstance(competition_type, str):
            competition_type = CompetitionType(competition_type)
        
        self._spec = DatasetSpec(
            competition_name=competition_name,
            dataset_path=self._data_path or "",
            competition_type=competition_type,
            num_samples=0,  # Will be updated when data is loaded
            num_features=0,  # Will be updated when data is loaded
            text_columns=text_columns or [],
            categorical_columns=categorical_columns or [],
            numerical_columns=numerical_columns or [],
            target_column=target_column,
            **kwargs
        )
        
        logger.debug(f"Set dataset spec: {competition_name} ({competition_type})")
        return self
    
    def with_template(
        self,
        template: Union[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> "DatasetBuilder":
        """Set text template for data conversion.
        
        Args:
            template: Template name or instance
            config: Template configuration
            
        Returns:
            Self for chaining
        """
        if isinstance(template, str):
            template_config = TemplateConfig(**config) if config else None
            self._template = get_template(template, template_config)
        else:
            self._template = template
        
        logger.debug(f"Set template: {template}")
        return self
    
    def with_augmentation(
        self,
        augmenter: Union[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> "DatasetBuilder":
        """Set data augmentation strategy.
        
        Args:
            augmenter: Augmenter name or instance
            config: Augmentation configuration
            
        Returns:
            Self for chaining
        """
        if isinstance(augmenter, str):
            registry = get_augmentation_registry()
            self._augmenter = registry.get(augmenter, config)
        else:
            self._augmenter = augmenter
        
        logger.debug(f"Set augmenter: {augmenter}")
        return self
    
    def with_cache(
        self,
        strategy: Union[str, CacheStrategy] = "lru",
        max_size: int = 1000,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> "DatasetBuilder":
        """Configure caching strategy.
        
        Args:
            strategy: Cache strategy
            max_size: Maximum cache size
            cache_dir: Cache directory for disk-based caches
            **kwargs: Additional cache options
            
        Returns:
            Self for chaining
        """
        self._cache_config = {
            "strategy": strategy,
            "max_size": max_size,
            "cache_dir": cache_dir,
            **kwargs
        }
        
        logger.debug(f"Set cache config: {strategy}")
        return self
    
    def with_validation(
        self,
        schema: Optional[Schema] = None,
        validator: Optional[SchemaValidator] = None
    ) -> "DatasetBuilder":
        """Configure data validation.
        
        Args:
            schema: Data schema for validation
            validator: Custom validator instance
            
        Returns:
            Self for chaining
        """
        if validator:
            self._validator = validator
        elif schema:
            self._validator = SchemaValidator(schema)
        
        logger.debug(f"Set validator: {type(self._validator).__name__ if self._validator else 'None'}")
        return self
    
    def with_pipeline(self, pipeline_builder: PipelineBuilder) -> "DatasetBuilder":
        """Set data processing pipeline.
        
        Args:
            pipeline_builder: Configured pipeline builder
            
        Returns:
            Self for chaining
        """
        self._pipeline = pipeline_builder.build()
        logger.debug("Set data processing pipeline")
        return self
    
    def add_transformation(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> "DatasetBuilder":
        """Add a custom transformation function.
        
        Args:
            func: Transformation function
            
        Returns:
            Self for chaining
        """
        self._transformations.append(func)
        logger.debug(f"Added transformation: {func.__name__}")
        return self
    
    def with_metadata(self, **metadata) -> "DatasetBuilder":
        """Add metadata to the dataset.
        
        Args:
            **metadata: Metadata key-value pairs
            
        Returns:
            Self for chaining
        """
        self._metadata.update(metadata)
        logger.debug(f"Added metadata: {list(metadata.keys())}")
        return self
    
    def auto_configure(self, sample_size: int = 1000) -> "DatasetBuilder":
        """Automatically configure the dataset based on data analysis.
        
        Args:
            sample_size: Number of rows to analyze for auto-configuration
            
        Returns:
            Self for chaining
        """
        if not self._data_path or not self._data_path.exists():
            logger.warning("Cannot auto-configure: data path not set or doesn't exist")
            return self
        
        # Load sample data for analysis
        sample_data = pd.read_csv(self._data_path, nrows=sample_size)
        
        # Auto-detect column types
        text_columns = []
        categorical_columns = []
        numerical_columns = []
        
        for col in sample_data.columns:
            if sample_data[col].dtype == 'object':
                # Check if it's likely text or categorical
                avg_length = sample_data[col].astype(str).str.len().mean()
                unique_ratio = sample_data[col].nunique() / len(sample_data)
                
                if avg_length > 50 or unique_ratio > 0.5:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        
        # Auto-detect target column
        target_column = None
        common_target_names = ['target', 'label', 'class', 'y', 'survived']
        for col in common_target_names:
            if col.lower() in [c.lower() for c in sample_data.columns]:
                target_column = col
                break
        
        # Auto-detect competition type
        competition_type = CompetitionType.UNKNOWN
        if target_column and target_column in sample_data.columns:
            unique_targets = sample_data[target_column].nunique()
            if unique_targets == 2:
                competition_type = CompetitionType.BINARY_CLASSIFICATION
            elif unique_targets > 2 and unique_targets < 20:
                competition_type = CompetitionType.MULTICLASS_CLASSIFICATION
            elif sample_data[target_column].dtype in ['float64', 'int64']:
                competition_type = CompetitionType.REGRESSION
        
        # Create spec if not already set
        if not self._spec:
            self.with_spec(
                competition_name=self._data_path.stem,
                competition_type=competition_type,
                text_columns=text_columns,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                target_column=target_column
            )
        
        # Set default template if not set
        if not self._template:
            if text_columns:
                self.with_template("natural")
            else:
                self.with_template("keyvalue")
        
        # Set default cache if not set
        if not self._cache_config:
            self.with_cache()
        
        logger.info(f"Auto-configured dataset: {len(text_columns)} text, {len(categorical_columns)} categorical, {len(numerical_columns)} numerical columns")
        return self
    
    def build(self) -> KaggleDataset:
        """Build the configured dataset.
        
        Returns:
            Configured dataset instance
        """
        if not self._data_path:
            raise ValueError("Data path must be set")
        
        if not self._spec:
            logger.info("No spec provided, auto-configuring...")
            self.auto_configure()
        
        # Create dataset
        from .factory import CSVDataset
        
        dataset = CSVDataset(
            csv_path=self._data_path,
            spec=self._spec,
            text_converter=self._template,
            augmenter=self._augmenter,
        )
        
        # Apply transformations
        if self._transformations:
            original_data = dataset._data.copy()
            for transform in self._transformations:
                dataset._data = transform(dataset._data)
            logger.info(f"Applied {len(self._transformations)} transformations")
        
        # Setup caching
        if self._cache_config:
            cache = create_cache(**self._cache_config)
            dataset.enable_caching(cache.config.cache_dir)
        
        # Validate data
        if self._validator:
            is_valid, errors = self._validator.validate(dataset._data)
            if not is_valid:
                logger.warning(f"Dataset validation failed: {errors}")
            else:
                logger.info("Dataset validation passed")
        
        # Apply pipeline if set
        if self._pipeline:
            dataset._data = self._pipeline.run(dataset._data)
            logger.info("Applied data processing pipeline")
        
        # Add metadata
        if self._metadata:
            if not hasattr(dataset, 'metadata'):
                dataset.metadata = {}
            dataset.metadata.update(self._metadata)
        
        logger.info(f"Built dataset '{self.name}' with {len(dataset)} samples")
        return dataset
    
    @classmethod
    def quick_csv(
        cls,
        path: Union[str, Path],
        competition_name: Optional[str] = None,
        auto_configure: bool = True
    ) -> "DatasetBuilder":
        """Quick builder for CSV datasets with sensible defaults.
        
        Args:
            path: Path to CSV file
            competition_name: Name of competition (defaults to filename)
            auto_configure: Whether to auto-configure the dataset
            
        Returns:
            Configured dataset builder
        """
        path = Path(path)
        name = competition_name or path.stem
        
        builder = cls(name).from_csv(path)
        
        if auto_configure:
            builder.auto_configure()
        
        return builder
    
    @classmethod  
    def for_competition(
        cls,
        competition_name: str,
        train_path: Union[str, Path],
        competition_type: Union[str, CompetitionType],
        target_column: str,
        **kwargs
    ) -> "DatasetBuilder":
        """Builder preset for competition datasets.
        
        Args:
            competition_name: Name of the competition
            train_path: Path to training data
            competition_type: Type of competition
            target_column: Target column name
            **kwargs: Additional spec parameters
            
        Returns:
            Configured dataset builder
        """
        return (cls(competition_name)
                .from_csv(train_path)
                .with_spec(
                    competition_name=competition_name,
                    competition_type=competition_type,
                    target_column=target_column,
                    **kwargs
                )
                .with_cache()
                .auto_configure())