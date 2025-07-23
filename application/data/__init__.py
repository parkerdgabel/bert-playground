"""Domain models and services for data handling.

This package contains pure business logic for:
- Data configuration and validation
- Data processing pipelines
- Dataset management
- Data quality checks

All data handling logic is framework-agnostic.
"""

from .data_models import (
    DatasetType,
    InputFormat,
    TaskDataType,
    DataConfig,
    DataStatistics,
    TextExample,
    TabularExample,
    TokenizedExample,
    DataProcessor,
    TextProcessor,
    TabularProcessor,
    DataAugmenter,
    DataSplit,
    DataValidator,
)

from .data_service import (
    DatasetInfo,
    DataService,
    BatchProcessor,
    CacheConfig,
    DataCache,
    DataPipeline,
    DatasetIterator,
    StreamingDataService,
    DataQualityReport,
)

# Import new domain models and interfaces
from .models import (
    CompetitionType, DatasetSpec, DataSample, DataBatch, Dataset,
    DatasetRepository, DataValidationResult, DataProcessor as DomainDataProcessor
)

from .interfaces import (
    DataRepository, DataCache as DataCacheInterface, TokenizerAdapter, DataLoader,
    DataValidatorAdapter, TextProcessorAdapter, DataAugmentationAdapter,
    TemplateEngine, DatasetFactory, MetricsCollector, FileSystemAdapter,
    ComputeBackendAdapter
)

from .services import (
    DataProcessingService, DataPreprocessingService, DataPipelineService
)

# Import components from reorganized structure
from .templates import Template, TemplateConfig, TemplateRegistry
from .augmentation import BaseAugmenter, FeatureType, AugmentationMode
from .validation import FieldType, SchemaField, Schema, SchemaValidator

__all__ = [
    # New Domain Models
    "CompetitionType", "DatasetSpec", "DataSample", "DataBatch", "Dataset",
    "DatasetRepository", "DataValidationResult", "DomainDataProcessor",
    
    # New Domain Interfaces  
    "DataRepository", "DataCacheInterface", "TokenizerAdapter", "DataLoader",
    "DataValidatorAdapter", "TextProcessorAdapter", "DataAugmentationAdapter",
    "TemplateEngine", "DatasetFactory", "MetricsCollector", "FileSystemAdapter",
    "ComputeBackendAdapter",
    
    # New Domain Services
    "DataProcessingService", "DataPreprocessingService", "DataPipelineService",
    
    # Legacy Data Models
    "DatasetType",
    "InputFormat",
    "TaskDataType",
    "DataConfig",
    "DataStatistics",
    "TextExample",
    "TabularExample", 
    "TokenizedExample",
    "DataProcessor",
    "TextProcessor",
    "TabularProcessor",
    "DataAugmenter",
    "DataSplit",
    "DataValidator",
    
    # Legacy Data Services
    "DatasetInfo",
    "DataService",
    "BatchProcessor",
    "CacheConfig",
    "DataCache",
    "DataPipeline",
    "DatasetIterator",
    "StreamingDataService",
    "DataQualityReport",
    
    # Templates
    "Template",
    "TemplateConfig", 
    "TemplateRegistry",
    
    # Augmentation
    "BaseAugmenter",
    "FeatureType",
    "AugmentationMode",
    
    # Validation
    "FieldType",
    "SchemaField",
    "Schema", 
    "SchemaValidator",
]