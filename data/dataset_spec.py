"""Dataset specification system for Kaggle competitions.

This module defines the configuration system for describing Kaggle datasets
and their characteristics for automated processing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
from loguru import logger


class ProblemType(Enum):
    """Types of ML problems supported."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    TIME_SERIES = "time_series"


class FeatureType(Enum):
    """Types of features in datasets."""
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TEXT = "text"
    DATETIME = "datetime"
    ID = "id"
    TARGET = "target"


class OptimizationProfile(Enum):
    """Data loading optimization profiles."""
    DEVELOPMENT = "development"  # Fast iteration, minimal optimization
    PRODUCTION = "production"    # Balanced performance and resource usage
    COMPETITION = "competition"  # Maximum performance optimization
    AUTO = "auto"               # Automatically choose based on dataset size


@dataclass
class KaggleDatasetSpec:
    """Specification for a Kaggle dataset configuration."""
    
    # Basic dataset information
    name: str
    problem_type: ProblemType
    target_column: str
    
    # Feature specifications
    feature_types: Dict[str, FeatureType] = field(default_factory=dict)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    text_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    id_columns: List[str] = field(default_factory=list)
    
    # Text generation strategy
    text_template_strategy: str = "auto"
    primary_text_features: List[str] = field(default_factory=list)
    
    # Optimization settings
    optimization_profile: OptimizationProfile = OptimizationProfile.AUTO
    
    # Dataset-specific settings
    has_missing_values: bool = True
    missing_value_strategy: str = "auto"
    
    # Performance hints
    expected_size: Optional[int] = None  # Number of rows
    memory_usage_hint: str = "medium"    # "low", "medium", "high"
    
    def __post_init__(self):
        """Validate and normalize the specification."""
        # Ensure enums are properly set
        if isinstance(self.problem_type, str):
            self.problem_type = ProblemType(self.problem_type)
        if isinstance(self.optimization_profile, str):
            self.optimization_profile = OptimizationProfile(self.optimization_profile)
            
        # Build feature_types from individual lists
        if not self.feature_types:
            self.feature_types = {}
            
        for col in self.categorical_columns:
            self.feature_types[col] = FeatureType.CATEGORICAL
        for col in self.numerical_columns:
            self.feature_types[col] = FeatureType.NUMERICAL
        for col in self.text_columns:
            self.feature_types[col] = FeatureType.TEXT
        for col in self.datetime_columns:
            self.feature_types[col] = FeatureType.DATETIME
        for col in self.id_columns:
            self.feature_types[col] = FeatureType.ID
            
        # Mark target column
        if self.target_column:
            self.feature_types[self.target_column] = FeatureType.TARGET
    
    @classmethod
    def from_csv_analysis(cls, csv_path: str, target_column: str, name: Optional[str] = None) -> "KaggleDatasetSpec":
        """Auto-generate specification by analyzing a CSV file."""
        csv_path = Path(csv_path)
        if name is None:
            name = csv_path.stem
            
        logger.info(f"Analyzing CSV file: {csv_path}")
        
        # Load a sample of the data for analysis
        try:
            df = pd.read_csv(csv_path, nrows=1000)  # Sample for analysis
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise
            
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Auto-detect problem type
        if target_column in df.columns:
            target_series = df[target_column].dropna()
            unique_values = target_series.nunique()
            
            if target_series.dtype in ['int64', 'float64']:
                if unique_values == 2:
                    problem_type = ProblemType.BINARY_CLASSIFICATION
                elif unique_values <= 10 and target_series.dtype == 'int64':  # Integer multiclass
                    problem_type = ProblemType.MULTICLASS_CLASSIFICATION
                elif unique_values / len(target_series) < 0.1:  # Low cardinality
                    problem_type = ProblemType.MULTICLASS_CLASSIFICATION
                else:
                    problem_type = ProblemType.REGRESSION
            else:
                if unique_values <= 20:
                    problem_type = ProblemType.MULTICLASS_CLASSIFICATION
                else:
                    problem_type = ProblemType.REGRESSION
        else:
            # Default for test data
            problem_type = ProblemType.BINARY_CLASSIFICATION
            
        logger.info(f"Detected problem type: {problem_type}")
        
        # Auto-detect feature types
        categorical_columns = []
        numerical_columns = []
        text_columns = []
        datetime_columns = []
        id_columns = []
        
        for col in df.columns:
            if col == target_column:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            # Check for ID columns (unique values, often named with 'id')
            # But exclude columns that look like text (Name, Title, etc.)
            if (series.nunique() == len(series) and 
                ('id' in col.lower() or col.lower().endswith('_id')) and
                col.lower() not in ['name', 'title', 'description']):
                id_columns.append(col)
                continue
                
            # Check for datetime
            if series.dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(series.iloc[:min(100, len(series))])
                    datetime_columns.append(col)
                    continue
                except:
                    pass
                    
            # Check data type and unique values
            if series.dtype in ['int64', 'float64']:
                # Numerical
                numerical_columns.append(col)
            elif series.dtype == 'object':
                unique_ratio = series.nunique() / len(series)
                avg_length = series.astype(str).str.len().mean()
                
                # Heuristics for text vs categorical
                if unique_ratio > 0.8 or avg_length > 50:
                    text_columns.append(col)
                else:
                    categorical_columns.append(col)
            else:
                # Default to categorical for other types
                categorical_columns.append(col)
        
        logger.info(f"Feature type detection:")
        logger.info(f"  Categorical: {categorical_columns}")
        logger.info(f"  Numerical: {numerical_columns}")
        logger.info(f"  Text: {text_columns}")
        logger.info(f"  Datetime: {datetime_columns}")
        logger.info(f"  ID: {id_columns}")
        
        # Determine primary text features for template generation
        primary_text_features = text_columns[:3]  # Use up to 3 main text features
        if not primary_text_features:
            # If no text columns, use top categorical and numerical
            primary_text_features = (categorical_columns[:2] + numerical_columns[:2])[:3]
            
        # Determine optimization profile based on dataset size
        dataset_size = len(df)
        if dataset_size < 5000:
            optimization_profile = OptimizationProfile.DEVELOPMENT
        elif dataset_size < 50000:
            optimization_profile = OptimizationProfile.PRODUCTION
        else:
            optimization_profile = OptimizationProfile.COMPETITION
            
        return cls(
            name=name,
            problem_type=problem_type,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            datetime_columns=datetime_columns,
            id_columns=id_columns,
            primary_text_features=primary_text_features,
            optimization_profile=optimization_profile,
            expected_size=dataset_size,
            has_missing_values=df.isnull().any().any(),
        )
    
    @classmethod
    def from_dataframe_analysis(cls, df: pd.DataFrame, target_column: str, name: Optional[str] = None) -> "KaggleDatasetSpec":
        """Create dataset specification from DataFrame analysis.
        
        Args:
            df: DataFrame to analyze
            target_column: Name of the target column
            name: Optional name for the dataset
            
        Returns:
            KaggleDatasetSpec instance
        """
        if name is None:
            name = f"dataset_{len(df)}_samples"
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Use the same analysis logic as from_csv_analysis
        logger.info(f"Analyzing DataFrame with {len(df)} samples and {len(df.columns)} columns")
        
        # Analyze target column to determine problem type
        target_series = df[target_column]
        unique_values = target_series.nunique()
        
        if target_series.dtype in ['object', 'category']:
            if unique_values == 2:
                problem_type = ProblemType.BINARY_CLASSIFICATION
            else:
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
        else:
            # Check if it's regression or classification with numeric labels
            if unique_values <= 10 and target_series.dtype in ['int64', 'int32']:
                problem_type = ProblemType.MULTICLASS_CLASSIFICATION
            else:
                problem_type = ProblemType.REGRESSION
        
        # Categorize columns by type
        categorical_columns = []
        numerical_columns = []
        text_columns = []
        datetime_columns = []
        id_columns = []
        
        for col in df.columns:
            if col == target_column:
                continue
                
            series = df[col]
            
            # Check for datetime
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(col)
            # Check for text (object type with high cardinality)
            elif series.dtype == 'object':
                # If most values are unique, likely text or ID
                if series.nunique() / len(series) > 0.9:
                    if 'id' in col.lower() or col.lower().endswith('id'):
                        id_columns.append(col)
                    else:
                        text_columns.append(col)
                else:
                    categorical_columns.append(col)
            # Check for numerical
            elif pd.api.types.is_numeric_dtype(series):
                # If low cardinality, might be categorical
                if series.nunique() <= 10 and series.dtype in ['int64', 'int32']:
                    categorical_columns.append(col)
                else:
                    numerical_columns.append(col)
            else:
                # Default to categorical
                categorical_columns.append(col)
        
        logger.info(f"  Problem type: {problem_type}")
        logger.info(f"  Categorical: {categorical_columns}")
        logger.info(f"  Numerical: {numerical_columns}")
        logger.info(f"  Text: {text_columns}")
        logger.info(f"  Datetime: {datetime_columns}")
        logger.info(f"  ID: {id_columns}")
        
        # Determine primary text features for template generation
        primary_text_features = text_columns[:3]  # Use up to 3 main text features
        if not primary_text_features:
            # If no text columns, use top categorical and numerical
            primary_text_features = (categorical_columns[:2] + numerical_columns[:2])[:3]
            
        # Determine optimization profile based on dataset size
        dataset_size = len(df)
        if dataset_size < 5000:
            optimization_profile = OptimizationProfile.DEVELOPMENT
        elif dataset_size < 50000:
            optimization_profile = OptimizationProfile.PRODUCTION
        else:
            optimization_profile = OptimizationProfile.COMPETITION
            
        return cls(
            name=name,
            problem_type=problem_type,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            datetime_columns=datetime_columns,
            id_columns=id_columns,
            primary_text_features=primary_text_features,
            optimization_profile=optimization_profile,
            expected_size=dataset_size,
            has_missing_values=df.isnull().any().any(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "problem_type": self.problem_type.value,
            "target_column": self.target_column,
            "feature_types": {k: v.value for k, v in self.feature_types.items()},
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "text_columns": self.text_columns,
            "datetime_columns": self.datetime_columns,
            "id_columns": self.id_columns,
            "text_template_strategy": self.text_template_strategy,
            "primary_text_features": self.primary_text_features,
            "optimization_profile": self.optimization_profile.value,
            "has_missing_values": bool(self.has_missing_values),  # Ensure Python bool
            "missing_value_strategy": self.missing_value_strategy,
            "expected_size": int(self.expected_size) if self.expected_size is not None else None,
            "memory_usage_hint": self.memory_usage_hint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KaggleDatasetSpec":
        """Create from dictionary."""
        # Convert enum strings back to enums
        data = data.copy()
        data["problem_type"] = ProblemType(data["problem_type"])
        data["optimization_profile"] = OptimizationProfile(data["optimization_profile"])
        
        # Convert feature_types back to enums
        if "feature_types" in data:
            data["feature_types"] = {k: FeatureType(v) for k, v in data["feature_types"].items()}
            
        return cls(**data)


# Pre-defined specifications for common Kaggle datasets
TITANIC_SPEC = KaggleDatasetSpec(
    name="titanic",
    problem_type=ProblemType.BINARY_CLASSIFICATION,
    target_column="Survived",
    categorical_columns=["Sex", "Embarked", "Pclass"],
    numerical_columns=["Age", "SibSp", "Parch", "Fare"],
    text_columns=["Name"],
    id_columns=["PassengerId"],
    primary_text_features=["Name", "Sex", "Pclass"],
    optimization_profile=OptimizationProfile.DEVELOPMENT,
    expected_size=891,
)

# Registry for common dataset specifications
DATASET_REGISTRY = {
    "titanic": TITANIC_SPEC,
}


def get_dataset_spec(name_or_path: str, target_column: Optional[str] = None) -> KaggleDatasetSpec:
    """Get dataset specification by name or auto-detect from CSV path."""
    if name_or_path in DATASET_REGISTRY:
        return DATASET_REGISTRY[name_or_path]
    
    # Assume it's a file path
    if target_column is None:
        raise ValueError("target_column is required when auto-detecting from CSV path")
        
    return KaggleDatasetSpec.from_csv_analysis(name_or_path, target_column)


def register_dataset_spec(spec: KaggleDatasetSpec) -> None:
    """Register a new dataset specification."""
    DATASET_REGISTRY[spec.name] = spec
    logger.info(f"Registered dataset specification: {spec.name}")