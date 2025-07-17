"""
Dataset registry and specifications for Kaggle competitions.

This module provides a centralized registry of dataset configurations
and metadata for easy dataset management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class ProblemType(Enum):
    """Types of ML problems."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification" 
    REGRESSION = "regression"
    RANKING = "ranking"


@dataclass
class DatasetSpec:
    """Specification for a Kaggle dataset."""
    
    # Basic info
    name: str
    problem_type: ProblemType
    description: str
    
    # Column specifications
    target_column: str
    feature_columns: List[str]
    id_columns: List[str] = field(default_factory=list)
    
    # Data characteristics
    num_classes: Optional[int] = None
    missing_values_expected: bool = True
    
    # Text generation hints
    text_columns: Optional[List[str]] = None
    column_descriptions: Dict[str, str] = field(default_factory=dict)
    value_mappings: Dict[str, Dict[Any, str]] = field(default_factory=dict)
    
    # Performance hints
    expected_size: Optional[int] = None
    recommended_batch_size: int = 32
    recommended_max_length: int = 128


# Predefined dataset specifications
DATASET_SPECS = {
    'titanic': DatasetSpec(
        name='titanic',
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        description='Titanic passenger survival prediction',
        target_column='Survived',
        feature_columns=['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'],
        id_columns=['PassengerId'],
        num_classes=2,
        text_columns=['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'],
        column_descriptions={
            'Pclass': 'Passenger Class',
            'Name': 'Passenger Name',
            'Sex': 'Gender',
            'Age': 'Age in years',
            'SibSp': 'Number of siblings/spouse aboard',
            'Parch': 'Number of parents/children aboard',
            'Ticket': 'Ticket number',
            'Fare': 'Ticket fare',
            'Embarked': 'Port of embarkation'
        },
        value_mappings={
            'Pclass': {1: 'first class', 2: 'second class', 3: 'third class'},
            'Embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
        },
        expected_size=891,
        recommended_batch_size=32,
        recommended_max_length=128
    ),
    
    'house_prices': DatasetSpec(
        name='house_prices',
        problem_type=ProblemType.REGRESSION,
        description='House price prediction',
        target_column='SalePrice',
        feature_columns=[
            'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
            'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
            'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
            'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
            'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
            'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
            'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
            'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
            'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
            'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold',
            'SaleType', 'SaleCondition'
        ],
        id_columns=['Id'],
        expected_size=1460,
        recommended_batch_size=16,
        recommended_max_length=256
    ),
    
    'digit_recognizer': DatasetSpec(
        name='digit_recognizer',
        problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
        description='MNIST digit recognition',
        target_column='label',
        feature_columns=[f'pixel{i}' for i in range(784)],
        id_columns=[],
        num_classes=10,
        missing_values_expected=False,
        expected_size=42000,
        recommended_batch_size=64,
        recommended_max_length=128
    ),
}


class DatasetRegistry:
    """Registry for managing dataset specifications."""
    
    def __init__(self):
        self._specs = DATASET_SPECS.copy()
    
    def register(self, spec: DatasetSpec) -> None:
        """Register a new dataset specification."""
        self._specs[spec.name] = spec
    
    def get(self, name: str) -> Optional[DatasetSpec]:
        """Get a dataset specification by name."""
        return self._specs.get(name)
    
    def list(self) -> List[str]:
        """List all registered dataset names."""
        return list(self._specs.keys())
    
    def get_config_for_dataloader(self, name: str) -> Dict[str, Any]:
        """Get dataloader configuration for a dataset."""
        spec = self.get(name)
        if not spec:
            return {}
        
        config = {
            'label_column': spec.target_column,
            'text_columns': spec.text_columns or spec.feature_columns,
            'batch_size': spec.recommended_batch_size,
            'max_length': spec.recommended_max_length,
        }
        
        return config


# Global registry instance
dataset_registry = DatasetRegistry()


# Convenience functions
def get_dataset_spec(name: str) -> Optional[DatasetSpec]:
    """Get a dataset specification."""
    return dataset_registry.get(name)


def register_dataset(spec: DatasetSpec) -> None:
    """Register a new dataset."""
    dataset_registry.register(spec)


def list_datasets() -> List[str]:
    """List all registered datasets."""
    return dataset_registry.list()