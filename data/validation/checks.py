"""Data quality checks for validation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import re

import pandas as pd
import numpy as np
from loguru import logger


class DataQualityCheck(ABC):
    """Abstract base class for data quality checks."""
    
    def __init__(self, name: Optional[str] = None):
        """Initialize check.
        
        Args:
            name: Name of the check
        """
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Perform the quality check.
        
        Args:
            data: Data to check
            
        Returns:
            Tuple of (is_valid, message)
        """
        pass
    
    def __call__(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Make check callable."""
        return self.check(data)


class CompletenessCheck(DataQualityCheck):
    """Check for data completeness (missing values)."""
    
    def __init__(
        self,
        max_missing_ratio: float = 0.0,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """Initialize completeness check.
        
        Args:
            max_missing_ratio: Maximum allowed ratio of missing values (0-1)
            columns: Specific columns to check (None = all columns)
            name: Name of the check
        """
        super().__init__(name)
        self.max_missing_ratio = max_missing_ratio
        self.columns = columns
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check for missing values."""
        if isinstance(data, pd.Series):
            missing_ratio = data.isna().sum() / len(data)
            if missing_ratio > self.max_missing_ratio:
                return False, f"Missing ratio {missing_ratio:.2%} exceeds threshold {self.max_missing_ratio:.2%}"
            return True, f"Missing ratio {missing_ratio:.2%} within threshold"
        
        elif isinstance(data, pd.DataFrame):
            columns_to_check = self.columns if self.columns else data.columns
            failed_columns = []
            
            for col in columns_to_check:
                if col in data.columns:
                    missing_ratio = data[col].isna().sum() / len(data)
                    if missing_ratio > self.max_missing_ratio:
                        failed_columns.append(f"{col} ({missing_ratio:.2%})")
            
            if failed_columns:
                return False, f"Columns with excessive missing values: {', '.join(failed_columns)}"
            return True, "All columns within missing value threshold"
        
        return False, f"Unsupported data type: {type(data)}"


class UniquenessCheck(DataQualityCheck):
    """Check for unique values."""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        min_unique_ratio: Optional[float] = None,
        max_duplicates: int = 0,
        name: Optional[str] = None
    ):
        """Initialize uniqueness check.
        
        Args:
            columns: Columns to check for uniqueness
            min_unique_ratio: Minimum ratio of unique values (0-1)
            max_duplicates: Maximum allowed duplicate values
            name: Name of the check
        """
        super().__init__(name)
        self.columns = columns
        self.min_unique_ratio = min_unique_ratio
        self.max_duplicates = max_duplicates
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check for unique values."""
        if isinstance(data, pd.Series):
            unique_ratio = data.nunique() / len(data)
            duplicates = len(data) - data.nunique()
            
            if self.min_unique_ratio and unique_ratio < self.min_unique_ratio:
                return False, f"Unique ratio {unique_ratio:.2%} below threshold {self.min_unique_ratio:.2%}"
            
            if duplicates > self.max_duplicates:
                return False, f"Found {duplicates} duplicates, exceeds threshold {self.max_duplicates}"
            
            return True, f"Uniqueness check passed (unique ratio: {unique_ratio:.2%})"
        
        elif isinstance(data, pd.DataFrame):
            if not self.columns:
                return False, "No columns specified for uniqueness check"
            
            failed_checks = []
            for col in self.columns:
                if col in data.columns:
                    is_valid, msg = self.check(data[col])
                    if not is_valid:
                        failed_checks.append(f"{col}: {msg}")
            
            if failed_checks:
                return False, "\n".join(failed_checks)
            return True, "All specified columns pass uniqueness check"
        
        return False, f"Unsupported data type: {type(data)}"


class RangeCheck(DataQualityCheck):
    """Check if values are within specified range."""
    
    def __init__(
        self,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """Initialize range check.
        
        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            columns: Columns to check (for DataFrame)
            name: Name of the check
        """
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.columns = columns
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check if values are within range."""
        if isinstance(data, pd.Series):
            # Only check numeric data
            if not pd.api.types.is_numeric_dtype(data):
                return True, "Non-numeric data, skipping range check"
            
            violations = []
            if self.min_value is not None:
                below_min = (data < self.min_value).sum()
                if below_min > 0:
                    violations.append(f"{below_min} values below minimum {self.min_value}")
            
            if self.max_value is not None:
                above_max = (data > self.max_value).sum()
                if above_max > 0:
                    violations.append(f"{above_max} values above maximum {self.max_value}")
            
            if violations:
                return False, "; ".join(violations)
            return True, "All values within specified range"
        
        elif isinstance(data, pd.DataFrame):
            columns_to_check = self.columns if self.columns else data.select_dtypes(include=[np.number]).columns
            failed_checks = []
            
            for col in columns_to_check:
                if col in data.columns:
                    is_valid, msg = self.check(data[col])
                    if not is_valid:
                        failed_checks.append(f"{col}: {msg}")
            
            if failed_checks:
                return False, "\n".join(failed_checks)
            return True, "All numeric columns within range"
        
        return False, f"Unsupported data type: {type(data)}"


class PatternCheck(DataQualityCheck):
    """Check if string values match specified pattern."""
    
    def __init__(
        self,
        pattern: str,
        columns: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """Initialize pattern check.
        
        Args:
            pattern: Regular expression pattern
            columns: Columns to check (for DataFrame)
            name: Name of the check
        """
        super().__init__(name)
        self.pattern = pattern
        self.columns = columns
        self._compiled_pattern = re.compile(pattern)
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check if values match pattern."""
        if isinstance(data, pd.Series):
            # Convert to string and check pattern
            str_data = data.astype(str)
            matches = str_data.apply(lambda x: bool(self._compiled_pattern.match(x)))
            non_matching = (~matches).sum()
            
            if non_matching > 0:
                return False, f"{non_matching} values don't match pattern '{self.pattern}'"
            return True, f"All values match pattern '{self.pattern}'"
        
        elif isinstance(data, pd.DataFrame):
            if not self.columns:
                return False, "No columns specified for pattern check"
            
            failed_checks = []
            for col in self.columns:
                if col in data.columns:
                    is_valid, msg = self.check(data[col])
                    if not is_valid:
                        failed_checks.append(f"{col}: {msg}")
            
            if failed_checks:
                return False, "\n".join(failed_checks)
            return True, "All specified columns match pattern"
        
        return False, f"Unsupported data type: {type(data)}"


class ConsistencyCheck(DataQualityCheck):
    """Check for data consistency across columns."""
    
    def __init__(
        self,
        consistency_rules: List[Callable[[pd.DataFrame], bool]],
        rule_descriptions: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """Initialize consistency check.
        
        Args:
            consistency_rules: List of functions that return True if consistent
            rule_descriptions: Descriptions of each rule
            name: Name of the check
        """
        super().__init__(name)
        self.consistency_rules = consistency_rules
        self.rule_descriptions = rule_descriptions or [f"Rule {i+1}" for i in range(len(consistency_rules))]
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check data consistency."""
        if not isinstance(data, pd.DataFrame):
            return False, "Consistency check requires DataFrame input"
        
        failed_rules = []
        for i, (rule, description) in enumerate(zip(self.consistency_rules, self.rule_descriptions)):
            try:
                if not rule(data):
                    failed_rules.append(description)
            except Exception as e:
                failed_rules.append(f"{description} (error: {str(e)})")
        
        if failed_rules:
            return False, f"Failed consistency rules: {', '.join(failed_rules)}"
        return True, "All consistency rules passed"


class StatisticalCheck(DataQualityCheck):
    """Check for statistical properties of data."""
    
    def __init__(
        self,
        columns: Optional[List[str]] = None,
        expected_mean: Optional[Dict[str, float]] = None,
        expected_std: Optional[Dict[str, float]] = None,
        mean_tolerance: float = 0.1,
        std_tolerance: float = 0.1,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        name: Optional[str] = None
    ):
        """Initialize statistical check.
        
        Args:
            columns: Columns to check
            expected_mean: Expected mean values per column
            expected_std: Expected standard deviation per column
            mean_tolerance: Tolerance for mean comparison
            std_tolerance: Tolerance for std comparison
            outlier_method: Method for outlier detection ('iqr' or 'zscore')
            outlier_threshold: Threshold for outlier detection
            name: Name of the check
        """
        super().__init__(name)
        self.columns = columns
        self.expected_mean = expected_mean or {}
        self.expected_std = expected_std or {}
        self.mean_tolerance = mean_tolerance
        self.std_tolerance = std_tolerance
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
    
    def check(self, data: Union[pd.Series, pd.DataFrame]) -> Tuple[bool, str]:
        """Check statistical properties."""
        if isinstance(data, pd.Series):
            if not pd.api.types.is_numeric_dtype(data):
                return True, "Non-numeric data, skipping statistical check"
            
            issues = []
            
            # Check mean
            if data.name in self.expected_mean:
                actual_mean = data.mean()
                expected = self.expected_mean[data.name]
                if abs(actual_mean - expected) / expected > self.mean_tolerance:
                    issues.append(f"Mean {actual_mean:.2f} deviates from expected {expected:.2f}")
            
            # Check std
            if data.name in self.expected_std:
                actual_std = data.std()
                expected = self.expected_std[data.name]
                if abs(actual_std - expected) / expected > self.std_tolerance:
                    issues.append(f"Std {actual_std:.2f} deviates from expected {expected:.2f}")
            
            # Check outliers
            outliers = self._detect_outliers(data)
            if len(outliers) > 0:
                issues.append(f"Found {len(outliers)} outliers")
            
            if issues:
                return False, "; ".join(issues)
            return True, "Statistical properties within expected range"
        
        elif isinstance(data, pd.DataFrame):
            columns_to_check = self.columns if self.columns else data.select_dtypes(include=[np.number]).columns
            failed_checks = []
            
            for col in columns_to_check:
                if col in data.columns:
                    is_valid, msg = self.check(data[col])
                    if not is_valid:
                        failed_checks.append(f"{col}: {msg}")
            
            if failed_checks:
                return False, "\n".join(failed_checks)
            return True, "All columns pass statistical checks"
        
        return False, f"Unsupported data type: {type(data)}"
    
    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """Detect outliers in data."""
        if self.outlier_method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            return data[(data < lower_bound) | (data > upper_bound)]
        
        elif self.outlier_method == "zscore":
            z_scores = np.abs((data - data.mean()) / data.std())
            return data[z_scores > self.outlier_threshold]
        
        else:
            return pd.Series([])