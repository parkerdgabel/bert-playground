"""Base classes and protocols for data preprocessing plugins."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, Protocol, Union
import pandas as pd
from loguru import logger


@dataclass
class DataPrepConfig:
    """Configuration for data preprocessing."""
    
    input_path: Path
    output_path: Path
    text_column: str = "text"
    label_column: Optional[str] = "label"
    id_column: Optional[str] = None
    additional_columns: List[str] = None
    
    def __post_init__(self):
        """Validate configuration."""
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        if self.additional_columns is None:
            self.additional_columns = []


class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing plugins."""
    
    # Plugin metadata
    name: str = "base"
    description: str = "Base data preprocessor"
    supported_competitions: List[str] = []
    
    def __init__(self, config: DataPrepConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        
    @abstractmethod
    def row_to_text(self, row: pd.Series) -> str:
        """Convert a single row to text representation.
        
        Args:
            row: Pandas Series representing one data sample
            
        Returns:
            Text representation of the row
        """
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the data has expected columns and format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            True if data is valid, False otherwise
        """
        pass
    
    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a batch of data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame compatible with dataloader
        """
        # Validate data
        if not self.validate_data(df):
            raise ValueError(f"Invalid data format for {self.name} preprocessor")
        
        # Convert to text
        logger.info(f"Converting {len(df)} rows to text...")
        processed_texts = df.apply(self.row_to_text, axis=1)
        
        # Create output DataFrame with dataloader-compatible columns
        # The dataloader expects 'text' column for text data
        output_data = {'text': processed_texts}
        
        # Add ID column if specified (preserve original name for compatibility)
        if self.config.id_column and self.config.id_column in df.columns:
            output_data[self.config.id_column] = df[self.config.id_column].values
        
        # Add labels if available (must be called 'labels' for dataloader)
        if self.config.label_column and self.config.label_column in df.columns:
            # The dataloader expects 'labels' column, not custom label column names
            output_data['labels'] = df[self.config.label_column].values
        
        # Add any additional columns requested
        for col in self.config.additional_columns:
            if col in df.columns and col not in output_data:
                output_data[col] = df[col].values
        
        output_df = pd.DataFrame(output_data)
        
        # Ensure text column is properly formatted for BERT tokenization
        output_df['text'] = output_df['text'].fillna('').astype(str)
        
        # Validate text content - check for empty texts
        empty_texts = output_df['text'].str.strip() == ''
        if empty_texts.any():
            logger.warning(f"Found {empty_texts.sum()} empty text entries after preprocessing")
            # Replace empty texts with a default message that includes [SEP] token
            output_df.loc[empty_texts, 'text'] = "Unknown passenger, traveling alone. [SEP]"
        
        # Ensure all text entries end with [SEP] for BERT compatibility
        needs_sep = ~output_df['text'].str.endswith(' [SEP]')
        if needs_sep.any():
            logger.debug(f"Adding [SEP] token to {needs_sep.sum()} text entries")
            output_df.loc[needs_sep, 'text'] = output_df.loc[needs_sep, 'text'] + ' [SEP]'
        
        return output_df
    
    def process_file(self, input_path: Optional[Path] = None, 
                    output_path: Optional[Path] = None) -> pd.DataFrame:
        """Process a complete file.
        
        Args:
            input_path: Override input path from config
            output_path: Override output path from config
            
        Returns:
            Processed DataFrame
        """
        input_path = input_path or self.config.input_path
        output_path = output_path or self.config.output_path
        
        # Read data
        logger.info(f"Reading data from {input_path}...")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")
        
        # Process
        processed_df = self.preprocess_batch(df)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        # Show examples
        self._show_examples(processed_df, n=3)
        
        return processed_df
    
    def _show_examples(self, df: pd.DataFrame, n: int = 3):
        """Show example processed rows."""
        logger.info(f"\nExample processed texts:")
        for i in range(min(n, len(df))):
            logger.info(f"\n{i+1}. {df.iloc[i][self.config.text_column][:200]}...")
            if self.config.label_column in df.columns:
                logger.info(f"   Label: {df.iloc[i][self.config.label_column]}")


class TabularToTextMixin:
    """Mixin providing utilities for converting tabular data to text."""
    
    @staticmethod
    def format_value(value: Any, column: str) -> str:
        """Format a single value for text representation."""
        if pd.isna(value):
            return None
        
        # Handle different data types
        if isinstance(value, bool):
            return "yes" if value else "no"
        elif isinstance(value, (int, float)):
            if column.lower() in ['age', 'year', 'years']:
                return f"{int(value)}"
            elif column.lower() in ['price', 'fare', 'cost', 'amount']:
                return f"${value:.2f}"
            else:
                return str(value)
        else:
            return str(value)
    
    @staticmethod
    def build_text_parts(parts: List[str], separator: str = " ") -> str:
        """Build text from parts, filtering out None values."""
        valid_parts = [p for p in parts if p is not None and p != ""]
        return separator.join(valid_parts)


class PreprocessorRegistry:
    """Registry for data preprocessing plugins."""
    
    _preprocessors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, preprocessor_class: type):
        """Register a preprocessor class.
        
        Args:
            name: Name of the preprocessor
            preprocessor_class: Preprocessor class
        """
        if not issubclass(preprocessor_class, DataPreprocessor):
            raise TypeError(f"{preprocessor_class} must be a subclass of DataPreprocessor")
        
        cls._preprocessors[name] = preprocessor_class
        logger.info(f"Registered preprocessor: {name}")
    
    @classmethod
    def get(cls, name: str) -> type:
        """Get a preprocessor class by name.
        
        Args:
            name: Name of the preprocessor
            
        Returns:
            Preprocessor class
        """
        if name not in cls._preprocessors:
            raise ValueError(f"Unknown preprocessor: {name}. Available: {list(cls._preprocessors.keys())}")
        
        return cls._preprocessors[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available preprocessors."""
        return list(cls._preprocessors.keys())
    
    @classmethod
    def create(cls, name: str, config: DataPrepConfig) -> DataPreprocessor:
        """Create a preprocessor instance.
        
        Args:
            name: Name of the preprocessor
            config: Configuration for the preprocessor
            
        Returns:
            Preprocessor instance
        """
        preprocessor_class = cls.get(name)
        return preprocessor_class(config)