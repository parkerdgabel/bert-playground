"""
Text generation utilities for converting tabular data to natural language.

This module provides flexible text generation strategies for different
Kaggle datasets and problem types.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import random
from abc import ABC, abstractmethod


class TextGenerator(ABC):
    """Abstract base class for text generation strategies."""
    
    @abstractmethod
    def generate(self, row: pd.Series) -> str:
        """Generate text from a data row."""
        pass
    
    @abstractmethod
    def generate_augmented(self, row: pd.Series) -> List[str]:
        """Generate multiple text variations for augmentation."""
        pass


class TabularTextGenerator(TextGenerator):
    """Generic text generator for tabular data."""
    
    def __init__(
        self,
        columns: List[str],
        column_descriptions: Optional[Dict[str, str]] = None,
        value_mappings: Optional[Dict[str, Dict[Any, str]]] = None,
        skip_values: List[str] = None,
    ):
        """
        Initialize the text generator.
        
        Args:
            columns: List of columns to include in text
            column_descriptions: Human-readable descriptions for columns
            value_mappings: Custom mappings for column values
            skip_values: Values to skip (e.g., 'nan', 'none')
        """
        self.columns = columns
        self.column_descriptions = column_descriptions or {}
        self.value_mappings = value_mappings or {}
        self.skip_values = skip_values or ['nan', 'none', '', 'na', 'null']
    
    def generate(self, row: pd.Series) -> str:
        """Generate natural language text from a data row."""
        text_parts = []
        
        for col in self.columns:
            if col not in row or pd.isna(row[col]):
                continue
            
            value = str(row[col]).strip()
            if value.lower() in self.skip_values:
                continue
            
            # Get column description
            col_desc = self.column_descriptions.get(col, col.replace('_', ' ').title())
            
            # Apply value mapping if available
            if col in self.value_mappings and row[col] in self.value_mappings[col]:
                value = self.value_mappings[col][row[col]]
            
            text_parts.append(f"{col_desc}: {value}")
        
        # Join with proper punctuation
        text = ". ".join(text_parts)
        if text:
            text += "."
        
        return text
    
    def generate_augmented(self, row: pd.Series) -> List[str]:
        """Generate multiple variations for data augmentation."""
        variations = []
        
        # Variation 1: Standard format
        variations.append(self.generate(row))
        
        # Variation 2: Sentence format
        sentence_parts = []
        for col in self.columns:
            if col not in row or pd.isna(row[col]):
                continue
            
            value = str(row[col]).strip()
            if value.lower() in self.skip_values:
                continue
            
            col_desc = self.column_descriptions.get(col, col.replace('_', ' '))
            
            if col in self.value_mappings and row[col] in self.value_mappings[col]:
                value = self.value_mappings[col][row[col]]
            
            sentence_parts.append(f"The {col_desc} is {value}")
        
        if sentence_parts:
            variations.append(". ".join(sentence_parts) + ".")
        
        # Variation 3: Narrative format (if we have enough context)
        if len(sentence_parts) >= 3:
            narrative = "This record shows " + ", ".join(sentence_parts[:-1])
            narrative += f", and {sentence_parts[-1].lower()}."
            variations.append(narrative)
        
        return variations


class TitanicTextGenerator(TabularTextGenerator):
    """Specialized text generator for Titanic dataset."""
    
    def __init__(self):
        super().__init__(
            columns=['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked'],
            column_descriptions={
                'Pclass': 'passenger class',
                'Name': 'passenger name',
                'Sex': 'gender',
                'Age': 'age',
                'SibSp': 'siblings/spouse aboard',
                'Parch': 'parents/children aboard',
                'Ticket': 'ticket number',
                'Fare': 'ticket fare',
                'Embarked': 'port of embarkation'
            },
            value_mappings={
                'Pclass': {1: 'first class', 2: 'second class', 3: 'third class'},
                'Sex': {'male': 'male', 'female': 'female'},
                'Embarked': {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
            }
        )
    
    def generate_narrative(self, row: pd.Series) -> str:
        """Generate a narrative description of a passenger."""
        parts = []
        
        # Name and basic info
        if pd.notna(row.get('Name')):
            parts.append(f"{row['Name']} was")
        else:
            parts.append("This passenger was")
        
        # Age and gender
        age_sex = []
        if pd.notna(row.get('Age')):
            age_sex.append(f"a {int(row['Age'])}-year-old")
        if pd.notna(row.get('Sex')):
            age_sex.append(row['Sex'])
        if age_sex:
            parts.append(" ".join(age_sex))
        
        # Class
        if pd.notna(row.get('Pclass')):
            class_map = {1: 'first', 2: 'second', 3: 'third'}
            parts.append(f"traveling in {class_map.get(row['Pclass'], 'unknown')} class")
        
        # Family
        family = []
        if pd.notna(row.get('SibSp')) and row['SibSp'] > 0:
            family.append(f"{row['SibSp']} sibling(s)/spouse")
        if pd.notna(row.get('Parch')) and row['Parch'] > 0:
            family.append(f"{row['Parch']} parent(s)/child(ren)")
        if family:
            parts.append(f"with {' and '.join(family)}")
        
        # Fare and embarkation
        if pd.notna(row.get('Fare')):
            parts.append(f"paying ${row['Fare']:.2f} for the journey")
        
        if pd.notna(row.get('Embarked')):
            port_map = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
            parts.append(f"embarking from {port_map.get(row['Embarked'], 'unknown port')}")
        
        return " ".join(parts) + "."
    
    def generate_augmented(self, row: pd.Series) -> List[str]:
        """Generate multiple text variations for Titanic data."""
        variations = []
        
        # Standard format
        variations.append(self.generate(row))
        
        # Narrative format
        variations.append(self.generate_narrative(row))
        
        # Question-answer format
        qa_parts = []
        if pd.notna(row.get('Name')):
            qa_parts.append(f"Who: {row['Name']}")
        if pd.notna(row.get('Age')):
            qa_parts.append(f"Age: {row['Age']} years")
        if pd.notna(row.get('Sex')):
            qa_parts.append(f"Gender: {row['Sex']}")
        if pd.notna(row.get('Pclass')):
            qa_parts.append(f"Class: {row['Pclass']}")
        if qa_parts:
            variations.append(" | ".join(qa_parts))
        
        return variations


# Registry of dataset-specific generators
TEXT_GENERATORS = {
    'titanic': TitanicTextGenerator,
    # Add more datasets here as needed
}


def get_text_generator(dataset_name: str, **kwargs) -> TextGenerator:
    """
    Get a text generator for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        **kwargs: Additional arguments for the generator
        
    Returns:
        Configured TextGenerator instance
    """
    if dataset_name in TEXT_GENERATORS:
        return TEXT_GENERATORS[dataset_name](**kwargs)
    else:
        # Return generic generator
        return TabularTextGenerator(**kwargs)