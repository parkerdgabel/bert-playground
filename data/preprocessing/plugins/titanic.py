"""Titanic competition data preprocessing plugin."""

import pandas as pd
from typing import Dict, Any
from loguru import logger
from ..base import DataPreprocessor, TabularToTextMixin, PreprocessorRegistry


class TitanicPreprocessor(DataPreprocessor, TabularToTextMixin):
    """Preprocessor for Titanic competition data."""
    
    name = "titanic"
    description = "Convert Titanic passenger data to natural language descriptions"
    supported_competitions = ["titanic", "titanic-machine-learning-from-disaster"]
    
    # Expected columns in Titanic dataset
    REQUIRED_COLUMNS = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 
                       'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    
    # Mappings for categorical values
    CLASS_MAP = {1: "first-class", 2: "second-class", 3: "third-class"}
    EMBARK_MAP = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate Titanic data format."""
        # Check if we have the required columns (minus Survived which is only in train)
        missing_cols = []
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            # Allow some flexibility - only PassengerId is truly required
            return 'PassengerId' in df.columns
        
        return True
    
    def row_to_text(self, row: pd.Series) -> str:
        """Convert a Titanic passenger row to BERT-optimized natural language."""
        parts = []
        
        # Start with passenger name (truncate if too long for BERT efficiency)
        name = row.get('Name', 'Unknown passenger')
        if len(name) > 80:
            name = name[:77] + "..."
        parts.append(f"Passenger {name}")
        
        # Add demographic info
        demo_parts = []
        
        # Age
        age = row.get('Age')
        if pd.notna(age):
            demo_parts.append(f"{int(age)}-year-old")
        
        # Sex
        sex = row.get('Sex', '').lower()
        if sex:
            demo_parts.append(sex)
        
        if demo_parts:
            parts.append(f"was a {' '.join(demo_parts)}")
        
        # Class
        pclass = row.get('Pclass')
        if pd.notna(pclass):
            class_desc = self.CLASS_MAP.get(pclass, f"class {pclass}")
            parts.append(f"traveling in {class_desc}")
        
        # Family information
        sibsp = row.get('SibSp', 0)
        parch = row.get('Parch', 0)
        
        if sibsp > 0 or parch > 0:
            family_parts = []
            if sibsp > 0:
                family_parts.append(f"{sibsp} {'sibling' if sibsp == 1 else 'siblings'}/spouse")
            if parch > 0:
                family_parts.append(f"{parch} {'parent/child' if parch == 1 else 'parents/children'}")
            parts.append(f"with {' and '.join(family_parts)}")
        else:
            parts.append("traveling alone")
        
        # Fare information with semantic categorization for BERT
        fare = row.get('Fare')
        if pd.notna(fare):
            if fare > 100:
                fare_desc = f"paid a high fare of ${fare:.2f}"
            elif fare < 10:
                fare_desc = f"paid a low fare of ${fare:.2f}"
            else:
                fare_desc = f"paid ${fare:.2f} for the ticket"
            parts.append(fare_desc)
        
        # Embarkation port
        embarked = row.get('Embarked')
        if pd.notna(embarked) and embarked in self.EMBARK_MAP:
            parts.append(f"embarking from {self.EMBARK_MAP[embarked]}")
        
        # Cabin information (truncate if too long)
        cabin = row.get('Cabin')
        if pd.notna(cabin):
            cabin_str = str(cabin)
            if len(cabin_str) > 40:
                cabin_str = cabin_str[:37] + "..."
            parts.append(f"in cabin {cabin_str}")
        
        # Ticket information (only if unusual and keep short)
        ticket = row.get('Ticket')
        if pd.notna(ticket) and not str(ticket).isdigit():
            ticket_str = str(ticket)
            if len(ticket_str) > 20:
                ticket_str = ticket_str[:17] + "..."
            parts.append(f"with ticket {ticket_str}")
        
        # Build final text with BERT-optimized structure
        text = self.build_text_parts(parts, separator=", ")
        
        # Ensure it ends with a period for proper sentence structure
        if not text.endswith('.'):
            text += '.'
        
        # Add BERT special token for text boundary (tokenizer will add [CLS])
        text = text + " [SEP]"
        
        # Conservative length check to stay well under 512 tokens
        # Rough estimate: 1 token ≈ 3-4 characters for English text
        if len(text) > 350:  # ~100 tokens buffer for safety
            # Truncate while preserving sentence structure
            truncated = text[:347] + "... [SEP]"
            text = truncated
        
        return text
    
    def preprocess_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Override to set specific columns for Titanic."""
        # Set Titanic-specific configuration
        self.config.id_column = "PassengerId"
        if "Survived" in df.columns:
            self.config.label_column = "Survived"
        
        # Call parent method (no need to rename columns - parent handles this)
        return super().preprocess_batch(df)


# Register the preprocessor
PreprocessorRegistry.register("titanic", TitanicPreprocessor)