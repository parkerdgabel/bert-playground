"""Competition-specific text templates.

This module provides pre-built templates for common Kaggle competition types
with optimizations for BERT models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.base import CompetitionType


@dataclass
class CompetitionTextTemplate:
    """Pre-built text template for a specific competition type.
    
    This class encapsulates competition-specific knowledge about how to
    best convert tabular data to text for BERT models.
    """
    
    competition_type: CompetitionType
    template_string: str
    description: str
    
    # Template metadata
    max_length: int = 512
    priority_columns: List[str] = None
    exclude_columns: List[str] = None
    
    # BERT optimization hints
    use_attention_pooling: bool = False
    recommended_head_type: str = "binary_classification"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.priority_columns is None:
            self.priority_columns = []
        if self.exclude_columns is None:
            self.exclude_columns = []


# Pre-built templates for common competitions

TITANIC_TEMPLATE = CompetitionTextTemplate(
    competition_type=CompetitionType.BINARY_CLASSIFICATION,
    template_string=(
        "[CLS] Passenger profile: {Name} is a {Age} year old {Sex} "
        "traveling in {Pclass} class with {SibSp} siblings/spouses and {Parch} parents/children. "
        "Ticket: {Ticket}, Fare: {Fare}, Embarked from: {Embarked}. "
        "Cabin: {Cabin}. [SEP]"
    ),
    description="Optimized template for Titanic survival prediction",
    max_length=256,
    priority_columns=["Name", "Age", "Sex", "Pclass"],
    exclude_columns=["PassengerId", "Survived"],
    recommended_head_type="binary_classification",
)

HOUSE_PRICES_TEMPLATE = CompetitionTextTemplate(
    competition_type=CompetitionType.REGRESSION,
    template_string=(
        "[CLS] Property details: {MSSubClass} type home built in {YearBuilt} "
        "with {TotalBsmtSF} sqft basement, {1stFlrSF} sqft first floor, "
        "{2ndFlrSF} sqft second floor. {BedroomAbvGr} bedrooms, {FullBath} full baths. "
        "Located in {Neighborhood} with {LotArea} sqft lot. "
        "Quality: Overall {OverallQual}, Kitchen {KitchenQual}. [SEP]"
    ),
    description="Optimized template for house price prediction",
    max_length=384,
    priority_columns=["TotalBsmtSF", "1stFlrSF", "GrLivArea", "OverallQual"],
    exclude_columns=["Id", "SalePrice"],
    recommended_head_type="regression",
)

DIGIT_RECOGNIZER_TEMPLATE = CompetitionTextTemplate(
    competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
    template_string=(
        "[CLS] Pixel intensity pattern: {pixel_summary} "
        "with key features in regions {active_regions}. "
        "Brightness distribution: {brightness_stats}. [SEP]"
    ),
    description="Optimized template for handwritten digit recognition",
    max_length=512,
    priority_columns=[],  # Special handling for pixel data
    exclude_columns=["label"],
    recommended_head_type="multiclass_classification",
)

SENTIMENT_ANALYSIS_TEMPLATE = CompetitionTextTemplate(
    competition_type=CompetitionType.MULTICLASS_CLASSIFICATION,
    template_string=(
        "[CLS] Review text: {review} "
        "Word count: {word_count}, Sentence count: {sentence_count}. "
        "Key phrases: {key_phrases}. [SEP]"
    ),
    description="Optimized template for sentiment analysis",
    max_length=512,
    priority_columns=["review", "text", "comment"],
    exclude_columns=["id", "sentiment", "label"],
    use_attention_pooling=True,
    recommended_head_type="multiclass_classification",
)

TIMESERIES_SALES_TEMPLATE = CompetitionTextTemplate(
    competition_type=CompetitionType.TIME_SERIES,
    template_string=(
        "[CLS] Sales record: Date {date}, Store {store_id}, Item {item_id}. "
        "Historical sales: {sales_history}, Recent trend: {trend}. "
        "Seasonal factors: {seasonal_info}, Promotions: {promo_info}. [SEP]"
    ),
    description="Optimized template for time series sales forecasting",
    max_length=384,
    priority_columns=["date", "store_id", "item_id", "sales"],
    exclude_columns=["id"],
    recommended_head_type="regression",
)

# Template registry
COMPETITION_TEMPLATES = {
    "titanic": TITANIC_TEMPLATE,
    "house-prices": HOUSE_PRICES_TEMPLATE,
    "house_prices": HOUSE_PRICES_TEMPLATE,
    "digit-recognizer": DIGIT_RECOGNIZER_TEMPLATE,
    "digit_recognizer": DIGIT_RECOGNIZER_TEMPLATE,
    "sentiment": SENTIMENT_ANALYSIS_TEMPLATE,
    "sentiment_analysis": SENTIMENT_ANALYSIS_TEMPLATE,
    "sales": TIMESERIES_SALES_TEMPLATE,
    "timeseries": TIMESERIES_SALES_TEMPLATE,
}


def get_template_for_competition(competition_name: str) -> Optional[CompetitionTextTemplate]:
    """Get a pre-built template for a competition.
    
    Args:
        competition_name: Name of the competition
        
    Returns:
        CompetitionTextTemplate or None if not found
    """
    # Normalize competition name
    normalized_name = competition_name.lower().replace("-", "_")
    
    # Check for exact match
    if normalized_name in COMPETITION_TEMPLATES:
        return COMPETITION_TEMPLATES[normalized_name]
        
    # Check for partial matches
    for template_name, template in COMPETITION_TEMPLATES.items():
        if template_name in normalized_name or normalized_name in template_name:
            return template
            
    return None


def get_template_for_type(competition_type: CompetitionType) -> Optional[CompetitionTextTemplate]:
    """Get a default template for a competition type.
    
    Args:
        competition_type: Type of competition
        
    Returns:
        CompetitionTextTemplate or None if not found
    """
    # Find first template matching the competition type
    for template in COMPETITION_TEMPLATES.values():
        if template.competition_type == competition_type:
            return template
            
    return None


def create_custom_template(
    competition_type: CompetitionType,
    template_string: str,
    description: str = "Custom template",
    **kwargs,
) -> CompetitionTextTemplate:
    """Create a custom competition template.
    
    Args:
        competition_type: Type of competition
        template_string: Template string with {column} placeholders
        description: Description of the template
        **kwargs: Additional template parameters
        
    Returns:
        CompetitionTextTemplate instance
    """
    return CompetitionTextTemplate(
        competition_type=competition_type,
        template_string=template_string,
        description=description,
        **kwargs,
    )


# Smart template suggestions based on column patterns
def suggest_template_from_columns(columns: List[str]) -> Optional[CompetitionTextTemplate]:
    """Suggest a template based on column names.
    
    Args:
        columns: List of column names in the dataset
        
    Returns:
        Suggested CompetitionTextTemplate or None
    """
    columns_lower = [col.lower() for col in columns]
    
    # Titanic patterns
    titanic_patterns = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'embarked']
    if sum(1 for pattern in titanic_patterns if pattern in columns_lower) >= 5:
        return TITANIC_TEMPLATE
        
    # House prices patterns
    house_patterns = ['saleprice', 'mszoning', 'lotarea', 'yearbuilt', 'overallqual', 'grlivarea']
    if sum(1 for pattern in house_patterns if pattern in columns_lower) >= 3:
        return HOUSE_PRICES_TEMPLATE
        
    # Sentiment analysis patterns
    sentiment_patterns = ['review', 'text', 'comment', 'sentiment', 'rating']
    if any(pattern in columns_lower for pattern in sentiment_patterns):
        return SENTIMENT_ANALYSIS_TEMPLATE
        
    # Time series patterns
    time_patterns = ['date', 'time', 'timestamp', 'sales', 'store', 'item']
    if sum(1 for pattern in time_patterns if pattern in columns_lower) >= 2:
        return TIMESERIES_SALES_TEMPLATE
        
    # Digit recognizer patterns (pixel columns)
    if any(col.startswith('pixel') for col in columns_lower) or len([col for col in columns if col.isdigit()]) > 100:
        return DIGIT_RECOGNIZER_TEMPLATE
        
    return None