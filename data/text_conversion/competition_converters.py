"""
Competition-specific text converters.
"""

from typing import Dict, Any, List, Optional, Union
from abc import abstractmethod
import random
from loguru import logger

from .base_converter import BaseTextConverter, TextConversionConfig
from .template_converter import TemplateConverter, TemplateConfig


class CompetitionConverter(BaseTextConverter):
    """Base class for competition-specific converters."""
    
    def __init__(
        self,
        competition_name: str,
        config: Optional[TextConversionConfig] = None
    ):
        """
        Initialize competition converter.
        
        Args:
            competition_name: Name of the competition
            config: Text conversion configuration
        """
        super().__init__(config)
        self.competition_name = competition_name
    
    @abstractmethod
    def get_competition_context(self) -> str:
        """Get competition-specific context."""
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.competition_name})"


class TitanicConverter(TemplateConverter):
    """Text converter for Titanic competition."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize Titanic converter."""
        if config is None:
            config = TemplateConfig()
        
        # Set up Titanic-specific templates
        config.templates = [
            # Basic template
            "A $age_desc $sex passenger traveling in $class_desc class. "
            "$family_desc $fare_desc $embark_desc",
            
            # Narrative style
            "$name_info was a $age_desc $sex who boarded the Titanic $embark_desc. "
            "Traveling in $class_desc class, $pronoun paid $$$fare for the journey. $family_desc",
            
            # Descriptive style
            "Passenger details: $sex, $age years old, $class_desc class ticket. "
            "Embarked from $embarked_full. $family_desc Fare paid: $$$fare.",
        ]
        
        config.augmentation_templates = [
            # Story style
            "On that fateful voyage, $a_an $age_desc $sex $embark_desc "
            "with a $class_desc class ticket costing $$$fare. $family_desc",
            
            # Report style
            "Record shows: $sex passenger, age $age, $class_desc class. "
            "Boarding location: $embarked_full. Family status: $family_desc",
        ]
        
        # Custom formatters
        config.field_formatters = {
            "age": self._format_age,
            "pclass": self._format_pclass,
            "embarked": self._format_embarked,
            "sex": self._format_sex,
            "fare": self._format_fare,
        }
        
        super().__init__(config)
        
        # Additional mappings
        self.embarkation_map = {
            "S": "Southampton",
            "C": "Cherbourg",
            "Q": "Queenstown"
        }
    
    def _prepare_template_vars(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare Titanic-specific template variables."""
        vars = super()._prepare_template_vars(data)
        
        # Add computed variables
        age = data.get("age")
        vars["age_desc"] = self._get_age_description(age)
        
        pclass = data.get("pclass")
        vars["class_desc"] = self._get_class_description(pclass)
        
        sex = data.get("sex", "").lower()
        vars["pronoun"] = "he" if sex == "male" else "she" if sex == "female" else "they"
        vars["a_an"] = "an" if vars.get("age_desc", "").startswith(("e", "i")) else "a"
        
        # Family description
        sibsp = data.get("sibsp", 0)
        parch = data.get("parch", 0)
        vars["family_desc"] = self._get_family_description(sibsp, parch)
        
        # Embarkation
        embarked = data.get("embarked")
        vars["embarked_full"] = self.embarkation_map.get(embarked, "unknown port")
        vars["embark_desc"] = f"embarked at {vars['embarked_full']}"
        
        # Fare description
        fare = data.get("fare")
        vars["fare_desc"] = self._get_fare_description(fare)
        
        # Name info
        name = data.get("name", "")
        vars["name_info"] = self._extract_name_info(name)
        
        return vars
    
    def _format_age(self, age: Any) -> str:
        """Format age value."""
        if age is None:
            return "unknown"
        return str(int(age)) if float(age).is_integer() else f"{age:.1f}"
    
    def _format_pclass(self, pclass: Any) -> str:
        """Format passenger class."""
        class_names = {1: "first", 2: "second", 3: "third"}
        return class_names.get(pclass, str(pclass))
    
    def _format_embarked(self, embarked: Any) -> str:
        """Format embarkation port."""
        return self.embarkation_map.get(embarked, str(embarked))
    
    def _format_sex(self, sex: Any) -> str:
        """Format sex/gender."""
        return str(sex).lower() if sex else "unknown"
    
    def _format_fare(self, fare: Any) -> str:
        """Format fare amount."""
        if fare is None:
            return "unknown"
        return f"{fare:.2f}"
    
    def _get_age_description(self, age: Any) -> str:
        """Get age description."""
        if age is None:
            return "passenger of unknown age"
        
        age_val = float(age)
        if age_val < 1:
            return "infant"
        elif age_val < 13:
            return "child"
        elif age_val < 20:
            return "teenager"
        elif age_val < 30:
            return "young adult"
        elif age_val < 50:
            return "middle-aged"
        elif age_val < 65:
            return "mature"
        else:
            return "elderly"
    
    def _get_class_description(self, pclass: Any) -> str:
        """Get passenger class description."""
        class_descriptions = {
            1: "luxurious first",
            2: "comfortable second",
            3: "economical third",
        }
        return class_descriptions.get(pclass, f"{pclass}th")
    
    def _get_family_description(self, sibsp: int, parch: int) -> str:
        """Get family status description."""
        if sibsp == 0 and parch == 0:
            return "Traveled alone."
        
        parts = []
        if sibsp > 0:
            parts.append(f"{sibsp} sibling(s)/spouse")
        if parch > 0:
            parts.append(f"{parch} parent(s)/child(ren)")
        
        return f"Traveled with {' and '.join(parts)}."
    
    def _get_fare_description(self, fare: Any) -> str:
        """Get fare description."""
        if fare is None:
            return "The fare is unknown."
        
        fare_val = float(fare)
        if fare_val < 10:
            return "The fare was very affordable."
        elif fare_val < 30:
            return "The fare was moderately priced."
        elif fare_val < 100:
            return "The fare was expensive."
        else:
            return "The fare was extremely expensive."
    
    def _extract_name_info(self, name: str) -> str:
        """Extract information from name."""
        if not name:
            return "A passenger"
        
        # Extract title
        if "Mr." in name:
            return "Mr."
        elif "Mrs." in name:
            return "Mrs."
        elif "Miss." in name or "Ms." in name:
            return "Miss"
        elif "Master." in name:
            return "Master"
        elif "Dr." in name:
            return "Dr."
        else:
            return "A passenger"


class SpaceshipTitanicConverter(TemplateConverter):
    """Text converter for Spaceship Titanic competition."""
    
    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize Spaceship Titanic converter."""
        if config is None:
            config = TemplateConfig()
        
        # Set up templates
        config.templates = [
            "Passenger $passenger_id is $demographics, traveling to $destination. "
            "$cryo_status $vip_status Cabin: $cabin. $spending_summary",
            
            "Record for $passenger_id: $age year old from $home_planet, "
            "destination $destination. $facilities_usage Total spent: $$$total_spent.",
        ]
        
        config.augmentation_templates = [
            "$passenger_id - $home_planet resident, age $age. "
            "Journey to $destination. $cryo_status $spending_details",
            
            "Spaceship passenger $passenger_id: $demographics. "
            "$vip_status $facilities_summary Cabin location: $cabin.",
        ]
        
        # Custom formatters
        config.field_formatters = {
            "cryosleep": self._format_cryosleep,
            "vip": self._format_vip,
            "home_planet": self._format_home_planet,
            "destination": self._format_destination,
        }
        
        super().__init__(config)
        
        # Planet descriptions
        self.planet_descriptions = {
            "Earth": "Earth",
            "Europa": "Jupiter's moon Europa",
            "Mars": "Mars",
        }
        
        self.destination_descriptions = {
            "TRAPPIST-1e": "TRAPPIST-1e exoplanet",
            "PSO J318.5-22": "rogue planet PSO J318.5-22",
            "55 Cancri e": "super-Earth 55 Cancri e",
        }
    
    def _prepare_template_vars(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare Spaceship Titanic template variables."""
        vars = super()._prepare_template_vars(data)
        
        # Demographics
        age = data.get("age", "unknown")
        home_planet = data.get("home_planet", "unknown")
        vars["demographics"] = f"a {age} year old from {self._format_home_planet(home_planet)}"
        
        # Status fields
        vars["cryo_status"] = self._get_cryo_status(data.get("cryosleep"))
        vars["vip_status"] = self._get_vip_status(data.get("vip"))
        
        # Spending summary
        spending_fields = ["roomservice", "foodcourt", "shoppingmall", "spa", "vrdeck"]
        total_spent = sum(data.get(f, 0) or 0 for f in spending_fields)
        vars["total_spent"] = f"{total_spent:.2f}"
        vars["spending_summary"] = self._get_spending_summary(data, spending_fields)
        vars["spending_details"] = self._get_spending_details(data, spending_fields)
        
        # Facilities usage
        vars["facilities_usage"] = self._get_facilities_usage(data, spending_fields)
        vars["facilities_summary"] = self._get_facilities_summary(data, spending_fields)
        
        return vars
    
    def _format_cryosleep(self, cryosleep: Any) -> str:
        """Format cryosleep status."""
        if cryosleep is True:
            return "in cryogenic sleep"
        elif cryosleep is False:
            return "active passenger"
        else:
            return "unknown sleep status"
    
    def _format_vip(self, vip: Any) -> str:
        """Format VIP status."""
        if vip is True:
            return "VIP passenger"
        elif vip is False:
            return "standard passenger"
        else:
            return "unknown VIP status"
    
    def _format_home_planet(self, planet: Any) -> str:
        """Format home planet."""
        return self.planet_descriptions.get(planet, str(planet))
    
    def _format_destination(self, destination: Any) -> str:
        """Format destination."""
        return self.destination_descriptions.get(destination, str(destination))
    
    def _get_cryo_status(self, cryosleep: Any) -> str:
        """Get cryosleep status description."""
        if cryosleep is True:
            return "Currently in cryogenic sleep."
        elif cryosleep is False:
            return "Active during the journey."
        else:
            return ""
    
    def _get_vip_status(self, vip: Any) -> str:
        """Get VIP status description."""
        if vip is True:
            return "Traveling as a VIP passenger."
        elif vip is False:
            return "Standard class passenger."
        else:
            return ""
    
    def _get_spending_summary(self, data: Dict[str, Any], fields: List[str]) -> str:
        """Get spending summary."""
        spending = {}
        for field in fields:
            value = data.get(field, 0) or 0
            if value > 0:
                spending[field] = value
        
        if not spending:
            return "No recorded spending on amenities."
        
        parts = []
        for facility, amount in spending.items():
            facility_name = facility.replace("roomservice", "room service").replace("foodcourt", "food court")
            facility_name = facility_name.replace("shoppingmall", "shopping mall").replace("vrdeck", "VR deck")
            parts.append(f"${amount:.2f} at {facility_name}")
        
        return f"Spent {', '.join(parts)}."
    
    def _get_spending_details(self, data: Dict[str, Any], fields: List[str]) -> str:
        """Get detailed spending information."""
        total = sum(data.get(f, 0) or 0 for f in fields)
        if total == 0:
            return "No amenities purchases recorded."
        
        return f"Total amenities spending: ${total:.2f}."
    
    def _get_facilities_usage(self, data: Dict[str, Any], fields: List[str]) -> str:
        """Get facilities usage description."""
        used_facilities = [f for f in fields if (data.get(f, 0) or 0) > 0]
        
        if not used_facilities:
            return "Did not use any paid facilities."
        
        facility_names = {
            "roomservice": "room service",
            "foodcourt": "food court",
            "shoppingmall": "shopping mall",
            "spa": "spa",
            "vrdeck": "VR deck"
        }
        
        names = [facility_names.get(f, f) for f in used_facilities]
        
        if len(names) == 1:
            return f"Used {names[0]}."
        else:
            return f"Used {', '.join(names[:-1])} and {names[-1]}."
    
    def _get_facilities_summary(self, data: Dict[str, Any], fields: List[str]) -> str:
        """Get facilities summary."""
        num_used = sum(1 for f in fields if (data.get(f, 0) or 0) > 0)
        
        if num_used == 0:
            return "No facility usage."
        elif num_used == len(fields):
            return "Used all available facilities."
        else:
            return f"Used {num_used} out of {len(fields)} facilities."


# Registry of competition converters
COMPETITION_CONVERTERS = {
    "titanic": TitanicConverter,
    "spaceship-titanic": SpaceshipTitanicConverter,
}


def get_competition_converter(
    competition: str,
    config: Optional[Union[TextConversionConfig, Dict[str, Any]]] = None
) -> BaseTextConverter:
    """
    Get converter for a specific competition.
    
    Args:
        competition: Competition name
        config: Configuration (dict or object)
        
    Returns:
        Competition-specific converter
    """
    converter_class = COMPETITION_CONVERTERS.get(competition.lower())
    
    if converter_class is None:
        raise ValueError(
            f"No converter found for competition '{competition}'. "
            f"Available: {list(COMPETITION_CONVERTERS.keys())}"
        )
    
    # Handle config
    if isinstance(config, dict):
        if converter_class == TitanicConverter or converter_class == SpaceshipTitanicConverter:
            config = TemplateConfig(**config)
        else:
            config = TextConversionConfig(**config)
    
    return converter_class(config)