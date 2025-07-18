"""
Text templates for converting Spaceship Titanic tabular data to natural language.
"""

from typing import Dict, List, Optional
import random
import pandas as pd
import numpy as np


class SpaceshipTitanicTextConverter:
    """Convert Spaceship Titanic tabular data to natural language descriptions."""
    
    def __init__(self, augment: bool = False):
        """
        Initialize text converter.
        
        Args:
            augment: Whether to use multiple template variations
        """
        self.augment = augment
        
        # Define templates for different aspects
        self.intro_templates = [
            "Passenger {passenger_id} is",
            "Record for passenger {passenger_id}:",
            "Passenger ID {passenger_id} -",
            "Data for passenger {passenger_id} shows",
            "Passenger {passenger_id} information:",
        ]
        
        self.demographics_templates = [
            "a {age} year old from {home_planet}",
            "aged {age}, originally from {home_planet}",
            "from {home_planet}, {age} years of age",
            "{age} years old, home planet: {home_planet}",
            "age {age}, departing from {home_planet}",
        ]
        
        self.destination_templates = [
            "traveling to {destination}",
            "headed for {destination}",
            "with destination {destination}",
            "journey endpoint: {destination}",
            "destined for {destination}",
        ]
        
        self.cryo_templates = {
            True: [
                "in cryogenic sleep",
                "currently in cryosleep",
                "suspended in cryogenic stasis",
                "frozen in cryosleep",
                "undergoing cryogenic suspension",
            ],
            False: [
                "awake during the journey",
                "not in cryosleep",
                "active passenger",
                "conscious during travel",
                "not using cryogenic services",
            ]
        }
        
        self.vip_templates = {
            True: [
                "with VIP status",
                "as a VIP passenger",
                "holding VIP privileges",
                "VIP class traveler",
                "premium VIP member",
            ],
            False: [
                "standard passenger",
                "regular class traveler",
                "non-VIP passenger",
                "standard fare passenger",
                "economy class traveler",
            ]
        }
        
        self.cabin_templates = [
            "assigned to cabin {cabin}",
            "in cabin {cabin}",
            "cabin location: {cabin}",
            "staying in cabin {cabin}",
            "accommodated in cabin {cabin}",
        ]
        
        self.spending_templates = [
            "Spending records: Room Service ${room_service}, Food Court ${food_court}, Shopping Mall ${shopping_mall}, Spa ${spa}, VR Deck ${vr_deck}",
            "Amenities usage: ${room_service} on room service, ${food_court} at food court, ${shopping_mall} shopping, ${spa} spa services, ${vr_deck} VR entertainment",
            "Luxury spending: Room Service (${room_service}), Food Court (${food_court}), Shopping (${shopping_mall}), Spa (${spa}), VR Deck (${vr_deck})",
            "Service charges: ${room_service} room service, ${food_court} dining, ${shopping_mall} retail, ${spa} wellness, ${vr_deck} virtual reality",
            "Expenditures - Room: ${room_service}, Dining: ${food_court}, Shopping: ${shopping_mall}, Spa: ${spa}, VR: ${vr_deck}",
        ]
        
        self.no_spending_templates = [
            "No luxury spending recorded (likely due to cryosleep)",
            "Zero amenity charges (consistent with cryosleep status)",
            "No service usage detected",
            "All luxury services unused",
            "Zero expenditure on ship amenities",
        ]
    
    def _select_template(self, templates: List[str]) -> str:
        """Select a template based on augmentation setting."""
        if self.augment:
            return random.choice(templates)
        return templates[0]
    
    def _format_cabin(self, cabin: str) -> str:
        """Format cabin information."""
        if pd.isna(cabin) or cabin == "":
            return "unknown cabin"
        
        parts = cabin.split('/')
        if len(parts) == 3:
            deck, num, side = parts
            side_full = "Port" if side == "P" else "Starboard"
            return f"{deck}/{num}/{side_full}"
        return cabin
    
    def _format_spending(self, row: pd.Series) -> str:
        """Format spending information."""
        spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        
        # Check if all spending is zero (common for cryosleep passengers)
        total_spending = sum(row.get(col, 0) or 0 for col in spending_cols)
        
        if total_spending == 0:
            return self._select_template(self.no_spending_templates)
        
        # Format spending values
        spending_dict = {}
        for col in spending_cols:
            value = row.get(col, 0) or 0
            spending_dict[col.lower().replace('service', '_service').replace('court', '_court')
                        .replace('mall', '_mall').replace('deck', '_deck')] = int(value)
        
        template = self._select_template(self.spending_templates)
        return template.format(**spending_dict)
    
    def convert_row_to_text(self, row: pd.Series) -> str:
        """
        Convert a single row to natural language text.
        
        Args:
            row: Pandas Series containing passenger data
            
        Returns:
            Natural language description of the passenger
        """
        parts = []
        
        # Introduction
        intro = self._select_template(self.intro_templates)
        parts.append(intro.format(passenger_id=row['PassengerId']))
        
        # Demographics
        if not pd.isna(row.get('Age')) and not pd.isna(row.get('HomePlanet')):
            demo = self._select_template(self.demographics_templates)
            parts.append(demo.format(
                age=int(row['Age']),
                home_planet=row['HomePlanet']
            ))
        elif not pd.isna(row.get('Age')):
            parts.append(f"{int(row['Age'])} years old")
        elif not pd.isna(row.get('HomePlanet')):
            parts.append(f"from {row['HomePlanet']}")
        
        # Destination
        if not pd.isna(row.get('Destination')):
            dest = self._select_template(self.destination_templates)
            parts.append(dest.format(destination=row['Destination']))
        
        # Cryosleep status
        if not pd.isna(row.get('CryoSleep')):
            cryo_status = bool(row['CryoSleep'])
            cryo = self._select_template(self.cryo_templates[cryo_status])
            parts.append(cryo)
        
        # VIP status
        if not pd.isna(row.get('VIP')):
            vip_status = bool(row['VIP'])
            vip = self._select_template(self.vip_templates[vip_status])
            parts.append(vip)
        
        # Cabin
        if not pd.isna(row.get('Cabin')):
            cabin = self._select_template(self.cabin_templates)
            parts.append(cabin.format(cabin=self._format_cabin(row['Cabin'])))
        
        # Combine basic info
        if len(parts) > 1:
            text = parts[0] + " " + ", ".join(parts[1:]) + "."
        else:
            text = parts[0] + "."
        
        # Add spending information
        spending_text = self._format_spending(row)
        text += " " + spending_text + "."
        
        # Add passenger name if available
        if not pd.isna(row.get('Name')):
            text += f" Passenger name: {row['Name']}."
        
        return text
    
    def convert_dataframe(self, df: pd.DataFrame) -> List[str]:
        """
        Convert entire dataframe to list of text descriptions.
        
        Args:
            df: DataFrame containing Spaceship Titanic data
            
        Returns:
            List of natural language descriptions
        """
        texts = []
        for _, row in df.iterrows():
            text = self.convert_row_to_text(row)
            texts.append(text)
        return texts
    
    def create_augmented_texts(self, row: pd.Series, n_augmentations: int = 3) -> List[str]:
        """
        Create multiple augmented versions of the same row.
        
        Args:
            row: Pandas Series containing passenger data
            n_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented text descriptions
        """
        original_augment = self.augment
        self.augment = True
        
        texts = []
        for _ in range(n_augmentations):
            text = self.convert_row_to_text(row)
            texts.append(text)
        
        self.augment = original_augment
        return texts


def create_spaceship_dataset_splits(
    train_path: str,
    test_path: Optional[str] = None,
    val_split: float = 0.2,
    augment_train: bool = True,
    n_augmentations: int = 2
) -> Dict[str, pd.DataFrame]:
    """
    Create train/val/test splits with text conversion.
    
    Args:
        train_path: Path to training CSV
        test_path: Optional path to test CSV
        val_split: Validation split ratio
        augment_train: Whether to augment training data
        n_augmentations: Number of augmentations per sample
        
    Returns:
        Dictionary with 'train', 'val', and optionally 'test' DataFrames
    """
    # Load data
    train_df = pd.read_csv(train_path)
    
    # Create text converter
    converter = SpaceshipTitanicTextConverter(augment=augment_train)
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=val_split,
        random_state=42,
        stratify=train_df['Transported']
    )
    
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()
    
    # Convert to text
    print("Converting training data to text...")
    if augment_train:
        train_texts = []
        train_labels = []
        for _, row in train_split.iterrows():
            # Original
            train_texts.append(converter.convert_row_to_text(row))
            train_labels.append(row['Transported'])
            
            # Augmentations
            aug_texts = converter.create_augmented_texts(row, n_augmentations)
            train_texts.extend(aug_texts)
            train_labels.extend([row['Transported']] * n_augmentations)
        
        train_split = pd.DataFrame({
            'text': train_texts,
            'label': train_labels,
            'passenger_id': ['aug'] * len(train_texts)
        })
    else:
        train_split['text'] = converter.convert_dataframe(train_split)
        train_split['label'] = train_split['Transported']
    
    # Convert validation (no augmentation)
    print("Converting validation data to text...")
    converter.augment = False
    val_split['text'] = converter.convert_dataframe(val_split)
    val_split['label'] = val_split['Transported']
    
    results = {
        'train': train_split[['text', 'label']],
        'val': val_split[['text', 'label']]
    }
    
    # Process test if provided
    if test_path:
        print("Converting test data to text...")
        test_df = pd.read_csv(test_path)
        test_df['text'] = converter.convert_dataframe(test_df)
        results['test'] = test_df[['PassengerId', 'text']]
    
    return results


if __name__ == "__main__":
    # Example usage
    converter = SpaceshipTitanicTextConverter(augment=False)
    
    # Create a sample row
    sample_row = pd.Series({
        'PassengerId': '0001_01',
        'HomePlanet': 'Earth',
        'CryoSleep': False,
        'Cabin': 'B/10/P',
        'Destination': 'TRAPPIST-1e',
        'Age': 35,
        'VIP': False,
        'RoomService': 125,
        'FoodCourt': 458,
        'ShoppingMall': 0,
        'Spa': 234,
        'VRDeck': 888,
        'Name': 'John Doe',
        'Transported': True
    })
    
    print("Sample text conversion:")
    print(converter.convert_row_to_text(sample_row))
    
    print("\n\nAugmented versions:")
    for i, text in enumerate(converter.create_augmented_texts(sample_row, 3), 1):
        print(f"\n{i}. {text}")