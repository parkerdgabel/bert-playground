"""
Simple test script for the text conversion system.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich import print as rprint
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.text_conversion import (
    TextConverterFactory,
    TitanicConverter,
    SpaceshipTitanicConverter,
    TemplateConverter,
    FeatureConverter,
    NaturalLanguageConverter,
)

console = Console()


def test_titanic_conversion():
    """Test Titanic text conversion."""
    console.print("\n[bold blue]Testing Titanic Text Conversion[/bold blue]")
    
    # Create converter
    converter = TextConverterFactory.create("titanic", augment=True)
    
    # Sample Titanic data
    sample = {
        "passengerid": 1,
        "survived": 0,
        "pclass": 3,
        "name": "Braund, Mr. Owen Harris",
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "ticket": "A/5 21171",
        "fare": 7.25,
        "cabin": None,
        "embarked": "S"
    }
    
    console.print("\n[yellow]Original data:[/yellow]")
    rprint(sample)
    
    console.print("\n[yellow]Text conversions:[/yellow]")
    for i in range(3):
        result = converter(sample)
        console.print(f"\n[cyan]Variation {i+1}:[/cyan]")
        console.print(result["text"])


def test_spaceship_conversion():
    """Test Spaceship Titanic conversion."""
    console.print("\n[bold blue]Testing Spaceship Titanic Text Conversion[/bold blue]")
    
    converter = TextConverterFactory.create("spaceship-titanic", augment=True)
    
    sample = {
        "passenger_id": "0001_01",
        "home_planet": "Europa",
        "cryosleep": False,
        "cabin": "B/0/P",
        "destination": "TRAPPIST-1e",
        "age": 39.0,
        "vip": False,
        "roomservice": 0.0,
        "foodcourt": 0.0,
        "shoppingmall": 0.0,
        "spa": 0.0,
        "vrdeck": 0.0,
        "name": "Maham Ofracculy",
        "transported": False
    }
    
    console.print("\n[yellow]Original data:[/yellow]")
    rprint(sample)
    
    console.print("\n[yellow]Text conversions:[/yellow]")
    for i in range(3):
        result = converter(sample)
        console.print(f"\n[cyan]Variation {i+1}:[/cyan]")
        console.print(result["text"])


def test_template_converter():
    """Test template-based conversion."""
    console.print("\n[bold blue]Testing Template Converter[/bold blue]")
    
    # Create custom template converter
    converter = TextConverterFactory.create(
        "template",
        templates=[
            "A ${age} year old ${sex} passenger in class ${pclass}.",
            "Passenger: ${sex}, age ${age}, traveling in ${pclass} class.",
        ],
        augmentation_templates=[
            "${sex} passenger, ${age} years old, ${pclass} class ticket.",
        ],
        augment=True,
    )
    
    sample = {"age": 25, "sex": "female", "pclass": 2}
    
    console.print("\n[yellow]Sample data:[/yellow]")
    rprint(sample)
    
    console.print("\n[yellow]Template conversions:[/yellow]")
    for i in range(3):
        result = converter(sample)
        console.print(f"Variation {i+1}: {result['text']}")


def test_feature_converter():
    """Test feature-based conversion."""
    console.print("\n[bold blue]Testing Feature Converter[/bold blue]")
    
    converter = TextConverterFactory.create(
        "feature",
        feature_names={
            "age": "Age",
            "fare": "Ticket price",
            "pclass": "Class",
        },
        feature_units={
            "age": "years",
            "fare": "USD",
        },
        numerical_bins={
            "age": [
                (0, 18, "child"),
                (18, 65, "adult"),
                (65, 100, "senior"),
            ]
        },
        prefix="Passenger details: ",
        separator=", ",
    )
    
    sample = {"age": 30, "fare": 72.50, "pclass": 1}
    
    console.print("\n[yellow]Sample data:[/yellow]")
    rprint(sample)
    
    result = converter(sample)
    console.print(f"\n[yellow]Feature conversion:[/yellow]\n{result['text']}")


def test_natural_language_converter():
    """Test natural language conversion."""
    console.print("\n[bold blue]Testing Natural Language Converter[/bold blue]")
    
    # Test different styles
    styles = ["descriptive", "narrative", "technical", "casual"]
    
    sample = {
        "name": "John Doe",
        "age": 35,
        "occupation": "Engineer",
        "location": "San Francisco",
        "experience": 10,
    }
    
    console.print("\n[yellow]Sample data:[/yellow]")
    rprint(sample)
    
    for style in styles:
        converter = TextConverterFactory.create(
            "natural_language",
            style=style,
            feature_templates={
                "age": ["is {value} years old", "aged {value}"],
                "occupation": ["works as a {value}", "is employed as a {value}"],
                "experience": ["has {value} years of experience"],
            }
        )
        
        result = converter(sample)
        console.print(f"\n[yellow]{style.capitalize()} style:[/yellow]")
        console.print(result["text"])


def test_with_real_data():
    """Test with real Titanic data."""
    console.print("\n[bold blue]Testing with Real Titanic Data[/bold blue]")
    
    # Load some real data
    try:
        df = pd.read_csv("data/titanic/train.csv")
        converter = TextConverterFactory.create("titanic", augment=True)
        
        console.print(f"\n[yellow]Converting first 5 samples from real data:[/yellow]")
        
        for idx in range(min(5, len(df))):
            row = df.iloc[idx].to_dict()
            result = converter(row)
            
            console.print(f"\n[cyan]Sample {idx+1} (PassengerID: {row['PassengerId']}):[/cyan]")
            console.print(f"Survived: {'Yes' if row['Survived'] else 'No'}")
            console.print(f"Text: {result['text']}")
            
    except Exception as e:
        console.print(f"[red]Could not load real data: {e}[/red]")


def test_converter_factory():
    """Test converter factory functionality."""
    console.print("\n[bold blue]Testing Converter Factory[/bold blue]")
    
    # List available converters
    available = TextConverterFactory.list_available()
    
    console.print("\n[yellow]Available converters:[/yellow]")
    for name, description in available.items():
        console.print(f"  - [cyan]{name}[/cyan]: {description}")
    
    # Test creating config template
    template = TextConverterFactory.create_config_template("template")
    console.print("\n[yellow]Template converter config template:[/yellow]")
    rprint(template)


def main():
    """Run all tests."""
    console.print("[bold green]Testing Text Conversion System[/bold green]")
    
    try:
        test_titanic_conversion()
        test_spaceship_conversion()
        test_template_converter()
        test_feature_converter()
        test_natural_language_converter()
        test_converter_factory()
        test_with_real_data()
        
        console.print("\n[bold green]✓ All tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed with error:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()