"""
Test the modular dataloader system with real Titanic data.
Shows how to use the configuration system and text conversion.
"""

import sys
from pathlib import Path
import pandas as pd
import mlx.core as mx
from rich.console import Console
from rich import print as rprint
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.text_conversion import TextConverterFactory
from data.configs import (
    ConfigFactory,
    DataLoaderConfig,
    DatasetConfig,
)

console = Console()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase."""
    df.columns = df.columns.str.lower()
    return df


def test_titanic_text_conversion():
    """Test text conversion with real Titanic data."""
    console.print("[bold blue]Testing Titanic Text Conversion[/bold blue]\n")
    
    # Load data
    df = pd.read_csv("data/titanic/train.csv")
    df = normalize_columns(df)
    
    # Create Titanic converter
    converter = TextConverterFactory.create("titanic", augment=True)
    
    # Convert first 5 samples
    console.print("[yellow]Converting samples from Titanic dataset:[/yellow]\n")
    
    for idx in range(min(5, len(df))):
        row = df.iloc[idx].to_dict()
        
        # Convert to text
        result = converter(row)
        
        console.print(f"[cyan]Sample {idx+1} (PassengerID: {row.get('passengerid')}):[/cyan]")
        console.print(f"  Survived: {'Yes' if row.get('survived') else 'No'}")
        console.print(f"  Class: {row.get('pclass')}")
        console.print(f"  Age: {row.get('age')}")
        console.print(f"  Sex: {row.get('sex')}")
        console.print(f"  [green]Text:[/green] {result['text']}\n")


def test_configuration_system():
    """Test the configuration system."""
    console.print("[bold blue]Testing Configuration System[/bold blue]\n")
    
    # Create competition configuration
    config = ConfigFactory.create_competition_config(
        "titanic",
        data_path="data/titanic/train.csv",
        batch_size=32,
        max_length=256,
        augment=True,
    )
    
    console.print("[yellow]Competition Configuration:[/yellow]")
    console.print(f"  Name: {config.name}")
    console.print(f"  Competition: {config.competition_name}")
    console.print(f"  Label column: {config.dataset.label_column}")
    console.print(f"  Text converter: {config.dataloader.text_converter}")
    console.print(f"  Batch size: {config.dataloader.batch_size}")
    console.print(f"  Max length: {config.dataloader.max_length}")
    console.print(f"  Augmentation: {config.dataloader.augment}\n")
    
    # Test different optimization profiles
    console.print("[yellow]Optimization Profiles:[/yellow]")
    
    profiles = ["speed", "memory", "balanced", "debug"]
    for profile in profiles:
        dl_config = DataLoaderConfig()
        dl_config.apply_optimization_profile(profile)
        
        console.print(f"\n  [cyan]{profile.upper()}:[/cyan]")
        console.print(f"    Batch size: {dl_config.batch_size}")
        console.print(f"    Workers: {dl_config.num_workers}")
        console.print(f"    Prefetch: {dl_config.prefetch_size}")
        console.print(f"    Buffer: {dl_config.buffer_size}")


def test_batch_creation():
    """Test creating batches with text conversion."""
    console.print("\n[bold blue]Testing Batch Creation[/bold blue]\n")
    
    # Load small sample
    df = pd.read_csv("data/titanic/train.csv", nrows=10)
    df = normalize_columns(df)
    
    # Create converter
    converter = TextConverterFactory.create("titanic", augment=False)
    
    # Convert to text and create simple batches
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        result = converter(row.to_dict())
        texts.append(result["text"])
        labels.append(row["survived"])
    
    console.print(f"[yellow]Created {len(texts)} text samples[/yellow]\n")
    
    # Show batch info
    console.print("[cyan]Sample texts:[/cyan]")
    for i in range(min(3, len(texts))):
        console.print(f"  {i+1}. {texts[i][:100]}...")
    
    # Create MLX arrays (simulating tokenization)
    console.print("\n[yellow]Simulating batch creation:[/yellow]")
    
    # Simulate tokenized data
    batch_size = 4
    seq_length = 128
    
    # Create dummy tokenized data
    input_ids = mx.random.randint(0, 1000, shape=(batch_size, seq_length))
    attention_mask = mx.ones((batch_size, seq_length))
    labels = mx.array(labels[:batch_size])
    
    console.print(f"  Input IDs shape: {input_ids.shape}")
    console.print(f"  Attention mask shape: {attention_mask.shape}")
    console.print(f"  Labels shape: {labels.shape}")
    console.print(f"  Labels: {labels.tolist()}")


def test_caching():
    """Test caching functionality."""
    console.print("\n[bold blue]Testing Caching System[/bold blue]\n")
    
    from data.utils.caching import MemoryCache
    
    # Create cache
    cache = MemoryCache(max_items=5)
    
    # Create converter with caching
    converter = TextConverterFactory.create("titanic")
    
    # Test data
    sample = {
        "passengerid": 1,
        "survived": 0,
        "pclass": 3,
        "sex": "male",
        "age": 22.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S"
    }
    
    # Convert multiple times
    console.print("[yellow]Testing conversion caching:[/yellow]")
    
    import time
    
    # First conversion (no cache)
    start = time.time()
    result1 = converter(sample)
    time1 = time.time() - start
    
    # Cache the result
    cache_key = str(sample.get("passengerid"))
    cache.set(cache_key, result1["text"])
    
    # Second conversion (from cache)
    start = time.time()
    cached = cache.get(cache_key)
    time2 = time.time() - start
    
    console.print(f"  First conversion: {time1*1000:.3f}ms")
    console.print(f"  Cached retrieval: {time2*1000:.3f}ms")
    console.print(f"  Speedup: {time1/time2:.1f}x")
    
    # Test cache eviction
    console.print("\n[yellow]Testing cache eviction:[/yellow]")
    for i in range(7):
        cache.set(f"key_{i}", f"value_{i}")
    
    # Check if first items were evicted
    console.print(f"  Cache size: {len(cache.cache)}")
    console.print(f"  First key exists: {cache.exists('key_0')}")
    console.print(f"  Last key exists: {cache.exists('key_6')}")


def test_presets():
    """Test preset configurations."""
    console.print("\n[bold blue]Testing Preset Configurations[/bold blue]\n")
    
    from data.configs.presets import get_preset_config, list_presets
    
    # List all presets
    all_presets = list_presets()
    
    console.print("[yellow]Available Presets:[/yellow]")
    for category, names in all_presets.items():
        console.print(f"\n  [cyan]{category}:[/cyan]")
        for name in names:
            console.print(f"    - {name}")
    
    # Test competition preset
    console.print("\n[yellow]Titanic Competition Preset:[/yellow]")
    titanic_preset = get_preset_config("competition", "titanic")
    
    console.print(f"  Dataset config:")
    console.print(f"    Label column: {titanic_preset['dataset']['label_column']}")
    console.print(f"    ID column: {titanic_preset['dataset']['id_column']}")
    console.print(f"  DataLoader config:")
    console.print(f"    Text converter: {titanic_preset['dataloader']['text_converter']}")
    console.print(f"    Batch size: {titanic_preset['dataloader']['batch_size']}")


def main():
    """Run all tests."""
    console.print("[bold green]Testing Modular DataLoader System with Titanic Data[/bold green]\n")
    
    try:
        test_titanic_text_conversion()
        test_configuration_system()
        test_batch_creation()
        test_caching()
        test_presets()
        
        console.print("\n[bold green]✓ All tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()