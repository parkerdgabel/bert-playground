"""
Test script for the modular dataloader system with Titanic dataset.
"""

import sys
import mlx.core as mx
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import print as rprint

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.configs import (
    ConfigFactory,
    get_preset_config,
    DataLoaderConfig,
    DatasetConfig,
)
from data.text_conversion import (
    TextConverterFactory,
    TitanicConverter,
)
from data.core.base_dataset import MLXDataset
from data.core.base_loader import MLXDataLoader
from data.transforms import Compose, Tokenize, ToMLXArray
from embeddings.tokenizer_wrapper import TokenizerWrapper

console = Console()


def test_text_conversion():
    """Test text conversion for Titanic data."""
    console.print("\n[bold blue]Testing Text Conversion[/bold blue]")
    
    # Create Titanic converter
    converter = TextConverterFactory.create("titanic", augment=True)
    
    # Sample data
    sample_data = {
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
    
    # Convert to text
    console.print("\n[yellow]Original data:[/yellow]")
    rprint(sample_data)
    
    console.print("\n[yellow]Converted text variations:[/yellow]")
    for i in range(3):
        result = converter(sample_data)
        console.print(f"\nVariation {i+1}:")
        console.print(result["text"])


def test_configuration_system():
    """Test configuration system."""
    console.print("\n[bold blue]Testing Configuration System[/bold blue]")
    
    # Create competition config
    config = ConfigFactory.create_competition_config(
        "titanic",
        data_path="data/titanic/train.csv",
        batch_size=16,
        max_length=128,
    )
    
    console.print("\n[yellow]Competition configuration:[/yellow]")
    console.print(f"Name: {config.name}")
    console.print(f"Competition: {config.competition_name}")
    console.print(f"Label column: {config.dataset.label_column}")
    console.print(f"Text converter: {config.dataloader.text_converter}")
    console.print(f"Batch size: {config.dataloader.batch_size}")
    console.print(f"Max length: {config.dataloader.max_length}")
    
    # Test preset configs
    console.print("\n[yellow]Available presets:[/yellow]")
    from data.configs.presets import list_presets
    presets = list_presets()
    
    table = Table(title="Configuration Presets")
    table.add_column("Category", style="cyan")
    table.add_column("Presets", style="green")
    
    for category, preset_names in presets.items():
        table.add_row(category, ", ".join(preset_names))
    
    console.print(table)


def test_dataloader_creation():
    """Test creating and using a dataloader."""
    console.print("\n[bold blue]Testing DataLoader Creation[/bold blue]")
    
    # Create configuration
    dataset_config = DatasetConfig(
        data_path="data/titanic/train.csv",
        label_column="survived",
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        max_samples=100,  # Limit for testing
    )
    
    dataloader_config = DataLoaderConfig(
        batch_size=8,
        shuffle=True,
        max_length=128,
        text_converter="titanic",
        text_converter_config={"augment": True},
        optimization_profile="balanced",
    )
    
    # Apply optimization profile
    dataloader_config.apply_optimization_profile("balanced")
    
    console.print("\n[yellow]DataLoader configuration:[/yellow]")
    console.print(f"Batch size: {dataloader_config.batch_size}")
    console.print(f"Prefetch size: {dataloader_config.prefetch_size}")
    console.print(f"Num workers: {dataloader_config.num_workers}")
    console.print(f"Buffer size: {dataloader_config.buffer_size}")
    
    # Create text converter
    converter = TextConverterFactory.create(
        dataloader_config.text_converter,
        **dataloader_config.text_converter_config
    )
    
    # Create tokenizer
    console.print("\n[yellow]Loading tokenizer...[/yellow]")
    tokenizer = TokenizerWrapper(
        tokenizer_name=dataloader_config.tokenizer_name,
        backend=dataloader_config.tokenizer_backend,
    )
    
    # Create transforms
    transforms = Compose([
        converter,
        Tokenize(
            tokenizer=tokenizer,
            max_length=dataloader_config.max_length,
            padding=dataloader_config.padding,
            truncation=dataloader_config.truncation,
        ),
        ToMLXArray(fields=["input_ids", "attention_mask", "label"]),
    ])
    
    # Create dataset
    console.print("\n[yellow]Creating dataset...[/yellow]")
    import pandas as pd
    df = pd.read_csv(dataset_config.data_path)
    
    # Limit samples for testing
    if dataset_config.max_samples:
        df = df.head(dataset_config.max_samples)
    
    # Create MLX dataset (using our base dataset as a simple wrapper)
    class SimpleDataset:
        def __init__(self, df, transforms):
            self.data = df.to_dict('records')
            self.transforms = transforms
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            if self.transforms:
                sample = self.transforms(sample)
            return sample
    
    dataset = SimpleDataset(df, transforms)
    
    console.print(f"Dataset size: {len(dataset)}")
    
    # Test a few samples
    console.print("\n[yellow]Sample data:[/yellow]")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        console.print(f"\nSample {i+1}:")
        console.print(f"  Text: {sample.get('text', 'N/A')[:100]}...")
        console.print(f"  Input IDs shape: {sample['input_ids'].shape}")
        console.print(f"  Label: {sample['label']}")


def test_mlx_optimization():
    """Test MLX-specific optimizations."""
    console.print("\n[bold blue]Testing MLX Optimizations[/bold blue]")
    
    # Test dtype utilities
    from data.utils.dtype_utils import (
        infer_dtype,
        convert_to_mlx,
        ensure_dtype_consistency,
        get_memory_usage,
        format_bytes,
    )
    
    # Test data
    test_data = {
        "input_ids": [101, 2054, 2003, 1996, 3633, 102],
        "attention_mask": [1, 1, 1, 1, 1, 1],
        "label": 0,
        "score": 0.95,
    }
    
    console.print("\n[yellow]Testing dtype inference:[/yellow]")
    for field, value in test_data.items():
        dtype = infer_dtype(value, field)
        console.print(f"{field}: {dtype}")
    
    # Convert to MLX arrays
    console.print("\n[yellow]Converting to MLX arrays:[/yellow]")
    mlx_data = ensure_dtype_consistency(test_data)
    
    for field, array in mlx_data.items():
        if isinstance(array, mx.array):
            console.print(f"{field}: shape={array.shape}, dtype={array.dtype}")
    
    # Check memory usage
    memory = get_memory_usage(mlx_data)
    console.print(f"\nTotal memory usage: {format_bytes(memory)}")


def test_caching_system():
    """Test caching system."""
    console.print("\n[bold blue]Testing Caching System[/bold blue]")
    
    from data.utils.caching import DiskCache, MemoryCache, TokenizationCache
    
    # Test memory cache
    console.print("\n[yellow]Testing memory cache:[/yellow]")
    mem_cache = MemoryCache(max_items=10)
    
    # Add items
    for i in range(5):
        mem_cache.set(f"key_{i}", f"value_{i}")
    
    # Retrieve items
    console.print("Retrieving cached items:")
    for i in range(3):
        value = mem_cache.get(f"key_{i}")
        console.print(f"  key_{i}: {value}")
    
    # Test tokenization cache
    console.print("\n[yellow]Testing tokenization cache:[/yellow]")
    tok_cache = TokenizationCache(
        cache_dir="./cache_test",
        tokenizer_name="test_tokenizer"
    )
    
    # Cache tokenized data
    text = "This is a test sentence."
    encoding = {"input_ids": [1, 2, 3, 4, 5], "attention_mask": [1, 1, 1, 1, 1]}
    
    tok_cache.set_tokenized(
        text=text,
        max_length=128,
        padding="max_length",
        truncation=True,
        encoding=encoding
    )
    
    # Retrieve cached
    cached = tok_cache.get_tokenized(
        text=text,
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    console.print(f"Cached tokenization: {cached}")
    
    # Clean up test cache
    import shutil
    if Path("./cache_test").exists():
        shutil.rmtree("./cache_test")


def main():
    """Run all tests."""
    console.print("[bold green]Testing Modular DataLoader System[/bold green]")
    
    try:
        test_text_conversion()
        test_configuration_system()
        test_dataloader_creation()
        test_mlx_optimization()
        test_caching_system()
        
        console.print("\n[bold green]✓ All tests completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✗ Test failed with error:[/bold red]")
        console.print(f"[red]{str(e)}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()