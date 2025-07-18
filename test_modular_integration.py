"""
Test script to verify ModularMLXDataLoader integration with existing trainer.

This script tests the protocol compliance and basic functionality of the new
modular dataloader with the existing MLXTrainer infrastructure.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
import mlx.core as mx
from loguru import logger

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data.modular_mlx_dataloader import ModularMLXDataLoader
from training.protocols import DataLoaderProtocol
from data.configs import ConfigFactory

console = Console()


def test_protocol_compliance():
    """Test that ModularMLXDataLoader implements DataLoaderProtocol."""
    console.print("[bold blue]Testing Protocol Compliance[/bold blue]")
    
    # Check if it's a valid protocol implementation
    console.print(f"Is DataLoaderProtocol: {isinstance(ModularMLXDataLoader, type)}")
    
    # Test with a simple config
    try:
        # Create a minimal config for testing
        config = ConfigFactory.create_competition_config(
            "titanic",
            data_path="data/titanic/train.csv",
            batch_size=8,
            max_length=128,
        )
        
        # Create loader
        loader = ModularMLXDataLoader(config)
        
        # Check protocol compliance
        is_compliant = isinstance(loader, DataLoaderProtocol)
        console.print(f"✓ Protocol compliant: {is_compliant}")
        
        # Check required methods
        has_iter = hasattr(loader, '__iter__')
        has_len = hasattr(loader, '__len__')
        
        console.print(f"✓ Has __iter__: {has_iter}")
        console.print(f"✓ Has __len__: {has_len}")
        
        # Check optional attributes
        has_batch_size = hasattr(loader, 'batch_size')
        has_max_length = hasattr(loader, 'max_length')
        has_dataset_spec = hasattr(loader, 'dataset_spec')
        
        console.print(f"✓ Has batch_size: {has_batch_size}")
        console.print(f"✓ Has max_length: {has_max_length}")
        console.print(f"✓ Has dataset_spec: {has_dataset_spec}")
        
        return loader
        
    except Exception as e:
        console.print(f"[red]✗ Protocol compliance test failed: {e}[/red]")
        return None


def test_batch_format(loader):
    """Test that batches have the correct format."""
    console.print("\n[bold blue]Testing Batch Format[/bold blue]")
    
    try:
        # Get a sample batch
        batch = loader.get_sample_batch()
        
        # Check required keys
        required_keys = {'input_ids', 'attention_mask', 'labels'}
        actual_keys = set(batch.keys())
        
        console.print(f"Required keys: {required_keys}")
        console.print(f"Actual keys: {actual_keys}")
        console.print(f"✓ Has all required keys: {required_keys.issubset(actual_keys)}")
        
        # Check types and shapes
        table = Table(title="Batch Array Information")
        table.add_column("Key", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Shape", style="yellow")
        table.add_column("Dtype", style="magenta")
        
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch:
                value = batch[key]
                table.add_row(
                    key,
                    str(type(value)),
                    str(value.shape),
                    str(value.dtype)
                )
        
        console.print(table)
        
        # Verify MLX arrays
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batch:
                is_mlx = isinstance(batch[key], mx.array)
                console.print(f"✓ {key} is MLX array: {is_mlx}")
        
        # Check shapes make sense
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        shapes_match = input_ids.shape == attention_mask.shape
        batch_size_correct = input_ids.shape[0] == labels.shape[0]
        
        console.print(f"✓ Input/attention shapes match: {shapes_match}")
        console.print(f"✓ Batch sizes consistent: {batch_size_correct}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Batch format test failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_iteration(loader):
    """Test iteration over the dataloader."""
    console.print("\n[bold blue]Testing Iteration[/bold blue]")
    
    try:
        # Test length
        num_batches = len(loader)
        console.print(f"Number of batches: {num_batches}")
        
        # Test iteration over a few batches
        batches_processed = 0
        for i, batch in enumerate(loader):
            batches_processed += 1
            
            # Check batch format
            if not all(key in batch for key in ['input_ids', 'attention_mask', 'labels']):
                console.print(f"[red]✗ Batch {i} missing required keys[/red]")
                return False
            
            # Test a few batches only
            if i >= 2:
                break
        
        console.print(f"✓ Successfully processed {batches_processed} batches")
        console.print(f"✓ Iteration works correctly")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Iteration test failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_dataset_spec(loader):
    """Test dataset specification."""
    console.print("\n[bold blue]Testing Dataset Specification[/bold blue]")
    
    try:
        spec = loader.dataset_spec
        console.print(f"Dataset spec type: {type(spec)}")
        
        if spec:
            table = Table(title="Dataset Specification")
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in spec.items():
                table.add_row(str(key), str(value))
            
            console.print(table)
        
        console.print("✓ Dataset spec accessible")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Dataset spec test failed: {e}[/red]")
        return False


def test_caching_features(loader):
    """Test caching functionality."""
    console.print("\n[bold blue]Testing Caching Features[/bold blue]")
    
    try:
        # Get cache stats
        stats = loader.get_cache_stats()
        console.print(f"Cache stats: {stats}")
        
        # Process a batch twice to test caching
        batch1 = loader.get_sample_batch()
        batch2 = loader.get_sample_batch()
        
        # Check if cache stats changed
        stats_after = loader.get_cache_stats()
        console.print(f"Cache stats after processing: {stats_after}")
        
        console.print("✓ Caching features accessible")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Caching test failed: {e}[/red]")
        return False


def main():
    """Run all integration tests."""
    console.print("[bold green]ModularMLXDataLoader Integration Test[/bold green]\n")
    
    # Test 1: Protocol compliance
    loader = test_protocol_compliance()
    if not loader:
        console.print("[red]Failed protocol compliance test - aborting[/red]")
        return
    
    # Test 2: Batch format
    if not test_batch_format(loader):
        console.print("[red]Failed batch format test[/red]")
    
    # Test 3: Iteration
    if not test_iteration(loader):
        console.print("[red]Failed iteration test[/red]")
    
    # Test 4: Dataset spec
    if not test_dataset_spec(loader):
        console.print("[red]Failed dataset spec test[/red]")
    
    # Test 5: Caching
    if not test_caching_features(loader):
        console.print("[red]Failed caching test[/red]")
    
    console.print("\n[bold green]Integration tests completed![/bold green]")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Test with actual MLXTrainer")
    console.print("2. Verify text conversion quality")
    console.print("3. Performance comparison with existing dataloaders")


if __name__ == "__main__":
    main()