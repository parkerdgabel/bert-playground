"""
Test consolidated integration between TrainingConfig and ModularMLXDataLoader.

This test verifies that the simplified, consolidated approach works correctly
without creating redundant "unified" configurations.
"""

import sys
from pathlib import Path
from rich.console import Console
import mlx.core as mx

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.config import TrainingConfig
from training.competition_factory import create_competition_config, auto_configure_from_data
from data.modular_mlx_dataloader import ModularMLXDataLoader
from training.protocols import DataLoaderProtocol

console = Console()


def test_competition_factory():
    """Test the simplified competition factory."""
    console.print("[bold blue]Testing Competition Factory[/bold blue]")
    
    try:
        # Test known competition config
        config = create_competition_config("titanic", "data/titanic/train.csv")
        
        console.print(f"✓ Created config for: {config.competition_name}")
        console.print(f"  Target column: {config.target_column}")
        console.print(f"  Text converter: {config.dataloader.text_converter}")
        console.print(f"  Task type: {config.task_type}")
        console.print(f"  Batch size: {config.batch_size}")
        
        return config
        
    except Exception as e:
        console.print(f"[red]✗ Competition factory test failed: {e}[/red]")
        return None


def test_enhanced_training_config():
    """Test the enhanced TrainingConfig with dataloader support."""
    console.print("\n[bold blue]Testing Enhanced TrainingConfig[/bold blue]")
    
    try:
        config = TrainingConfig()
        
        # Check that dataloader config exists
        has_dataloader = hasattr(config, 'dataloader')
        console.print(f"✓ Has dataloader config: {has_dataloader}")
        
        if has_dataloader:
            console.print(f"  Text converter: {config.dataloader.text_converter}")
            console.print(f"  Caching enabled: {config.dataloader.enable_caching}")
            console.print(f"  Optimization profile: {config.dataloader.optimization_profile}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Enhanced config test failed: {e}[/red]")
        return False


def test_modular_dataloader_with_training_config():
    """Test ModularMLXDataLoader with TrainingConfig."""
    console.print("\n[bold blue]Testing ModularMLXDataLoader with TrainingConfig[/bold blue]")
    
    try:
        # Create training config
        config = create_competition_config("titanic", "data/titanic/train.csv")
        
        # Create dataloader with training config
        loader = ModularMLXDataLoader(config)
        
        # Check protocol compliance
        is_compliant = isinstance(loader, DataLoaderProtocol)
        console.print(f"✓ Protocol compliant: {is_compliant}")
        
        # Test basic functionality
        num_batches = len(loader)
        console.print(f"✓ Number of batches: {num_batches}")
        
        # Test sample batch
        batch = loader.get_sample_batch()
        console.print(f"✓ Sample batch shapes:")
        console.print(f"  Input IDs: {batch['input_ids'].shape}")
        console.print(f"  Attention mask: {batch['attention_mask'].shape}")
        console.print(f"  Labels: {batch['labels'].shape}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Dataloader integration test failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")
        return False


def test_auto_configuration():
    """Test auto-configuration from data."""
    console.print("\n[bold blue]Testing Auto-Configuration[/bold blue]")
    
    try:
        # Test auto-configuration
        config = auto_configure_from_data("data/titanic/train.csv")
        
        console.print(f"✓ Auto-detected target: {config.target_column}")
        console.print(f"  Competition type: {config.competition_type}")
        console.print(f"  Task type: {config.task_type}")
        console.print(f"  Num labels: {config.num_labels}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Auto-configuration test failed: {e}[/red]")
        return False


def main():
    """Run consolidated integration tests."""
    console.print("[bold green]Consolidated Integration Test[/bold green]\n")
    
    # Test 1: Competition factory
    config = test_competition_factory()
    if not config:
        return
    
    # Test 2: Enhanced TrainingConfig
    if not test_enhanced_training_config():
        return
    
    # Test 3: ModularMLXDataLoader integration
    if not test_modular_dataloader_with_training_config():
        return
    
    # Test 4: Auto-configuration
    if not test_auto_configuration():
        return
    
    console.print("\n[bold green]✓ All consolidated integration tests passed![/bold green]")
    console.print("\n[yellow]Benefits of consolidated approach:[/yellow]")
    console.print("• Single TrainingConfig instead of multiple config systems")
    console.print("• ModularMLXDataLoader works with existing trainer infrastructure")
    console.print("• Competition factory creates standard TrainingConfig objects")
    console.print("• No redundant 'unified' or parallel configuration systems")
    console.print("• Cleaner, more maintainable codebase")


if __name__ == "__main__":
    main()