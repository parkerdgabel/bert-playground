"""Refactored training command using dependency injection.

This is an example of how to refactor commands to use the new BaseCommand
with dependency injection.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...commands.base import BaseCommand
from ...commands.factory import register_command
from ...config.schemas import KBertConfig
from ...config.manager import RefactoredConfigManager
from ...plugins import ComponentRegistry


@register_command("train")
class TrainCommand(BaseCommand):
    """Training command with dependency injection."""
    
    def execute(
        self,
        config: Optional[Path] = None,
        experiment: Optional[str] = None,
        train_data: Optional[Path] = None,
        val_data: Optional[Path] = None,
        epochs: Optional[int] = None,
        output_dir: Optional[Path] = None,
        no_config: bool = False,
        dry_run: bool = False,
        debug: bool = False,
    ) -> None:
        """Execute training command.
        
        Args:
            config: Configuration file path
            experiment: Experiment name from config
            train_data: Training data path override
            val_data: Validation data path override
            epochs: Number of epochs override
            output_dir: Output directory override
            no_config: Run without configuration
            dry_run: Show config without training
            debug: Enable debug logging
        """
        # Inject services
        config_manager = self.inject(RefactoredConfigManager)
        registry = self.inject(ComponentRegistry)
        
        # Setup debug logging if requested
        if debug:
            logger.remove()
            logger.add(sys.stderr, level="DEBUG")
        
        # Load configuration
        if no_config:
            self.console.print("[yellow]Running without configuration file[/yellow]")
            merged_config = self._get_minimal_config(train_data, val_data, epochs, output_dir)
        else:
            merged_config = self._load_configuration(config_manager, config, experiment)
        
        # Apply CLI overrides
        merged_config = self._apply_overrides(
            merged_config,
            train_data,
            val_data,
            epochs,
            output_dir
        )
        
        # Show configuration if dry run
        if dry_run:
            self._show_configuration(merged_config)
            return
        
        # Load project plugins if in project directory
        self._load_plugins(registry, merged_config)
        
        # Run training
        self._run_training(merged_config, registry)
    
    def _load_configuration(
        self,
        config_manager: RefactoredConfigManager,
        config_path: Optional[Path],
        experiment: Optional[str]
    ) -> KBertConfig:
        """Load and merge configuration.
        
        Args:
            config_manager: Configuration manager
            config_path: Optional config file path
            experiment: Optional experiment name
            
        Returns:
            Merged configuration
        """
        self.console.print("Loading configuration...")
        
        # Get merged config
        cli_overrides = {"experiment": experiment} if experiment else None
        merged_config = config_manager.get_merged_config(
            cli_overrides=cli_overrides,
            project_path=config_path
        )
        
        # If experiment specified, load experiment config
        if experiment and hasattr(merged_config, "experiments"):
            if experiment in merged_config.experiments:
                exp_config = merged_config.experiments[experiment]
                self.console.print(f"Using experiment: [green]{experiment}[/green]")
                # Merge experiment config
                # This would need implementation in the config merger
            else:
                raise ValueError(f"Experiment '{experiment}' not found in configuration")
        
        return merged_config
    
    def _get_minimal_config(
        self,
        train_data: Optional[Path],
        val_data: Optional[Path],
        epochs: Optional[int],
        output_dir: Optional[Path]
    ) -> KBertConfig:
        """Create minimal configuration for no-config mode.
        
        Args:
            train_data: Training data path
            val_data: Validation data path
            epochs: Number of epochs
            output_dir: Output directory
            
        Returns:
            Minimal configuration
        """
        from ...config.defaults import get_default_config
        
        config = get_default_config()
        
        # Require at least training data
        if not train_data:
            raise ValueError("--train is required when using --no-config")
        
        # Update with provided values
        config.data.train_path = train_data
        if val_data:
            config.data.val_path = val_data
        if epochs:
            config.training.epochs = epochs
        if output_dir:
            config.training.output_dir = output_dir
        
        return config
    
    def _apply_overrides(
        self,
        config: KBertConfig,
        train_data: Optional[Path],
        val_data: Optional[Path],
        epochs: Optional[int],
        output_dir: Optional[Path]
    ) -> KBertConfig:
        """Apply CLI overrides to configuration.
        
        Args:
            config: Base configuration
            train_data: Training data override
            val_data: Validation data override
            epochs: Epochs override
            output_dir: Output directory override
            
        Returns:
            Updated configuration
        """
        if train_data:
            config.data.train_path = train_data
            self.console.print(f"Override: train_path = {train_data}")
        
        if val_data:
            config.data.val_path = val_data
            self.console.print(f"Override: val_path = {val_data}")
        
        if epochs:
            config.training.epochs = epochs
            self.console.print(f"Override: epochs = {epochs}")
        
        if output_dir:
            config.training.output_dir = output_dir
            self.console.print(f"Override: output_dir = {output_dir}")
        
        return config
    
    def _show_configuration(self, config: KBertConfig) -> None:
        """Display configuration for dry run.
        
        Args:
            config: Configuration to display
        """
        from rich.syntax import Syntax
        import yaml
        
        # Convert to dict and format as YAML
        config_dict = config.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        # Display with syntax highlighting
        syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
        
        self.console.print(
            Panel(
                syntax,
                title="[bold]Training Configuration[/bold]",
                border_style="blue",
            )
        )
        
        self.console.print("\n[yellow]Dry run mode - no training performed[/yellow]")
    
    def _load_plugins(self, registry: ComponentRegistry, config: KBertConfig) -> None:
        """Load project plugins if available.
        
        Args:
            registry: Component registry
            config: Configuration
        """
        project_root = Path.cwd()
        src_dir = project_root / "src"
        
        if src_dir.exists():
            self.console.print("Loading project plugins...")
            from ...plugins import load_project_plugins
            loaded = load_project_plugins(project_root)
            
            if loaded:
                self.console.print(f"[green]Loaded {len(loaded)} plugins[/green]")
                for plugin_type, plugin_name in loaded:
                    self.console.print(f"  • {plugin_type}: {plugin_name}")
    
    def _run_training(self, config: KBertConfig, registry: ComponentRegistry) -> None:
        """Run the actual training.
        
        Args:
            config: Training configuration
            registry: Component registry
        """
        # Import here to avoid circular imports
        from bert_playground.training import Trainer
        from bert_playground.data import create_dataloaders
        from bert_playground.models import create_model
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Create output directory
            output_dir = Path(config.training.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            config_path = output_dir / "config.yaml"
            with open(config_path, "w") as f:
                import yaml
                yaml.dump(config.to_dict(), f, default_flow_style=False)
            
            # Load data
            task = progress.add_task("Loading data...", total=None)
            train_loader, val_loader, num_classes = create_dataloaders(
                train_path=config.data.train_path,
                val_path=config.data.val_path,
                batch_size=config.training.batch_size,
                max_length=config.data.max_length,
                tokenizer=config.models.default_model,
                num_workers=config.data.num_workers,
            )
            progress.update(task, completed=True, description="Data loaded")
            
            # Create model
            task = progress.add_task("Creating model...", total=None)
            model = create_model(
                model_name=config.models.default_model,
                num_classes=num_classes,
                architecture=config.models.default_architecture,
                use_lora=config.models.use_lora_by_default,
                lora_config=config.models.lora_config,
                registry=registry,
            )
            progress.update(task, completed=True, description="Model created")
            
            # Create trainer
            task = progress.add_task("Initializing trainer...", total=None)
            trainer = Trainer(
                model=model,
                config=config.training,
                output_dir=output_dir,
                use_mlflow=config.mlflow.auto_log,
                mlflow_config=config.mlflow,
            )
            progress.update(task, completed=True, description="Trainer initialized")
        
        # Run training
        self.console.print(
            Panel(
                f"[bold]Starting Training[/bold]\n\n"
                f"Model: {config.models.default_model}\n"
                f"Epochs: {config.training.epochs}\n"
                f"Batch Size: {config.training.batch_size}\n"
                f"Output: {output_dir}",
                border_style="green",
            )
        )
        
        try:
            trainer.train(train_loader, val_loader)
            
            self.console.print(
                f"\n[green]✓[/green] Training completed successfully!"
            )
            self.console.print(f"Results saved to: {output_dir}")
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Training interrupted by user[/yellow]")
            raise
        except Exception as e:
            self.console.print(f"\n[red]Training failed: {str(e)}[/red]")
            raise