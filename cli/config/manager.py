"""Refactored configuration manager using dependency injection.

This module provides the main configuration management functionality
with injected dependencies for better testability and flexibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .schemas import KBertConfig, ProjectConfig
from .defaults import get_default_config, get_competition_defaults
from .loader import ConfigLoaderProtocol
from .validator import ConfigValidatorProtocol
from .merger import ConfigMergerProtocol
from .resolver import ConfigResolverProtocol


class RefactoredConfigManager:
    """Configuration manager with injected dependencies."""
    
    # Configuration file search paths
    CONFIG_SEARCH_PATHS = [
        Path("k-bert.yaml"),
        Path("k-bert.yml"),
        Path("k-bert.json"),
        Path("pyproject.toml"),
        Path(".k-bert/config.yaml"),
        Path(".k-bert/config.yml"),
        Path(".k-bert/config.json"),
    ]
    
    # User configuration path
    USER_CONFIG_PATH = Path("~/.k-bert/config.yaml").expanduser()
    
    def __init__(
        self,
        loader: ConfigLoaderProtocol,
        validator: ConfigValidatorProtocol,
        merger: ConfigMergerProtocol,
        resolver: ConfigResolverProtocol,
    ):
        """Initialize configuration manager with dependencies.
        
        Args:
            loader: Configuration loader
            validator: Configuration validator
            merger: Configuration merger
            resolver: Configuration resolver
        """
        self._loader = loader
        self._validator = validator
        self._merger = merger
        self._resolver = resolver
        
        # Caches
        self._user_config: Optional[KBertConfig] = None
        self._project_config: Optional[ProjectConfig] = None
        self._merged_config: Optional[KBertConfig] = None
    
    @property
    def user_config_path(self) -> Path:
        """Get user configuration path."""
        return self.USER_CONFIG_PATH
    
    @property
    def project_config_path(self) -> Optional[Path]:
        """Find project configuration path."""
        return self._loader.find_config_file(self.CONFIG_SEARCH_PATHS)
    
    def load_user_config(self, force_reload: bool = False) -> KBertConfig:
        """Load user configuration.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            User configuration
        """
        if not force_reload and self._user_config is not None:
            return self._user_config
        
        if not self.user_config_path.exists():
            logger.debug("No user configuration found, using defaults")
            self._user_config = get_default_config()
            return self._user_config
        
        try:
            # Load configuration file
            config_dict = self._loader.load_file(self.user_config_path)
            
            # Resolve dynamic values
            config_dict = self._resolver.resolve(config_dict)
            
            # Create config object
            self._user_config = KBertConfig.from_dict(config_dict)
            
            # Validate
            errors = self._validator.validate(self._user_config)
            if errors:
                logger.warning(f"User configuration has validation errors: {errors}")
            
            logger.debug(f"Loaded user configuration from {self.user_config_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load user configuration: {e}")
            self._user_config = get_default_config()
        
        return self._user_config
    
    def load_project_config(self, path: Optional[Path] = None) -> Optional[ProjectConfig]:
        """Load project configuration.
        
        Args:
            path: Specific path to load from (optional)
            
        Returns:
            Project configuration if found
        """
        if path is None:
            path = self.project_config_path
        
        if path is None or not path.exists():
            return None
        
        try:
            # Load configuration file
            config_dict = self._loader.load_file(path)
            
            # Resolve dynamic values
            config_dict = self._resolver.resolve(config_dict)
            
            # Add name if not present
            if "name" not in config_dict:
                config_dict["name"] = path.parent.name
            
            # Create config object
            self._project_config = ProjectConfig(**config_dict)
            
            # Validate
            errors = self._validator.validate(self._project_config)
            if errors:
                logger.warning(f"Project configuration has validation errors: {errors}")
            
            logger.debug(f"Loaded project configuration from {path}")
            
        except Exception as e:
            logger.warning(f"Failed to load project configuration: {e}")
            return None
        
        return self._project_config
    
    def get_merged_config(
        self,
        cli_overrides: Optional[Dict[str, Any]] = None,
        project_path: Optional[Path] = None,
        validate: bool = True,
    ) -> KBertConfig:
        """Get merged configuration from all sources.
        
        Configuration priority (highest to lowest):
        1. CLI arguments
        2. Environment variables
        3. Project configuration
        4. User configuration
        5. System defaults
        
        Args:
            cli_overrides: CLI argument overrides
            project_path: Specific project config path
            validate: Whether to validate the final config
            
        Returns:
            Merged configuration
        """
        # Start with defaults
        default_dict = get_default_config().to_dict()
        
        # Load user configuration
        user_config = self.load_user_config()
        user_dict = user_config.to_dict()
        
        # Load project configuration if available
        project_dict = {}
        project_config = self.load_project_config(project_path)
        if project_config:
            # Convert project config sections to dict
            for field in ["kaggle", "models", "training", "mlflow", "data", "logging"]:
                value = getattr(project_config, field, None)
                if value is not None:
                    project_dict[field] = value.model_dump(exclude_none=True)
            
            # Apply competition-specific defaults if specified
            if project_config.competition:
                comp_defaults = get_competition_defaults(project_config.competition)
                project_dict = self._merger.merge(comp_defaults.to_dict(), project_dict)
        
        # Get environment variable overrides
        env_overrides = self._resolver.get_env_overrides() if hasattr(self._resolver, 'get_env_overrides') else {}
        
        # Extract CLI overrides if needed
        if cli_overrides and hasattr(self._merger, 'extract_cli_overrides'):
            cli_overrides = self._merger.extract_cli_overrides(cli_overrides)
        
        # Merge all configurations
        if hasattr(self._merger, 'merge_cli_config'):
            merged_dict = self._merger.merge_cli_config(
                defaults=default_dict,
                user_config=user_dict,
                project_config=project_dict,
                env_vars=env_overrides,
                cli_args=cli_overrides,
            )
        else:
            # Fallback to standard merge
            configs = [default_dict, user_dict, project_dict]
            if env_overrides:
                configs.append(env_overrides)
            if cli_overrides:
                configs.append(cli_overrides)
            merged_dict = self._merger.merge(*configs)
        
        # Create final config object
        config = KBertConfig.from_dict(merged_dict)
        
        # Validate if requested
        if validate:
            errors = self._validator.validate(config)
            if errors:
                from .validators import ConfigValidationError
                raise ConfigValidationError(errors)
        
        self._merged_config = config
        return config
    
    def save_user_config(self, config: KBertConfig) -> None:
        """Save user configuration.
        
        Args:
            config: Configuration to save
        """
        import yaml
        
        # Ensure directory exists
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and remove None values
        config_dict = config.to_dict()
        
        # Convert Path objects to strings for YAML serialization
        config_dict = self._convert_paths_to_strings(config_dict)
        
        # Save as YAML
        with open(self.user_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved user configuration to {self.user_config_path}")
        
        # Clear cache
        self._user_config = None
    
    def save_project_config(
        self,
        config: Union[ProjectConfig, Dict[str, Any]],
        path: Optional[Path] = None
    ) -> None:
        """Save project configuration.
        
        Args:
            config: Configuration to save
            path: Path to save to (defaults to k-bert.yaml)
        """
        import yaml
        import json
        
        if path is None:
            path = Path("k-bert.yaml")
        
        # Convert to dict if needed
        if isinstance(config, ProjectConfig):
            config_dict = config.model_dump(exclude_none=True)
        else:
            config_dict = config
        
        # Determine format from extension
        if path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif path.name == "pyproject.toml":
            # For pyproject.toml, we need to be careful not to overwrite other content
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            import tomli_w
            
            if path.exists():
                with open(path, "rb") as f:
                    pyproject = tomllib.load(f)
            else:
                pyproject = {}
            
            if "tool" not in pyproject:
                pyproject["tool"] = {}
            
            pyproject["tool"]["k-bert"] = config_dict
            
            with open(path, "wb") as f:
                tomli_w.dump(pyproject, f)
        else:
            raise ValueError(f"Unsupported configuration format: {path.suffix}")
        
        logger.info(f"Saved project configuration to {path}")
        
        # Clear cache
        self._project_config = None
    
    def init_user_config(self, interactive: bool = True) -> KBertConfig:
        """Initialize user configuration.
        
        Args:
            interactive: Whether to prompt for values
            
        Returns:
            Initialized configuration
        """
        if self.user_config_path.exists():
            logger.warning(f"User configuration already exists at {self.user_config_path}")
            if interactive:
                from rich.prompt import Confirm
                if not Confirm.ask("Overwrite existing configuration?"):
                    return self.load_user_config()
        
        config = get_default_config()
        
        if interactive:
            from rich.prompt import Prompt, Confirm
            from rich.console import Console
            
            console = Console()
            console.print("\n[bold]Welcome to k-bert![/bold] Let's set up your configuration.\n")
            
            # Kaggle credentials
            username = Prompt.ask("Kaggle username", default="")
            if username:
                config.kaggle.username = username
                
                key = Prompt.ask("Kaggle API key (will be stored securely)", password=True, default="")
                if key:
                    config.kaggle.key = key
            
            # Default model
            model = Prompt.ask(
                "Default model",
                default=config.models.default_model,
                choices=[
                    "answerdotai/ModernBERT-base",
                    "answerdotai/ModernBERT-large",
                    "bert-base-uncased",
                ]
            )
            config.models.default_model = model
            
            # Output directory
            output_dir = Prompt.ask("Default output directory", default=str(config.training.output_dir))
            config.training.output_dir = Path(output_dir)
            
            # MLflow tracking
            enable_mlflow = Confirm.ask("Enable MLflow tracking?", default=True)
            config.mlflow.auto_log = enable_mlflow
            
            console.print(f"\n[green]Configuration saved to {self.user_config_path}[/green]")
        
        self.save_user_config(config)
        return config
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Dot-separated key path (e.g., "kaggle.username")
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        config = self.get_merged_config()
        
        # Navigate through nested keys
        value = config
        for part in key.split("."):
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set_value(self, key: str, value: Any, save: bool = True) -> None:
        """Set a configuration value.
        
        Args:
            key: Dot-separated key path (e.g., "kaggle.username")
            value: Value to set
            save: Whether to save immediately
        """
        config = self.load_user_config()
        
        # Navigate to the parent object
        parts = key.split(".")
        parent = config
        for part in parts[:-1]:
            if hasattr(parent, part):
                parent = getattr(parent, part)
            else:
                raise ValueError(f"Invalid configuration key: {key}")
        
        # Set the value
        final_key = parts[-1]
        if hasattr(parent, final_key):
            setattr(parent, final_key, value)
        else:
            raise ValueError(f"Invalid configuration key: {key}")
        
        if save:
            self.save_user_config(config)
    
    def list_all_values(self) -> Dict[str, Any]:
        """Get all configuration values as a flat dictionary.
        
        Returns:
            Flat dictionary of all configuration values
        """
        config = self.get_merged_config()
        return self._flatten_config(config.to_dict())
    
    def validate_project_config(self, path: Optional[Path] = None) -> List[str]:
        """Validate project configuration.
        
        Args:
            path: Path to project config file (optional)
            
        Returns:
            List of validation errors (empty if valid)
        """
        try:
            config = self.load_project_config(path)
            if config is None:
                return ["No project configuration found"]
            
            return self._validator.validate(config)
        except Exception as e:
            return [f"Failed to load configuration: {str(e)}"]
    
    def _flatten_config(self, obj: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration to dot-notation keys.
        
        Args:
            obj: Configuration object
            prefix: Current key prefix
            
        Returns:
            Flat dictionary
        """
        result = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(self._flatten_config(value, new_prefix))
                else:
                    result[new_prefix] = value
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(value, (dict, list)):
                    result.update(self._flatten_config(value, new_prefix))
                else:
                    result[new_prefix] = value
        else:
            result[prefix] = obj
        
        return result
    
    def _convert_paths_to_strings(self, obj: Any) -> Any:
        """Convert Path objects to strings for serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            Object with paths converted to strings
        """
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_paths_to_strings(v) for v in obj]
        else:
            return obj