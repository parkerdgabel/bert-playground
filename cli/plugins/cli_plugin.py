"""CLI plugin system for extending commands dynamically."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import typer
from loguru import logger

from cli.middleware.base import Middleware
from cli.pipeline.base import CommandHook
from cli.plugins.base import Plugin


class CLIPlugin(Plugin):
    """Base class for CLI plugins that add commands."""
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get commands provided by this plugin.
        
        Returns:
            Dictionary of command name to command function
        """
        return {}
    
    def get_middleware(self) -> List[Middleware]:
        """Get middleware provided by this plugin.
        
        Returns:
            List of middleware instances
        """
        return []
    
    def get_hooks(self) -> List[CommandHook]:
        """Get command hooks provided by this plugin.
        
        Returns:
            List of command hook instances
        """
        return []
    
    def get_app_extensions(self) -> Dict[str, typer.Typer]:
        """Get Typer app extensions (sub-apps).
        
        Returns:
            Dictionary of app name to Typer instance
        """
        return {}


class CLIPluginLoader:
    """Loader for CLI plugins."""
    
    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """Initialize plugin loader."""
        self.plugin_dirs = plugin_dirs or []
        self.plugins: Dict[str, CLIPlugin] = {}
        self.commands: Dict[str, Callable] = {}
        self.middleware: List[Middleware] = []
        self.hooks: List[CommandHook] = []
        self.app_extensions: Dict[str, typer.Typer] = {}
    
    def add_plugin_dir(self, path: Path) -> None:
        """Add plugin directory."""
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins."""
        discovered = []
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # Look for Python files
            for py_file in plugin_dir.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                module_name = py_file.stem
                discovered.append(module_name)
            
            # Look for packages
            for pkg_dir in plugin_dir.iterdir():
                if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                    discovered.append(pkg_dir.name)
        
        return discovered
    
    def load_plugin(self, name: str) -> Optional[CLIPlugin]:
        """Load a single plugin."""
        if name in self.plugins:
            return self.plugins[name]
        
        for plugin_dir in self.plugin_dirs:
            try:
                # Try loading as module
                module_path = plugin_dir / f"{name}.py"
                if module_path.exists():
                    spec = importlib.util.spec_from_file_location(name, module_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return self._extract_plugin_from_module(module, name)
                
                # Try loading as package
                pkg_path = plugin_dir / name
                if pkg_path.is_dir() and (pkg_path / "__init__.py").exists():
                    spec = importlib.util.spec_from_file_location(
                        name,
                        pkg_path / "__init__.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        return self._extract_plugin_from_module(module, name)
                
            except Exception as e:
                logger.error(f"Failed to load plugin {name}: {e}")
        
        return None
    
    def _extract_plugin_from_module(self, module: Any, name: str) -> Optional[CLIPlugin]:
        """Extract plugin from module."""
        # Look for CLIPlugin subclasses
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr) and
                issubclass(attr, CLIPlugin) and
                attr is not CLIPlugin
            ):
                try:
                    plugin = attr()
                    self.plugins[name] = plugin
                    return plugin
                except Exception as e:
                    logger.error(f"Failed to instantiate plugin {attr_name}: {e}")
        
        # Look for plugin factory function
        if hasattr(module, "create_plugin"):
            try:
                plugin = module.create_plugin()
                if isinstance(plugin, CLIPlugin):
                    self.plugins[name] = plugin
                    return plugin
            except Exception as e:
                logger.error(f"Failed to create plugin via factory: {e}")
        
        return None
    
    def load_all_plugins(self) -> None:
        """Load all discovered plugins."""
        for plugin_name in self.discover_plugins():
            plugin = self.load_plugin(plugin_name)
            if plugin:
                self._register_plugin_components(plugin, plugin_name)
    
    def _register_plugin_components(self, plugin: CLIPlugin, name: str) -> None:
        """Register plugin components."""
        # Register commands
        for cmd_name, cmd_func in plugin.get_commands().items():
            full_name = f"{name}:{cmd_name}"
            self.commands[full_name] = cmd_func
            logger.info(f"Registered command: {full_name}")
        
        # Register middleware
        for middleware in plugin.get_middleware():
            self.middleware.append(middleware)
            logger.info(f"Registered middleware: {middleware.name}")
        
        # Register hooks
        for hook in plugin.get_hooks():
            self.hooks.append(hook)
            logger.info(f"Registered hook: {hook.name}")
        
        # Register app extensions
        for app_name, app in plugin.get_app_extensions().items():
            self.app_extensions[app_name] = app
            logger.info(f"Registered app extension: {app_name}")
    
    def validate_plugin(self, plugin: CLIPlugin) -> List[str]:
        """Validate plugin components."""
        errors = []
        
        # Validate commands
        for cmd_name, cmd_func in plugin.get_commands().items():
            if not callable(cmd_func):
                errors.append(f"Command {cmd_name} is not callable")
            
            # Check for Typer decorator
            if not hasattr(cmd_func, "__typer_meta__"):
                logger.warning(f"Command {cmd_name} missing Typer decorators")
        
        # Validate middleware
        for middleware in plugin.get_middleware():
            if not isinstance(middleware, Middleware):
                errors.append(f"Invalid middleware type: {type(middleware)}")
        
        # Validate hooks
        for hook in plugin.get_hooks():
            if not isinstance(hook, CommandHook):
                errors.append(f"Invalid hook type: {type(hook)}")
        
        return errors
    
    def get_command(self, name: str) -> Optional[Callable]:
        """Get command by name."""
        return self.commands.get(name)
    
    def apply_to_app(self, app: typer.Typer) -> None:
        """Apply all plugins to a Typer app."""
        # Add individual commands
        for cmd_name, cmd_func in self.commands.items():
            # Extract base name for command
            base_name = cmd_name.split(":")[-1]
            app.command(name=base_name)(cmd_func)
        
        # Add app extensions as sub-commands
        for app_name, sub_app in self.app_extensions.items():
            app.add_typer(sub_app, name=app_name)


class DynamicCommandRegistry:
    """Registry for dynamically adding commands."""
    
    def __init__(self):
        """Initialize registry."""
        self._commands: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        aliases: Optional[List[str]] = None,
        **metadata
    ) -> None:
        """Register a command."""
        self._commands[name] = func
        self._metadata[name] = metadata
        
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name
    
    def unregister(self, name: str) -> None:
        """Unregister a command."""
        if name in self._commands:
            del self._commands[name]
            del self._metadata[name]
            
            # Remove aliases
            self._aliases = {
                alias: target
                for alias, target in self._aliases.items()
                if target != name
            }
    
    def get(self, name: str) -> Optional[Callable]:
        """Get command by name or alias."""
        if name in self._commands:
            return self._commands[name]
        
        if name in self._aliases:
            return self._commands.get(self._aliases[name])
        
        return None
    
    def list_commands(self) -> List[str]:
        """List all registered commands."""
        return list(self._commands.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get command metadata."""
        return self._metadata.get(name, {})


# Example plugin implementation
class ExampleCLIPlugin(CLIPlugin):
    """Example CLI plugin implementation."""
    
    def get_commands(self) -> Dict[str, Callable]:
        """Get example commands."""
        
        @typer.command()
        def hello(name: str = typer.Argument(..., help="Name to greet")):
            """Say hello to someone."""
            typer.echo(f"Hello, {name}!")
        
        @typer.command()
        def goodbye(name: str = typer.Argument(..., help="Name to say goodbye to")):
            """Say goodbye to someone."""
            typer.echo(f"Goodbye, {name}!")
        
        return {
            "hello": hello,
            "goodbye": goodbye,
        }
    
    def get_middleware(self) -> List[Middleware]:
        """Get example middleware."""
        from cli.middleware.logging import LoggingMiddleware
        
        return [
            LoggingMiddleware(
                name="ExampleLogging",
                structured=True
            )
        ]
    
    def get_hooks(self) -> List[CommandHook]:
        """Get example hooks."""
        from cli.pipeline.hooks import ValidationHook
        
        return [
            ValidationHook(
                name="ExampleValidation",
                required_args=["name"]
            )
        ]