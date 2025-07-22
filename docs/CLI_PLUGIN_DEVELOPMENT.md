# CLI Plugin Development Guide

This guide covers how to develop plugins for the k-bert CLI system, enabling you to extend the CLI with custom commands, middleware, hooks, and functionality.

## Plugin Architecture Overview

The k-bert CLI supports a flexible plugin system that allows you to:

- **Add Custom Commands**: Create new CLI commands with full Typer integration
- **Register Middleware**: Add middleware for logging, validation, error handling
- **Define Hooks**: Create hooks that execute at specific pipeline phases
- **Extend Applications**: Add entire sub-applications to the CLI

## Creating a Basic Plugin

### 1. Plugin Structure

Create a plugin by extending the `CLIPlugin` base class:

```python
# my_plugin.py
from cli.plugins.cli_plugin import CLIPlugin
import typer

class MyPlugin(CLIPlugin):
    """Example plugin for k-bert CLI."""
    
    def get_commands(self):
        """Return commands provided by this plugin."""
        
        @typer.command()
        def hello(name: str = typer.Argument(..., help="Name to greet")):
            """Say hello to someone."""
            typer.echo(f"Hello, {name}! Greetings from MyPlugin.")
        
        @typer.command()
        def analyze(
            data_file: str = typer.Argument(..., help="Data file to analyze"),
            output: str = typer.Option("output.json", help="Output file")
        ):
            """Analyze data file with custom logic."""
            # Your analysis logic here
            typer.echo(f"Analyzing {data_file}, saving to {output}")
            return {"status": "completed", "output": output}
        
        return {
            "hello": hello,
            "analyze": analyze,
        }
```

### 2. Directory Structure

For more complex plugins, use a directory structure:

```
my_plugin/
├── __init__.py          # Plugin entry point
├── commands/            # Command modules
│   ├── __init__.py
│   ├── data_commands.py
│   └── model_commands.py
├── middleware/          # Custom middleware
│   ├── __init__.py
│   └── my_middleware.py
├── hooks/              # Custom hooks
│   ├── __init__.py
│   └── my_hooks.py
└── utils/              # Utility modules
    ├── __init__.py
    └── helpers.py
```

## Adding Middleware

Plugins can provide custom middleware for cross-cutting concerns:

```python
from cli.middleware.base import Middleware, MiddlewareResult
from cli.plugins.cli_plugin import CLIPlugin

class CustomLoggingMiddleware(Middleware):
    """Custom logging middleware for plugin commands."""
    
    async def process(self, context, next_handler):
        # Pre-processing
        print(f"[PLUGIN] Executing: {context.command_name}")
        
        try:
            result = next_handler(context)
            print(f"[PLUGIN] Completed: {context.command_name}")
            return result
        except Exception as e:
            print(f"[PLUGIN] Failed: {context.command_name} - {e}")
            raise

class MyPlugin(CLIPlugin):
    def get_middleware(self):
        return [CustomLoggingMiddleware(name="PluginLogger")]
```

## Adding Command Hooks

Create hooks that execute at specific phases of command execution:

```python
from cli.pipeline.base import CommandHook, HookPhase
from cli.plugins.cli_plugin import CLIPlugin

class DataValidationHook(CommandHook):
    """Hook to validate data files before command execution."""
    
    def __init__(self):
        super().__init__(
            name="DataValidationHook",
            phases={HookPhase.PRE_EXECUTE},
            priority=10
        )
    
    async def execute(self, phase, context):
        # Check if data file argument exists
        if "data_file" in context.kwargs:
            data_file = context.kwargs["data_file"]
            if not Path(data_file).exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")
            
            # Add file size to context
            file_size = Path(data_file).stat().st_size
            context.set("data_file_size", file_size)

class MyPlugin(CLIPlugin):
    def get_hooks(self):
        return [DataValidationHook()]
```

## Creating Sub-Applications

For complex functionality, create entire sub-applications:

```python
import typer
from cli.plugins.cli_plugin import CLIPlugin

# Create sub-application
data_app = typer.Typer(name="data", help="Data management commands")

@data_app.command()
def import_data(source: str):
    """Import data from source."""
    typer.echo(f"Importing data from {source}")

@data_app.command()  
def export_data(destination: str):
    """Export data to destination."""
    typer.echo(f"Exporting data to {destination}")

class MyPlugin(CLIPlugin):
    def get_app_extensions(self):
        return {"data": data_app}
```

## Plugin Configuration

Plugins can define their own configuration schemas:

```python
from pydantic import BaseModel
from cli.plugins.cli_plugin import CLIPlugin

class PluginConfig(BaseModel):
    """Configuration for MyPlugin."""
    api_key: str
    endpoint: str = "https://api.example.com"
    timeout: int = 30

class MyPlugin(CLIPlugin):
    def __init__(self):
        self.config = None
    
    def initialize(self, config_data: dict):
        """Initialize plugin with configuration."""
        self.config = PluginConfig(**config_data.get("my_plugin", {}))
    
    def get_commands(self):
        @typer.command()
        def api_call():
            """Make API call using plugin config."""
            if not self.config:
                raise ValueError("Plugin not configured")
            
            # Use self.config.api_key, self.config.endpoint, etc.
            return f"API call to {self.config.endpoint}"
        
        return {"api_call": api_call}
```

## Dependency Injection

Plugins can utilize the CLI's dependency injection system:

```python
from cli.factory import injectable
from cli.plugins.cli_plugin import CLIPlugin

@injectable
class DataProcessor:
    """Injectable service for data processing."""
    
    def process(self, data):
        return f"Processed: {data}"

class MyPlugin(CLIPlugin):
    def get_commands(self):
        @typer.command()
        def process_data(
            data: str,
            processor: DataProcessor = typer.Depends()  # Will be injected
        ):
            """Process data using injected service."""
            result = processor.process(data)
            typer.echo(result)
            return result
        
        return {"process": process_data}
    
    def register_services(self, factory):
        """Register plugin services with DI container."""
        factory.register_service(DataProcessor)
```

## Plugin Testing

Create comprehensive tests for your plugins:

```python
# test_my_plugin.py
import pytest
from cli.testing import CLIRunner, assert_success
from cli.plugins.cli_plugin import CLIPluginLoader
from my_plugin import MyPlugin

class TestMyPlugin:
    def test_plugin_commands(self):
        """Test plugin command registration."""
        plugin = MyPlugin()
        commands = plugin.get_commands()
        
        assert "hello" in commands
        assert "analyze" in commands
    
    def test_hello_command(self):
        """Test hello command execution."""
        plugin = MyPlugin()
        commands = plugin.get_commands()
        
        result = commands["hello"]("Alice")
        assert "Hello, Alice!" in result
    
    def test_plugin_loading(self):
        """Test plugin loading through loader."""
        loader = CLIPluginLoader()
        plugin = MyPlugin()
        
        loader.plugins["my_plugin"] = plugin
        loader._register_plugin_components(plugin, "my_plugin")
        
        assert "my_plugin:hello" in loader.commands
        
        # Test command execution
        cmd = loader.get_command("my_plugin:hello")
        assert cmd is not None
    
    def test_middleware_integration(self):
        """Test plugin middleware."""
        plugin = MyPlugin()
        middleware_list = plugin.get_middleware()
        
        assert len(middleware_list) > 0
        assert middleware_list[0].name == "PluginLogger"
```

## Plugin Distribution

### Package Structure

Create a proper Python package for distribution:

```
my-bert-plugin/
├── setup.py
├── README.md
├── my_bert_plugin/
│   ├── __init__.py
│   ├── plugin.py
│   └── commands/
└── tests/
    └── test_plugin.py
```

### Setup Configuration

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-bert-plugin",
    version="0.1.0",
    description="Custom plugin for k-bert CLI",
    packages=find_packages(),
    install_requires=[
        "k-bert>=1.0.0",
        "typer>=0.9.0",
    ],
    entry_points={
        "k_bert_plugins": [
            "my_plugin = my_bert_plugin.plugin:MyPlugin",
        ],
    },
    python_requires=">=3.8",
)
```

## Best Practices

### 1. Command Design

- Use clear, descriptive command names
- Provide comprehensive help text
- Use type hints for all parameters
- Handle errors gracefully with meaningful messages

### 2. Configuration

- Define clear configuration schemas
- Provide sensible defaults
- Validate configuration early
- Document all configuration options

### 3. Error Handling

- Use specific exception types
- Provide actionable error messages
- Log errors appropriately
- Handle edge cases gracefully

### 4. Testing

- Test all command paths
- Mock external dependencies
- Test error conditions
- Use the CLI testing framework

### 5. Documentation

- Document all commands and options
- Provide usage examples
- Document configuration requirements
- Include troubleshooting information

## Plugin Validation

The CLI system validates plugins automatically:

```python
def validate_plugin():
    """Example of what the system checks."""
    
    # 1. Commands are callable
    for name, func in plugin.get_commands().items():
        assert callable(func), f"Command {name} must be callable"
    
    # 2. Middleware extends base class
    for middleware in plugin.get_middleware():
        assert isinstance(middleware, Middleware)
    
    # 3. Hooks extend base class
    for hook in plugin.get_hooks():
        assert isinstance(hook, CommandHook)
    
    # 4. Apps are Typer instances
    for name, app in plugin.get_app_extensions().items():
        assert isinstance(app, typer.Typer)
```

## Advanced Features

### 1. Dynamic Command Registration

```python
class DynamicPlugin(CLIPlugin):
    def __init__(self):
        self._dynamic_commands = {}
    
    def register_command(self, name, func):
        """Dynamically register a command."""
        self._dynamic_commands[name] = func
    
    def get_commands(self):
        return self._dynamic_commands
```

### 2. Plugin Communication

```python
class PluginRegistry:
    """Registry for plugin communication."""
    
    def __init__(self):
        self.plugins = {}
        self.events = {}
    
    def emit_event(self, event_name, data):
        """Emit event to all listening plugins."""
        if event_name in self.events:
            for callback in self.events[event_name]:
                callback(data)
```

### 3. Resource Management

```python
class ResourcePlugin(CLIPlugin):
    def __init__(self):
        self.resources = []
    
    def acquire_resource(self):
        """Acquire a resource."""
        resource = SomeResource()
        self.resources.append(resource)
        return resource
    
    def cleanup(self):
        """Clean up plugin resources."""
        for resource in self.resources:
            resource.close()
        self.resources.clear()
```

This guide provides a comprehensive foundation for developing plugins for the k-bert CLI system. Start with simple command plugins and gradually add more advanced features as needed.