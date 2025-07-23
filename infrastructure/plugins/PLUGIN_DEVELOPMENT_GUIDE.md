# k-bert Plugin Development Guide

This guide covers how to develop plugins for the k-bert unified plugin system.

## Table of Contents

1. [Overview](#overview)
2. [Plugin Architecture](#plugin-architecture)
3. [Plugin Types](#plugin-types)
4. [Creating Your First Plugin](#creating-your-first-plugin)
5. [Plugin Lifecycle](#plugin-lifecycle)
6. [Configuration](#configuration)
7. [Dependency Injection](#dependency-injection)
8. [Testing Plugins](#testing-plugins)
9. [Packaging and Distribution](#packaging-and-distribution)
10. [Best Practices](#best-practices)
11. [Examples](#examples)

## Overview

The k-bert plugin system allows you to extend k-bert with custom functionality including:

- **Model Architectures**: Custom BERT variants and neural networks
- **Task Heads**: Custom output layers for specific tasks
- **Data Processing**: Custom data loaders and augmentation strategies
- **Training Components**: Custom metrics, callbacks, and training strategies
- **CLI Commands**: Custom command-line interfaces

## Plugin Architecture

### Core Components

```
infrastructure/plugins/
├── base.py           # Base classes and protocols
├── loader.py         # Plugin discovery and loading
├── registry.py       # Plugin management
├── validators.py     # Plugin validation
├── config.py         # Configuration support
├── integration.py    # Legacy system integration
└── templates/        # Development templates
```

### Plugin Lifecycle

Every plugin follows this lifecycle:

1. **Discover** - Found by the plugin system
2. **Validate** - Checked for correctness
3. **Load** - Class instantiated
4. **Initialize** - Plugin setup with dependencies
5. **Start** - Plugin begins operation
6. **Stop** - Plugin ends operation
7. **Cleanup** - Resources released

## Plugin Types

### 1. Model Plugins

Provide custom model architectures:

```python
from infrastructure.plugins import PluginBase
from infrastructure.protocols.plugins import ModelPlugin

class CustomModelPlugin(PluginBase, ModelPlugin):
    NAME = "custom_model"
    CATEGORY = "model"
    
    def build_model(self, config):
        # Return custom model instance
        return CustomModel(**config)
    
    def get_default_config(self):
        return {"hidden_size": 768, "num_layers": 12}
```

### 2. Head Plugins

Provide custom task-specific output layers:

```python
from infrastructure.plugins import PluginBase
from infrastructure.protocols.plugins import HeadPlugin

class CustomHeadPlugin(PluginBase, HeadPlugin):
    NAME = "custom_head"
    CATEGORY = "head"
    
    def __call__(self, hidden_states, **kwargs):
        # Forward pass logic
        return {"logits": self.classifier(hidden_states)}
    
    def compute_loss(self, logits, labels, **kwargs):
        return cross_entropy_loss(logits, labels)
```

### 3. Data Processing Plugins

Provide custom data handling:

```python
from infrastructure.plugins import PluginBase
from infrastructure.protocols.plugins import AugmenterPlugin

class CustomAugmenterPlugin(PluginBase, AugmenterPlugin):
    NAME = "custom_augmenter"
    CATEGORY = "augmenter"
    
    def augment(self, data, training=True, **kwargs):
        if training:
            # Apply augmentation
            return self.transform(data)
        return data
```

## Creating Your First Plugin

### Step 1: Set Up Project Structure

```
my_project/
├── k-bert.yaml          # Project configuration
├── src/
│   └── plugins/         # Plugin directory
│       └── my_plugin.py # Your plugin
└── tests/
    └── test_my_plugin.py
```

### Step 2: Create Plugin Class

```python
# src/plugins/my_plugin.py
from typing import Any, Dict, Optional
from infrastructure.plugins import PluginBase, PluginContext

class MyPlugin(PluginBase):
    """My custom k-bert plugin."""
    
    NAME = "my_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "My custom plugin for k-bert"
    AUTHOR = "Your Name"
    CATEGORY = "custom"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.resource = None
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize plugin resources."""
        # Setup your plugin here
        setting = self.config.get("my_setting", "default")
        self.resource = self.create_resource(setting)
    
    def _start(self, context: PluginContext) -> None:
        """Start plugin operations."""
        # Start any background operations
        pass
    
    def _stop(self, context: PluginContext) -> None:
        """Stop plugin operations."""
        # Stop background operations
        pass
    
    def _cleanup(self, context: PluginContext) -> None:
        """Clean up resources."""
        if self.resource:
            self.resource.close()
            self.resource = None
    
    # Plugin-specific methods
    def process(self, data):
        """Process data using this plugin."""
        return self.resource.process(data)
```

### Step 3: Configure Plugin

```yaml
# k-bert.yaml
plugins:
  auto_initialize: true
  configs:
    my_plugin:
      my_setting: "custom_value"
```

### Step 4: Test Plugin

```python
# tests/test_my_plugin.py
import pytest
from infrastructure.plugins import PluginContext
from infrastructure.di import Container
from src.plugins.my_plugin import MyPlugin

def test_my_plugin():
    config = {"my_setting": "test_value"}
    plugin = MyPlugin(config=config)
    
    context = PluginContext(container=Container())
    
    # Test lifecycle
    plugin.validate(context)
    plugin.initialize(context)
    plugin.start(context)
    
    # Test functionality
    result = plugin.process("test_data")
    assert result is not None
    
    # Cleanup
    plugin.stop(context)
    plugin.cleanup(context)
```

## Plugin Lifecycle

### Validation

Override `_validate()` to check configuration and environment:

```python
def _validate(self, context: PluginContext) -> None:
    if "required_setting" not in self.config:
        raise PluginError("Missing required_setting")
    
    # Check dependencies
    try:
        service = context.resolve(RequiredService)
    except KeyError:
        raise PluginError("Required service not available")
```

### Initialization

Override `_initialize()` to set up resources:

```python
def _initialize(self, context: PluginContext) -> None:
    # Load model, open connections, etc.
    self.model = self.load_model(self.config["model_path"])
    
    # Register services with DI container
    context.container.register(MyService, self.model)
```

### Error Handling

Use `PluginError` for plugin-specific errors:

```python
from infrastructure.plugins import PluginError

def risky_operation(self):
    try:
        # Some operation that might fail
        result = dangerous_function()
    except Exception as e:
        raise PluginError(
            f"Operation failed: {e}",
            plugin_name=self.NAME,
            cause=e
        )
```

## Configuration

### Plugin-Specific Configuration

```yaml
# k-bert.yaml
plugins:
  configs:
    my_plugin:
      setting1: "value1"
      setting2: 42
      nested:
        subsetting: true
```

Access in plugin:
```python
def _initialize(self, context: PluginContext):
    setting1 = self.config.get("setting1", "default")
    nested = self.config.get("nested", {})
    subsetting = nested.get("subsetting", False)
```

### Global Configuration

```yaml
# k-bert.yaml
plugins:
  auto_initialize: true
  auto_start: false
  validate_on_load: true
  categories: ["model", "head"]  # Only load these types
```

### Configuration Schema

Define configuration schema for validation:

```python
from pydantic import BaseModel, Field

class MyPluginConfig(BaseModel):
    setting1: str = Field(..., description="Required setting")
    setting2: int = Field(42, description="Optional setting")
    
    class Config:
        extra = "allow"

class MyPlugin(PluginBase):
    def _validate(self, context: PluginContext):
        # Validate configuration
        MyPluginConfig(**self.config)
```

## Dependency Injection

### Using Services

```python
def _initialize(self, context: PluginContext):
    # Resolve services from DI container
    logger = context.resolve(Logger)
    config_service = context.resolve(ConfigService)
    
    # Use services
    logger.info(f"Initializing {self.NAME}")
```

### Providing Services

```python
def _initialize(self, context: PluginContext):
    # Create service
    my_service = MyService(self.config)
    
    # Register with container
    context.container.register(
        service_type=MyService,
        implementation=my_service,
        instance=True
    )
```

### Auto-wiring

```python
from infrastructure.di import get_container

class MyService:
    def __init__(self, logger: Logger, config: ConfigService):
        self.logger = logger
        self.config = config

# Auto-wire dependencies
container = get_container()
service = container.auto_wire(MyService)
```

## Testing Plugins

### Unit Tests

```python
import pytest
from unittest.mock import Mock
from infrastructure.plugins import PluginContext
from infrastructure.di import Container

def test_plugin_initialization():
    plugin = MyPlugin(config={"setting": "value"})
    context = PluginContext(container=Container())
    
    plugin.validate(context)
    plugin.initialize(context)
    
    assert plugin.state == PluginState.INITIALIZED
    assert plugin.resource is not None
```

### Integration Tests

```python
def test_plugin_with_registry():
    from infrastructure.plugins import PluginRegistry
    
    registry = PluginRegistry()
    plugin = MyPlugin()
    
    registry.register(plugin)
    
    retrieved = registry.get("my_plugin")
    assert retrieved is plugin
```

### Mock Dependencies

```python
def test_plugin_with_mocked_service():
    mock_service = Mock()
    container = Container()
    container.register(RequiredService, mock_service, instance=True)
    
    context = PluginContext(container=container)
    plugin = MyPlugin()
    
    plugin.initialize(context)
    
    # Verify mock was used
    mock_service.some_method.assert_called()
```

## Packaging and Distribution

### Entry Points

Define entry points in `pyproject.toml`:

```toml
[project.entry-points."k_bert.plugins"]
my_plugin = "my_package.plugins:MyPlugin"
custom_head = "my_package.heads:CustomHead"
```

### Package Structure

```
my_plugin_package/
├── pyproject.toml
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── plugins.py
│       └── heads.py
└── tests/
```

### Installation

```bash
# Install in development mode
pip install -e .

# Or build and install
python -m build
pip install dist/my_plugin_package-*.whl
```

## Best Practices

### 1. Use Descriptive Names

```python
class BertForSequenceClassificationHead(PluginBase, HeadPlugin):
    NAME = "bert_sequence_classification"  # Clear, specific name
```

### 2. Provide Good Metadata

```python
NAME = "custom_augmenter"
VERSION = "1.2.0"
DESCRIPTION = "Text augmentation with synonym replacement and noise injection"
AUTHOR = "Your Name <email@domain.com>"
TAGS = ["augmentation", "text", "nlp"]
REQUIREMENTS = ["nltk>=3.6", "transformers>=4.20"]
```

### 3. Handle Errors Gracefully

```python
def _initialize(self, context: PluginContext):
    try:
        self.model = self.load_model()
    except FileNotFoundError as e:
        raise PluginError(
            f"Model file not found: {e}",
            plugin_name=self.NAME,
            cause=e
        )
    except Exception as e:
        raise PluginError(
            f"Failed to load model: {e}",
            plugin_name=self.NAME,
            cause=e
        )
```

### 4. Use Type Hints

```python
from typing import Dict, Any, Optional
import mlx.core as mx

def process(self, data: Dict[str, Any]) -> mx.array:
    """Process input data and return MLX array."""
    return self.model(data["input_ids"])
```

### 5. Document Your Plugin

```python
class MyPlugin(PluginBase):
    """Custom plugin for specialized text processing.
    
    This plugin provides advanced text processing capabilities including:
    - Custom tokenization
    - Domain-specific preprocessing
    - Advanced augmentation strategies
    
    Configuration:
        tokenizer_path (str): Path to custom tokenizer
        augment_prob (float): Probability of applying augmentation (0.0-1.0)
        
    Example:
        >>> plugin = MyPlugin({
        ...     "tokenizer_path": "/path/to/tokenizer",
        ...     "augment_prob": 0.3
        ... })
        >>> result = plugin.process({"text": "Hello world"})
    """
```

## Examples

### Complete Model Plugin

```python
from typing import Any, Dict, Optional
import mlx.core as mx
import mlx.nn as nn
from infrastructure.plugins import PluginBase, PluginContext
from infrastructure.protocols.plugins import ModelPlugin

class CustomTransformerPlugin(PluginBase, ModelPlugin):
    """Custom transformer model plugin."""
    
    NAME = "custom_transformer"
    VERSION = "1.0.0"
    DESCRIPTION = "Custom transformer architecture"
    CATEGORY = "model"
    PROVIDES = ["custom_transformer_v1"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_class = None
    
    def _initialize(self, context: PluginContext):
        """Initialize model class."""
        self.model_class = self._create_model_class()
    
    def build_model(self, config: Dict[str, Any]) -> Any:
        """Build model instance."""
        merged_config = {**self.get_default_config(), **config}
        return self.model_class(**merged_config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1,
        }
    
    def _create_model_class(self):
        """Create custom model class."""
        
        class CustomTransformer(nn.Module):
            def __init__(self, hidden_size=768, num_layers=12, **kwargs):
                super().__init__()
                self.hidden_size = hidden_size
                self.layers = [
                    self._create_layer(hidden_size)
                    for _ in range(num_layers)
                ]
            
            def _create_layer(self, hidden_size):
                return nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=12,
                )
            
            def __call__(self, input_ids: mx.array) -> Dict[str, mx.array]:
                # Custom forward pass
                x = input_ids
                for layer in self.layers:
                    x = layer(x)
                
                return {
                    "last_hidden_state": x,
                    "pooled_output": x[:, 0],  # [CLS] token
                }
        
        return CustomTransformer
```

### Complete Head Plugin

```python
from typing import Dict, List, Optional
import mlx.core as mx
import mlx.nn as nn
from infrastructure.plugins import PluginBase, PluginContext
from infrastructure.protocols.plugins import HeadPlugin

class MultiTaskHeadPlugin(PluginBase, HeadPlugin):
    """Multi-task classification head."""
    
    NAME = "multi_task_head"
    VERSION = "1.0.0"
    DESCRIPTION = "Head for multi-task learning"
    CATEGORY = "head"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.heads = {}
        self.task_configs = self.config.get("tasks", {})
    
    def _initialize(self, context: PluginContext):
        """Initialize task heads."""
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = self._create_task_head(task_config)
    
    def _create_task_head(self, task_config: Dict[str, Any]) -> nn.Module:
        """Create head for specific task."""
        input_size = task_config.get("input_size", 768)
        output_size = task_config.get("output_size", 2)
        
        return nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size),
        )
    
    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Dict[str, mx.array]:
        """Forward pass through all task heads."""
        # Pool hidden states
        pooled = hidden_states[:, 0]  # [CLS] token
        
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[f"{task_name}_logits"] = head(pooled)
        
        return outputs
    
    def compute_loss(
        self,
        logits: mx.array,
        labels: mx.array,
        **kwargs
    ) -> mx.array:
        """Compute multi-task loss."""
        total_loss = 0
        task_weights = kwargs.get("task_weights", {})
        
        for task_name in self.task_configs:
            task_logits_key = f"{task_name}_logits"
            task_labels_key = f"{task_name}_labels"
            
            if task_logits_key in logits and task_labels_key in labels:
                task_loss = nn.losses.cross_entropy(
                    logits[task_logits_key],
                    labels[task_labels_key]
                )
                weight = task_weights.get(task_name, 1.0)
                total_loss += weight * task_loss
        
        return total_loss
    
    def get_output_size(self) -> Dict[str, int]:
        """Get output sizes for all tasks."""
        return {
            task_name: config.get("output_size", 2)
            for task_name, config in self.task_configs.items()
        }
    
    def get_metrics(self) -> List[str]:
        """Get metrics for all tasks."""
        base_metrics = ["accuracy", "f1"]
        metrics = []
        
        for task_name in self.task_configs:
            for metric in base_metrics:
                metrics.append(f"{task_name}_{metric}")
        
        return metrics
```

This guide provides comprehensive coverage of k-bert plugin development. For more examples, see the `infrastructure/plugins/templates/` directory and existing plugins in the k-bert codebase.