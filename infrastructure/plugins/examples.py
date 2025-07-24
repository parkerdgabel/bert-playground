"""Examples demonstrating the k-bert unified plugin system.

This module provides practical examples of how to use the plugin system
for various scenarios.
"""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from infrastructure.di import Container, get_container
from infrastructure.plugins import (
    PluginBase,
    PluginContext,
    PluginError,
    PluginRegistry,
    load_plugins,
    setup_plugin_system,
)


# Example 1: Simple Plugin
class ExamplePlugin(PluginBase):
    """A simple example plugin."""
    
    NAME = "example_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "An example plugin demonstrating basic functionality"
    AUTHOR = "k-bert Team"
    CATEGORY = "example"
    TAGS = ["example", "demo"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.counter = 0
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize the plugin."""
        initial_value = self.config.get("initial_counter", 0)
        self.counter = initial_value
        logger.info(f"Example plugin initialized with counter = {self.counter}")
    
    def _start(self, context: PluginContext) -> None:
        """Start the plugin."""
        logger.info("Example plugin started")
    
    def increment(self) -> int:
        """Increment the counter and return new value."""
        self.counter += 1
        return self.counter
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status."""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "counter": self.counter,
            "state": self.state.name,
        }


# Example 2: Plugin with Dependencies
class DatabasePlugin(PluginBase):
    """Plugin that provides database functionality."""
    
    NAME = "database_plugin" 
    VERSION = "1.0.0"
    DESCRIPTION = "Provides database connectivity"
    CATEGORY = "data"
    PROVIDES = ["database_connection", "query_executor"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.connection = None
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize database connection."""
        db_url = self.config.get("database_url", "sqlite:///:memory:")
        # Mock database connection
        self.connection = f"Connection to {db_url}"
        
        # Register service with DI container
        context.container.register(
            service_type=str,  # Mock database service
            implementation=self.connection,
            instance=True
        )
        
        logger.info(f"Database plugin connected to {db_url}")
    
    def query(self, sql: str) -> str:
        """Execute a query."""
        return f"Result of: {sql}"


class AnalyticsPlugin(PluginBase):
    """Plugin that depends on database plugin."""
    
    NAME = "analytics_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Provides analytics functionality"
    CATEGORY = "analytics"
    DEPENDS_ON = ["database_plugin"]
    CONSUMES = ["database_connection"]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.db_service = None
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize analytics with database dependency."""
        # Resolve database service from DI container
        try:
            self.db_service = context.resolve(str)  # Mock database service
            logger.info(f"Analytics plugin initialized with database: {self.db_service}")
        except KeyError:
            raise PluginError(
                "Database service not available",
                plugin_name=self.NAME
            )
    
    def generate_report(self) -> str:
        """Generate an analytics report."""
        if not self.db_service:
            raise RuntimeError("Database not initialized")
        
        # Mock analytics
        return f"Analytics report using {self.db_service}"


def example_basic_plugin_usage():
    """Demonstrate basic plugin usage."""
    logger.info("=== Basic Plugin Usage Example ===")
    
    # Create and configure plugin
    config = {"initial_counter": 5}
    plugin = ExamplePlugin(config=config)
    
    # Create context
    container = Container()
    context = PluginContext(container=container)
    
    # Run through lifecycle
    plugin.validate(context)
    plugin.initialize(context)
    plugin.start(context)
    
    # Use plugin
    logger.info(f"Plugin status: {plugin.get_status()}")
    logger.info(f"Increment result: {plugin.increment()}")
    logger.info(f"Increment result: {plugin.increment()}")
    logger.info(f"Final status: {plugin.get_status()}")
    
    # Shutdown
    plugin.stop(context)
    plugin.cleanup(context)


def example_plugin_registry():
    """Demonstrate plugin registry usage."""
    logger.info("=== Plugin Registry Example ===")
    
    # Create registry
    registry = PluginRegistry()
    
    # Create plugins
    example_plugin = ExamplePlugin({"initial_counter": 10})
    db_plugin = DatabasePlugin({"database_url": "sqlite:///example.db"})
    
    # Register plugins with proper initialization
    registry.register(example_plugin, initialize=False)
    registry.register(db_plugin, initialize=False)
    
    # Initialize plugins explicitly
    context1 = PluginContext(container=registry.container)
    context2 = PluginContext(container=registry.container)
    
    example_plugin.validate(context1)
    example_plugin.initialize(context1)
    
    db_plugin.validate(context2)
    db_plugin.initialize(context2)
    
    # List registered plugins
    plugins = registry.list_plugins()
    logger.info(f"Registered plugins: {list(plugins.keys())}")
    
    # Get plugins by category
    data_plugins = registry.get_by_category("data")
    logger.info(f"Data plugins: {[p.metadata.name for p in data_plugins]}")
    
    # Start all plugins
    registry.start_all()
    
    # Use plugins
    example = registry.get("example_plugin")
    if example:
        logger.info(f"Example plugin counter: {example.increment()}")
    
    db = registry.get("database_plugin")
    if db:
        result = db.query("SELECT * FROM users")
        logger.info(f"Query result: {result}")
    
    # Stop all plugins
    registry.stop_all()


def example_plugin_dependencies():
    """Demonstrate plugin dependencies."""
    logger.info("=== Plugin Dependencies Example ===")
    
    registry = PluginRegistry()
    
    # Create plugins with dependency relationship
    db_plugin = DatabasePlugin({
        "database_url": "postgresql://localhost/analytics"
    })
    analytics_plugin = AnalyticsPlugin({
        "report_format": "json"
    })
    
    # Register database plugin first (dependency)
    registry.register(db_plugin, initialize=False)
    
    # Register analytics plugin (depends on database)
    registry.register(analytics_plugin, initialize=False)
    
    # Initialize plugins in dependency order
    db_context = PluginContext(container=registry.container)
    db_plugin.validate(db_context)
    db_plugin.initialize(db_context)
    
    analytics_context = PluginContext(container=registry.container)
    analytics_plugin.validate(analytics_context)
    analytics_plugin.initialize(analytics_context)
    
    # Start plugins
    registry.start_all()
    
    # Use analytics plugin
    analytics = registry.get("analytics_plugin")
    if analytics:
        report = analytics.generate_report()
        logger.info(f"Analytics report: {report}")
    
    registry.stop_all()


def example_project_based_loading():
    """Demonstrate loading plugins from a project structure."""
    logger.info("=== Project-Based Plugin Loading Example ===")
    
    # Create temporary project structure
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Create plugin directory
        plugin_dir = project_root / "src" / "plugins"
        plugin_dir.mkdir(parents=True)
        
        # Create a plugin file
        plugin_code = '''
from infrastructure.plugins import PluginBase

class ProjectPlugin(PluginBase):
    """Plugin loaded from project directory."""
    
    NAME = "project_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Plugin loaded from project"
    CATEGORY = "project"
    
    def _initialize(self, context):
        self.data = "Project plugin initialized"
        
    def get_data(self):
        return self.data
'''
        
        # Write plugin file
        (plugin_dir / "project_plugin.py").write_text(plugin_code)
        
        # Create configuration
        config_content = '''
plugins:
  auto_initialize: true
  configs:
    project_plugin:
      custom_setting: "from_config"
'''
        (project_root / "k-bert.yaml").write_text(config_content)
        
        # Load plugins from project
        try:
            plugins = load_plugins(project_root=project_root)
            logger.info(f"Loaded {len(plugins)} plugins from project")
            
            for name, plugin in plugins.items():
                logger.info(f"- {name}: {plugin.metadata.description}")
                
                # Initialize and use plugin if it has expected methods
                if hasattr(plugin, 'get_data'):
                    # Initialize plugin first
                    if plugin.state.name == 'LOADED':
                        context = PluginContext(container=Container())
                        plugin.validate(context)
                        plugin.initialize(context)
                    logger.info(f"  Data: {plugin.get_data()}")
                    
        except Exception as e:
            logger.error(f"Failed to load project plugins: {e}")


def example_unified_system_setup():
    """Demonstrate setting up the unified plugin system."""
    logger.info("=== Unified System Setup Example ===")
    
    # Create temporary project
    with tempfile.TemporaryDirectory() as temp_dir:
        project_root = Path(temp_dir)
        
        # Set up unified plugin system
        try:
            registry = setup_plugin_system(
                project_root=project_root,
                migrate_old=False,  # No old plugins in this example
            )
            
            logger.info("Unified plugin system set up successfully")
            
            # Check what was loaded
            plugins = registry.list_plugins()
            categories = registry.list_categories()
            
            logger.info(f"Loaded plugins: {list(plugins.keys())}")
            logger.info(f"Categories: {list(categories.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to set up plugin system: {e}")


if __name__ == "__main__":
    """Run all examples."""
    
    logger.info("Running k-bert Plugin System Examples")
    logger.info("=" * 50)
    
    try:
        example_basic_plugin_usage()
        logger.info("")
        
        example_plugin_registry()
        logger.info("")
        
        example_plugin_dependencies()
        logger.info("")
        
        example_project_based_loading()
        logger.info("")
        
        example_unified_system_setup()
        
        logger.info("=" * 50)
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()