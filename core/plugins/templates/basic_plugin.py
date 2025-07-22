"""Template for creating a basic k-bert plugin.

This template demonstrates:
- Plugin metadata definition
- Lifecycle implementation
- Configuration handling
- Error handling
"""

from typing import Any, Dict, Optional

from loguru import logger

from core.plugins import PluginBase, PluginContext, PluginError, PluginMetadata


class MyPlugin(PluginBase):
    """Example plugin implementation.
    
    This plugin demonstrates the basic structure and lifecycle
    of a k-bert plugin.
    """
    
    # Plugin metadata (optional - can use _create_metadata instead)
    NAME = "my_plugin"
    VERSION = "1.0.0"
    DESCRIPTION = "An example k-bert plugin"
    AUTHOR = "Your Name"
    EMAIL = "your.email@example.com"
    TAGS = ["example", "template"]
    CATEGORY = "example"
    
    # Dependencies and capabilities
    DEPENDS_ON = []  # List of required plugins
    CONFLICTS_WITH = []  # List of conflicting plugins
    PROVIDES = ["example_capability"]  # Capabilities this plugin provides
    CONSUMES = []  # Capabilities this plugin needs
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        super().__init__(config)
        
        # Initialize plugin-specific attributes
        self.resource = None
        self.is_configured = False
    
    def _validate(self, context: PluginContext) -> None:
        """Validate plugin configuration and environment.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If validation fails
        """
        # Validate configuration
        required_keys = ["setting1", "setting2"]
        for key in required_keys:
            if key not in self.config:
                raise PluginError(
                    f"Missing required configuration: {key}",
                    plugin_name=self.NAME
                )
        
        # Validate environment
        try:
            # Check for required services in DI container
            # service = context.resolve(RequiredService)
            pass
        except KeyError as e:
            raise PluginError(
                f"Required service not available: {e}",
                plugin_name=self.NAME
            )
        
        logger.debug(f"{self.NAME}: Validation successful")
    
    def _initialize(self, context: PluginContext) -> None:
        """Initialize plugin resources.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If initialization fails
        """
        try:
            # Initialize resources
            self.resource = self._create_resource(context)
            
            # Configure plugin
            self._configure(context)
            
            logger.info(f"{self.NAME}: Initialized successfully")
            
        except Exception as e:
            raise PluginError(
                f"Failed to initialize: {e}",
                plugin_name=self.NAME,
                cause=e
            )
    
    def _start(self, context: PluginContext) -> None:
        """Start plugin operations.
        
        Args:
            context: Plugin context
            
        Raises:
            PluginError: If startup fails
        """
        if not self.is_configured:
            raise PluginError(
                "Plugin not properly configured",
                plugin_name=self.NAME
            )
        
        try:
            # Start any background operations
            self._start_operations(context)
            
            logger.info(f"{self.NAME}: Started successfully")
            
        except Exception as e:
            raise PluginError(
                f"Failed to start: {e}",
                plugin_name=self.NAME,
                cause=e
            )
    
    def _stop(self, context: PluginContext) -> None:
        """Stop plugin operations.
        
        Args:
            context: Plugin context
        """
        try:
            # Stop any background operations
            self._stop_operations(context)
            
            logger.info(f"{self.NAME}: Stopped successfully")
            
        except Exception as e:
            logger.error(f"{self.NAME}: Error during stop: {e}")
    
    def _cleanup(self, context: PluginContext) -> None:
        """Clean up plugin resources.
        
        Args:
            context: Plugin context
        """
        try:
            # Clean up resources
            if self.resource:
                self._cleanup_resource(self.resource)
                self.resource = None
            
            self.is_configured = False
            
            logger.info(f"{self.NAME}: Cleaned up successfully")
            
        except Exception as e:
            logger.error(f"{self.NAME}: Error during cleanup: {e}")
    
    # Plugin-specific methods
    
    def _create_resource(self, context: PluginContext) -> Any:
        """Create the main resource for this plugin.
        
        Args:
            context: Plugin context
            
        Returns:
            Created resource
        """
        # Example: Create a connection, load a model, etc.
        return "resource_placeholder"
    
    def _configure(self, context: PluginContext) -> None:
        """Configure the plugin.
        
        Args:
            context: Plugin context
        """
        # Apply configuration
        setting1 = self.config.get("setting1", "default1")
        setting2 = self.config.get("setting2", "default2")
        
        # Configure resource
        # self.resource.configure(setting1=setting1, setting2=setting2)
        
        self.is_configured = True
    
    def _start_operations(self, context: PluginContext) -> None:
        """Start plugin operations.
        
        Args:
            context: Plugin context
        """
        # Start any background tasks, listeners, etc.
        pass
    
    def _stop_operations(self, context: PluginContext) -> None:
        """Stop plugin operations.
        
        Args:
            context: Plugin context
        """
        # Stop any background tasks, listeners, etc.
        pass
    
    def _cleanup_resource(self, resource: Any) -> None:
        """Clean up a resource.
        
        Args:
            resource: Resource to clean up
        """
        # Close connections, release memory, etc.
        pass
    
    # Public API methods (specific to your plugin's functionality)
    
    def process(self, data: Any) -> Any:
        """Process data using this plugin.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
            
        Raises:
            PluginError: If processing fails
        """
        if self.state != self.state.RUNNING:
            raise PluginError(
                f"Plugin must be running to process data (current state: {self.state})",
                plugin_name=self.NAME
            )
        
        try:
            # Process the data
            result = self._do_processing(data)
            return result
            
        except Exception as e:
            raise PluginError(
                f"Processing failed: {e}",
                plugin_name=self.NAME,
                cause=e
            )
    
    def _do_processing(self, data: Any) -> Any:
        """Internal processing logic.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        # Implement your processing logic here
        return data


# Example usage
if __name__ == "__main__":
    # Create plugin with configuration
    config = {
        "setting1": "value1",
        "setting2": "value2",
    }
    plugin = MyPlugin(config=config)
    
    # Create context (normally done by the plugin system)
    from core.di import get_container
    context = PluginContext(container=get_container())
    
    # Run through lifecycle
    try:
        plugin.validate(context)
        plugin.initialize(context)
        plugin.start(context)
        
        # Use plugin
        result = plugin.process("test_data")
        print(f"Result: {result}")
        
        # Shutdown
        plugin.stop(context)
        plugin.cleanup(context)
        
    except PluginError as e:
        print(f"Plugin error: {e}")
        if e.cause:
            print(f"Caused by: {e.cause}")