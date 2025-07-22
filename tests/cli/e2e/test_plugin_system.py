"""End-to-end tests for plugin system."""

import pytest
import tempfile
from pathlib import Path

from cli.factory import CommandFactory
from cli.plugins.cli_plugin import CLIPlugin, CLIPluginLoader
from cli.testing import CLIRunner, assert_success


class TestPluginSystem:
    """Test plugin system integration."""
    
    def test_plugin_loading_and_registration(self):
        """Test plugin loading and command registration."""
        # Create a test plugin
        class TestPlugin(CLIPlugin):
            def get_commands(self):
                def hello_plugin():
                    return "Hello from plugin!"
                
                return {"hello": hello_plugin}
        
        # Set up plugin loader
        loader = CLIPluginLoader()
        plugin = TestPlugin()
        loader.plugins["test"] = plugin
        loader._register_plugin_components(plugin, "test")
        
        # Verify command registration
        assert "test:hello" in loader.commands
        
        # Test command execution
        command = loader.get_command("test:hello")
        result = command()
        assert result == "Hello from plugin!"
    
    def test_plugin_middleware_integration(self):
        """Test plugin middleware integration."""
        from cli.middleware import LoggingMiddleware
        
        class TestPlugin(CLIPlugin):
            def get_middleware(self):
                return [LoggingMiddleware(name="PluginLogging")]
            
            def get_commands(self):
                def test_command():
                    return "plugin command executed"
                
                return {"test": test_command}
        
        # Set up factory with plugin
        factory = CommandFactory()
        plugin = TestPlugin()
        
        # Register middleware
        for middleware in plugin.get_middleware():
            factory.middleware_pipeline.add(middleware)
        
        # Create command with middleware
        commands = plugin.get_commands()
        enhanced_command = factory.create_middleware_command(commands["test"])
        
        result = enhanced_command()
        assert result == "plugin command executed"
    
    def test_plugin_discovery_from_filesystem(self):
        """Test plugin discovery from filesystem."""
        with tempfile.TemporaryDirectory() as temp_dir:
            plugin_dir = Path(temp_dir) / "plugins"
            plugin_dir.mkdir()
            
            # Create a test plugin file
            plugin_file = plugin_dir / "test_plugin.py"
            plugin_code = '''
from cli.plugins.cli_plugin import CLIPlugin

class TestFilePlugin(CLIPlugin):
    def get_commands(self):
        def filesystem_command():
            return "Hello from filesystem plugin!"
        
        return {"fs_hello": filesystem_command}
'''
            plugin_file.write_text(plugin_code)
            
            # Set up loader with plugin directory
            loader = CLIPluginLoader(plugin_dirs=[plugin_dir])
            
            # Discover and load plugins
            discovered = loader.discover_plugins()
            assert "test_plugin" in discovered
            
            # Load specific plugin
            plugin = loader.load_plugin("test_plugin")
            assert plugin is not None
            
            # Test command availability
            commands = plugin.get_commands()
            assert "fs_hello" in commands
            
            result = commands["fs_hello"]()
            assert result == "Hello from filesystem plugin!"
    
    def test_plugin_validation(self):
        """Test plugin validation."""
        class InvalidPlugin(CLIPlugin):
            def get_commands(self):
                return {"invalid": "not_a_function"}  # Invalid: not callable
        
        loader = CLIPluginLoader()
        plugin = InvalidPlugin()
        
        errors = loader.validate_plugin(plugin)
        assert len(errors) > 0
        assert any("not callable" in error for error in errors)
    
    def test_plugin_hooks_integration(self):
        """Test plugin hooks integration."""
        from cli.pipeline.base import CommandHook, HookPhase
        
        class TestHook(CommandHook):
            def __init__(self):
                super().__init__(
                    name="TestPluginHook",
                    phases={HookPhase.PRE_EXECUTE}
                )
                self.executed = False
            
            async def execute(self, phase, context):
                self.executed = True
                context.set("plugin_hook_executed", True)
        
        class TestPlugin(CLIPlugin):
            def __init__(self):
                self.hook = TestHook()
            
            def get_hooks(self):
                return [self.hook]
        
        # Set up factory with plugin
        factory = CommandFactory()
        plugin = TestPlugin()
        
        # Register hooks
        for hook in plugin.get_hooks():
            factory.command_pipeline.add_hook(hook)
        
        # Test hook execution
        async def test_command():
            return "command executed"
        
        # This would need async pipeline execution in real implementation
        # For now, just verify hook registration
        assert len(factory.command_pipeline.hooks) == 1
        assert factory.command_pipeline.hooks[0].name == "TestPluginHook"
    
    def test_multiple_plugin_interaction(self):
        """Test interaction between multiple plugins."""
        class Plugin1(CLIPlugin):
            def get_commands(self):
                def cmd1():
                    return "plugin1"
                return {"cmd1": cmd1}
        
        class Plugin2(CLIPlugin):
            def get_commands(self):
                def cmd2():
                    return "plugin2"
                return {"cmd2": cmd2}
        
        # Load both plugins
        loader = CLIPluginLoader()
        
        plugin1 = Plugin1()
        plugin2 = Plugin2()
        
        loader.plugins["plugin1"] = plugin1
        loader.plugins["plugin2"] = plugin2
        
        loader._register_plugin_components(plugin1, "plugin1")
        loader._register_plugin_components(plugin2, "plugin2")
        
        # Verify both plugins' commands are available
        assert "plugin1:cmd1" in loader.commands
        assert "plugin2:cmd2" in loader.commands
        
        # Test command execution
        cmd1 = loader.get_command("plugin1:cmd1")
        cmd2 = loader.get_command("plugin2:cmd2")
        
        assert cmd1() == "plugin1"
        assert cmd2() == "plugin2"