"""End-to-end tests for command pipeline."""

import pytest

from cli.pipeline import CommandPipeline, HookPhase
from cli.pipeline.hooks import ConfigurationHook, EnvironmentHook, ValidationHook
from cli.pipeline.composer import CommandComposer, CompositionStrategy
from cli.testing import PipelineTestRunner, assert_success


class TestCommandPipeline:
    """Test command pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_basic_pipeline_execution(self):
        """Test basic pipeline execution with hooks."""
        pipeline = CommandPipeline()
        
        # Add configuration hook
        config_hook = ConfigurationHook(required_keys=["test_key"])
        pipeline.add_hook(config_hook)
        
        # Test command
        async def test_command(config):
            return f"Config received: {config.get('test_key', 'not found')}"
        
        # Execute with required config
        result = await pipeline.execute(
            "test_command",
            test_command,
            config={"test_key": "test_value"}
        )
        
        assert "test_value" in result
    
    @pytest.mark.asyncio
    async def test_hook_execution_phases(self):
        """Test hooks execute in correct phases."""
        pipeline = CommandPipeline()
        execution_log = []
        
        class LoggingHook:
            def __init__(self, name, phases):
                self.name = name
                self.phases = phases
                self.priority = 50
            
            def should_execute(self, phase):
                return phase in self.phases
            
            async def execute(self, phase, context):
                execution_log.append(f"{self.name}:{phase.value}")
        
        # Add hooks for different phases
        pipeline.add_hook(LoggingHook("PreHook", {HookPhase.PRE_EXECUTE}))
        pipeline.add_hook(LoggingHook("PostHook", {HookPhase.POST_EXECUTE}))
        pipeline.add_hook(LoggingHook("CleanupHook", {HookPhase.POST_CLEANUP}))
        
        async def test_command():
            execution_log.append("command_executed")
            return "success"
        
        result = await pipeline.execute("test", test_command)
        
        assert result == "success"
        assert "PreHook:pre_execute" in execution_log
        assert "command_executed" in execution_log
        assert "PostHook:post_execute" in execution_log
        assert "CleanupHook:post_cleanup" in execution_log
    
    @pytest.mark.asyncio
    async def test_environment_hook(self):
        """Test environment hook functionality."""
        import os
        
        pipeline = CommandPipeline()
        
        # Add environment hook
        env_hook = EnvironmentHook(
            set_vars={"TEST_VAR": "test_value"},
            required_vars=["TEST_VAR"]
        )
        pipeline.add_hook(env_hook)
        
        async def test_command():
            return os.environ.get("TEST_VAR", "not_found")
        
        result = await pipeline.execute("test", test_command)
        
        assert result == "test_value"
        # Environment should be restored after execution
        assert "TEST_VAR" not in os.environ
    
    @pytest.mark.asyncio
    async def test_validation_hook(self):
        """Test validation hook functionality."""
        pipeline = CommandPipeline()
        
        # Add validation hook
        validation_hook = ValidationHook(
            required_args=["name", "age"],
            validators={
                "age": lambda x: isinstance(x, int) and x > 0
            }
        )
        pipeline.add_hook(validation_hook)
        
        async def test_command(name, age):
            return f"{name} is {age} years old"
        
        # Valid execution
        result = await pipeline.execute(
            "test",
            test_command,
            name="Alice",
            age=30
        )
        assert result == "Alice is 30 years old"
        
        # Invalid execution - missing required arg
        with pytest.raises(Exception):
            await pipeline.execute(
                "test",
                test_command,
                name="Alice"  # missing age
            )
    
    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Test hook error handling."""
        pipeline = CommandPipeline()
        
        class FailingHook:
            def __init__(self):
                self.name = "FailingHook"
                self.phases = {HookPhase.PRE_EXECUTE}
                self.priority = 50
            
            def should_execute(self, phase):
                return phase in self.phases
            
            async def execute(self, phase, context):
                raise ValueError("Hook failed intentionally")
        
        pipeline.add_hook(FailingHook())
        
        async def test_command():
            return "success"
        
        # Should propagate hook error
        with pytest.raises(ValueError, match="Validation failed"):
            await pipeline.execute("test", test_command)


class TestCommandComposition:
    """Test command composition functionality."""
    
    @pytest.mark.asyncio
    async def test_sequential_composition(self):
        """Test sequential command composition."""
        composer = CommandComposer()
        
        async def cmd1(x):
            return x * 2
        
        async def cmd2(x):
            return x + 10
        
        async def cmd3(x):
            return x * 3
        
        # Compose commands
        composed = composer.sequential(cmd1, cmd2, cmd3)
        
        # Execute: 5 -> 10 -> 20 -> 60
        result = await composed.execute(5)
        assert result == 60
    
    @pytest.mark.asyncio
    async def test_parallel_composition(self):
        """Test parallel command composition."""
        composer = CommandComposer()
        
        async def cmd1(x):
            return x * 2
        
        async def cmd2(x):
            return x + 5
        
        async def cmd3(x):
            return x * 3
        
        # Compose commands in parallel
        composed = composer.parallel(cmd1, cmd2, cmd3)
        
        # Execute: all commands get input 10
        results = await composed.execute(10)
        
        assert results == [20, 15, 30]  # [10*2, 10+5, 10*3]
    
    @pytest.mark.asyncio
    async def test_conditional_composition(self):
        """Test conditional command composition."""
        composer = CommandComposer()
        
        async def cmd1():
            return "cmd1 executed"
        
        async def cmd2():
            return "cmd2 executed"
        
        # Compose with conditions
        composed = composer.conditional(
            cmd1, cmd2,
            conditions=[
                lambda **kwargs: kwargs.get("execute_cmd1", False),
                lambda **kwargs: kwargs.get("execute_cmd2", False)
            ]
        )
        
        # Test first condition
        result = await composed.execute(execute_cmd1=True)
        assert result == "cmd1 executed"
        
        # Test second condition
        result = await composed.execute(execute_cmd2=True)
        assert result == "cmd2 executed"
        
        # Test no conditions met
        result = await composed.execute()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_fallback_composition(self):
        """Test fallback command composition."""
        composer = CommandComposer()
        
        async def failing_cmd():
            raise ValueError("Command failed")
        
        async def fallback_cmd():
            return "fallback executed"
        
        # Compose with fallback
        composed = composer.fallback(failing_cmd, fallback_cmd)
        
        # Should execute fallback when first fails
        result = await composed.execute()
        assert result == "fallback executed"
    
    @pytest.mark.asyncio
    async def test_reduce_composition(self):
        """Test reduce command composition."""
        composer = CommandComposer()
        
        async def cmd1():
            return [1, 2, 3]
        
        async def cmd2():
            return [4, 5, 6]
        
        async def cmd3():
            return [7, 8, 9]
        
        # Compose with custom reducer
        def list_merger(acc, value):
            if acc is None:
                return value
            return acc + value
        
        composed = composer.reduce(
            cmd1, cmd2, cmd3,
            reducer=list_merger
        )
        
        result = await composed.execute()
        assert result == [1, 2, 3, 4, 5, 6, 7, 8, 9]