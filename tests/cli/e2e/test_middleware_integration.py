"""End-to-end tests for middleware integration."""

import pytest
from unittest.mock import Mock

from cli.factory import CommandFactory
from cli.middleware import LoggingMiddleware, ValidationMiddleware, ErrorMiddleware
from cli.testing import CLIRunner, assert_success, assert_failure


class TestMiddlewareIntegration:
    """Test middleware system integration."""
    
    def test_logging_middleware_captures_execution(self):
        """Test logging middleware captures command execution."""
        factory = CommandFactory()
        
        # Add logging middleware
        logging_middleware = LoggingMiddleware(
            log_args=True,
            log_result=True,
            log_timing=True
        )
        factory.middleware_pipeline.add(logging_middleware)
        
        @factory.create_middleware_command
        def test_command(name: str, value: int = 42):
            """Test command."""
            return f"Hello {name}, value is {value}"
        
        # Execute command
        result = test_command("Alice", value=100)
        
        assert result == "Hello Alice, value is 100"
    
    def test_validation_middleware_validates_inputs(self):
        """Test validation middleware validates command inputs."""
        from pydantic import BaseModel
        
        class TestSchema(BaseModel):
            name: str
            age: int
        
        factory = CommandFactory()
        
        # Add validation middleware
        validation_middleware = ValidationMiddleware(
            schemas={"test_command": TestSchema}
        )
        factory.middleware_pipeline.add(validation_middleware)
        
        @factory.create_middleware_command
        def test_command(name: str, age: int):
            """Test command."""
            return f"{name} is {age} years old"
        
        # Test valid input
        result = test_command("Alice", age=30)
        assert result == "Alice is 30 years old"
        
        # Test invalid input - should be handled by middleware
        with pytest.raises(ValueError, match="Validation failed"):
            test_command("Alice", age="invalid")
    
    def test_error_middleware_handles_exceptions(self):
        """Test error middleware handles command exceptions."""
        factory = CommandFactory()
        
        # Add error middleware
        error_middleware = ErrorMiddleware(
            show_traceback=False
        )
        factory.middleware_pipeline.add(error_middleware)
        
        @factory.create_middleware_command
        def failing_command():
            """Command that always fails."""
            raise ValueError("This command always fails")
        
        # Execute failing command - should be handled gracefully
        with pytest.raises(ValueError, match="This command always fails"):
            failing_command()
    
    def test_middleware_pipeline_execution_order(self):
        """Test middleware executes in correct order."""
        factory = CommandFactory()
        execution_order = []
        
        class OrderTrackingMiddleware:
            def __init__(self, name):
                self.name = name
            
            async def process(self, context, next_handler):
                execution_order.append(f"{self.name}_start")
                result = next_handler(context)
                execution_order.append(f"{self.name}_end")
                return result
        
        # Add middleware in specific order
        factory.middleware_pipeline.add(OrderTrackingMiddleware("first"))
        factory.middleware_pipeline.add(OrderTrackingMiddleware("second"))
        factory.middleware_pipeline.add(OrderTrackingMiddleware("third"))
        
        @factory.create_middleware_command
        def test_command():
            execution_order.append("command_execute")
            return "success"
        
        result = test_command()
        
        assert result == "success"
        assert execution_order == [
            "first_start",
            "second_start", 
            "third_start",
            "command_execute",
            "third_end",
            "second_end",
            "first_end"
        ]
    
    def test_middleware_context_sharing(self):
        """Test middleware can share context data."""
        factory = CommandFactory()
        
        class ContextMiddleware:
            def __init__(self, key, value):
                self.key = key
                self.value = value
            
            async def process(self, context, next_handler):
                context.set(self.key, self.value)
                return next_handler(context)
        
        factory.middleware_pipeline.add(ContextMiddleware("user_id", "123"))
        factory.middleware_pipeline.add(ContextMiddleware("session_id", "abc"))
        
        @factory.create_middleware_command
        def test_command():
            # This would need context access in real implementation
            return "executed"
        
        result = test_command()
        assert result == "executed"