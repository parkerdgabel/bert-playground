"""CLI testing framework.

Provides utilities for testing CLI commands including:
- Command testing helpers
- Mock factories
- Integration test support
- Fixture management
"""

from cli.testing.fixtures import (
    CLIFixture,
    ConfigFixture,
    EnvironmentFixture,
    FileSystemFixture,
)
from cli.testing.helpers import (
    CLIRunner,
    CommandResult,
    assert_success,
    assert_failure,
    assert_output_contains,
)
from cli.testing.mocks import (
    MockContext,
    MockMiddleware,
    MockHook,
    MockPlugin,
    create_mock_config,
    create_mock_command,
)

__all__ = [
    # Fixtures
    "CLIFixture",
    "ConfigFixture",
    "EnvironmentFixture",
    "FileSystemFixture",
    # Helpers
    "CLIRunner",
    "CommandResult",
    "assert_success",
    "assert_failure",
    "assert_output_contains",
    # Mocks
    "MockContext",
    "MockMiddleware",
    "MockHook",
    "MockPlugin",
    "create_mock_config",
    "create_mock_command",
]