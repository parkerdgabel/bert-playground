"""Unit tests for console utilities."""

import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich.syntax import Syntax

from cli.utils.console import (
    get_console,
    print_error,
    print_success,
    print_warning,
    print_info,
    create_table,
    create_progress,
    print_code,
    confirm,
    format_bytes,
    format_timestamp
)
import cli.utils.console


class TestGetConsole:
    """Test console instance management."""
    
    def setup_method(self):
        """Reset global console before each test."""
        cli.utils.console._console = None
    
    def test_get_console_creates_instance(self):
        """Test console is created on first call."""
        console = get_console()
        assert isinstance(console, Console)
        assert console is not None
    
    def test_get_console_returns_same_instance(self):
        """Test same console instance is returned."""
        console1 = get_console()
        console2 = get_console()
        assert console1 is console2
    
    def test_get_console_quiet_mode(self):
        """Test console creation in quiet mode."""
        with patch.dict(os.environ, {"BERT_CLI_QUIET": "1"}):
            cli.utils.console._console = None  # Reset
            console = get_console()
            assert console.quiet is True
    
    def test_get_console_normal_mode(self):
        """Test console creation in normal mode."""
        with patch.dict(os.environ, {"BERT_CLI_QUIET": "0"}):
            cli.utils.console._console = None  # Reset
            console = get_console()
            assert console.quiet is False
    
    def test_get_console_no_env_var(self):
        """Test console creation without environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            cli.utils.console._console = None  # Reset
            console = get_console()
            assert console.quiet is False


class TestPrintFunctions:
    """Test print utility functions."""
    
    def test_print_error(self):
        """Test error message printing."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            print_error("Something went wrong", "Test Error")
            
            mock_console.print.assert_called_once()
            panel = mock_console.print.call_args[0][0]
            
            assert isinstance(panel, Panel)
            assert "Something went wrong" in str(panel.renderable)
            assert panel.border_style == "red"
            assert "Test Error" in str(panel.title)
    
    def test_print_error_default_title(self):
        """Test error message with default title."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            print_error("Error message")
            
            panel = mock_console.print.call_args[0][0]
            assert "Error" in str(panel.title)
    
    def test_print_success(self):
        """Test success message printing."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            print_success("Operation completed", "Great!")
            
            mock_console.print.assert_called_once()
            panel = mock_console.print.call_args[0][0]
            
            assert isinstance(panel, Panel)
            assert "Operation completed" in str(panel.renderable)
            assert panel.border_style == "green"
            assert "Great!" in str(panel.title)
    
    def test_print_warning(self):
        """Test warning message printing."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            print_warning("This might cause issues", "Caution")
            
            mock_console.print.assert_called_once()
            panel = mock_console.print.call_args[0][0]
            
            assert isinstance(panel, Panel)
            assert "This might cause issues" in str(panel.renderable)
            assert panel.border_style == "yellow"
            assert "Caution" in str(panel.title)
    
    def test_print_info(self):
        """Test info message printing."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            print_info("Here's some information", "FYI")
            
            mock_console.print.assert_called_once()
            panel = mock_console.print.call_args[0][0]
            
            assert isinstance(panel, Panel)
            assert "Here's some information" in str(panel.renderable)
            assert panel.border_style == "blue"
            assert "FYI" in str(panel.title)


class TestCreateTable:
    """Test table creation."""
    
    def test_create_table_basic(self):
        """Test basic table creation."""
        table = create_table("Test Table", ["Column 1", "Column 2", "Column 3"])
        
        assert isinstance(table, Table)
        assert table.title == "Test Table"
        assert table.show_header is True
        assert table.header_style == "bold magenta"
        assert len(table.columns) == 3
        assert table.columns[0].header == "Column 1"
        assert table.columns[1].header == "Column 2"
        assert table.columns[2].header == "Column 3"
    
    def test_create_table_empty_columns(self):
        """Test table creation with empty column list."""
        table = create_table("Empty Table", [])
        
        assert isinstance(table, Table)
        assert table.title == "Empty Table"
        assert len(table.columns) == 0
    
    def test_create_table_single_column(self):
        """Test table creation with single column."""
        table = create_table("Single Column", ["Only Column"])
        
        assert len(table.columns) == 1
        assert table.columns[0].header == "Only Column"


class TestCreateProgress:
    """Test progress bar creation."""
    
    def test_create_progress(self):
        """Test progress bar creation."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            progress = create_progress()
            
            assert isinstance(progress, Progress)
            assert progress.console is mock_console
            
            # Check columns
            column_types = [type(col).__name__ for col in progress.columns]
            assert "SpinnerColumn" in column_types
            assert "TextColumn" in column_types
            assert "BarColumn" in column_types
            assert "TaskProgressColumn" in column_types


class TestPrintCode:
    """Test code printing with syntax highlighting."""
    
    def test_print_code_basic(self):
        """Test basic code printing."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            code = "def hello():\n    print('Hello, World!')"
            print_code(code)
            
            mock_console.print.assert_called_once()
            syntax = mock_console.print.call_args[0][0]
            
            assert isinstance(syntax, Syntax)
            assert syntax.code == code
            assert syntax.lexer.name == "Python"
            # Check that theme is set (it's a PygmentsSyntaxTheme object)
            assert syntax._theme is not None
            assert syntax.line_numbers is True
    
    def test_print_code_with_language(self):
        """Test code printing with specific language."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            code = "console.log('Hello, World!');"
            print_code(code, language="javascript")
            
            syntax = mock_console.print.call_args[0][0]
            assert syntax.lexer.name == "JavaScript"
    
    def test_print_code_with_title(self):
        """Test code printing with title."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console
            
            code = "print('Hello')"
            print_code(code, title="Example Code")
            
            panel = mock_console.print.call_args[0][0]
            assert isinstance(panel, Panel)
            assert panel.title == "Example Code"
            assert isinstance(panel.renderable, Syntax)


class TestConfirm:
    """Test confirmation prompt."""
    
    def test_confirm_yes_default_false(self):
        """Test confirmation with 'yes' response (default false)."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = "y"
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=False)
            
            assert result is True
            mock_console.input.assert_called_once_with("Continue? (y/N): ")
    
    def test_confirm_yes_full_default_false(self):
        """Test confirmation with 'yes' full response."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = "yes"
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=False)
            assert result is True
    
    def test_confirm_no_default_false(self):
        """Test confirmation with 'no' response."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = "n"
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=False)
            assert result is False
    
    def test_confirm_empty_default_false(self):
        """Test confirmation with empty response (default false)."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = ""
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=False)
            assert result is False
    
    def test_confirm_empty_default_true(self):
        """Test confirmation with empty response (default true)."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = ""
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=True)
            
            assert result is True
            mock_console.input.assert_called_once_with("Continue? (Y/n): ")
    
    def test_confirm_invalid_response(self):
        """Test confirmation with invalid response."""
        with patch("cli.utils.console.get_console") as mock_get_console:
            mock_console = Mock()
            mock_console.input.return_value = "maybe"
            mock_get_console.return_value = mock_console
            
            result = confirm("Continue?", default=False)
            assert result is False


class TestFormatBytes:
    """Test byte formatting."""
    
    def test_format_bytes_basic(self):
        """Test basic byte formatting."""
        assert format_bytes(0) == "0.0 B"
        assert format_bytes(1) == "1.0 B"
        assert format_bytes(1023) == "1023.0 B"
        
    def test_format_bytes_kb(self):
        """Test kilobyte formatting."""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1536) == "1.5 KB"
        assert format_bytes(1024 * 1023) == "1023.0 KB"
    
    def test_format_bytes_mb(self):
        """Test megabyte formatting."""
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1.5) == "1.5 MB"
        assert format_bytes(1024 * 1024 * 999) == "999.0 MB"
    
    def test_format_bytes_gb(self):
        """Test gigabyte formatting."""
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(1024 * 1024 * 1024 * 2.5) == "2.5 GB"
    
    def test_format_bytes_tb(self):
        """Test terabyte formatting."""
        assert format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"
        assert format_bytes(1024 * 1024 * 1024 * 1024 * 10) == "10.0 TB"
    
    def test_format_bytes_pb(self):
        """Test petabyte formatting."""
        size = 1024 * 1024 * 1024 * 1024 * 1024 * 2
        assert format_bytes(size) == "2.0 PB"
    
    def test_format_bytes_large(self):
        """Test very large byte values."""
        # Larger than PB
        size = 1024 * 1024 * 1024 * 1024 * 1024 * 1024 * 5
        result = format_bytes(size)
        assert result.endswith(" PB")
        assert float(result.split()[0]) > 1024


class TestFormatTimestamp:
    """Test timestamp formatting."""
    
    def test_format_timestamp_basic(self):
        """Test basic timestamp formatting."""
        # Use a known timestamp
        timestamp = 1609459200.0  # 2021-01-01 00:00:00 UTC
        
        with patch("datetime.datetime") as mock_datetime:
            mock_dt = Mock()
            mock_dt.strftime.return_value = "2021-01-01 00:00:00"
            mock_datetime.fromtimestamp.return_value = mock_dt
            
            result = format_timestamp(timestamp)
            
            mock_datetime.fromtimestamp.assert_called_once_with(timestamp)
            mock_dt.strftime.assert_called_once_with("%Y-%m-%d %H:%M:%S")
            assert result == "2021-01-01 00:00:00"
    
    def test_format_timestamp_various(self):
        """Test formatting various timestamps."""
        from datetime import datetime
        
        # Current time
        now = datetime.now().timestamp()
        result = format_timestamp(now)
        assert len(result) == 19  # YYYY-MM-DD HH:MM:SS
        assert result[4] == '-' and result[7] == '-'
        assert result[10] == ' '
        assert result[13] == ':' and result[16] == ':'
        
    def test_format_timestamp_epoch(self):
        """Test formatting epoch timestamp."""
        from datetime import datetime
        
        result = format_timestamp(0)
        # This will vary by timezone, so just check format
        assert len(result) == 19
        assert result[4] == '-' and result[7] == '-'