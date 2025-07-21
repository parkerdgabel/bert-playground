"""Unit tests for CLI decorators."""

import os
import time
from unittest.mock import Mock, patch

import pytest
import typer

from cli.utils.decorators import (
    confirm_action,
    handle_errors,
    require_auth,
    requires_project,
    track_time,
)


class TestHandleErrors:
    """Test error handling decorator."""

    def test_successful_execution(self):
        """Test decorator doesn't interfere with successful execution."""

        @handle_errors
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_keyboard_interrupt(self):
        """Test handling of KeyboardInterrupt."""

        @handle_errors
        def interrupted_func():
            raise KeyboardInterrupt()

        with patch("cli.utils.decorators.print_error") as mock_print_error:
            with pytest.raises(typer.Exit) as exc_info:
                interrupted_func()

            assert exc_info.value.exit_code == 130  # SIGINT exit code
            mock_print_error.assert_called_once()
            call_args = mock_print_error.call_args[0]
            assert "cancelled by user" in call_args[0].lower()

    def test_file_not_found_error(self):
        """Test handling of FileNotFoundError."""

        @handle_errors
        def file_not_found_func():
            raise FileNotFoundError(2, "No such file", "missing.txt")

        with patch("cli.utils.decorators.print_error") as mock_print_error:
            with pytest.raises(typer.Exit) as exc_info:
                file_not_found_func()

            assert exc_info.value.exit_code == 1
            mock_print_error.assert_called_once()
            call_args = mock_print_error.call_args[0]
            assert "file not found" in call_args[0].lower()
            assert "missing.txt" in call_args[0]

    def test_permission_error(self):
        """Test handling of PermissionError."""

        @handle_errors
        def permission_denied_func():
            raise PermissionError(13, "Permission denied", "/restricted/file")

        with patch("cli.utils.decorators.print_error") as mock_print_error:
            with pytest.raises(typer.Exit) as exc_info:
                permission_denied_func()

            assert exc_info.value.exit_code == 1
            mock_print_error.assert_called_once()
            call_args = mock_print_error.call_args[0]
            assert "permission denied" in call_args[0].lower()
            assert "/restricted/file" in call_args[0]

    def test_import_error(self):
        """Test handling of ImportError."""

        @handle_errors
        def import_error_func():
            raise ImportError("No module named 'missing_module'")

        with patch("cli.utils.decorators.print_error") as mock_print_error:
            with pytest.raises(typer.Exit) as exc_info:
                import_error_func()

            assert exc_info.value.exit_code == 1
            mock_print_error.assert_called_once()
            call_args = mock_print_error.call_args[0]
            assert "missing dependency" in call_args[0].lower()
            assert "missing_module" in call_args[0]
            assert "uv pip install" in call_args[0]

    def test_generic_exception(self):
        """Test handling of generic exceptions."""

        @handle_errors
        def generic_error_func():
            raise ValueError("Something went wrong")

        with patch("cli.utils.decorators.print_error") as mock_print_error:
            with patch("cli.utils.decorators.logger.exception") as mock_logger:
                with pytest.raises(typer.Exit) as exc_info:
                    generic_error_func()

                assert exc_info.value.exit_code == 1
                mock_logger.assert_called_once_with("Unexpected error")
                mock_print_error.assert_called_once()
                call_args = mock_print_error.call_args[0]
                assert "unexpected error" in call_args[0].lower()
                assert "ValueError" in call_args[0]
                assert "Something went wrong" in call_args[0]
                assert "--verbose" in call_args[0]

    def test_preserves_function_attributes(self):
        """Test decorator preserves function attributes."""

        @handle_errors
        def documented_func():
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."


class TestTrackTime:
    """Test time tracking decorator."""

    def test_successful_execution_with_default_message(self):
        """Test time tracking with default message."""

        @track_time()
        def timed_func():
            time.sleep(0.1)
            return "done"

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            result = timed_func()

            assert result == "done"
            assert mock_console.print.call_count == 2

            # Check starting message
            start_call = mock_console.print.call_args_list[0][0][0]
            assert "Running timed_func" in start_call

            # Check completion message
            end_call = mock_console.print.call_args_list[1][0][0]
            assert "Completed" in end_call
            assert "0.1s" in end_call or "0.2s" in end_call  # Allow some variance

    def test_successful_execution_with_custom_message(self):
        """Test time tracking with custom message."""

        @track_time("Processing data")
        def process_func():
            return "processed"

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            result = process_func()

            assert result == "processed"

            # Check custom starting message
            start_call = mock_console.print.call_args_list[0][0][0]
            assert "Processing data" in start_call

    def test_time_formatting_seconds(self):
        """Test time formatting for seconds."""

        @track_time()
        def quick_func():
            time.sleep(0.5)

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            quick_func()

            end_call = mock_console.print.call_args_list[1][0][0]
            assert "0.5s" in end_call or "0.6s" in end_call

    def test_time_formatting_minutes(self):
        """Test time formatting for minutes."""

        @track_time()
        def medium_func():
            pass

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            # Mock time to simulate 2 minutes 30 seconds
            with patch("time.time") as mock_time:
                mock_time.side_effect = [0, 150]  # 2.5 minutes

                medium_func()

                end_call = mock_console.print.call_args_list[1][0][0]
                assert "2m 30s" in end_call

    def test_time_formatting_hours(self):
        """Test time formatting for hours."""

        @track_time()
        def long_func():
            pass

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            # Mock time to simulate 1 hour 45 minutes
            with patch("time.time") as mock_time:
                mock_time.side_effect = [0, 6300]  # 1h 45m

                long_func()

                end_call = mock_console.print.call_args_list[1][0][0]
                assert "1h 45m" in end_call

    def test_exception_handling(self):
        """Test time tracking when function raises exception."""

        @track_time()
        def failing_func():
            time.sleep(0.1)
            raise ValueError("Test error")

        with patch("cli.utils.decorators.get_console") as mock_get_console:
            mock_console = Mock()
            mock_get_console.return_value = mock_console

            with pytest.raises(ValueError, match="Test error"):
                failing_func()

            # Check failure message
            end_call = mock_console.print.call_args_list[1][0][0]
            assert "Failed after" in end_call
            assert "0.1s" in end_call or "0.2s" in end_call


class TestRequireAuth:
    """Test authentication requirement decorator."""

    def test_kaggle_auth_success(self, tmp_path):
        """Test successful Kaggle authentication check."""

        @require_auth("kaggle")
        def kaggle_func():
            return "kaggle_success"

        # Create mock kaggle.json
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        kaggle_json = kaggle_dir / "kaggle.json"
        kaggle_json.write_text('{"username": "test", "key": "test_key"}')

        with patch("os.path.expanduser") as mock_expanduser:
            mock_expanduser.return_value = str(kaggle_json)

            result = kaggle_func()
            assert result == "kaggle_success"

    def test_kaggle_auth_missing(self):
        """Test Kaggle authentication check when credentials missing."""

        @require_auth("kaggle")
        def kaggle_func():
            return "should_not_reach"

        with patch("os.path.expanduser") as mock_expanduser:
            mock_expanduser.return_value = "/nonexistent/.kaggle/kaggle.json"

            with patch("cli.utils.decorators.print_error") as mock_print_error:
                with pytest.raises(typer.Exit) as exc_info:
                    kaggle_func()

                assert exc_info.value.exit_code == 1
                mock_print_error.assert_called_once()
                call_args = mock_print_error.call_args[0]
                assert "kaggle credentials not found" in call_args[0].lower()
                assert "kaggle config set" in call_args[0]

    def test_mlflow_auth_with_tracking_uri(self):
        """Test MLflow authentication with tracking URI."""

        @require_auth("mlflow")
        def mlflow_func():
            return "mlflow_success"

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            result = mlflow_func()
            assert result == "mlflow_success"

    def test_mlflow_auth_without_tracking_uri(self):
        """Test MLflow authentication without tracking URI."""

        @require_auth("mlflow")
        def mlflow_func():
            return "mlflow_success"

        with patch.dict(os.environ, {}, clear=True):
            # Should still succeed as MLflow can work locally
            result = mlflow_func()
            assert result == "mlflow_success"

    def test_unknown_service(self):
        """Test with unknown service (should just pass through)."""

        @require_auth("unknown_service")
        def unknown_func():
            return "unknown_success"

        result = unknown_func()
        assert result == "unknown_success"


class TestConfirmAction:
    """Test action confirmation decorator."""

    def test_confirmation_accepted(self):
        """Test when user confirms action."""

        @confirm_action("Delete all data?")
        def destructive_func():
            return "deleted"

        # Mock sys.stdin.isatty to return True (interactive mode)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("typer.confirm") as mock_confirm:
                mock_confirm.return_value = True

                result = destructive_func()
                assert result == "deleted"
                mock_confirm.assert_called_once_with("Delete all data?")

    def test_confirmation_rejected(self):
        """Test when user rejects action."""

        @confirm_action("Delete all data?")
        def destructive_func():
            return "should_not_reach"

        # Mock sys.stdin.isatty to return True (interactive mode)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("typer.confirm") as mock_confirm:
                mock_confirm.return_value = False

                with patch("cli.utils.decorators.print_info") as mock_print_info:
                    # typer.Exit may be a click.exceptions.Exit
                    with pytest.raises((typer.Exit, Exception)) as exc_info:
                        destructive_func()

                    # Check if it's either typer.Exit or click's Exit
                    if hasattr(exc_info.value, "exit_code"):
                        assert exc_info.value.exit_code == 0
                    mock_print_info.assert_called_once_with("Operation cancelled")

    def test_skip_confirmation_with_yes_flag(self):
        """Test skipping confirmation with --yes flag."""

        @confirm_action("Delete all data?")
        def destructive_func(yes=False):
            return "deleted"

        with patch("typer.confirm") as mock_confirm:
            result = destructive_func(yes=True)
            assert result == "deleted"
            mock_confirm.assert_not_called()

    def test_skip_confirmation_non_interactive(self):
        """Test skipping confirmation in non-interactive mode."""

        @confirm_action("Delete all data?")
        def destructive_func():
            return "deleted"

        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False

            with patch("typer.confirm") as mock_confirm:
                result = destructive_func()
                assert result == "deleted"
                mock_confirm.assert_not_called()

    def test_custom_message(self):
        """Test with custom confirmation message."""
        custom_msg = "This will remove all files. Continue?"

        @confirm_action(custom_msg)
        def remove_func():
            return "removed"

        # Mock sys.stdin.isatty to return True (interactive mode)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True

            with patch("typer.confirm") as mock_confirm:
                mock_confirm.return_value = True

                result = remove_func()
                assert result == "removed"
                mock_confirm.assert_called_once_with(custom_msg)


class TestRequiresProject:
    """Test project requirement decorator."""

    def test_in_project_with_pyproject(self, tmp_path):
        """Test when in project with pyproject.toml."""

        @requires_project()
        def project_func():
            return "in_project"

        # Create project marker
        (tmp_path / "pyproject.toml").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path

            result = project_func()
            assert result == "in_project"

    def test_in_project_with_bert_yaml(self, tmp_path):
        """Test when in project with bert.yaml."""

        @requires_project()
        def project_func():
            return "in_project"

        # Create project marker
        (tmp_path / "bert.yaml").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path

            result = project_func()
            assert result == "in_project"

    def test_in_project_with_configs_dir(self, tmp_path):
        """Test when in project with configs directory."""

        @requires_project()
        def project_func():
            return "in_project"

        # Create project marker
        (tmp_path / "configs").mkdir()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path

            result = project_func()
            assert result == "in_project"

    def test_in_subdirectory_of_project(self, tmp_path):
        """Test when in subdirectory of project."""

        @requires_project()
        def project_func():
            return "in_project"

        # Create project structure
        project_root = tmp_path / "myproject"
        project_root.mkdir()
        (project_root / "pyproject.toml").touch()

        subdir = project_root / "src" / "components"
        subdir.mkdir(parents=True)

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = subdir

            result = project_func()
            assert result == "in_project"

    def test_not_in_project(self, tmp_path):
        """Test when not in a project."""

        @requires_project()
        def project_func():
            return "should_not_reach"

        # No project markers
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path

            with patch("cli.utils.decorators.print_error") as mock_print_error:
                with pytest.raises(typer.Exit) as exc_info:
                    project_func()

                assert exc_info.value.exit_code == 1
                mock_print_error.assert_called_once()
                call_args = mock_print_error.call_args[0]
                assert "must be run within a bert project" in call_args[0].lower()
                assert "bert init" in call_args[0]

    def test_all_project_markers(self, tmp_path):
        """Test recognition of all project marker types."""

        @requires_project()
        def project_func():
            return "in_project"

        markers = ["pyproject.toml", "bert.yaml", "bert.yml", ".bertrc", "configs/"]

        for marker in markers:
            # Create fresh temp directory for each test
            test_dir = tmp_path / f"test_{marker.replace('/', '_')}"
            test_dir.mkdir()

            # Create marker
            marker_path = test_dir / marker
            if marker.endswith("/"):
                marker_path.mkdir()
            else:
                marker_path.touch()

            with patch("pathlib.Path.cwd") as mock_cwd:
                mock_cwd.return_value = test_dir

                result = project_func()
                assert result == "in_project"
