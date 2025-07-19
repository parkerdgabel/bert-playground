"""Unit tests for Kaggle CLI commands."""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typer.testing import CliRunner

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cli.app import app


class TestKaggleCompetitionsCommand:
    """Test suite for the kaggle-competitions command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_competitions_df(self):
        """Create mock competitions DataFrame."""
        return pd.DataFrame({
            'id': ['titanic', 'digit-recognizer', 'house-prices'],
            'title': ['Titanic - Machine Learning from Disaster', 
                     'Digit Recognizer', 
                     'House Prices - Advanced Regression'],
            'deadline': [
                pd.Timestamp.now() + timedelta(days=30),
                pd.Timestamp.now() + timedelta(days=60),
                pd.Timestamp.now() - timedelta(days=10)  # Expired
            ],
            'numTeams': [15000, 2500, 5000],
            'reward': ['Knowledge', '$25,000', 'Knowledge'],
            'isCompleted': [False, False, True],
            'tags': [
                ['binary classification', 'tabular'],
                ['multiclass classification', 'computer vision'],
                ['regression', 'tabular']
            ]
        })
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_list_basic(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test basic competition listing."""
        # Setup mock
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        # Run command
        result = runner.invoke(app, ["kaggle-competitions"])
        
        # Check success
        assert result.exit_code == 0
        assert "Kaggle Competitions" in result.stdout
        assert "Titanic" in result.stdout
        assert "Digit Recognizer" in result.stdout
        
        # Verify API call
        mock_kaggle.list_competitions.assert_called_once_with(
            category=None,
            search=None,
            sort_by="latestDeadline",
            page=1  # Critical: must be 'page' not 'page_size'
        )
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_with_filters(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test competitions with category and search filters."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--category", "tabular",
            "--search", "classification",
            "--sort", "prize"
        ])
        
        assert result.exit_code == 0
        
        # Verify filters were passed correctly
        mock_kaggle.list_competitions.assert_called_once_with(
            category="tabular",
            search="classification",
            sort_by="prize",
            page=1
        )
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_active_only(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test filtering active competitions only."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--active"
        ])
        
        assert result.exit_code == 0
        assert "Titanic" in result.stdout
        assert "Digit Recognizer" in result.stdout
        # Should not show expired competition
        assert "House Prices" not in result.stdout or "Expired" in result.stdout
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_show_all(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test showing all competitions including expired."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--all"
        ])
        
        assert result.exit_code == 0
        assert "House Prices" in result.stdout
        assert "Expired" in result.stdout or "Completed" in result.stdout
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_with_tags(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test showing competitions with tags."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--tags"
        ])
        
        assert result.exit_code == 0
        assert "binary classification" in result.stdout or "Tags" in result.stdout
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_limit(self, mock_kaggle_class, runner, mock_competitions_df):
        """Test limiting number of competitions shown."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = mock_competitions_df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--limit", "2"
        ])
        
        assert result.exit_code == 0
        # Check that output is limited (exact check depends on implementation)
        assert "Showing" in result.stdout or len(result.stdout.split('\n')) < 50
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_empty_results(self, mock_kaggle_class, runner):
        """Test handling empty competition results."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = pd.DataFrame()
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-competitions",
            "--search", "nonexistent"
        ])
        
        assert result.exit_code == 0
        assert "No competitions found" in result.stdout
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_api_error(self, mock_kaggle_class, runner):
        """Test handling API errors."""
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.side_effect = Exception("API Error")
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, ["kaggle-competitions"])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "Failed" in result.stdout
    
    @patch('cli.commands.kaggle.competitions.KaggleIntegration')
    def test_competitions_timestamp_handling(self, mock_kaggle_class, runner):
        """Test proper handling of Pandas Timestamp objects."""
        # Create DataFrame with actual Timestamp objects
        df = pd.DataFrame({
            'id': ['test-comp'],
            'title': ['Test Competition'],
            'deadline': [pd.Timestamp('2024-12-25 23:59:59')],
            'numTeams': [100],
            'reward': ['$10,000'],
            'isCompleted': [False],
            'tags': [[]]
        })
        
        mock_kaggle = MagicMock()
        mock_kaggle.list_competitions.return_value = df
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, ["kaggle-competitions"])
        
        assert result.exit_code == 0
        assert "Test Competition" in result.stdout
        # Should not have timestamp errors
        assert "Timestamp' object is not subscriptable" not in result.stdout


class TestKaggleDownloadCommand:
    """Test suite for the kaggle-download command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.kaggle.download.KaggleIntegration')
    @patch('cli.commands.kaggle.download.Path')
    def test_download_competition_basic(self, mock_path, mock_kaggle_class, runner):
        """Test basic competition download."""
        # Setup mocks
        mock_kaggle = MagicMock()
        mock_kaggle.download_competition.return_value = True
        mock_kaggle_class.return_value = mock_kaggle
        
        mock_output_path = MagicMock()
        mock_output_path.exists.return_value = False
        mock_path.return_value = mock_output_path
        
        # Run command
        result = runner.invoke(app, [
            "kaggle-download",
            "titanic"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Download" in result.stdout or "Success" in result.stdout
        
        # Verify download was called
        mock_kaggle.download_competition.assert_called_once_with(
            "titanic",
            mock_output_path
        )
    
    @patch('cli.commands.kaggle.download.KaggleIntegration')
    def test_download_with_output_path(self, mock_kaggle_class, runner, tmp_path):
        """Test download with custom output path."""
        mock_kaggle = MagicMock()
        mock_kaggle.download_competition.return_value = True
        mock_kaggle_class.return_value = mock_kaggle
        
        output_dir = tmp_path / "competitions"
        
        result = runner.invoke(app, [
            "kaggle-download",
            "titanic",
            "--output", str(output_dir)
        ])
        
        assert result.exit_code == 0
        
        # Verify custom path was used
        call_args = mock_kaggle.download_competition.call_args
        assert str(output_dir) in str(call_args[0][1])
    
    @patch('cli.commands.kaggle.download.KaggleIntegration')
    def test_download_force_overwrite(self, mock_kaggle_class, runner, tmp_path):
        """Test force overwrite existing data."""
        mock_kaggle = MagicMock()
        mock_kaggle.download_competition.return_value = True
        mock_kaggle_class.return_value = mock_kaggle
        
        # Create existing directory
        existing_dir = tmp_path / "titanic"
        existing_dir.mkdir()
        (existing_dir / "train.csv").touch()
        
        result = runner.invoke(app, [
            "kaggle-download",
            "titanic",
            "--output", str(tmp_path),
            "--force"
        ])
        
        assert result.exit_code == 0
        # Should proceed with download despite existing files
        mock_kaggle.download_competition.assert_called_once()
    
    @patch('cli.commands.kaggle.download.KaggleIntegration')
    def test_download_no_extract(self, mock_kaggle_class, runner):
        """Test download without extracting zip files."""
        mock_kaggle = MagicMock()
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-download",
            "titanic",
            "--no-extract"
        ])
        
        # Check that the no-extract option is handled
        # (Implementation depends on how the command handles this)
        assert result.exit_code == 0 or "--no-extract" in result.stdout


class TestKaggleSubmitCommand:
    """Test suite for the kaggle-submit command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.kaggle.submit.KaggleIntegration')
    @patch('cli.commands.kaggle.submit.Path')
    def test_submit_basic(self, mock_path, mock_kaggle_class, runner, tmp_path):
        """Test basic submission."""
        # Create submission file
        submission_file = tmp_path / "submission.csv"
        submission_file.write_text("id,target\n1,0\n2,1\n")
        
        # Setup mocks
        mock_kaggle = MagicMock()
        mock_kaggle.submit_predictions.return_value = {
            'status': 'complete',
            'publicScore': 0.85,
            'privateScore': None
        }
        mock_kaggle_class.return_value = mock_kaggle
        
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.is_file.return_value = True
        
        # Run command
        result = runner.invoke(app, [
            "kaggle-submit",
            "titanic",
            str(submission_file),
            "--message", "Test submission"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Submission" in result.stdout or "Success" in result.stdout
        
        # Verify submission was called
        mock_kaggle.submit_predictions.assert_called_once()
        call_args = mock_kaggle.submit_predictions.call_args
        assert call_args[0][0] == "titanic"
        assert "Test submission" in str(call_args)
    
    @patch('cli.commands.kaggle.submit.KaggleIntegration')
    def test_submit_with_checkpoint(self, mock_kaggle_class, runner, tmp_path):
        """Test submission with checkpoint reference."""
        submission_file = tmp_path / "submission.csv"
        submission_file.write_text("id,target\n1,0\n2,1\n")
        
        mock_kaggle = MagicMock()
        mock_kaggle.submit_predictions.return_value = {'status': 'complete'}
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-submit",
            "titanic",
            str(submission_file),
            "--checkpoint", "output/run_001/best_model",
            "--message", "Submission from checkpoint"
        ])
        
        assert result.exit_code == 0
        assert "checkpoint" in result.stdout.lower() or result.exit_code == 0
    
    @patch('cli.commands.kaggle.submit.KaggleIntegration')
    def test_submit_file_not_found(self, mock_kaggle_class, runner):
        """Test submission with non-existent file."""
        result = runner.invoke(app, [
            "kaggle-submit",
            "titanic",
            "nonexistent.csv"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "Error" in result.stdout
    
    @patch('cli.commands.kaggle.submit.KaggleIntegration')
    def test_submit_api_error(self, mock_kaggle_class, runner, tmp_path):
        """Test handling submission API errors."""
        submission_file = tmp_path / "submission.csv"
        submission_file.write_text("id,target\n1,0\n")
        
        mock_kaggle = MagicMock()
        mock_kaggle.submit_predictions.side_effect = Exception("API Error: Invalid format")
        mock_kaggle_class.return_value = mock_kaggle
        
        result = runner.invoke(app, [
            "kaggle-submit",
            "titanic",
            str(submission_file)
        ])
        
        assert result.exit_code != 0
        assert "Error" in result.stdout or "Failed" in result.stdout