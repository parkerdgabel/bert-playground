"""Integration tests for competition commands."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
import yaml
from typer.testing import CliRunner

from cli.app import app


class TestCompetitionIntegration:
    """Integration tests for competition commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_kaggle_api(self):
        """Mock Kaggle API."""
        with patch('kaggle.api') as mock_api:
            # Mock competition list
            mock_api.competition_list.return_value = [
                MagicMock(id="titanic", title="Titanic - Machine Learning from Disaster"),
                MagicMock(id="house-prices", title="House Prices - Advanced Regression"),
                MagicMock(id="nlp-getting-started", title="Real or Not? NLP with Disaster Tweets")
            ]
            
            # Mock competition info
            def mock_competition_list_files(competition):
                if competition == "titanic":
                    return [
                        MagicMock(name="train.csv", size=61194),
                        MagicMock(name="test.csv", size=28629),
                        MagicMock(name="gender_submission.csv", size=3258)
                    ]
                return []
            
            mock_api.competition_list_files.side_effect = mock_competition_list_files
            
            # Mock submission result
            mock_api.competition_submit.return_value = MagicMock(
                ref="12345",
                message="Successfully submitted to titanic"
            )
            
            # Mock leaderboard
            mock_api.competition_leaderboard_view.return_value = [
                {"teamName": "Team1", "score": "0.85432"},
                {"teamName": "test_user", "score": "0.82156"}
            ]
            
            yield mock_api

    @pytest.fixture
    def project_with_competition(self, tmp_path):
        """Create project with competition config."""
        config = {
            "name": "titanic-solution",
            "competition": "titanic",
            "data": {
                "train_path": "data/train.csv",
                "test_path": "data/test.csv"
            }
        }
        
        with open(tmp_path / "k-bert.yaml", "w") as f:
            yaml.dump(config, f)
        
        return tmp_path

    def test_competition_list(self, runner, mock_kaggle_api):
        """Test listing competitions."""
        result = runner.invoke(app, ["competition", "list"])
        
        assert result.exit_code == 0
        assert "titanic" in result.stdout
        assert "Titanic - Machine Learning from Disaster" in result.stdout
        assert "house-prices" in result.stdout
        mock_kaggle_api.competition_list.assert_called_once()

    def test_competition_list_with_search(self, runner, mock_kaggle_api):
        """Test listing competitions with search."""
        result = runner.invoke(app, ["competition", "list", "--search", "nlp"])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_list.assert_called_with(search="nlp")

    def test_competition_list_active_only(self, runner, mock_kaggle_api):
        """Test listing only active competitions."""
        result = runner.invoke(app, ["competition", "list", "--active"])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_list.assert_called_with(group="entered")

    def test_competition_info(self, runner, mock_kaggle_api):
        """Test getting competition info."""
        result = runner.invoke(app, ["competition", "info", "titanic"])
        
        assert result.exit_code == 0
        assert "titanic" in result.stdout
        assert "train.csv" in result.stdout
        assert "61194" in result.stdout  # File size
        mock_kaggle_api.competition_list_files.assert_called_with("titanic")

    def test_competition_download(self, runner, mock_kaggle_api, tmp_path):
        """Test downloading competition data."""
        os.chdir(tmp_path)
        
        result = runner.invoke(app, ["competition", "download", "titanic"])
        
        assert result.exit_code == 0
        assert "Downloading competition: titanic" in result.stdout
        mock_kaggle_api.competition_download_files.assert_called_with(
            "titanic", 
            path=Path("data")
        )

    def test_competition_download_custom_path(self, runner, mock_kaggle_api, tmp_path):
        """Test downloading to custom path."""
        os.chdir(tmp_path)
        custom_path = "custom/data"
        
        result = runner.invoke(app, ["competition", "download", "titanic", "--path", custom_path])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_download_files.assert_called_with(
            "titanic",
            path=Path(custom_path)
        )

    def test_competition_download_specific_file(self, runner, mock_kaggle_api, tmp_path):
        """Test downloading specific file."""
        os.chdir(tmp_path)
        
        result = runner.invoke(app, ["competition", "download", "titanic", "--file", "train.csv"])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_download_file.assert_called_with(
            "titanic",
            "train.csv",
            path=Path("data")
        )

    def test_competition_submit(self, runner, mock_kaggle_api, tmp_path):
        """Test submitting to competition."""
        os.chdir(tmp_path)
        
        # Create submission file
        submission_file = tmp_path / "submission.csv"
        submission_file.write_text("PassengerId,Survived\n892,0\n893,1")
        
        result = runner.invoke(app, ["competition", "submit", "titanic", str(submission_file), "-m", "Test submission"])
        
        assert result.exit_code == 0
        assert "Successfully submitted" in result.stdout
        mock_kaggle_api.competition_submit.assert_called_once()

    def test_competition_submit_from_project(self, runner, mock_kaggle_api, project_with_competition):
        """Test submitting using project competition setting."""
        os.chdir(project_with_competition)
        
        submission_file = project_with_competition / "submission.csv"
        submission_file.write_text("PassengerId,Survived\n892,0")
        
        # Should use competition from project config
        result = runner.invoke(app, ["competition", "submit", str(submission_file)])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_submit.assert_called_with(
            "titanic",
            str(submission_file),
            message=None
        )

    def test_competition_submit_no_competition(self, runner, tmp_path):
        """Test submit without specifying competition."""
        os.chdir(tmp_path)
        submission_file = tmp_path / "submission.csv"
        submission_file.write_text("id,target\n1,0")
        
        result = runner.invoke(app, ["competition", "submit", str(submission_file)])
        
        assert result.exit_code == 1
        assert "No competition specified" in result.stdout

    def test_competition_init(self, runner, mock_kaggle_api, tmp_path):
        """Test initializing competition project."""
        os.chdir(tmp_path)
        
        with patch('cli.commands.competition.init._create_competition_config') as mock_create:
            result = runner.invoke(app, ["competition", "init", "titanic"])
            
            assert result.exit_code == 0
            assert "Initialized titanic competition" in result.stdout
            mock_create.assert_called_once()

    def test_competition_init_with_download(self, runner, mock_kaggle_api, tmp_path):
        """Test init with automatic download."""
        os.chdir(tmp_path)
        
        with patch('cli.commands.competition.init._create_competition_config'):
            result = runner.invoke(app, ["competition", "init", "titanic", "--download"])
            
            assert result.exit_code == 0
            mock_kaggle_api.competition_download_files.assert_called_with(
                "titanic",
                path=Path("data")
            )

    def test_competition_track(self, runner, mock_kaggle_api, project_with_competition):
        """Test tracking competition status."""
        os.chdir(project_with_competition)
        
        result = runner.invoke(app, ["competition", "track"])
        
        assert result.exit_code == 0
        assert "Competition: titanic" in result.stdout
        assert "Your best score:" in result.stdout
        mock_kaggle_api.competition_submissions_list.assert_called_with("titanic")

    def test_competition_track_specific(self, runner, mock_kaggle_api):
        """Test tracking specific competition."""
        result = runner.invoke(app, ["competition", "track", "titanic"])
        
        assert result.exit_code == 0
        mock_kaggle_api.competition_submissions_list.assert_called_with("titanic")

    def test_competition_leaderboard(self, runner, mock_kaggle_api):
        """Test viewing competition leaderboard."""
        result = runner.invoke(app, ["competition", "leaderboard", "titanic"])
        
        assert result.exit_code == 0
        assert "Team1" in result.stdout
        assert "0.85432" in result.stdout
        mock_kaggle_api.competition_leaderboard_view.assert_called_with("titanic")

    def test_competition_leaderboard_top_n(self, runner, mock_kaggle_api):
        """Test viewing top N leaderboard entries."""
        result = runner.invoke(app, ["competition", "leaderboard", "titanic", "--top", "10"])
        
        assert result.exit_code == 0
        # Check that output is limited to top 10
        mock_kaggle_api.competition_leaderboard_view.assert_called_with("titanic")

    def test_competition_workflow_integration(self, runner, mock_kaggle_api, tmp_path):
        """Test complete competition workflow."""
        os.chdir(tmp_path)
        
        # Initialize competition
        with patch('cli.commands.competition.init._create_competition_config'):
            result = runner.invoke(app, ["competition", "init", "titanic", "--download"])
            assert result.exit_code == 0
        
        # Create dummy data files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "train.csv").write_text("PassengerId,Survived\n1,0")
        (data_dir / "test.csv").write_text("PassengerId\n892")
        
        # Submit
        submission = tmp_path / "submission.csv"
        submission.write_text("PassengerId,Survived\n892,0")
        
        result = runner.invoke(app, ["competition", "submit", "titanic", str(submission)])
        assert result.exit_code == 0
        
        # Track progress
        result = runner.invoke(app, ["competition", "track", "titanic"])
        assert result.exit_code == 0

    def test_competition_error_handling(self, runner, mock_kaggle_api):
        """Test error handling in competition commands."""
        # Mock API error
        mock_kaggle_api.competition_download_files.side_effect = Exception("API Error")
        
        result = runner.invoke(app, ["competition", "download", "invalid-comp"])
        
        assert result.exit_code == 1
        assert "Error" in result.stdout or "Failed" in result.stdout

    def test_competition_authentication_required(self, runner):
        """Test that commands require authentication."""
        with patch('kaggle.api', side_effect=ImportError("No module named 'kaggle'")):
            result = runner.invoke(app, ["competition", "list"])
            
            assert result.exit_code == 1
            assert "Kaggle API" in result.stdout or "authentication" in result.stdout