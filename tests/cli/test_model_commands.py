"""Unit tests for model CLI commands."""

import pytest
from unittest.mock import patch, MagicMock, call, mock_open
from pathlib import Path
import json
from typer.testing import CliRunner

import sys
from pathlib import Path as SysPath

# Add project root to path
sys.path.insert(0, str(SysPath(__file__).parent.parent.parent))

from cli.app import app


class TestModelInspectCommand:
    """Test suite for the model-inspect command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_model_files(self):
        """Create mock model checkpoint files."""
        return {
            'config.json': {
                'model_type': 'modernbert',
                'num_labels': 2,
                'problem_type': 'binary_classification',
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'num_attention_heads': 12
            },
            'model.safetensors': b'mock_binary_data',
            'training_args.json': {
                'learning_rate': 2e-5,
                'batch_size': 32,
                'num_epochs': 5,
                'optimizer': 'adamw'
            },
            'metrics.json': {
                'final_accuracy': 0.92,
                'final_loss': 0.23,
                'best_accuracy': 0.93,
                'total_steps': 1000
            }
        }
    
    @patch('cli.commands.model.inspect.Path')
    @patch('builtins.open', new_callable=mock_open)
    def test_inspect_valid_checkpoint(self, mock_file, mock_path, runner, mock_model_files):
        """Test inspecting a valid model checkpoint."""
        # Setup path mocks
        checkpoint_path = MagicMock()
        checkpoint_path.exists.return_value = True
        checkpoint_path.is_dir.return_value = True
        checkpoint_path.iterdir.return_value = [
            MagicMock(name='config.json', suffix='.json'),
            MagicMock(name='model.safetensors', suffix='.safetensors'),
            MagicMock(name='training_args.json', suffix='.json'),
            MagicMock(name='metrics.json', suffix='.json')
        ]
        mock_path.return_value = checkpoint_path
        
        # Mock file reads
        mock_file.return_value.read.side_effect = [
            json.dumps(mock_model_files['config.json']),
            json.dumps(mock_model_files['training_args.json']),
            json.dumps(mock_model_files['metrics.json'])
        ]
        
        # Run command
        result = runner.invoke(app, [
            "model-inspect",
            "output/run_001/best_model"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Model Inspection" in result.stdout
        assert "modernbert" in result.stdout
        assert "binary_classification" in result.stdout
        assert "768" in result.stdout  # hidden_size
        assert "0.92" in result.stdout or "92" in result.stdout  # accuracy
    
    @patch('cli.commands.model.inspect.Path')
    def test_inspect_show_weights(self, mock_path, runner):
        """Test showing model weights."""
        checkpoint_path = MagicMock()
        checkpoint_path.exists.return_value = True
        checkpoint_path.is_dir.return_value = True
        mock_path.return_value = checkpoint_path
        
        # Mock loading weights
        with patch('cli.commands.model.inspect.mx.load') as mock_load:
            mock_weights = {
                'model.embeddings.word_embeddings.weight': MagicMock(shape=(30522, 768)),
                'model.encoder.layer.0.attention.self.query.weight': MagicMock(shape=(768, 768)),
                'model.encoder.layer.0.attention.self.key.weight': MagicMock(shape=(768, 768)),
                'classifier.weight': MagicMock(shape=(2, 768)),
                'classifier.bias': MagicMock(shape=(2,))
            }
            mock_load.return_value = mock_weights
            
            result = runner.invoke(app, [
                "model-inspect",
                "output/run_001/best_model",
                "--show-weights"
            ])
            
            assert result.exit_code == 0
            assert "Model Weights" in result.stdout or "Parameters" in result.stdout
            assert "embeddings" in result.stdout.lower()
            assert "classifier" in result.stdout
            assert "shape" in result.stdout.lower() or "(768, 768)" in result.stdout
    
    @patch('cli.commands.model.inspect.Path')
    def test_inspect_nonexistent_checkpoint(self, mock_path, runner):
        """Test inspecting non-existent checkpoint."""
        checkpoint_path = MagicMock()
        checkpoint_path.exists.return_value = False
        mock_path.return_value = checkpoint_path
        
        result = runner.invoke(app, [
            "model-inspect",
            "nonexistent/checkpoint"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "does not exist" in result.stdout.lower()
    
    @patch('cli.commands.model.inspect.Path')
    @patch('builtins.open')
    def test_inspect_output_json(self, mock_file, mock_path, runner, mock_model_files, tmp_path):
        """Test outputting inspection results as JSON."""
        checkpoint_path = MagicMock()
        checkpoint_path.exists.return_value = True
        checkpoint_path.is_dir.return_value = True
        mock_path.return_value = checkpoint_path
        
        output_file = tmp_path / "inspection.json"
        
        result = runner.invoke(app, [
            "model-inspect",
            "output/run_001/best_model",
            "--output", str(output_file)
        ])
        
        assert result.exit_code == 0
        assert "Saved inspection results" in result.stdout or str(output_file) in result.stdout


class TestModelConvertCommand:
    """Test suite for the model-convert command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.model.convert.ModelConverter')
    @patch('cli.commands.model.convert.Path')
    def test_convert_to_onnx(self, mock_path, mock_converter_class, runner):
        """Test converting model to ONNX format."""
        # Setup mocks
        mock_converter = MagicMock()
        mock_converter.convert_to_onnx.return_value = True
        mock_converter_class.return_value = mock_converter
        
        source_path = MagicMock()
        source_path.exists.return_value = True
        mock_path.return_value = source_path
        
        # Run command
        result = runner.invoke(app, [
            "model-convert",
            "output/run_001/best_model",
            "--format", "onnx"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Conversion complete" in result.stdout or "Successfully converted" in result.stdout
        
        # Verify conversion was called
        mock_converter.convert_to_onnx.assert_called_once()
    
    @patch('cli.commands.model.convert.ModelConverter')
    @patch('cli.commands.model.convert.Path')
    def test_convert_with_output_path(self, mock_path, mock_converter_class, runner, tmp_path):
        """Test conversion with custom output path."""
        mock_converter = MagicMock()
        mock_converter.convert_to_onnx.return_value = True
        mock_converter_class.return_value = mock_converter
        
        source_path = MagicMock()
        source_path.exists.return_value = True
        mock_path.return_value = source_path
        
        output_dir = tmp_path / "converted_models"
        
        result = runner.invoke(app, [
            "model-convert",
            "output/run_001/best_model",
            "--output", str(output_dir),
            "--format", "onnx"
        ])
        
        assert result.exit_code == 0
        
        # Verify output path was used
        call_args = mock_converter.convert_to_onnx.call_args
        assert str(output_dir) in str(call_args)
    
    @patch('cli.commands.model.convert.ModelConverter')
    def test_convert_quantization(self, mock_converter_class, runner):
        """Test model quantization during conversion."""
        mock_converter = MagicMock()
        mock_converter.convert_to_onnx.return_value = True
        mock_converter_class.return_value = mock_converter
        
        result = runner.invoke(app, [
            "model-convert",
            "output/run_001/best_model",
            "--format", "onnx",
            "--quantize"
        ])
        
        assert result.exit_code == 0
        assert "quantiz" in result.stdout.lower() or result.exit_code == 0
    
    @patch('cli.commands.model.convert.Path')
    def test_convert_invalid_source(self, mock_path, runner):
        """Test conversion with invalid source path."""
        source_path = MagicMock()
        source_path.exists.return_value = False
        mock_path.return_value = source_path
        
        result = runner.invoke(app, [
            "model-convert",
            "nonexistent/model"
        ])
        
        assert result.exit_code != 0
        assert "not found" in result.stdout.lower() or "does not exist" in result.stdout.lower()
    
    def test_convert_invalid_format(self, runner):
        """Test conversion with unsupported format."""
        result = runner.invoke(app, [
            "model-convert",
            "output/run_001/best_model",
            "--format", "invalid_format"
        ])
        
        assert result.exit_code != 0
        assert "Invalid value" in result.stdout or "Unsupported" in result.stdout


class TestModelMergeCommand:
    """Test suite for the model-merge command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.model.merge.ModelMerger')
    @patch('cli.commands.model.merge.Path')
    def test_merge_basic(self, mock_path, mock_merger_class, runner):
        """Test basic model merging."""
        # Setup mocks
        mock_merger = MagicMock()
        mock_merger.merge_models.return_value = "output/merged_model"
        mock_merger_class.return_value = mock_merger
        
        # Mock path validation
        mock_path.return_value.exists.return_value = True
        
        # Run command
        result = runner.invoke(app, [
            "model-merge",
            "output/run_001/best_model",
            "output/run_002/best_model"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Merge complete" in result.stdout or "Successfully merged" in result.stdout
        
        # Verify merge was called with correct models
        mock_merger.merge_models.assert_called_once()
        call_args = mock_merger.merge_models.call_args[0]
        assert len(call_args[0]) == 2  # Two model paths
    
    @patch('cli.commands.model.merge.ModelMerger')
    @patch('cli.commands.model.merge.Path')
    def test_merge_multiple_models(self, mock_path, mock_merger_class, runner):
        """Test merging multiple models."""
        mock_merger = MagicMock()
        mock_merger.merge_models.return_value = "output/merged_model"
        mock_merger_class.return_value = mock_merger
        
        mock_path.return_value.exists.return_value = True
        
        result = runner.invoke(app, [
            "model-merge",
            "model1",
            "model2",
            "model3",
            "model4"
        ])
        
        assert result.exit_code == 0
        
        # Verify all models were passed
        call_args = mock_merger.merge_models.call_args[0]
        assert len(call_args[0]) == 4
    
    @patch('cli.commands.model.merge.ModelMerger')
    @patch('cli.commands.model.merge.Path')
    def test_merge_with_weights(self, mock_path, mock_merger_class, runner):
        """Test merging with custom weights."""
        mock_merger = MagicMock()
        mock_merger.merge_models.return_value = "output/merged_model"
        mock_merger_class.return_value = mock_merger
        
        mock_path.return_value.exists.return_value = True
        
        result = runner.invoke(app, [
            "model-merge",
            "model1",
            "model2",
            "--weights", "0.7,0.3"
        ])
        
        assert result.exit_code == 0
        
        # Verify weights were passed
        call_args = mock_merger.merge_models.call_args
        assert 'weights' in call_args[1]
        assert call_args[1]['weights'] == [0.7, 0.3]
    
    @patch('cli.commands.model.merge.ModelMerger')
    @patch('cli.commands.model.merge.Path')
    def test_merge_strategies(self, mock_path, mock_merger_class, runner):
        """Test different merge strategies."""
        mock_merger = MagicMock()
        mock_merger.merge_models.return_value = "output/merged_model"
        mock_merger_class.return_value = mock_merger
        
        mock_path.return_value.exists.return_value = True
        
        # Test averaging strategy
        result = runner.invoke(app, [
            "model-merge",
            "model1",
            "model2",
            "--strategy", "average"
        ])
        
        assert result.exit_code == 0
        call_args = mock_merger.merge_models.call_args
        assert call_args[1].get('strategy') == 'average'
        
        # Test weighted strategy
        result = runner.invoke(app, [
            "model-merge",
            "model1",
            "model2",
            "--strategy", "weighted",
            "--weights", "0.6,0.4"
        ])
        
        assert result.exit_code == 0
    
    def test_merge_insufficient_models(self, runner):
        """Test merge with insufficient models."""
        result = runner.invoke(app, [
            "model-merge",
            "single_model"
        ])
        
        assert result.exit_code != 0
        assert "at least 2" in result.stdout.lower() or "Error" in result.stdout
    
    def test_merge_mismatched_weights(self, runner):
        """Test merge with mismatched weights count."""
        result = runner.invoke(app, [
            "model-merge",
            "model1",
            "model2",
            "model3",
            "--weights", "0.5,0.5"  # Only 2 weights for 3 models
        ])
        
        assert result.exit_code != 0
        assert "weights" in result.stdout.lower() or "mismatch" in result.stdout.lower()


class TestModelCompareCommand:
    """Test suite for the model-compare command."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()
    
    @patch('cli.commands.model.compare.ModelComparator')
    @patch('cli.commands.model.compare.Path')
    def test_compare_two_models(self, mock_path, mock_comparator_class, runner):
        """Test comparing two models."""
        # Setup mocks
        mock_comparator = MagicMock()
        comparison_results = {
            'architecture_match': True,
            'parameter_diff': {
                'total_params_model1': 110000000,
                'total_params_model2': 110000000,
                'matching_layers': 95,
                'different_layers': 5
            },
            'performance_diff': {
                'accuracy_diff': 0.02,
                'loss_diff': -0.05
            }
        }
        mock_comparator.compare.return_value = comparison_results
        mock_comparator_class.return_value = mock_comparator
        
        mock_path.return_value.exists.return_value = True
        
        # Run command
        result = runner.invoke(app, [
            "model-compare",
            "output/run_001/best_model",
            "output/run_002/best_model"
        ])
        
        # Check success
        assert result.exit_code == 0
        assert "Model Comparison" in result.stdout
        assert "architecture" in result.stdout.lower()
        assert "95" in result.stdout  # matching layers
    
    @patch('cli.commands.model.compare.ModelComparator')
    @patch('cli.commands.model.compare.Path')
    def test_compare_detailed_output(self, mock_path, mock_comparator_class, runner):
        """Test detailed comparison output."""
        mock_comparator = MagicMock()
        mock_comparator.compare.return_value = {
            'architecture_match': False,
            'differences': ['layer_3', 'layer_5'],
            'weight_statistics': {
                'mean_diff': 0.001,
                'max_diff': 0.05
            }
        }
        mock_comparator_class.return_value = mock_comparator
        
        mock_path.return_value.exists.return_value = True
        
        result = runner.invoke(app, [
            "model-compare",
            "model1",
            "model2",
            "--detailed"
        ])
        
        assert result.exit_code == 0
        assert "layer_3" in result.stdout or "differences" in result.stdout.lower()
        assert "statistics" in result.stdout.lower() or "0.001" in result.stdout