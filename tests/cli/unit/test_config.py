"""Unit tests for configuration utilities."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import yaml

from cli.utils.config import (
    _deep_merge,
    _process_includes,
    _substitute_env_vars,
    get_default_config_path,
    load_config,
    save_config,
    validate_config,
)


class TestGetDefaultConfigPath:
    """Test default config path resolution."""

    def test_env_variable_override(self):
        """Test config path from environment variable."""
        with patch.dict(os.environ, {"BERT_CONFIG_PATH": "/custom/path/config.yaml"}):
            path = get_default_config_path()
            assert path == Path("/custom/path/config.yaml")

    def test_find_bert_yaml(self, tmp_path):
        """Test finding bert.yaml in current directory."""
        (tmp_path / "bert.yaml").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == tmp_path / "bert.yaml"

    def test_find_bert_yml(self, tmp_path):
        """Test finding bert.yml in current directory."""
        (tmp_path / "bert.yml").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == tmp_path / "bert.yml"

    def test_find_bert_json(self, tmp_path):
        """Test finding bert.json in current directory."""
        (tmp_path / "bert.json").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == tmp_path / "bert.json"

    def test_find_bertrc(self, tmp_path):
        """Test finding .bertrc in current directory."""
        (tmp_path / ".bertrc").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == tmp_path / ".bertrc"

    def test_find_in_configs_dir(self, tmp_path):
        """Test finding config in configs directory."""
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        (configs_dir / "default.yaml").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == configs_dir / "default.yaml"

    def test_default_fallback(self, tmp_path):
        """Test fallback to bert.yaml when nothing found."""
        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            path = get_default_config_path()
            assert path == tmp_path / "bert.yaml"

    def test_priority_order(self, tmp_path):
        """Test config file priority order."""
        # Create multiple config files
        (tmp_path / "bert.yaml").touch()
        (tmp_path / "bert.yml").touch()
        (tmp_path / "bert.json").touch()

        with patch("pathlib.Path.cwd") as mock_cwd:
            mock_cwd.return_value = tmp_path
            # Should prefer bert.yaml
            path = get_default_config_path()
            assert path == tmp_path / "bert.yaml"


class TestLoadConfig:
    """Test configuration loading."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config_data = {
            "model": {"type": "bert", "hidden_size": 768},
            "training": {"epochs": 5, "batch_size": 32},
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch("cli.utils.config.logger"):
            config = load_config(config_file)
            assert config == config_data

    def test_load_json_config(self, tmp_path):
        """Test loading JSON configuration."""
        config_data = {
            "model": {"type": "bert", "hidden_size": 768},
            "training": {"epochs": 5, "batch_size": 32},
        }

        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch("cli.utils.config.logger"):
            config = load_config(config_file)
            assert config == config_data

    def test_load_nonexistent_config(self, tmp_path):
        """Test loading non-existent config returns empty dict."""
        config_file = tmp_path / "nonexistent.yaml"

        with patch("cli.utils.config.logger") as mock_logger:
            config = load_config(config_file)
            assert config == {}
            mock_logger.debug.assert_called_once()

    def test_load_config_no_extension(self, tmp_path):
        """Test loading config file without extension."""
        config_data = {"test": "value"}

        # Test with YAML content
        config_file = tmp_path / "config"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch("cli.utils.config.logger"):
            config = load_config(config_file)
            assert config == config_data

        # Test with JSON content
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with patch("cli.utils.config.logger"):
            config = load_config(config_file)
            assert config == config_data

    def test_load_with_env_substitution(self, tmp_path):
        """Test loading config with environment variable substitution."""
        config_data = {
            "database": {
                "host": "${DB_HOST}",
                "port": "$DB_PORT",
                "name": "${DB_NAME:-mydb}",
            }
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with patch.dict(os.environ, {"DB_HOST": "localhost", "DB_PORT": "5432"}):
            with patch("cli.utils.config.logger"):
                config = load_config(config_file)

                assert config["database"]["host"] == "localhost"
                assert config["database"]["port"] == "5432"
                assert config["database"]["name"] == "mydb"  # Default value

    def test_load_with_includes(self, tmp_path):
        """Test loading config with include files."""
        # Create base config
        base_config = {"model": {"type": "bert"}, "training": {"epochs": 5}}
        base_file = tmp_path / "base.yaml"
        with open(base_file, "w") as f:
            yaml.dump(base_config, f)

        # Create main config with include
        main_config = {"include": ["base.yaml"], "training": {"batch_size": 32}}
        main_file = tmp_path / "main.yaml"
        with open(main_file, "w") as f:
            yaml.dump(main_config, f)

        with patch("cli.utils.config.logger"):
            config = load_config(main_file)

            # Should have merged config
            assert config["model"]["type"] == "bert"
            assert config["training"]["epochs"] == 5
            assert config["training"]["batch_size"] == 32


class TestSubstituteEnvVars:
    """Test environment variable substitution."""

    def test_substitute_simple_var(self):
        """Test simple variable substitution."""
        with patch.dict(os.environ, {"MY_VAR": "test_value"}):
            result = _substitute_env_vars("${MY_VAR}")
            assert result == "test_value"

            result = _substitute_env_vars("$MY_VAR")
            assert result == "test_value"

    def test_substitute_with_default(self):
        """Test substitution with default value."""
        result = _substitute_env_vars("${MISSING_VAR:-default_value}")
        assert result == "default_value"

        with patch.dict(os.environ, {"EXISTING_VAR": "actual_value"}):
            result = _substitute_env_vars("${EXISTING_VAR:-default_value}")
            assert result == "actual_value"

    def test_substitute_in_dict(self):
        """Test substitution in dictionary."""
        config = {
            "host": "${HOST}",
            "port": "$PORT",
            "database": {"name": "${DB_NAME:-testdb}", "user": "$DB_USER"},
        }

        with patch.dict(
            os.environ, {"HOST": "localhost", "PORT": "8080", "DB_USER": "admin"}
        ):
            with patch("cli.utils.config.logger"):
                result = _substitute_env_vars(config)

                assert result["host"] == "localhost"
                assert result["port"] == "8080"
                assert result["database"]["name"] == "testdb"
                assert result["database"]["user"] == "admin"

    def test_substitute_in_list(self):
        """Test substitution in list."""
        config = ["$VAR1", "${VAR2}", "static", "${VAR3:-default}"]

        with patch.dict(os.environ, {"VAR1": "value1", "VAR2": "value2"}):
            with patch("cli.utils.config.logger"):
                result = _substitute_env_vars(config)

                assert result == ["value1", "value2", "static", "default"]

    def test_missing_var_warning(self):
        """Test warning for missing environment variable."""
        with patch("cli.utils.config.logger") as mock_logger:
            result = _substitute_env_vars("${MISSING_VAR}")
            assert result == "${MISSING_VAR}"  # Original string
            mock_logger.warning.assert_called_once()

    def test_preserve_non_vars(self):
        """Test non-variable strings are preserved."""
        result = _substitute_env_vars("This is a $test string")
        assert result == "This is a $test string"

        result = _substitute_env_vars("Price: $100")
        assert result == "Price: $100"


class TestDeepMerge:
    """Test deep merge functionality."""

    def test_simple_merge(self):
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        """Test nested dictionary merge."""
        base = {
            "model": {"type": "bert", "hidden_size": 768},
            "training": {"epochs": 5},
        }
        override = {"model": {"hidden_size": 512}, "training": {"batch_size": 32}}

        result = _deep_merge(base, override)
        assert result == {
            "model": {"type": "bert", "hidden_size": 512},
            "training": {"epochs": 5, "batch_size": 32},
        }

    def test_override_non_dict(self):
        """Test overriding non-dict values."""
        base = {"a": {"b": 1}, "c": [1, 2, 3]}
        override = {"a": "string", "c": [4, 5]}

        result = _deep_merge(base, override)
        assert result == {"a": "string", "c": [4, 5]}

    def test_empty_dicts(self):
        """Test merging with empty dictionaries."""
        assert _deep_merge({}, {"a": 1}) == {"a": 1}
        assert _deep_merge({"a": 1}, {}) == {"a": 1}
        assert _deep_merge({}, {}) == {}


class TestProcessIncludes:
    """Test include file processing."""

    def test_single_include(self, tmp_path):
        """Test processing single include file."""
        # Create include file
        include_config = {"model": {"type": "bert"}}
        include_file = tmp_path / "include.yaml"
        with open(include_file, "w") as f:
            yaml.dump(include_config, f)

        # Main config with include
        config = {"include": "include.yaml", "training": {"epochs": 5}}

        with patch("cli.utils.config.logger"):
            result = _process_includes(config, tmp_path)

            assert "include" not in result  # Include directive removed
            assert result["model"]["type"] == "bert"
            assert result["training"]["epochs"] == 5

    def test_multiple_includes(self, tmp_path):
        """Test processing multiple include files."""
        # Create include files
        include1 = {"model": {"type": "bert"}}
        (tmp_path / "include1.yaml").write_text(yaml.dump(include1))

        include2 = {"training": {"epochs": 5}}
        (tmp_path / "include2.yaml").write_text(yaml.dump(include2))

        # Main config with includes
        config = {"include": ["include1.yaml", "include2.yaml"]}

        with patch("cli.utils.config.logger"):
            result = _process_includes(config, tmp_path)

            assert result["model"]["type"] == "bert"
            assert result["training"]["epochs"] == 5

    def test_missing_include(self, tmp_path):
        """Test handling missing include file."""
        config = {"include": "missing.yaml", "test": "value"}

        with patch("cli.utils.config.logger") as mock_logger:
            result = _process_includes(config, tmp_path)

            assert result == {"test": "value"}
            mock_logger.warning.assert_called_once()


class TestValidateConfig:
    """Test configuration validation."""

    def test_validate_with_required_fields(self):
        """Test validation with required fields."""
        config = {"model": {"type": "bert"}, "training": {"epochs": 5}}
        schema = {"required": ["model", "training"]}

        with patch("cli.utils.config.logger"):
            assert validate_config(config, schema) is True

    def test_validate_missing_required(self):
        """Test validation with missing required fields."""
        config = {"model": {"type": "bert"}}
        schema = {"required": ["model", "training"]}

        with patch("cli.utils.config.logger") as mock_logger:
            assert validate_config(config, schema) is False
            mock_logger.error.assert_called_once()

    def test_validate_with_default_schema(self):
        """Test validation with default schema."""
        config = {"test": "value"}

        with patch("cli.utils.config.get_default_schema") as mock_get_schema:
            mock_get_schema.return_value = {"required": []}

            with patch("cli.utils.config.logger"):
                assert validate_config(config) is True
                mock_get_schema.assert_called_once()


class TestSaveConfig:
    """Test configuration saving."""

    def test_save_yaml_config(self, tmp_path):
        """Test saving configuration as YAML."""
        config = {"model": {"type": "bert"}, "training": {"epochs": 5}}
        config_file = tmp_path / "output.yaml"

        with patch("cli.utils.config.logger"):
            save_config(config, config_file)

        assert config_file.exists()
        with open(config_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_save_json_config(self, tmp_path):
        """Test saving configuration as JSON."""
        config = {"model": {"type": "bert"}, "training": {"epochs": 5}}
        config_file = tmp_path / "output.json"

        with patch("cli.utils.config.logger"):
            save_config(config, config_file)

        assert config_file.exists()
        with open(config_file) as f:
            loaded = json.load(f)
        assert loaded == config

    def test_save_creates_parent_dirs(self, tmp_path):
        """Test saving creates parent directories."""
        config = {"test": "value"}
        config_file = tmp_path / "nested" / "dirs" / "config.yaml"

        with patch("cli.utils.config.logger"):
            save_config(config, config_file)

        assert config_file.exists()
        assert config_file.parent.exists()
