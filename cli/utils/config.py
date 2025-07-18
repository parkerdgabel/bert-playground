"""Configuration utilities for CLI."""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger

def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    # Check environment variable first
    if env_path := os.environ.get("BERT_CONFIG_PATH"):
        return Path(env_path)
    
    # Check common locations
    for config_name in ["bert.yaml", "bert.yml", "bert.json", ".bertrc"]:
        if (config_file := Path.cwd() / config_name).exists():
            return config_file
    
    # Check configs directory
    if (configs_dir := Path.cwd() / "configs").exists():
        for config_name in ["default.yaml", "default.yml", "default.json"]:
            if (config_file := configs_dir / config_name).exists():
                return config_file
    
    return Path.cwd() / "bert.yaml"

def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from file with environment variable substitution."""
    if config_path is None:
        config_path = get_default_config_path()
    
    if not config_path.exists():
        logger.debug(f"No config file found at {config_path}, using defaults")
        return {}
    
    logger.info(f"Loading configuration from {config_path}")
    
    # Load based on file extension
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            # Try YAML first, then JSON
            content = f.read()
            try:
                config = yaml.safe_load(content)
            except:
                config = json.loads(content)
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    # Process includes
    if 'include' in config:
        config = _process_includes(config, config_path.parent)
    
    return config

def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute environment variables in config."""
    if isinstance(obj, str):
        # Replace ${VAR} or $VAR with environment variable
        import re
        pattern = r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def replacer(match):
            var_name = match.group(1) or match.group(2)
            default = None
            
            # Handle ${VAR:-default} syntax
            if ':' in var_name:
                var_name, default = var_name.split(':', 1)
                if default.startswith('-'):
                    default = default[1:]
            
            value = os.environ.get(var_name, default)
            if value is None:
                logger.warning(f"Environment variable {var_name} not found")
                return match.group(0)
            return value
        
        return re.sub(pattern, replacer, obj)
    
    elif isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    
    return obj

def _process_includes(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Process !include directives in config."""
    includes = config.pop('include', [])
    if isinstance(includes, str):
        includes = [includes]
    
    # Load all includes
    for include_path in includes:
        include_file = base_dir / include_path
        if include_file.exists():
            include_config = load_config(include_file)
            # Deep merge with current config
            config = _deep_merge(include_config, config)
        else:
            logger.warning(f"Include file not found: {include_file}")
    
    return config

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def validate_config(config: Dict[str, Any], schema: Optional[Dict[str, Any]] = None) -> bool:
    """Validate configuration against schema."""
    if schema is None:
        # Use default schema
        from .schemas import get_default_schema
        schema = get_default_schema()
    
    # TODO: Implement schema validation using jsonschema or similar
    # For now, just check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in config:
            logger.error(f"Required field '{field}' missing from configuration")
            return False
    
    return True

def save_config(config: Dict[str, Any], path: Path):
    """Save configuration to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {path}")