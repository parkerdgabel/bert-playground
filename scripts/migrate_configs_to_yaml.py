#!/usr/bin/env python3
"""Migrate JSON configuration files to YAML format.

This script converts existing JSON configuration files to YAML format,
preserving all settings and adding helpful comments.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader


def add_comments_to_yaml(yaml_path: Path, config_name: str) -> None:
    """Add helpful comments to the generated YAML file."""
    
    # Read the YAML content
    with open(yaml_path, 'r') as f:
        lines = f.readlines()
    
    # Prepare header comments based on config name
    headers = {
        'production': [
            "# Production Training Configuration\n",
            "# Standard settings for production model training\n",
            "\n"
        ],
        'titanic_production': [
            "# Titanic Production Configuration\n",
            "# Optimized settings for Titanic dataset classification\n",
            "\n"
        ],
        'default': [
            "# Training Configuration\n",
            "# Converted from JSON format\n",
            "\n"
        ]
    }
    
    # Get appropriate header
    header = headers.get(config_name.lower(), headers['default'])
    
    # Add section comments
    section_comments = {
        'model_name:': '# Model configuration',
        'batch_size:': '\n# Training hyperparameters',
        'memory:': '\n# Memory optimization settings',
        'mlx_optimization:': '\n# MLX-specific optimizations',
        'monitoring:': '\n# Monitoring and logging',
        'checkpoint:': '\n# Checkpointing strategy',
        'evaluation:': '\n# Evaluation configuration',
        'train_path:': '\n# Data paths',
        'output_dir:': '\n# Output configuration',
    }
    
    # Process lines and add comments
    output_lines = header.copy()
    
    for line in lines:
        # Add section comments
        for key, comment in section_comments.items():
            if line.strip().startswith(key):
                output_lines.append(comment + '\n')
                break
        
        output_lines.append(line)
    
    # Write back with comments
    with open(yaml_path, 'w') as f:
        f.writelines(output_lines)


def migrate_config(json_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Migrate a single JSON config to YAML.
    
    Args:
        json_path: Path to JSON configuration file
        output_dir: Output directory for YAML files (default: same as JSON)
        
    Returns:
        Path to created YAML file
    """
    logger.info(f"Migrating {json_path}")
    
    # Determine output path
    if output_dir:
        yaml_path = output_dir / json_path.with_suffix('.yaml').name
    else:
        yaml_path = json_path.with_suffix('.yaml')
    
    # Convert to YAML
    yaml_path = ConfigLoader.convert_json_to_yaml(json_path, yaml_path)
    
    # Add helpful comments
    add_comments_to_yaml(yaml_path, json_path.stem)
    
    logger.success(f"Created {yaml_path}")
    return yaml_path


def find_json_configs(directory: Path) -> List[Path]:
    """Find all JSON configuration files in a directory."""
    return list(directory.glob("**/*.json"))


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate JSON configurations to YAML format"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to JSON file or directory containing JSON configs"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for YAML files (default: same location as JSON)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search for JSON files in subdirectories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually converting"
    )
    
    args = parser.parse_args()
    
    # Find JSON files to migrate
    if args.path.is_file():
        json_files = [args.path]
    elif args.path.is_dir():
        if args.recursive:
            json_files = find_json_configs(args.path)
        else:
            json_files = list(args.path.glob("*.json"))
    else:
        logger.error(f"Path not found: {args.path}")
        return 1
    
    # Filter to only config files (skip data files)
    config_files = [
        f for f in json_files 
        if 'config' in f.stem.lower() or f.parent.name == 'configs'
    ]
    
    if not config_files:
        logger.warning("No configuration files found to migrate")
        return 0
    
    logger.info(f"Found {len(config_files)} configuration file(s) to migrate")
    
    if args.dry_run:
        logger.info("Dry run - would migrate:")
        for f in config_files:
            output_path = args.output / f.with_suffix('.yaml').name if args.output else f.with_suffix('.yaml')
            logger.info(f"  {f} -> {output_path}")
        return 0
    
    # Create output directory if needed
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    # Migrate each file
    migrated = []
    for json_path in config_files:
        try:
            yaml_path = migrate_config(json_path, args.output)
            migrated.append(yaml_path)
        except Exception as e:
            logger.error(f"Failed to migrate {json_path}: {e}")
    
    # Summary
    logger.success(f"\nMigrated {len(migrated)} configuration(s) to YAML:")
    for path in migrated:
        logger.info(f"  - {path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())