#!/usr/bin/env python3
"""Unified training script using the consolidated MLX trainer.

This is the recommended way to train models going forward.
All other training scripts are deprecated in favor of this unified approach.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run training using the unified CLI."""
    # Build command
    cmd = [sys.executable, "mlx_bert_cli_v2.py", "train"]
    
    # Pass through all arguments
    cmd.extend(sys.argv[1:])
    
    # Run the command
    subprocess.run(cmd)


if __name__ == "__main__":
    main()