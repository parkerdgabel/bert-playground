"""MLX BERT CLI - A modular command-line interface for BERT Kaggle competitions.

This package provides a comprehensive CLI for training, evaluating, and deploying
BERT models optimized for Apple Silicon using MLX framework.
"""

from .app import app

__version__ = "2.0.0"

def main():
    """Entry point for the CLI."""
    app()

__all__ = ["app", "main"]