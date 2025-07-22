"""MLX BERT CLI - A modular command-line interface for BERT Kaggle competitions.

This package provides a comprehensive CLI for training, evaluating, and deploying
BERT models optimized for Apple Silicon using MLX framework.
"""

from ._version import __version__, __version_info__
from .app import app


def main():
    """Entry point for the CLI."""
    app()


__all__ = ["app", "main", "__version__", "__version_info__"]
