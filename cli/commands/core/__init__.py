"""Core BERT operations commands."""

import typer

# Create the core commands app
app = typer.Typer(
    help="Core BERT operations",
    no_args_is_help=True,
)

# Import commands (imported in app.py to avoid issues)
__all__ = ["app"]