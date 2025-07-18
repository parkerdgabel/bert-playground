"""Kaggle datasets commands."""

import typer

datasets_app = typer.Typer()

@datasets_app.command(name="list")
def list_datasets():
    """List available datasets."""
    typer.echo("List datasets - implementation pending")

@datasets_app.command(name="download")
def download_dataset():
    """Download a dataset."""
    typer.echo("Download dataset - implementation pending")