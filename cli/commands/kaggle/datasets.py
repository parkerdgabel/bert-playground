"""Dataset management commands."""

import sys
from pathlib import Path

import typer

from ...utils import get_console, handle_errors, print_error, print_info
from ...utils.console import create_table

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@handle_errors
def datasets_command(
    search: str | None = typer.Option(
        None, "--search", "-s", help="Search datasets by keyword"
    ),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Filter by tag (e.g., 'nlp', 'tabular')"
    ),
    user: str | None = typer.Option(None, "--user", "-u", help="Filter by username"),
    sort_by: str = typer.Option(
        "votes", "--sort", help="Sort by: votes, updated, size, relevance"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Number of datasets to show"),
    file_type: str | None = typer.Option(
        None, "--file-type", help="Filter by file type (csv, json, etc.)"
    ),
    min_size: int | None = typer.Option(None, "--min-size", help="Minimum size in MB"),
    max_size: int | None = typer.Option(None, "--max-size", help="Maximum size in MB"),
):
    """List and search Kaggle datasets.

    Browse available datasets on Kaggle with various filtering and sorting options.
    This helps you discover datasets for training and experimentation.

    Examples:
        # Search for NLP classification datasets
        bert kaggle datasets --search "nlp classification" --tag nlp

        # Find CSV datasets under 100MB
        bert kaggle datasets --file-type csv --max-size 100

        # Browse datasets from a specific user
        bert kaggle datasets --user rdizzl3 --sort updated

        # Find popular tabular datasets
        bert kaggle datasets --tag tabular --sort votes --limit 30
    """
    console = get_console()

    console.print("\n[bold blue]Kaggle Datasets[/bold blue]")
    console.print("=" * 60)

    try:
        from utils.kaggle_integration import KaggleDatasetManager, KaggleIntegration
    except ImportError:
        print_error(
            "Failed to import Kaggle integration. Make sure kaggle is installed:\n"
            "pip install kaggle",
            title="Import Error",
        )
        raise typer.Exit(1)

    try:
        dataset_manager = KaggleDatasetManager()

        # Build search parameters
        search_params = {
            "sort_by": sort_by,
            "size": limit,
        }

        if search:
            search_params["search"] = search
        if tag:
            search_params["tag_ids"] = tag
        if user:
            search_params["user"] = user
        if file_type:
            search_params["file_type"] = file_type

        # Search datasets
        with console.status("[yellow]Searching datasets...[/yellow]"):
            datasets = dataset_manager.search_datasets(**search_params)

        if not datasets:
            print_info("No datasets found matching your criteria.")
            return

        # Filter by size if specified
        if min_size is not None or max_size is not None:
            filtered = []
            for ds in datasets:
                size_bytes = ds.get("totalBytes", 0)
                size_mb = size_bytes / (1024 * 1024)

                if min_size is not None and size_mb < min_size:
                    continue
                if max_size is not None and size_mb > max_size:
                    continue

                filtered.append(ds)
            datasets = filtered

        if not datasets:
            print_info("No datasets found after applying size filters.")
            return

        # Create datasets table
        ds_table = create_table(
            "Available Datasets",
            ["Dataset", "Author", "Size", "Votes", "Updated", "Files"],
        )

        for ds in datasets[:limit]:
            # Parse dataset info
            ref = ds.get("ref", "unknown/dataset")
            author, name = ref.split("/") if "/" in ref else ("unknown", ref)

            title = ds.get("title", name)[:40]

            # Format size
            size_bytes = ds.get("totalBytes", 0)
            if size_bytes > 1e9:
                size_str = f"{size_bytes / 1e9:.1f} GB"
            else:
                size_str = f"{size_bytes / 1e6:.1f} MB"

            votes = str(ds.get("voteCount", 0))

            # Format update date
            last_updated = ds.get("lastUpdated", "")
            if last_updated:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
                    updated_str = dt.strftime("%Y-%m-%d")
                except:
                    updated_str = last_updated[:10]
            else:
                updated_str = "Unknown"

            # File count
            file_count = ds.get("fileCount", 0)
            files_str = f"{file_count} files"

            # Add row
            ds_table.add_row(
                f"[cyan]{ref}[/cyan]\n{title}",
                author,
                size_str,
                votes,
                updated_str,
                files_str,
            )

        console.print(ds_table)

        # Show summary
        console.print(
            f"\n[cyan]Showing {len(datasets[:limit])} of {len(datasets)} datasets[/cyan]"
        )

        # Show tags if available
        if datasets and "tags" in datasets[0]:
            all_tags = set()
            for ds in datasets[:10]:
                all_tags.update(ds.get("tags", []))

            if all_tags:
                console.print(
                    f"\n[cyan]Common tags: {', '.join(sorted(all_tags)[:10])}[/cyan]"
                )

        # Show next steps
        console.print("\n[bold]Next steps:[/bold]")
        console.print(
            "  • Download a dataset: [cyan]bert kaggle download-dataset DATASET_REF[/cyan]"
        )
        console.print(
            "  • Create data loader for dataset: [cyan]bert model create-loader --dataset DATASET_REF[/cyan]"
        )
        console.print(
            "  • View dataset on Kaggle: [cyan]https://kaggle.com/datasets/DATASET_REF[/cyan]"
        )

    except Exception as e:
        print_error(f"Failed to search datasets: {str(e)}", title="Dataset Error")
        raise typer.Exit(1)
