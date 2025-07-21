"""Data preparation command using the plugin system."""

from pathlib import Path
from typing import Optional, List
import typer
from loguru import logger
import sys

from ...utils import (
    handle_errors, track_time, requires_project,
    validate_path
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

@handle_errors
@requires_project()
@track_time("Preparing data")
def prepare_command(
    # Required arguments (make them optional when using --list)
    preprocessor: Optional[str] = typer.Argument(None, help="Name of the preprocessor plugin to use"),
    input_path: Optional[Path] = typer.Argument(None, help="Input data file (CSV)"),
    output_path: Optional[Path] = typer.Argument(None, help="Output data file (CSV)"),
    
    # Optional arguments
    text_column: str = typer.Option("text", "--text-col", help="Name for the text column"),
    label_column: Optional[str] = typer.Option("label", "--label-col", help="Name for the label column"),
    id_column: Optional[str] = typer.Option(None, "--id-col", help="Name for the ID column"),
    batch: bool = typer.Option(False, "--batch", "-b", help="Process all files in directory"),
    list_plugins: bool = typer.Option(False, "--list", "-l", help="List available preprocessors"),
    show_info: bool = typer.Option(False, "--info", "-i", help="Show preprocessor information"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Prepare data for BERT training using preprocessing plugins.
    
    This command uses a plugin system to convert various data formats into
    text suitable for BERT training. Each plugin handles a specific data format
    or competition.
    
    Examples:
        # Prepare Titanic data
        bert prepare titanic data/titanic/train.csv data/titanic_text/train.csv
        
        # Process all CSV files in a directory
        bert prepare titanic data/titanic/ data/titanic_text/ --batch
        
        # List available preprocessors
        bert prepare --list
        
        # Show info about a preprocessor
        bert prepare titanic --info
    """
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Import preprocessing modules
    from data.preprocessing.base import DataPrepConfig
    from data.preprocessing.loader import (
        list_available_preprocessors, 
        get_preprocessor,
        create_preprocessor
    )
    
    # Handle --list option
    if list_plugins:
        logger.info("Available preprocessors:")
        for name in list_available_preprocessors():
            preprocessor_class = get_preprocessor(name)
            logger.info(f"  - {name}: {preprocessor_class.description}")
            if preprocessor_class.supported_competitions:
                logger.info(f"    Competitions: {', '.join(preprocessor_class.supported_competitions)}")
        return
    
    # Handle --info option
    if show_info:
        if not preprocessor:
            logger.error("Preprocessor name required for --info")
            raise typer.Exit(1)
        try:
            preprocessor_class = get_preprocessor(preprocessor)
            logger.info(f"\nPreprocessor: {preprocessor}")
            logger.info(f"Description: {preprocessor_class.description}")
            if preprocessor_class.supported_competitions:
                logger.info(f"Supported competitions: {', '.join(preprocessor_class.supported_competitions)}")
            return
        except ValueError as e:
            logger.error(str(e))
            raise typer.Exit(1)
    
    # For non-list operations, require all arguments
    if not preprocessor or not input_path or not output_path:
        logger.error("Missing required arguments: preprocessor, input_path, output_path")
        logger.info("Use --help for usage information or --list to see available preprocessors")
        raise typer.Exit(1)
    
    # Validate preprocessor exists
    available = list_available_preprocessors()
    if preprocessor not in available:
        logger.error(f"Unknown preprocessor: {preprocessor}")
        logger.info(f"Available preprocessors: {', '.join(available)}")
        logger.info("Use --list to see more details")
        raise typer.Exit(1)
    
    # Handle batch processing
    if batch:
        if not input_path.is_dir():
            logger.error("Input path must be a directory for batch processing")
            raise typer.Exit(1)
        
        # Process all CSV files
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {input_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        for csv_file in csv_files:
            output_file = output_path / csv_file.name
            logger.info(f"\nProcessing {csv_file.name}...")
            
            # Create config for this file
            config = DataPrepConfig(
                input_path=csv_file,
                output_path=output_file,
                text_column=text_column,
                label_column=label_column,
                id_column=id_column,
            )
            
            # Create and run preprocessor
            prep = create_preprocessor(preprocessor, config)
            prep.process_file()
        
        logger.info(f"\n✅ Batch processing complete! Processed {len(csv_files)} files")
        logger.info(f"Output saved to: {output_path}")
    
    else:
        # Single file processing
        config = DataPrepConfig(
            input_path=input_path,
            output_path=output_path,
            text_column=text_column,
            label_column=label_column,
            id_column=id_column,
        )
        
        # Create and run preprocessor
        prep = create_preprocessor(preprocessor, config)
        result_df = prep.process_file()
        
        logger.info(f"\n✅ Processing complete!")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Total samples: {len(result_df)}")


# Register the command
prepare_app = typer.Typer()
prepare_app.command(name="prepare")(prepare_command)