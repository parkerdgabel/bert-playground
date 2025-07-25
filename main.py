#!/usr/bin/env python
"""
K-BERT Main Entry Point - Hexagonal Architecture Implementation

This is the main entry point for the k-bert application using the new
hexagonal architecture with dependency injection.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
from infrastructure.bootstrap import initialize_application, get_service
from application.ports.secondary.monitoring import MonitoringService


def setup_logging():
    """Configure logging for the application."""
    # Remove default logger
    logger.remove()
    
    # Add console logger with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logger
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "k-bert.log",
        rotation="10 MB",
        retention="1 week",
        level="DEBUG"
    )


def main():
    """Main application entry point."""
    # Setup basic logging
    setup_logging()
    
    logger.info("Starting k-bert application with hexagonal architecture")
    
    try:
        # Initialize the application with all dependencies using auto-discovery
        container = initialize_application(
            auto_discover=True,
            package_paths=[
                "domain",
                "application",
                "infrastructure", 
                "adapters",
                "ports",
            ],
            profiles=None,  # Use default profile
        )
        
        logger.info("Application initialized with auto-discovery")
        
        # Get monitoring service to confirm it's working
        monitoring = get_service(MonitoringService)
        monitoring.log_info("Application initialized successfully")
        
        # Import and run the CLI app
        from infrastructure.adapters.primary.cli.app import app, main_with_di
        
        # Run the CLI with DI container
        main_with_di(container)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()