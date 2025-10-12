"""
Main Flask application entry point for NeuroDx-MultiModal API.
"""

from src.api.app import create_app
from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """Main application entry point."""
    settings = get_settings()
    
    # Create Flask application
    app = create_app()
    
    logger.info(f"Starting NeuroDx-MultiModal API server on {settings.api.api_host}:{settings.api.api_port}")
    
    # Run the application
    app.run(
        host=settings.api.api_host,
        port=settings.api.api_port,
        debug=settings.api.flask_debug
    )


if __name__ == '__main__':
    main()