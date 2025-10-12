"""
Flask application factory and configuration for NeuroDx-MultiModal API.
"""

from flask import Flask
from flask_cors import CORS
import logging
from typing import Dict, Any

from src.config.settings import get_settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_app(config_override: Dict[str, Any] = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    settings = get_settings()
    
    # Configure Flask
    app.config['SECRET_KEY'] = settings.security.jwt_secret_key
    app.config['DEBUG'] = settings.api.flask_debug
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
    
    # Apply configuration overrides
    if config_override:
        app.config.update(config_override)
    
    # Configure CORS
    CORS(app, origins=settings.api.cors_origins)
    
    # Register blueprints
    from src.api.routes.image_processing import image_bp
    from src.api.routes.wearable_data import wearable_bp
    from src.api.routes.diagnostics import diagnostics_bp
    from src.api.routes.health import health_bp
    from src.api.routes.healthcare_integration import healthcare_bp
    
    app.register_blueprint(health_bp, url_prefix='/api/v1/health')
    app.register_blueprint(image_bp, url_prefix='/api/v1/images')
    app.register_blueprint(wearable_bp, url_prefix='/api/v1/wearable')
    app.register_blueprint(diagnostics_bp, url_prefix='/api/v1/diagnostics')
    app.register_blueprint(healthcare_bp, url_prefix='/api/v1/healthcare')
    
    # Configure error handlers
    register_error_handlers(app)
    
    logger.info("Flask application created successfully")
    return app


# Create default app instance for testing
app = create_app()

def register_error_handlers(app: Flask) -> None:
    """Register global error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return {
            'error': 'Bad Request',
            'message': 'The request was invalid or malformed',
            'status_code': 400
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        return {
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404
        }, 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {
            'error': 'File Too Large',
            'message': 'The uploaded file exceeds the maximum allowed size',
            'status_code': 413
        }, 413
    
    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error: {error}")
        return {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }, 500