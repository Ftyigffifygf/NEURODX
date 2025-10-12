"""
Main entry point for NeuroDx-MultiModal system.
"""

import sys
import asyncio
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import get_settings, validate_nvidia_setup, validate_monai_setup
from src.utils.logging_config import get_logger
from src.services.nvidia_integration.nvidia_service import get_nvidia_service

logger = get_logger("main")


def run_api_server():
    """Run the Flask API server."""
    from src.api.main import main as api_main
    api_main()


async def main():
    """Main application entry point."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NeuroDx-MultiModal System')
    parser.add_argument('--mode', choices=['api', 'test'], default='test',
                       help='Run mode: api (start API server) or test (run system tests)')
    args = parser.parse_args()
    
    logger.info("Starting NeuroDx-MultiModal system")
    
    # Load settings
    settings = get_settings()
    logger.info(f"Loaded configuration for environment: {settings.environment}")
    
    # Validate setup
    nvidia_valid = validate_nvidia_setup()
    monai_valid = validate_monai_setup()
    
    if not nvidia_valid:
        logger.warning("NVIDIA setup validation failed - some features may be unavailable")
    
    if not monai_valid:
        logger.warning("MONAI setup validation failed - creating required directories")
    
    # Initialize NVIDIA services
    nvidia_service = get_nvidia_service()
    health_status = nvidia_service.get_service_health()
    logger.info(f"NVIDIA services health: {health_status}")
    
    if args.mode == 'api':
        logger.info("Starting API server mode")
        run_api_server()
    else:
        # Test basic functionality
        await test_system_components()
        logger.info("NeuroDx-MultiModal system initialized successfully")


async def test_system_components():
    """Test basic system components."""
    
    logger.info("Testing system components...")
    
    try:
        # Test NVIDIA service availability
        nvidia_service = get_nvidia_service()
        
        # Test Palmyra client (if available)
        palmyra_client = nvidia_service.get_palmyra_client()
        if palmyra_client:
            logger.info("Palmyra client available")
        else:
            logger.warning("Palmyra client not available - check API key configuration")
        
        # Test genomics client (if available)
        genomics_client = nvidia_service.get_genomics_client()
        if genomics_client:
            logger.info("Genomics client available")
        else:
            logger.warning("Genomics client not available - check workflow configuration")
        
        # Test configuration loading
        settings = get_settings()
        logger.info(f"Configuration loaded: {settings.app_name} v{settings.app_version}")
        
        logger.info("System component tests completed")
        
    except Exception as e:
        logger.error(f"System component test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())