#!/usr/bin/env python3
"""
Script to set up NVIDIA genomics analysis workflow.
"""

import os
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clone_genomics_blueprint():
    """Clone NVIDIA genomics analysis blueprint repository."""
    
    repo_url = "https://github.com/clara-parabricks-workflows/genomics-analysis-blueprint.git"
    target_dir = Path("genomics-analysis-blueprint")
    
    if target_dir.exists():
        logger.info(f"Genomics blueprint already exists at {target_dir}")
        return True
    
    try:
        logger.info(f"Cloning genomics blueprint from {repo_url}")
        subprocess.run(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Successfully cloned genomics analysis blueprint")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone genomics blueprint: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Git not found. Please install git to clone the genomics blueprint.")
        return False


def setup_genomics_environment():
    """Set up genomics analysis environment."""
    
    blueprint_dir = Path("genomics-analysis-blueprint")
    
    if not blueprint_dir.exists():
        logger.error("Genomics blueprint not found. Run clone_genomics_blueprint() first.")
        return False
    
    try:
        # Create necessary directories
        data_dir = Path("data/genomics")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        results_dir = Path("results/genomics")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create configuration file
        config_content = """
# NVIDIA Genomics Analysis Configuration
reference_genome: GRCh38
analysis_type: neurodegenerative
quality_threshold: 30.0
output_format: vcf

# Neurodegenerative disease gene panel
target_genes:
  - APOE
  - MAPT
  - GRN
  - C9orf72
  - SNCA
  - LRRK2
  - PSEN1
  - PSEN2
  - APP

# Analysis parameters
variant_calling:
  min_base_quality: 20
  min_mapping_quality: 30
  min_coverage: 10

annotation:
  databases:
    - ClinVar
    - gnomAD
    - OMIM
    - HGMD
"""
        
        config_file = blueprint_dir / "neurodx_config.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info("Genomics environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup genomics environment: {e}")
        return False


def validate_genomics_setup():
    """Validate genomics analysis setup."""
    
    blueprint_dir = Path("genomics-analysis-blueprint")
    
    # Check if blueprint directory exists
    if not blueprint_dir.exists():
        logger.error("Genomics blueprint directory not found")
        return False
    
    # Check for required files
    required_files = [
        "workflow.cwl",
        "neurodx_config.yaml"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = blueprint_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.warning(f"Missing genomics workflow files: {missing_files}")
        return False
    
    logger.info("Genomics setup validation passed")
    return True


def main():
    """Main setup function."""
    
    logger.info("Setting up NVIDIA genomics analysis workflow")
    
    # Clone genomics blueprint
    if not clone_genomics_blueprint():
        logger.error("Failed to clone genomics blueprint")
        sys.exit(1)
    
    # Setup environment
    if not setup_genomics_environment():
        logger.error("Failed to setup genomics environment")
        sys.exit(1)
    
    # Validate setup
    if not validate_genomics_setup():
        logger.error("Genomics setup validation failed")
        sys.exit(1)
    
    logger.info("NVIDIA genomics analysis workflow setup completed successfully")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Configure your NVIDIA API keys in .env file")
    print("2. Install additional genomics dependencies if needed")
    print("3. Test the genomics workflow with sample data")


if __name__ == "__main__":
    main()