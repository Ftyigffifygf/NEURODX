#!/usr/bin/env python3
"""
MONAI Deploy SDK Setup for NeuroDx-MultiModal
Complete setup script for integrating NeuroDx with MONAI Deploy
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} - Failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None


def setup_monai_deploy():
    """Complete MONAI Deploy SDK setup"""
    
    print("🧠 NeuroDx-MultiModal MONAI Deploy Setup")
    print("=" * 60)
    
    # Step 1: Install MONAI Deploy SDK
    logger.info("📦 Installing MONAI Deploy SDK...")
    run_command("pip install monai-deploy-app-sdk", "Installing MONAI Deploy SDK")
    
    # Step 2: Install additional dependencies
    logger.info("📦 Installing additional dependencies...")
    dependencies = [
        "matplotlib",
        "Pillow", 
        "scikit-image",
        "monai[all]",
        "torch",
        "torchvision"
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"Installing {dep}")
    
    # Step 3: Clone MONAI Deploy App SDK repository
    logger.info("📂 Cloning MONAI Deploy App SDK repository...")
    if not Path("monai-deploy-app-sdk").exists():
        run_command(
            "git clone https://github.com/Project-MONAI/monai-deploy-app-sdk.git",
            "Cloning MONAI Deploy App SDK"
        )
    else:
        logger.info("✅ MONAI Deploy App SDK already exists")
    
    # Step 4: Create NeuroDx MONAI Deploy App structure
    logger.info("🏗️ Creating NeuroDx MONAI Deploy App structure...")
    create_neurodx_deploy_app()
    
    print("\n🎉 MONAI Deploy SDK setup completed!")
    print("\n📋 Next Steps:")
    print("1. Review the created NeuroDx Deploy App")
    print("2. Test the application locally")
    print("3. Package the application")
    print("4. Deploy to clinical environment")


def create_neurodx_deploy_app():
    """Create NeuroDx-specific MONAI Deploy application"""
    
    # Create app directory structure
    app_dir = Path("monai_deploy_apps/neurodx_multimodal")
    app_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (app_dir / "models").mkdir(exist_ok=True)
    (app_dir / "config").mkdir(exist_ok=True)
    (app_dir / "tests").mkdir(exist_ok=True)
    
    logger.info("✅ Created NeuroDx Deploy App directory structure")


if __name__ == "__main__":
    setup_monai_deploy()