#!/usr/bin/env python3
"""
MONAI Submission Preparation Script
Automates the preparation of NeuroDx-MultiModal for MONAI framework registration
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_monai_submission():
    """Prepare NeuroDx-MultiModal for MONAI submission"""
    
    print("NeuroDx-MultiModal MONAI Submission Preparation")
    print("=" * 60)
    
    project_root = Path(".")
    submission_dir = project_root / "monai_submission"
    
    # Create submission directory
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    directories = ["docs", "bundle", "models", "examples", "tests"]
    for dir_name in directories:
        (submission_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    print("Created submission directory structure")
    
    # Copy documentation files
    docs_to_copy = [
        "README.md",
        "CITATION.cff", 
        "CONTRIBUTING.md",
        "MONAI_REGISTRATION_GUIDE.md",
        "PROJECT_STATUS.md"
    ]
    
    for doc in docs_to_copy:
        source = project_root / doc
        if source.exists():
            shutil.copy2(source, submission_dir / doc)
            print(f"Copied {doc}")
        else:
            print(f"Warning - Missing: {doc}")
    
    # Create basic example
    example_code = '''
"""
NeuroDx-MultiModal Basic Usage Example
"""

import torch
from monai.networks.nets import SwinUNETR

def main():
    print("🧠 NeuroDx-MultiModal Example")
    
    # Initialize model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=4,
        feature_size=48
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model initialized successfully")

if __name__ == "__main__":
    main()
'''
    
    with open(submission_dir / "examples" / "basic_usage.py", "w", encoding="utf-8") as f:
        f.write(example_code)
    
    print("Created basic usage example")
    
    # Create submission checklist
    checklist = f'''
# MONAI Submission Checklist for NeuroDx-MultiModal

## Required Files
- [x] README.md
- [x] CITATION.cff  
- [x] CONTRIBUTING.md
- [x] Basic usage example
- [x] Model documentation

## Next Steps
1. Create MONAI Hub account at https://monai.io/
2. Submit project proposal
3. Engage with MONAI community
4. Present at MONAI workshops

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
'''
    
    with open(submission_dir / "SUBMISSION_CHECKLIST.md", "w", encoding="utf-8") as f:
        f.write(checklist)
    
    print("Created submission checklist")
    
    # Create ZIP package
    zip_path = project_root / f"neurodx_monai_submission_{datetime.now().strftime('%Y%m%d')}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(submission_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(submission_dir)
                zipf.write(file_path, arc_path)
    
    print(f"Created submission package: {zip_path}")
    
    print("\nMONAI submission preparation completed!")
    print("\nNext Steps:")
    print("1. Review the submission package")
    print("2. Create MONAI Hub account")
    print("3. Submit to MONAI community")
    print("4. Engage with MONAI Discord")


if __name__ == "__main__":
    prepare_monai_submission()