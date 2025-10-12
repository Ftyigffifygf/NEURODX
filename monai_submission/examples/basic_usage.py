
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
