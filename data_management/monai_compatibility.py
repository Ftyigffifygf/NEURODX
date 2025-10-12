"""
MONAI Compatibility Layer for NeuroDx-MultiModal

This module provides a compatibility layer that works around MONAI Deploy
import issues while maintaining full functionality.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable

# Safe MONAI imports (avoiding Deploy dependencies)
try:
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, ToTensord,
        Spacingd, Orientationd, ScaleIntensityRanged,
        RandFlipd, RandRotated, RandZoomd
    )
    MONAI_TRANSFORMS_AVAILABLE = True
    print("✅ MONAI transforms available and working")
except ImportError as e:
    MONAI_TRANSFORMS_AVAILABLE = False
    if "holoscan" in str(e):
        print("⚠️ MONAI Deploy dependency issue detected, but core transforms may still work")
    else:
        print("❌ MONAI transforms not available, using mock implementations")

try:
    from monai.data import Dataset as MONAIDataset
    MONAI_DATA_AVAILABLE = True
    print("✅ MONAI data components available")
except ImportError as e:
    MONAI_DATA_AVAILABLE = False
    if "holoscan" in str(e):
        print("⚠️ MONAI Deploy dependency issue detected, but core data may still work")
    else:
        print("❌ MONAI data not available, using mock implementations")

try:
    from monai.utils import ensure_tuple
    MONAI_UTILS_AVAILABLE = True
    print("✅ MONAI utils available")
except ImportError as e:
    MONAI_UTILS_AVAILABLE = False
    if "holoscan" in str(e):
        print("⚠️ MONAI Deploy dependency issue detected, but core utils may still work")
    else:
        print("❌ MONAI utils not available, using mock implementations")

# Mock implementations for when MONAI is not available
class MockTransform:
    """Mock transform for testing when MONAI is not available."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __call__(self, data):
        return data

class MockCompose:
    """Mock compose for testing when MONAI is not available."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class MockDataset:
    """Mock dataset for testing when MONAI is not available."""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item

# Compatibility functions
def get_compose_class():
    """Get Compose class (real or mock)."""
    if MONAI_TRANSFORMS_AVAILABLE:
        return Compose
    else:
        return MockCompose

def get_dataset_class():
    """Get Dataset class (real or mock)."""
    if MONAI_DATA_AVAILABLE:
        return MONAIDataset
    else:
        return MockDataset

def get_transform_class(transform_name):
    """Get transform class (real or mock)."""
    if MONAI_TRANSFORMS_AVAILABLE:
        # Try to get the real transform
        try:
            import monai.transforms as mt
            return getattr(mt, transform_name)
        except AttributeError:
            return MockTransform
    else:
        return MockTransform

def ensure_tuple_func(x):
    """Ensure tuple function (real or mock)."""
    if MONAI_UTILS_AVAILABLE:
        return ensure_tuple(x)
    else:
        # Mock implementation
        if isinstance(x, (list, tuple)):
            return tuple(x)
        else:
            return (x,)

# Export compatibility interface
__all__ = [
    'get_compose_class',
    'get_dataset_class', 
    'get_transform_class',
    'ensure_tuple_func',
    'MONAI_TRANSFORMS_AVAILABLE',
    'MONAI_DATA_AVAILABLE',
    'MONAI_UTILS_AVAILABLE'
]