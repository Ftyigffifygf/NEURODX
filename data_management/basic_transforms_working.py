"""
Working Basic MONAI Transforms Module

This module provides basic MONAI transforms that work around the holoscan import issue.
"""

# Import transforms using specific paths to avoid holoscan dependency
try:
    from monai.transforms.compose import Compose
    from monai.transforms.io.dictionary import LoadImaged
    from monai.transforms.utility.dictionary import ToTensord, EnsureChannelFirstd
    from monai.transforms.spatial.dictionary import Orientationd, Spacingd
    from monai.transforms.intensity.dictionary import ScaleIntensityRanged
    
    BASIC_TRANSFORMS_AVAILABLE = True
    print("Basic MONAI transforms successfully imported via specific paths")
    
except ImportError as e:
    print(f"Some basic transforms not available: {e}")
    BASIC_TRANSFORMS_AVAILABLE = False

def create_basic_preprocessing_pipeline(keys=['image']):
    """Create a basic preprocessing pipeline that works."""
    if not BASIC_TRANSFORMS_AVAILABLE:
        raise ImportError("Basic transforms not available")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ToTensord(keys=keys)
    ]
    
    return Compose(transforms)

def create_medical_preprocessing_pipeline(keys=['image']):
    """Create a medical preprocessing pipeline."""
    if not BASIC_TRANSFORMS_AVAILABLE:
        raise ImportError("Basic transforms not available")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0)),
        ScaleIntensityRanged(
            keys=keys,
            a_min=-1000, a_max=4000,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        ToTensord(keys=keys)
    ]
    
    return Compose(transforms)

# Export the working components
__all__ = [
    'Compose', 'LoadImaged', 'ToTensord', 'EnsureChannelFirstd',
    'Orientationd', 'Spacingd', 'ScaleIntensityRanged',
    'create_basic_preprocessing_pipeline',
    'create_medical_preprocessing_pipeline',
    'BASIC_TRANSFORMS_AVAILABLE'
]