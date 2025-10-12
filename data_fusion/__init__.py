"""
Data fusion service for multi-modal medical data integration.

This package provides services for fusing medical imaging data with wearable
sensor data, including feature alignment, tensor building, and MONAI-compatible
data preparation.
"""

from .multi_modal_fusion import (
    MultiModalFusion,
    FusionConfig,
    FusionStrategy,
    ModalityWeight,
    FusionResult,
    SpatialTemporalAlignment
)

from .feature_alignment import (
    FeatureAlignment,
    AlignmentConfig,
    AlignmentResult,
    FeatureRepresentation,
    AlignmentStrategy,
    MissingDataStrategy,
    NormalizationMethod,
    InputTensorBuilder
)

__all__ = [
    # Multi-modal fusion
    "MultiModalFusion",
    "FusionConfig", 
    "FusionStrategy",
    "ModalityWeight",
    "FusionResult",
    "SpatialTemporalAlignment",
    
    # Feature alignment
    "FeatureAlignment",
    "AlignmentConfig",
    "AlignmentResult", 
    "FeatureRepresentation",
    "AlignmentStrategy",
    "MissingDataStrategy",
    "NormalizationMethod",
    "InputTensorBuilder"
]