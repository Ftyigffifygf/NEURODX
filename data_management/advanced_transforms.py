#!/usr/bin/env python3
"""
Advanced MONAI Transforms Integration for NeuroDx-MultiModal

This module integrates all advanced MONAI transform capabilities including:
- Comprehensive spatial and intensity transforms
- Advanced augmentation strategies
- Multi-modal transform coordination
- Custom medical imaging transforms
- Performance-optimized transform pipelines
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path

# MONAI Core Transforms
from monai.transforms import (
    # Core transforms
    Compose, Transform, MapTransform, Randomizable,
    
    # Loading and I/O
    LoadImaged, SaveImaged, LoadImage, SaveImage,
    
    # Array transforms
    EnsureChannelFirstd, EnsureChannelFirst,
    AddChanneld, RemoveRepeatedChanneld, RepeatChanneld,
    SqueezeDimd, ExpandDimsd, TransposeTransform,
    
    # Spatial transforms
    Spacingd, Spacing, Orientationd, Orientation,
    Resized, Resize, ResizeWithPadOrCropd, ResizeWithPadOrCrop,
    CropForegroundd, CropForeground, CenterSpatialCropd, CenterSpatialCrop,
    SpatialPadd, SpatialPad, BorderPadd, BorderPad,
    SpatialCropd, SpatialCrop, RandSpatialCropd, RandSpatialCrop,
    RandSpatialCropSamplesd, RandSpatialCropSamples,
    
    # Intensity transforms
    NormalizeIntensityd, NormalizeIntensity,
    ScaleIntensityd, ScaleIntensity,
    ScaleIntensityRanged, ScaleIntensityRange,
    ScaleIntensityRangePercentilesd, ScaleIntensityRangePercentiles,
    ClipIntensityd, ClipIntensity,
    ThresholdIntensityd, ThresholdIntensity,
    MaskIntensityd, MaskIntensity,
    
    # Augmentation transforms
    RandFlipd, RandFlip, RandRotated, RandRotate,
    RandZoomd, RandZoom, RandAffined, RandAffine,
    RandElasticDeformd, RandElasticDeform,
    RandGaussianNoised, RandGaussianNoise,
    RandGaussianSmoothd, RandGaussianSmooth,
    RandShiftIntensityd, RandShiftIntensity,
    RandScaleIntensityd, RandScaleIntensity,
    RandBiasFieldd, RandBiasField,
    RandGibbsNoised, RandGibbsNoise,
    RandKSpaceSpikeNoised, RandKSpaceSpikeNoise,
    RandMotionGhostingd, RandMotionGhosting,
    
    # Utility transforms
    ToTensord, ToTensor, ToNumpyd, ToNumpy,
    ToPILd, ToPIL, ToDeviced, ToDevice,
    CastToTyped, CastToType,
    ConvertToMultiChannelBasedOnBratsClassesd,
    
    # Dictionary transforms
    SelectItemsd, DeleteItemsd, CopyItemsd,
    RenameKeysd, EnsureTyped, EnsureType,
    
    # Lambda and custom transforms
    Lambdad, Lambda, LabelToMaskd, LabelToMask,
    MaskIntensityd, MaskIntensity,
    
    # Medical specific transforms
    RandCoarseDropoutd, RandCoarseDropout,
    RandCoarseShuffled, RandCoarseShuffle,
    RandWeightedCropd, RandWeightedCrop,
    
    # Post-processing transforms
    Activationsd, Activations, AsDiscreted, AsDiscrete,
    KeepLargestConnectedComponentd, KeepLargestConnectedComponent,
    RemoveSmallObjectsd, RemoveSmallObjects,
    FillHolesd, FillHoles,
    
    # Inverse transforms
    InvertibleTransform, TraceableTransform,
    
    # Batch transforms
    BatchInverseTransform,
    
    # GPU transforms
    CuCIMTransforms,
)

# MONAI Utils
from monai.utils import (
    ensure_tuple, ensure_tuple_rep, ensure_tuple_size,
    GridSampleMode, GridSamplePadMode, InterpolateMode,
    BlendMode, NumpyPadMode, PytorchPadMode,
    TransformBackends, convert_to_tensor
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TransformConfig:
    """Configuration for transform pipelines."""
    # Spatial configuration
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    target_size: Tuple[int, int, int] = (96, 96, 96)
    orientation: str = "RAS"
    
    # Intensity configuration
    intensity_range: Tuple[float, float] = (-1000, 4000)
    normalize_intensity: bool = True
    clip_intensity: bool = True
    
    # Augmentation configuration
    augmentation_probability: float = 0.5
    spatial_aug_prob: float = 0.3
    intensity_aug_prob: float = 0.2
    noise_aug_prob: float = 0.1
    
    # Performance configuration
    cache_transforms: bool = True
    use_gpu_transforms: bool = False
    num_workers: int = 4
    
    # Multi-modal configuration
    modality_specific: bool = True
    align_modalities: bool = True


class NeuroDxTransformPipeline:
    """
    Advanced transform pipeline for NeuroDx multi-modal data.
    Integrates all MONAI transform capabilities with medical imaging best practices.
    """
    
    def __init__(self, config: TransformConfig):
        """
        Initialize transform pipeline.
        
        Args:
            config: Transform configuration
        """
        self.config = config
        self.modality_configs = self._initialize_modality_configs()
        self.transform_cache = {}
        
        logger.info("Initialized NeuroDx transform pipeline")
    
    def _initialize_modality_configs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize modality-specific configurations."""
        return {
            'MRI': {
                'intensity_range': (-1000, 4000),
                'spacing': (1.0, 1.0, 1.0),
                'orientation': 'RAS',
                'augmentation_strength': 0.3
            },
            'CT': {
                'intensity_range': (-1024, 3071),
                'spacing': (1.0, 1.0, 1.0),
                'orientation': 'RAS',
                'augmentation_strength': 0.2
            },
            'Ultrasound': {
                'intensity_range': (0, 255),
                'spacing': (0.5, 0.5, 0.5),
                'orientation': 'RAS',
                'augmentation_strength': 0.4
            },
            'fMRI': {
                'intensity_range': (-1000, 4000),
                'spacing': (2.0, 2.0, 2.0),
                'orientation': 'RAS',
                'augmentation_strength': 0.1
            },
            'DTI': {
                'intensity_range': (0, 10000),
                'spacing': (1.5, 1.5, 1.5),
                'orientation': 'RAS',
                'augmentation_strength': 0.2
            }
        }
    
    def create_preprocessing_pipeline(self, 
                                    keys: List[str],
                                    modality: str = "MRI") -> Compose:
        """
        Create comprehensive preprocessing pipeline.
        
        Args:
            keys: Data dictionary keys to process
            modality: Imaging modality
            
        Returns:
            Composed preprocessing pipeline
        """
        modality_config = self.modality_configs.get(modality, self.modality_configs['MRI'])
        
        transforms = [
            # Loading and basic setup
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
            EnsureTyped(keys=keys, data_type="tensor"),
            
            # Spatial preprocessing
            Orientationd(keys=keys, axcodes=modality_config['orientation']),
            Spacingd(
                keys=keys,
                pixdim=modality_config['spacing'],
                mode=["bilinear"] * len(keys)
            ),
            
            # Crop foreground to remove unnecessary background
            CropForegroundd(keys=keys, source_key=keys[0]),
            
            # Resize to target size
            ResizeWithPadOrCropd(
                keys=keys,
                spatial_size=self.config.target_size,
                mode=["bilinear"] * len(keys)
            ),
            
            # Intensity preprocessing
            ScaleIntensityRanged(
                keys=keys,
                a_min=modality_config['intensity_range'][0],
                a_max=modality_config['intensity_range'][1],
                b_min=0.0,
                b_max=1.0,
                clip=self.config.clip_intensity
            ),
        ]
        
        # Add normalization if requested
        if self.config.normalize_intensity:
            transforms.append(
                NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True)
            )
        
        # Final tensor conversion
        transforms.append(ToTensord(keys=keys))
        
        return Compose(transforms)
    
    def create_augmentation_pipeline(self,
                                   keys: List[str],
                                   modality: str = "MRI",
                                   training: bool = True) -> Compose:
        """
        Create comprehensive augmentation pipeline.
        
        Args:
            keys: Data dictionary keys to process
            modality: Imaging modality
            training: Whether this is for training (enables augmentation)
            
        Returns:
            Composed augmentation pipeline
        """
        if not training:
            return Compose([])  # No augmentation for inference
        
        modality_config = self.modality_configs.get(modality, self.modality_configs['MRI'])
        aug_strength = modality_config['augmentation_strength']
        
        transforms = []
        
        # Spatial augmentations
        if self.config.spatial_aug_prob > 0:
            transforms.extend([
                # Flipping
                RandFlipd(
                    keys=keys,
                    prob=self.config.spatial_aug_prob,
                    spatial_axis=[0, 1, 2]
                ),
                
                # Rotation
                RandRotated(
                    keys=keys,
                    prob=self.config.spatial_aug_prob * 0.8,
                    range_x=np.pi * 0.1 * aug_strength,
                    range_y=np.pi * 0.1 * aug_strength,
                    range_z=np.pi * 0.1 * aug_strength,
                    mode=["bilinear"] * len(keys),
                    padding_mode="border"
                ),
                
                # Zooming
                RandZoomd(
                    keys=keys,
                    prob=self.config.spatial_aug_prob * 0.6,
                    min_zoom=1.0 - 0.2 * aug_strength,
                    max_zoom=1.0 + 0.2 * aug_strength,
                    mode=["bilinear"] * len(keys)
                ),
                
                # Elastic deformation
                RandElasticDeformd(
                    keys=keys,
                    prob=self.config.spatial_aug_prob * 0.4,
                    sigma_range=(5, 8),
                    magnitude_range=(100 * aug_strength, 200 * aug_strength),
                    mode=["bilinear"] * len(keys)
                ),
                
                # Affine transformation
                RandAffined(
                    keys=keys,
                    prob=self.config.spatial_aug_prob * 0.5,
                    rotate_range=(np.pi * 0.05 * aug_strength,) * 3,
                    scale_range=(0.1 * aug_strength,) * 3,
                    translate_range=(10 * aug_strength,) * 3,
                    mode=["bilinear"] * len(keys),
                    padding_mode="border"
                )
            ])
        
        # Intensity augmentations
        if self.config.intensity_aug_prob > 0:
            transforms.extend([
                # Gaussian noise
                RandGaussianNoised(
                    keys=keys,
                    prob=self.config.intensity_aug_prob,
                    mean=0.0,
                    std=0.05 * aug_strength
                ),
                
                # Intensity shifting
                RandShiftIntensityd(
                    keys=keys,
                    prob=self.config.intensity_aug_prob * 0.8,
                    offsets=0.1 * aug_strength
                ),
                
                # Intensity scaling
                RandScaleIntensityd(
                    keys=keys,
                    prob=self.config.intensity_aug_prob * 0.8,
                    factors=0.2 * aug_strength
                ),
                
                # Gaussian smoothing
                RandGaussianSmoothd(
                    keys=keys,
                    prob=self.config.intensity_aug_prob * 0.6,
                    sigma_x=(0.5, 1.5),
                    sigma_y=(0.5, 1.5),
                    sigma_z=(0.5, 1.5)
                ),
                
                # Bias field simulation
                RandBiasFieldd(
                    keys=keys,
                    prob=self.config.intensity_aug_prob * 0.4,
                    degree=3,
                    coeff_range=(0.0, 0.1 * aug_strength)
                )
            ])
        
        # Medical imaging specific augmentations
        if self.config.noise_aug_prob > 0:
            transforms.extend([
                # Gibbs noise (ringing artifacts)
                RandGibbsNoised(
                    keys=keys,
                    prob=self.config.noise_aug_prob * 0.3,
                    alpha=(0.0, 0.5 * aug_strength)
                ),
                
                # K-space spike noise
                RandKSpaceSpikeNoised(
                    keys=keys,
                    prob=self.config.noise_aug_prob * 0.2,
                    intensity_range=(0.95, 1.05)
                ),
                
                # Motion ghosting
                RandMotionGhostingd(
                    keys=keys,
                    prob=self.config.noise_aug_prob * 0.2,
                    degrees=10 * aug_strength,
                    translation=10 * aug_strength
                ),
                
                # Coarse dropout
                RandCoarseDropoutd(
                    keys=keys,
                    prob=self.config.noise_aug_prob * 0.3,
                    holes=8,
                    spatial_size=(8, 8, 8),
                    dropout_holes=True,
                    fill_value=0
                )
            ])
        
        return Compose(transforms)
    
    def create_multi_modal_pipeline(self,
                                  modality_keys: Dict[str, List[str]],
                                  training: bool = True) -> Compose:
        """
        Create pipeline for multi-modal data processing.
        
        Args:
            modality_keys: Dictionary mapping modality names to their keys
            training: Whether this is for training
            
        Returns:
            Composed multi-modal pipeline
        """
        transforms = []
        
        # Process each modality with its specific pipeline
        for modality, keys in modality_keys.items():
            # Preprocessing for this modality
            preprocessing = self.create_preprocessing_pipeline(keys, modality)
            transforms.append(preprocessing)
            
            # Augmentation for this modality (if training)
            if training:
                augmentation = self.create_augmentation_pipeline(keys, modality, training)
                if len(augmentation.transforms) > 0:
                    transforms.append(augmentation)
        
        # Multi-modal alignment and fusion
        if self.config.align_modalities and len(modality_keys) > 1:
            transforms.extend(self._create_alignment_transforms(modality_keys))
        
        return Compose(transforms)
    
    def _create_alignment_transforms(self,
                                   modality_keys: Dict[str, List[str]]) -> List[Transform]:
        """Create transforms for multi-modal alignment."""
        alignment_transforms = []
        
        # Ensure all modalities have the same spatial properties
        all_keys = []
        for keys in modality_keys.values():
            all_keys.extend(keys)
        
        # Spatial alignment
        alignment_transforms.extend([
            # Ensure consistent orientation
            Orientationd(keys=all_keys, axcodes=self.config.orientation),
            
            # Ensure consistent spacing
            Spacingd(
                keys=all_keys,
                pixdim=self.config.target_spacing,
                mode=["bilinear"] * len(all_keys)
            ),
            
            # Ensure consistent size
            ResizeWithPadOrCropd(
                keys=all_keys,
                spatial_size=self.config.target_size,
                mode=["bilinear"] * len(all_keys)
            )
        ])
        
        return alignment_transforms
    
    def create_post_processing_pipeline(self,
                                      keys: List[str],
                                      task_type: str = "segmentation") -> Compose:
        """
        Create post-processing pipeline for model outputs.
        
        Args:
            keys: Data dictionary keys to process
            task_type: Type of task (segmentation, classification)
            
        Returns:
            Composed post-processing pipeline
        """
        transforms = []
        
        if task_type == "segmentation":
            transforms.extend([
                # Convert to discrete labels
                AsDiscreted(keys=keys, argmax=True),
                
                # Remove small objects
                RemoveSmallObjectsd(
                    keys=keys,
                    min_size=64,
                    connectivity=1
                ),
                
                # Keep largest connected component
                KeepLargestConnectedComponentd(
                    keys=keys,
                    applied_labels=[1, 2, 3]  # Assuming multi-class segmentation
                ),
                
                # Fill holes
                FillHolesd(keys=keys)
            ])
        
        elif task_type == "classification":
            transforms.extend([
                # Apply softmax activation
                Activationsd(keys=keys, softmax=True),
                
                # Convert to discrete predictions
                AsDiscreted(keys=keys, argmax=True)
            ])
        
        return Compose(transforms)
    
    def create_inference_pipeline(self,
                                keys: List[str],
                                modality: str = "MRI") -> Compose:
        """
        Create optimized pipeline for inference.
        
        Args:
            keys: Data dictionary keys to process
            modality: Imaging modality
            
        Returns:
            Composed inference pipeline
        """
        # Use preprocessing without augmentation
        preprocessing = self.create_preprocessing_pipeline(keys, modality)
        
        # Add GPU optimization if available
        transforms = list(preprocessing.transforms)
        
        if self.config.use_gpu_transforms and torch.cuda.is_available():
            # Move to GPU early in pipeline
            transforms.insert(-1, ToDeviced(keys=keys, device="cuda"))
        
        return Compose(transforms)
    
    def create_validation_pipeline(self,
                                 keys: List[str],
                                 modality: str = "MRI") -> Compose:
        """
        Create pipeline for validation data.
        
        Args:
            keys: Data dictionary keys to process
            modality: Imaging modality
            
        Returns:
            Composed validation pipeline
        """
        # Use preprocessing with minimal augmentation
        preprocessing = self.create_preprocessing_pipeline(keys, modality)
        
        # Add light augmentation for test-time augmentation
        light_augmentation = Compose([
            RandFlipd(keys=keys, prob=0.1, spatial_axis=[0, 1, 2])
        ])
        
        return Compose([preprocessing, light_augmentation])
    
    def create_custom_transform(self,
                              transform_func: Callable,
                              keys: List[str],
                              **kwargs) -> MapTransform:
        """
        Create custom transform from function.
        
        Args:
            transform_func: Custom transform function
            keys: Data dictionary keys to process
            **kwargs: Additional arguments for transform
            
        Returns:
            Custom map transform
        """
        return Lambdad(keys=keys, func=transform_func, **kwargs)
    
    def optimize_pipeline_performance(self,
                                    pipeline: Compose,
                                    sample_data: Dict[str, Any]) -> Compose:
        """
        Optimize pipeline performance through profiling and optimization.
        
        Args:
            pipeline: Transform pipeline to optimize
            sample_data: Sample data for profiling
            
        Returns:
            Optimized pipeline
        """
        logger.info("Optimizing transform pipeline performance...")
        
        # Profile individual transforms
        transform_times = []
        
        for i, transform in enumerate(pipeline.transforms):
            import time
            start_time = time.time()
            
            try:
                # Test transform on sample data
                _ = transform(sample_data.copy())
                end_time = time.time()
                transform_times.append((i, end_time - start_time, str(transform)))
            except Exception as e:
                logger.warning(f"Transform {i} failed during profiling: {e}")
                transform_times.append((i, float('inf'), str(transform)))
        
        # Identify slow transforms
        slow_transforms = [
            (i, time, name) for i, time, name in transform_times
            if time > 0.1  # Transforms taking more than 100ms
        ]
        
        if slow_transforms:
            logger.info(f"Identified {len(slow_transforms)} slow transforms:")
            for i, time, name in slow_transforms:
                logger.info(f"  Transform {i}: {time:.3f}s - {name}")
        
        # Apply optimizations
        optimized_transforms = []
        
        for i, transform in enumerate(pipeline.transforms):
            # Add GPU acceleration for compatible transforms
            if (self.config.use_gpu_transforms and 
                torch.cuda.is_available() and
                hasattr(transform, 'to')):
                try:
                    transform = transform.to('cuda')
                except:
                    pass  # Keep on CPU if GPU transfer fails
            
            optimized_transforms.append(transform)
        
        optimized_pipeline = Compose(optimized_transforms)
        logger.info("Pipeline optimization completed")
        
        return optimized_pipeline
    
    def get_pipeline_summary(self, pipeline: Compose) -> Dict[str, Any]:
        """
        Get summary information about a transform pipeline.
        
        Args:
            pipeline: Transform pipeline
            
        Returns:
            Pipeline summary information
        """
        summary = {
            'num_transforms': len(pipeline.transforms),
            'transform_types': [],
            'randomizable_transforms': 0,
            'gpu_transforms': 0,
            'estimated_memory_usage': 0
        }
        
        for transform in pipeline.transforms:
            transform_type = type(transform).__name__
            summary['transform_types'].append(transform_type)
            
            if isinstance(transform, Randomizable):
                summary['randomizable_transforms'] += 1
            
            if hasattr(transform, 'device') and 'cuda' in str(transform.device):
                summary['gpu_transforms'] += 1
        
        return summary


# Factory functions for common pipelines
def create_neurodx_training_pipeline(modality: str = "MRI") -> Compose:
    """Create standard training pipeline for NeuroDx."""
    config = TransformConfig(
        augmentation_probability=0.8,
        spatial_aug_prob=0.5,
        intensity_aug_prob=0.3,
        noise_aug_prob=0.2
    )
    
    pipeline = NeuroDxTransformPipeline(config)
    keys = ["image"]
    
    preprocessing = pipeline.create_preprocessing_pipeline(keys, modality)
    augmentation = pipeline.create_augmentation_pipeline(keys, modality, training=True)
    
    return Compose([preprocessing, augmentation])


def create_neurodx_inference_pipeline(modality: str = "MRI") -> Compose:
    """Create standard inference pipeline for NeuroDx."""
    config = TransformConfig(
        augmentation_probability=0.0,
        use_gpu_transforms=True
    )
    
    pipeline = NeuroDxTransformPipeline(config)
    keys = ["image"]
    
    return pipeline.create_inference_pipeline(keys, modality)


def create_multi_modal_training_pipeline() -> Compose:
    """Create multi-modal training pipeline."""
    config = TransformConfig(
        modality_specific=True,
        align_modalities=True,
        augmentation_probability=0.6
    )
    
    pipeline = NeuroDxTransformPipeline(config)
    modality_keys = {
        "MRI": ["mri_image"],
        "CT": ["ct_image"],
        "Ultrasound": ["us_image"]
    }
    
    return pipeline.create_multi_modal_pipeline(modality_keys, training=True)


# Example usage
if __name__ == "__main__":
    # Create training pipeline
    training_pipeline = create_neurodx_training_pipeline("MRI")
    
    # Create inference pipeline
    inference_pipeline = create_neurodx_inference_pipeline("MRI")
    
    # Create multi-modal pipeline
    multimodal_pipeline = create_multi_modal_training_pipeline()
    
    print("Advanced MONAI transforms integration completed successfully!")
    print(f"Training pipeline: {len(training_pipeline.transforms)} transforms")
    print(f"Inference pipeline: {len(inference_pipeline.transforms)} transforms")
    print(f"Multi-modal pipeline: {len(multimodal_pipeline.transforms)} transforms")