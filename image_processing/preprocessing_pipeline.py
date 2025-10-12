"""
Medical image preprocessing pipeline using MONAI transforms.
Supports configurable transform composition for different modalities and data augmentation.
"""

import logging
from typing import Dict, List, Optional, Union, Any, Callable
from enum import Enum
import torch
import numpy as np

from monai.transforms import (
    # Core transforms
    Compose, 
    LoadImaged,
    EnsureChannelFirstd,
    ToTensord,
    
    # Spatial transforms
    Spacingd,
    Orientationd,
    CropForegroundd,
    SpatialPadd,
    
    # Intensity transforms
    NormalizeIntensityd,
    ScaleIntensityRanged,
    ScaleIntensityd,
    
    # Augmentation transforms
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
    
    # Utility transforms
    DeleteItemsd,
    SelectItemsd
)

from monai.data import MetaTensor
from monai.utils import ensure_tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModalityType(Enum):
    """Supported imaging modalities."""
    MRI = "mri"
    CT = "ct"
    ULTRASOUND = "ultrasound"
    FMRI = "fmri"
    DTI = "dti"


class ProcessingMode(Enum):
    """Processing modes for different use cases."""
    INFERENCE = "inference"
    TRAINING = "training"
    VALIDATION = "validation"


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors."""
    pass


class PreprocessingPipeline:
    """
    Configurable preprocessing pipeline for medical images using MONAI transforms.
    Supports different modalities and processing modes with appropriate transforms.
    """
    
    # Standard target spacing for different modalities (in mm)
    DEFAULT_SPACING = {
        ModalityType.MRI: (1.0, 1.0, 1.0),
        ModalityType.CT: (1.0, 1.0, 1.0),
        ModalityType.ULTRASOUND: (0.5, 0.5, 0.5),
        ModalityType.FMRI: (2.0, 2.0, 2.0),
        ModalityType.DTI: (1.5, 1.5, 1.5)
    }
    
    # Standard intensity ranges for normalization
    INTENSITY_RANGES = {
        ModalityType.MRI: (-1000, 4000),
        ModalityType.CT: (-1024, 3071),
        ModalityType.ULTRASOUND: (0, 255),
        ModalityType.FMRI: (-1000, 4000),
        ModalityType.DTI: (0, 10000)
    }
    
    def __init__(self, 
                 target_spacing: Optional[Dict[ModalityType, tuple]] = None,
                 target_size: Optional[tuple] = None,
                 device: str = "cpu"):
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_spacing: Custom spacing for each modality
            target_size: Target spatial size for all images
            device: Device for tensor operations
        """
        self.target_spacing = target_spacing or self.DEFAULT_SPACING
        self.target_size = target_size or (96, 96, 96)  # Standard size for SwinUNETR
        self.device = device
        
        # Cache for compiled transforms
        self._transform_cache = {}
        
    def _get_base_transforms(self, keys: List[str]) -> List[Callable]:
        """
        Get base transforms that are common to all processing modes.
        
        Args:
            keys: List of data dictionary keys to process
            
        Returns:
            List of base transforms
        """
        return [
            LoadImaged(keys=keys, image_only=False),
            EnsureChannelFirstd(keys=keys),
        ]
    
    def _get_spatial_transforms(self, 
                              keys: List[str], 
                              modality: ModalityType) -> List[Callable]:
        """
        Get spatial preprocessing transforms.
        
        Args:
            keys: List of data dictionary keys to process
            modality: Imaging modality type
            
        Returns:
            List of spatial transforms
        """
        spacing = self.target_spacing.get(modality, (1.0, 1.0, 1.0))
        
        transforms = [
            # Standardize orientation
            Orientationd(keys=keys, axcodes="RAS"),
            
            # Resample to target spacing
            Spacingd(keys=keys, pixdim=spacing, mode=("bilinear",) * len(keys)),
            
            # Crop foreground to remove unnecessary background
            CropForegroundd(keys=keys, source_key=keys[0]),
            
            # Pad to target size
            SpatialPadd(keys=keys, spatial_size=self.target_size, mode="constant"),
        ]
        
        return transforms
    
    def _get_intensity_transforms(self, 
                                keys: List[str], 
                                modality: ModalityType) -> List[Callable]:
        """
        Get intensity preprocessing transforms.
        
        Args:
            keys: List of data dictionary keys to process
            modality: Imaging modality type
            
        Returns:
            List of intensity transforms
        """
        intensity_range = self.INTENSITY_RANGES.get(modality, (-1000, 4000))
        
        transforms = [
            # Scale intensity to [0, 1] range with clipping
            ScaleIntensityRanged(
                keys=keys,
                a_min=intensity_range[0],
                a_max=intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            
            # Z-score normalization
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True),
        ]
        
        return transforms
    
    def _get_augmentation_transforms(self, keys: List[str]) -> List[Callable]:
        """
        Get data augmentation transforms for training.
        
        Args:
            keys: List of data dictionary keys to process
            
        Returns:
            List of augmentation transforms
        """
        return [
            # Spatial augmentations
            RandFlipd(keys=keys, prob=0.5, spatial_axis=[0, 1, 2]),
            RandRotated(keys=keys, prob=0.3, range_x=0.2, range_y=0.2, range_z=0.2, mode="bilinear"),
            RandZoomd(keys=keys, prob=0.3, min_zoom=0.8, max_zoom=1.2, mode="bilinear"),
            
            # Intensity augmentations
            RandGaussianNoised(keys=keys, prob=0.2, std=0.1),
            RandShiftIntensityd(keys=keys, prob=0.3, offsets=0.1),
        ]
    
    def _get_finalization_transforms(self, keys: List[str]) -> List[Callable]:
        """
        Get finalization transforms.
        
        Args:
            keys: List of data dictionary keys to process
            
        Returns:
            List of finalization transforms
        """
        return [
            ToTensord(keys=keys),
        ]
    
    def create_transform_pipeline(self, 
                                keys: List[str],
                                modality: ModalityType,
                                mode: ProcessingMode = ProcessingMode.INFERENCE,
                                custom_transforms: Optional[List[Callable]] = None) -> Compose:
        """
        Create a complete transform pipeline for the specified configuration.
        
        Args:
            keys: List of data dictionary keys to process
            modality: Imaging modality type
            mode: Processing mode (inference, training, validation)
            custom_transforms: Additional custom transforms to include
            
        Returns:
            Composed transform pipeline
        """
        cache_key = (tuple(keys), modality, mode)
        
        if cache_key in self._transform_cache:
            logger.debug(f"Using cached transform pipeline for {cache_key}")
            return self._transform_cache[cache_key]
        
        logger.info(f"Creating {mode.value} transform pipeline for {modality.value} with keys: {keys}")
        
        # Build transform list
        transforms = []
        
        # Base transforms
        transforms.extend(self._get_base_transforms(keys))
        
        # Spatial transforms
        transforms.extend(self._get_spatial_transforms(keys, modality))
        
        # Intensity transforms
        transforms.extend(self._get_intensity_transforms(keys, modality))
        
        # Add augmentation for training mode
        if mode == ProcessingMode.TRAINING:
            transforms.extend(self._get_augmentation_transforms(keys))
        
        # Custom transforms
        if custom_transforms:
            transforms.extend(custom_transforms)
        
        # Finalization transforms
        transforms.extend(self._get_finalization_transforms(keys))
        
        # Create composed pipeline
        pipeline = Compose(transforms)
        
        # Cache the pipeline
        self._transform_cache[cache_key] = pipeline
        
        logger.info(f"Created transform pipeline with {len(transforms)} transforms")
        return pipeline
    
    def process_single_image(self, 
                           data_dict: Dict[str, Any],
                           modality: ModalityType,
                           mode: ProcessingMode = ProcessingMode.INFERENCE) -> Dict[str, Any]:
        """
        Process a single image using the preprocessing pipeline.
        
        Args:
            data_dict: Dictionary containing image data and metadata
            modality: Imaging modality type
            mode: Processing mode
            
        Returns:
            Processed data dictionary
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            keys = list(data_dict.keys())
            
            # Create transform pipeline
            transform_pipeline = self.create_transform_pipeline(keys, modality, mode)
            
            # Apply transforms
            processed_data = transform_pipeline(data_dict)
            
            logger.info(f"Successfully processed {modality.value} image")
            return processed_data
            
        except Exception as e:
            raise PreprocessingError(f"Failed to process {modality.value} image: {str(e)}")
    
    def process_multi_modal_study(self, 
                                study_data: Dict[str, Dict[str, Any]],
                                mode: ProcessingMode = ProcessingMode.INFERENCE) -> Dict[str, Any]:
        """
        Process a multi-modal imaging study.
        
        Args:
            study_data: Dictionary with modality keys and image data
                       e.g., {'mri': {'image': tensor}, 'ct': {'image': tensor}}
            mode: Processing mode
            
        Returns:
            Processed multi-modal data dictionary
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            processed_modalities = {}
            
            for modality_name, image_data in study_data.items():
                try:
                    modality = ModalityType(modality_name.lower())
                except ValueError:
                    logger.warning(f"Unknown modality: {modality_name}, using MRI defaults")
                    modality = ModalityType.MRI
                
                # Process each modality
                processed_data = self.process_single_image(image_data, modality, mode)
                processed_modalities[modality_name] = processed_data
            
            # Combine processed modalities
            combined_data = self._combine_modalities(processed_modalities)
            
            logger.info(f"Successfully processed multi-modal study with {len(processed_modalities)} modalities")
            return combined_data
            
        except Exception as e:
            raise PreprocessingError(f"Failed to process multi-modal study: {str(e)}")
    
    def _combine_modalities(self, processed_modalities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine processed modalities into a unified data structure.
        
        Args:
            processed_modalities: Dictionary of processed modality data
            
        Returns:
            Combined data dictionary
        """
        combined_data = {
            'modalities': processed_modalities,
            'combined_image': None,
            'metadata': {}
        }
        
        # Stack images along channel dimension for multi-modal input
        images = []
        for modality_name, data in processed_modalities.items():
            if 'image' in data:
                images.append(data['image'])
            
            # Collect metadata
            if 'metadata' in data:
                combined_data['metadata'][modality_name] = data['metadata']
        
        if images:
            # Concatenate along channel dimension
            combined_data['combined_image'] = torch.cat(images, dim=0)
            logger.debug(f"Combined image shape: {combined_data['combined_image'].shape}")
        
        return combined_data
    
    def get_pipeline_info(self, 
                         keys: List[str],
                         modality: ModalityType,
                         mode: ProcessingMode = ProcessingMode.INFERENCE) -> Dict[str, Any]:
        """
        Get information about the transform pipeline configuration.
        
        Args:
            keys: List of data dictionary keys
            modality: Imaging modality type
            mode: Processing mode
            
        Returns:
            Pipeline configuration information
        """
        pipeline = self.create_transform_pipeline(keys, modality, mode)
        
        return {
            'modality': modality.value,
            'mode': mode.value,
            'keys': keys,
            'target_spacing': self.target_spacing.get(modality),
            'target_size': self.target_size,
            'intensity_range': self.INTENSITY_RANGES.get(modality),
            'num_transforms': len(pipeline.transforms),
            'device': self.device
        }
    
    def clear_cache(self):
        """Clear the transform pipeline cache."""
        self._transform_cache.clear()
        logger.info("Transform pipeline cache cleared")