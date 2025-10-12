"""
Medical image ingestion handler using MONAI I/O capabilities.
Supports NIfTI and DICOM file formats with validation and error handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch

from monai.data import MetaTensor
from monai.transforms import (
    LoadImage, 
    LoadImaged,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    ToTensor,
    ToTensord
)
from monai.utils import ensure_tuple
import nibabel as nib
import pydicom
from pydicom.errors import InvalidDicomError

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageIngestionError(Exception):
    """Custom exception for image ingestion errors."""
    pass


class UnsupportedFormatError(ImageIngestionError):
    """Exception raised for unsupported file formats."""
    pass


class CorruptedFileError(ImageIngestionError):
    """Exception raised for corrupted or invalid files."""
    pass


class ImageIngestionHandler:
    """
    Handles medical image ingestion with MONAI I/O capabilities.
    Supports NIfTI (.nii, .nii.gz) and DICOM (.dcm) formats.
    """
    
    SUPPORTED_EXTENSIONS = {'.nii', '.nii.gz', '.dcm', '.dicom'}
    NIFTI_EXTENSIONS = {'.nii', '.nii.gz'}
    DICOM_EXTENSIONS = {'.dcm', '.dicom'}
    
    def __init__(self):
        """Initialize the image ingestion handler."""
        self.loader = LoadImage(image_only=False, ensure_channel_first=True)
        self.dict_loader = LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)
        
    def validate_file_format(self, file_path: Union[str, Path]) -> str:
        """
        Validate if the file format is supported.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            File format type ('nifti' or 'dicom')
            
        Raises:
            UnsupportedFormatError: If file format is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ImageIngestionError(f"File does not exist: {file_path}")
            
        # Check file extension
        if file_path.suffix.lower() == '.gz' and file_path.stem.endswith('.nii'):
            extension = '.nii.gz'
        else:
            extension = file_path.suffix.lower()
            
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
            
        if extension in self.NIFTI_EXTENSIONS:
            return 'nifti'
        elif extension in self.DICOM_EXTENSIONS:
            return 'dicom'
        else:
            raise UnsupportedFormatError(f"Unknown format classification for: {extension}")
    
    def _validate_nifti_file(self, file_path: Path) -> None:
        """
        Validate NIfTI file integrity.
        
        Args:
            file_path: Path to NIfTI file
            
        Raises:
            CorruptedFileError: If file is corrupted or invalid
        """
        try:
            # Try to load with nibabel for validation
            img = nib.load(str(file_path))
            
            # Basic validation checks
            if img.get_fdata().size == 0:
                raise CorruptedFileError(f"NIfTI file contains no data: {file_path}")
                
            # Check for reasonable dimensions (medical images typically 2D-4D)
            if len(img.shape) < 2 or len(img.shape) > 4:
                logger.warning(f"Unusual image dimensions {img.shape} for file: {file_path}")
                
        except Exception as e:
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"Invalid NIfTI file {file_path}: {str(e)}")
    
    def _validate_dicom_file(self, file_path: Path) -> None:
        """
        Validate DICOM file integrity.
        
        Args:
            file_path: Path to DICOM file
            
        Raises:
            CorruptedFileError: If file is corrupted or invalid
        """
        try:
            # Try to read DICOM file
            ds = pydicom.dcmread(str(file_path), force=True)
            
            # Check if it has pixel data
            if not hasattr(ds, 'pixel_array'):
                raise CorruptedFileError(f"DICOM file has no pixel data: {file_path}")
                
            # Try to access pixel array
            pixel_array = ds.pixel_array
            if pixel_array.size == 0:
                raise CorruptedFileError(f"DICOM file contains no pixel data: {file_path}")
                
        except InvalidDicomError as e:
            raise CorruptedFileError(f"Invalid DICOM file {file_path}: {str(e)}")
        except Exception as e:
            if isinstance(e, CorruptedFileError):
                raise
            raise CorruptedFileError(f"Error reading DICOM file {file_path}: {str(e)}")
    
    def load_single_image(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single medical image using MONAI.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing image data and metadata
            
        Raises:
            ImageIngestionError: If loading fails
            UnsupportedFormatError: If format is not supported
            CorruptedFileError: If file is corrupted
        """
        file_path = Path(file_path)
        
        try:
            # Validate file format
            format_type = self.validate_file_format(file_path)
            logger.info(f"Loading {format_type} image: {file_path}")
            
            # Validate file integrity
            if format_type == 'nifti':
                self._validate_nifti_file(file_path)
            elif format_type == 'dicom':
                self._validate_dicom_file(file_path)
            
            # Load image using MONAI
            image_data, metadata = self.loader(str(file_path))
            
            # Ensure we have a tensor
            if not isinstance(image_data, torch.Tensor):
                image_data = torch.tensor(image_data)
            
            # Validate loaded data
            if image_data.numel() == 0:
                raise CorruptedFileError(f"Loaded image contains no data: {file_path}")
            
            result = {
                'image': image_data,
                'metadata': metadata,
                'file_path': str(file_path),
                'format_type': format_type,
                'shape': tuple(image_data.shape),
                'dtype': str(image_data.dtype)
            }
            
            logger.info(f"Successfully loaded image: {file_path}, shape: {result['shape']}")
            return result
            
        except (UnsupportedFormatError, CorruptedFileError):
            raise
        except Exception as e:
            raise ImageIngestionError(f"Failed to load image {file_path}: {str(e)}")
    
    def load_multiple_images(self, file_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Load multiple medical images.
        
        Args:
            file_paths: List of paths to image files
            
        Returns:
            List of dictionaries containing image data and metadata
            
        Raises:
            ImageIngestionError: If any loading fails
        """
        results = []
        errors = []
        
        for file_path in file_paths:
            try:
                result = self.load_single_image(file_path)
                results.append(result)
            except Exception as e:
                error_msg = f"Failed to load {file_path}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if errors:
            error_summary = f"Failed to load {len(errors)} out of {len(file_paths)} images"
            if len(errors) == len(file_paths):
                raise ImageIngestionError(f"{error_summary}. All files failed to load.")
            else:
                logger.warning(f"{error_summary}. Continuing with {len(results)} successful loads.")
        
        return results
    
    def load_study_images(self, study_data: Dict[str, Union[str, Path]]) -> Dict[str, Any]:
        """
        Load images for a complete imaging study using MONAI dictionary transforms.
        
        Args:
            study_data: Dictionary with modality keys and file paths
                       e.g., {'mri': '/path/to/mri.nii', 'ct': '/path/to/ct.dcm'}
            
        Returns:
            Dictionary containing loaded images and metadata for each modality
            
        Raises:
            ImageIngestionError: If loading fails
        """
        try:
            # Validate all files first
            validated_data = {}
            for modality, file_path in study_data.items():
                format_type = self.validate_file_format(file_path)
                validated_data[modality] = {
                    'path': str(file_path),
                    'format': format_type
                }
            
            # Prepare data for MONAI dictionary transform
            data_dict = {modality: info['path'] for modality, info in validated_data.items()}
            
            # Load using MONAI dictionary loader
            # Note: We'll load each individually for better error handling
            results = {}
            for modality, file_path in data_dict.items():
                try:
                    image_result = self.load_single_image(file_path)
                    results[modality] = image_result
                except Exception as e:
                    logger.error(f"Failed to load {modality} image from {file_path}: {str(e)}")
                    raise ImageIngestionError(f"Failed to load {modality} image: {str(e)}")
            
            logger.info(f"Successfully loaded study with {len(results)} modalities")
            return results
            
        except Exception as e:
            if isinstance(e, ImageIngestionError):
                raise
            raise ImageIngestionError(f"Failed to load study images: {str(e)}")
    
    def get_image_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic information about an image file without loading the full data.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing image information
            
        Raises:
            ImageIngestionError: If file cannot be read
        """
        file_path = Path(file_path)
        
        try:
            format_type = self.validate_file_format(file_path)
            
            info = {
                'file_path': str(file_path),
                'format_type': format_type,
                'file_size': file_path.stat().st_size
            }
            
            if format_type == 'nifti':
                img = nib.load(str(file_path))
                info.update({
                    'shape': img.shape,
                    'dtype': str(img.get_data_dtype()),
                    'affine_shape': img.affine.shape if hasattr(img, 'affine') else None,
                    'header_info': dict(img.header) if hasattr(img, 'header') else None
                })
            elif format_type == 'dicom':
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                info.update({
                    'modality': getattr(ds, 'Modality', 'Unknown'),
                    'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                    'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
                    'patient_id': getattr(ds, 'PatientID', 'Unknown')
                })
            
            return info
            
        except Exception as e:
            raise ImageIngestionError(f"Failed to get image info for {file_path}: {str(e)}")