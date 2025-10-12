#!/usr/bin/env python3
"""
Data Quality Assurance for MONAI Integration

This module provides comprehensive data quality checks, validation,
and automated quality assurance for medical imaging data using MONAI utilities.
"""

import logging
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# MONAI imports for data quality
from monai.data import MetaTensor, DatasetSummary
from monai.utils import (
    ensure_tuple, convert_to_numpy,
    GridSampleMode, InterpolateMode
)
from monai.transforms import (
    LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensity, ToTensor,
    SpatialCrop, CenterSpatialCrop,
    Compose
)

# Medical imaging specific
import nibabel as nib
import pydicom
from scipy import ndimage
import SimpleITK as sitk

from src.utils.logging_config import get_logger
from src.models.patient import PatientRecord, ImagingStudy

logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """Data quality metrics for medical images."""
    # Image properties
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    orientation: str
    data_type: str
    
    # Intensity statistics
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    intensity_range: float
    
    # Quality indicators
    snr: float  # Signal-to-noise ratio
    cnr: float  # Contrast-to-noise ratio
    sharpness: float
    uniformity: float
    
    # Artifacts detection
    motion_artifacts: float
    noise_level: float
    bias_field: float
    truncation_artifacts: bool
    
    # Metadata completeness
    metadata_completeness: float
    dicom_compliance: bool
    
    # Overall quality score
    quality_score: float
    quality_grade: str  # A, B, C, D, F
    
    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DatasetQualityReport:
    """Comprehensive dataset quality report."""
    dataset_name: str
    total_samples: int
    processed_samples: int
    failed_samples: int
    
    # Quality distribution
    quality_distribution: Dict[str, int]
    
    # Aggregate metrics
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    
    # Common issues
    common_issues: Dict[str, int]
    critical_issues: List[str]
    
    # Recommendations
    recommendations: List[str]
    
    # Processing time
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)


class MedicalImageQualityAssessment:
    """
    Comprehensive quality assessment for medical images using MONAI.
    """
    
    def __init__(self):
        """Initialize quality assessment tools."""
        self.loader = LoadImage(image_only=False)
        self.quality_thresholds = self._initialize_quality_thresholds()
        
    def _initialize_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize quality thresholds for different modalities."""
        return {
            'MRI': {
                'min_snr': 20.0,
                'min_cnr': 5.0,
                'max_noise_level': 0.1,
                'min_sharpness': 0.5,
                'min_uniformity': 0.8
            },
            'CT': {
                'min_snr': 15.0,
                'min_cnr': 3.0,
                'max_noise_level': 0.15,
                'min_sharpness': 0.4,
                'min_uniformity': 0.7
            },
            'Ultrasound': {
                'min_snr': 10.0,
                'min_cnr': 2.0,
                'max_noise_level': 0.2,
                'min_sharpness': 0.3,
                'min_uniformity': 0.6
            }
        }
    
    def assess_image_quality(self, 
                           image_path: Union[str, Path],
                           modality: str = "MRI") -> QualityMetrics:
        """
        Perform comprehensive quality assessment on a medical image.
        
        Args:
            image_path: Path to the medical image
            modality: Imaging modality (MRI, CT, Ultrasound)
            
        Returns:
            Quality metrics for the image
        """
        try:
            # Load image with metadata
            image_data = self.loader(image_path)
            
            if isinstance(image_data, tuple):
                image, metadata = image_data
            else:
                image = image_data
                metadata = {}
            
            # Convert to numpy for analysis
            image_array = convert_to_numpy(image)
            
            # Ensure 3D image
            if image_array.ndim == 4:
                image_array = image_array[0]  # Remove channel dimension
            elif image_array.ndim == 2:
                image_array = image_array[np.newaxis, ...]  # Add depth dimension
            
            # Calculate quality metrics
            metrics = self._calculate_quality_metrics(
                image_array, metadata, modality, image_path
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to assess image quality for {image_path}: {e}")
            return self._create_failed_metrics(str(image_path), str(e))
    
    def _calculate_quality_metrics(self,
                                 image: np.ndarray,
                                 metadata: Dict[str, Any],
                                 modality: str,
                                 image_path: Union[str, Path]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        
        # Basic image properties
        shape = image.shape
        spacing = self._extract_spacing(metadata)
        orientation = self._extract_orientation(metadata)
        data_type = str(image.dtype)
        
        # Intensity statistics
        mean_intensity = float(np.mean(image))
        std_intensity = float(np.std(image))
        min_intensity = float(np.min(image))
        max_intensity = float(np.max(image))
        intensity_range = max_intensity - min_intensity
        
        # Quality indicators
        snr = self._calculate_snr(image)
        cnr = self._calculate_cnr(image)
        sharpness = self._calculate_sharpness(image)
        uniformity = self._calculate_uniformity(image)
        
        # Artifact detection
        motion_artifacts = self._detect_motion_artifacts(image)
        noise_level = self._estimate_noise_level(image)
        bias_field = self._detect_bias_field(image)
        truncation_artifacts = self._detect_truncation_artifacts(image)
        
        # Metadata assessment
        metadata_completeness = self._assess_metadata_completeness(metadata)
        dicom_compliance = self._check_dicom_compliance(image_path, metadata)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(
            snr, cnr, sharpness, uniformity, motion_artifacts,
            noise_level, bias_field, metadata_completeness, modality
        )
        
        quality_grade = self._assign_quality_grade(quality_score)
        
        # Identify issues and warnings
        issues, warnings = self._identify_issues_and_warnings(
            snr, cnr, sharpness, uniformity, motion_artifacts,
            noise_level, bias_field, truncation_artifacts,
            metadata_completeness, modality
        )
        
        return QualityMetrics(
            shape=shape,
            spacing=spacing,
            orientation=orientation,
            data_type=data_type,
            mean_intensity=mean_intensity,
            std_intensity=std_intensity,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            intensity_range=intensity_range,
            snr=snr,
            cnr=cnr,
            sharpness=sharpness,
            uniformity=uniformity,
            motion_artifacts=motion_artifacts,
            noise_level=noise_level,
            bias_field=bias_field,
            truncation_artifacts=truncation_artifacts,
            metadata_completeness=metadata_completeness,
            dicom_compliance=dicom_compliance,
            quality_score=quality_score,
            quality_grade=quality_grade,
            issues=issues,
            warnings=warnings
        )
    
    def _extract_spacing(self, metadata: Dict[str, Any]) -> Tuple[float, ...]:
        """Extract pixel/voxel spacing from metadata."""
        if 'spacing' in metadata:
            return tuple(float(s) for s in metadata['spacing'])
        elif 'pixdim' in metadata:
            return tuple(float(s) for s in metadata['pixdim'][1:4])
        else:
            return (1.0, 1.0, 1.0)  # Default spacing
    
    def _extract_orientation(self, metadata: Dict[str, Any]) -> str:
        """Extract image orientation from metadata."""
        if 'orientation' in metadata:
            return str(metadata['orientation'])
        elif 'qform_code' in metadata and metadata['qform_code'] > 0:
            return "RAS"  # Assume RAS for valid qform
        else:
            return "Unknown"
    
    def _calculate_snr(self, image: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        # Use central region as signal, corners as noise
        center = tuple(s // 2 for s in image.shape)
        signal_region = image[
            center[0]-10:center[0]+10,
            center[1]-10:center[1]+10,
            center[2]-10:center[2]+10
        ]
        
        # Noise from corners
        corner_size = 10
        corners = [
            image[:corner_size, :corner_size, :corner_size],
            image[-corner_size:, :corner_size, :corner_size],
            image[:corner_size, -corner_size:, :corner_size],
            image[:corner_size, :corner_size, -corner_size:]
        ]
        
        signal_mean = np.mean(signal_region)
        noise_std = np.std(np.concatenate([c.flatten() for c in corners]))
        
        if noise_std == 0:
            return float('inf')
        
        return float(signal_mean / noise_std)
    
    def _calculate_cnr(self, image: np.ndarray) -> float:
        """Calculate Contrast-to-Noise Ratio."""
        # Use histogram-based approach to find tissue regions
        hist, bins = np.histogram(image.flatten(), bins=100)
        
        # Find two main peaks (assuming two main tissue types)
        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append((bins[i], hist[i]))
        
        if len(peaks) < 2:
            return 0.0
        
        # Sort by histogram count and take top 2
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:2]
        
        tissue1_value, tissue2_value = peaks[0][0], peaks[1][0]
        
        # Create masks for each tissue type
        mask1 = np.abs(image - tissue1_value) < 0.1 * np.std(image)
        mask2 = np.abs(image - tissue2_value) < 0.1 * np.std(image)
        
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            return 0.0
        
        mean1 = np.mean(image[mask1])
        mean2 = np.mean(image[mask2])
        std_noise = np.std(image[mask1 | mask2])
        
        if std_noise == 0:
            return float('inf')
        
        return float(abs(mean1 - mean2) / std_noise)
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using gradient magnitude."""
        # Calculate 3D gradient
        grad_x = np.gradient(image, axis=0)
        grad_y = np.gradient(image, axis=1)
        grad_z = np.gradient(image, axis=2)
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Normalize by image intensity range
        intensity_range = np.max(image) - np.min(image)
        if intensity_range == 0:
            return 0.0
        
        return float(np.mean(grad_magnitude) / intensity_range)
    
    def _calculate_uniformity(self, image: np.ndarray) -> float:
        """Calculate image uniformity (inverse of coefficient of variation)."""
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        return float(1.0 / (1.0 + cv))  # Normalize to [0, 1]
    
    def _detect_motion_artifacts(self, image: np.ndarray) -> float:
        """Detect motion artifacts using frequency domain analysis."""
        # Apply FFT to detect periodic artifacts
        fft_image = np.fft.fftn(image)
        fft_magnitude = np.abs(fft_image)
        
        # Look for high-frequency spikes that indicate motion
        high_freq_threshold = np.percentile(fft_magnitude, 95)
        high_freq_ratio = np.sum(fft_magnitude > high_freq_threshold) / fft_magnitude.size
        
        return float(high_freq_ratio)
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using Laplacian method."""
        # Apply Laplacian filter
        laplacian = ndimage.laplace(image)
        
        # Estimate noise as standard deviation of Laplacian
        noise_estimate = np.std(laplacian) / np.sqrt(6)  # Normalize for 3D
        
        # Normalize by image intensity range
        intensity_range = np.max(image) - np.min(image)
        if intensity_range == 0:
            return 1.0
        
        return float(noise_estimate / intensity_range)
    
    def _detect_bias_field(self, image: np.ndarray) -> float:
        """Detect bias field using low-frequency analysis."""
        # Apply low-pass filter to extract bias field
        from scipy.ndimage import gaussian_filter
        
        smoothed = gaussian_filter(image, sigma=10)
        bias_field = image - smoothed
        
        # Calculate bias field strength
        bias_strength = np.std(bias_field) / np.mean(image) if np.mean(image) > 0 else 0
        
        return float(bias_strength)
    
    def _detect_truncation_artifacts(self, image: np.ndarray) -> bool:
        """Detect truncation artifacts at image boundaries."""
        # Check if image values at boundaries are significantly different
        # from interior values
        
        boundary_thickness = 5
        
        # Extract boundary voxels
        boundaries = [
            image[:boundary_thickness, :, :],
            image[-boundary_thickness:, :, :],
            image[:, :boundary_thickness, :],
            image[:, -boundary_thickness:, :],
            image[:, :, :boundary_thickness],
            image[:, :, -boundary_thickness:]
        ]
        
        # Extract interior voxels
        interior = image[
            boundary_thickness:-boundary_thickness,
            boundary_thickness:-boundary_thickness,
            boundary_thickness:-boundary_thickness
        ]
        
        if interior.size == 0:
            return False
        
        boundary_mean = np.mean([np.mean(b) for b in boundaries])
        interior_mean = np.mean(interior)
        
        # Check for significant difference
        threshold = 0.2 * np.std(image)
        return abs(boundary_mean - interior_mean) > threshold
    
    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> float:
        """Assess completeness of metadata."""
        required_fields = [
            'spacing', 'origin', 'direction', 'datatype',
            'dim', 'pixdim', 'qform_code', 'sform_code'
        ]
        
        present_fields = sum(1 for field in required_fields if field in metadata)
        completeness = present_fields / len(required_fields)
        
        return float(completeness)
    
    def _check_dicom_compliance(self, 
                              image_path: Union[str, Path],
                              metadata: Dict[str, Any]) -> bool:
        """Check DICOM compliance for DICOM files."""
        path = Path(image_path)
        
        if path.suffix.lower() in ['.dcm', '.dicom']:
            try:
                # Try to read as DICOM
                ds = pydicom.dcmread(str(path))
                
                # Check for required DICOM tags
                required_tags = [
                    'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID',
                    'SOPInstanceUID', 'Modality'
                ]
                
                for tag in required_tags:
                    if not hasattr(ds, tag):
                        return False
                
                return True
                
            except Exception:
                return False
        
        return True  # Non-DICOM files are considered compliant
    
    def _calculate_overall_quality_score(self,
                                       snr: float, cnr: float, sharpness: float,
                                       uniformity: float, motion_artifacts: float,
                                       noise_level: float, bias_field: float,
                                       metadata_completeness: float,
                                       modality: str) -> float:
        """Calculate overall quality score (0-100)."""
        
        thresholds = self.quality_thresholds.get(modality, self.quality_thresholds['MRI'])
        
        # Normalize metrics to [0, 1] scale
        snr_score = min(snr / thresholds['min_snr'], 1.0)
        cnr_score = min(cnr / thresholds['min_cnr'], 1.0)
        sharpness_score = min(sharpness / thresholds['min_sharpness'], 1.0)
        uniformity_score = uniformity  # Already in [0, 1]
        
        # Invert negative metrics
        motion_score = max(0, 1.0 - motion_artifacts * 10)
        noise_score = max(0, 1.0 - noise_level / thresholds['max_noise_level'])
        bias_score = max(0, 1.0 - bias_field * 5)
        
        # Weighted combination
        weights = {
            'snr': 0.2,
            'cnr': 0.15,
            'sharpness': 0.15,
            'uniformity': 0.1,
            'motion': 0.15,
            'noise': 0.1,
            'bias': 0.1,
            'metadata': 0.05
        }
        
        quality_score = (
            weights['snr'] * snr_score +
            weights['cnr'] * cnr_score +
            weights['sharpness'] * sharpness_score +
            weights['uniformity'] * uniformity_score +
            weights['motion'] * motion_score +
            weights['noise'] * noise_score +
            weights['bias'] * bias_score +
            weights['metadata'] * metadata_completeness
        )
        
        return float(quality_score * 100)  # Scale to 0-100
    
    def _assign_quality_grade(self, quality_score: float) -> str:
        """Assign quality grade based on score."""
        if quality_score >= 90:
            return "A"
        elif quality_score >= 80:
            return "B"
        elif quality_score >= 70:
            return "C"
        elif quality_score >= 60:
            return "D"
        else:
            return "F"
    
    def _identify_issues_and_warnings(self,
                                    snr: float, cnr: float, sharpness: float,
                                    uniformity: float, motion_artifacts: float,
                                    noise_level: float, bias_field: float,
                                    truncation_artifacts: bool,
                                    metadata_completeness: float,
                                    modality: str) -> Tuple[List[str], List[str]]:
        """Identify quality issues and warnings."""
        
        issues = []
        warnings = []
        thresholds = self.quality_thresholds.get(modality, self.quality_thresholds['MRI'])
        
        # Critical issues
        if snr < thresholds['min_snr'] * 0.5:
            issues.append(f"Very low SNR: {snr:.2f} (minimum: {thresholds['min_snr']})")
        elif snr < thresholds['min_snr']:
            warnings.append(f"Low SNR: {snr:.2f} (recommended: {thresholds['min_snr']})")
        
        if cnr < thresholds['min_cnr'] * 0.5:
            issues.append(f"Very low CNR: {cnr:.2f} (minimum: {thresholds['min_cnr']})")
        elif cnr < thresholds['min_cnr']:
            warnings.append(f"Low CNR: {cnr:.2f} (recommended: {thresholds['min_cnr']})")
        
        if sharpness < thresholds['min_sharpness'] * 0.5:
            issues.append(f"Very low sharpness: {sharpness:.2f}")
        elif sharpness < thresholds['min_sharpness']:
            warnings.append(f"Low sharpness: {sharpness:.2f}")
        
        if uniformity < thresholds['min_uniformity'] * 0.8:
            issues.append(f"Poor uniformity: {uniformity:.2f}")
        elif uniformity < thresholds['min_uniformity']:
            warnings.append(f"Suboptimal uniformity: {uniformity:.2f}")
        
        if motion_artifacts > 0.1:
            issues.append(f"Significant motion artifacts detected: {motion_artifacts:.3f}")
        elif motion_artifacts > 0.05:
            warnings.append(f"Possible motion artifacts: {motion_artifacts:.3f}")
        
        if noise_level > thresholds['max_noise_level'] * 1.5:
            issues.append(f"Very high noise level: {noise_level:.3f}")
        elif noise_level > thresholds['max_noise_level']:
            warnings.append(f"High noise level: {noise_level:.3f}")
        
        if bias_field > 0.2:
            issues.append(f"Significant bias field: {bias_field:.3f}")
        elif bias_field > 0.1:
            warnings.append(f"Possible bias field: {bias_field:.3f}")
        
        if truncation_artifacts:
            warnings.append("Possible truncation artifacts detected")
        
        if metadata_completeness < 0.5:
            issues.append(f"Incomplete metadata: {metadata_completeness:.2f}")
        elif metadata_completeness < 0.8:
            warnings.append(f"Limited metadata: {metadata_completeness:.2f}")
        
        return issues, warnings
    
    def _create_failed_metrics(self, image_path: str, error_msg: str) -> QualityMetrics:
        """Create metrics for failed quality assessment."""
        return QualityMetrics(
            shape=(0, 0, 0),
            spacing=(0.0, 0.0, 0.0),
            orientation="Unknown",
            data_type="Unknown",
            mean_intensity=0.0,
            std_intensity=0.0,
            min_intensity=0.0,
            max_intensity=0.0,
            intensity_range=0.0,
            snr=0.0,
            cnr=0.0,
            sharpness=0.0,
            uniformity=0.0,
            motion_artifacts=1.0,
            noise_level=1.0,
            bias_field=1.0,
            truncation_artifacts=True,
            metadata_completeness=0.0,
            dicom_compliance=False,
            quality_score=0.0,
            quality_grade="F",
            issues=[f"Failed to process image: {error_msg}"],
            warnings=[]
        )


class DatasetQualityAssessment:
    """
    Comprehensive quality assessment for entire datasets.
    """
    
    def __init__(self):
        """Initialize dataset quality assessment."""
        self.image_qa = MedicalImageQualityAssessment()
    
    def assess_dataset_quality(self,
                             patient_records: List[PatientRecord],
                             output_dir: Optional[Path] = None) -> DatasetQualityReport:
        """
        Assess quality of entire dataset.
        
        Args:
            patient_records: List of patient records
            output_dir: Directory to save detailed reports
            
        Returns:
            Dataset quality report
        """
        start_time = datetime.now()
        
        # Collect all imaging studies
        all_studies = []
        for patient in patient_records:
            for study in patient.imaging_studies:
                all_studies.append((patient, study))
        
        total_samples = len(all_studies)
        processed_samples = 0
        failed_samples = 0
        
        quality_scores = []
        quality_grades = []
        all_issues = []
        all_warnings = []
        
        individual_reports = []
        
        logger.info(f"Starting quality assessment for {total_samples} imaging studies")
        
        for patient, study in all_studies:
            try:
                # Assess image quality
                metrics = self.image_qa.assess_image_quality(
                    study.file_path,
                    study.modality
                )
                
                quality_scores.append(metrics.quality_score)
                quality_grades.append(metrics.quality_grade)
                all_issues.extend(metrics.issues)
                all_warnings.extend(metrics.warnings)
                
                # Store individual report
                individual_report = {
                    'patient_id': patient.patient_id,
                    'study_id': study.study_id,
                    'modality': study.modality,
                    'file_path': study.file_path,
                    'metrics': metrics
                }
                individual_reports.append(individual_report)
                
                processed_samples += 1
                
                if processed_samples % 10 == 0:
                    logger.info(f"Processed {processed_samples}/{total_samples} studies")
                
            except Exception as e:
                logger.error(f"Failed to assess {study.study_id}: {e}")
                failed_samples += 1
        
        # Calculate aggregate statistics
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        min_quality_score = np.min(quality_scores) if quality_scores else 0.0
        max_quality_score = np.max(quality_scores) if quality_scores else 0.0
        
        # Quality distribution
        quality_distribution = {}
        for grade in quality_grades:
            quality_distribution[grade] = quality_distribution.get(grade, 0) + 1
        
        # Common issues
        common_issues = {}
        for issue in all_issues:
            common_issues[issue] = common_issues.get(issue, 0) + 1
        
        # Identify critical issues
        critical_issues = [
            issue for issue, count in common_issues.items()
            if count > total_samples * 0.1  # Issues affecting >10% of data
        ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            quality_distribution, common_issues, avg_quality_score
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create report
        report = DatasetQualityReport(
            dataset_name="NeuroDx-MultiModal",
            total_samples=total_samples,
            processed_samples=processed_samples,
            failed_samples=failed_samples,
            quality_distribution=quality_distribution,
            avg_quality_score=avg_quality_score,
            min_quality_score=min_quality_score,
            max_quality_score=max_quality_score,
            common_issues=common_issues,
            critical_issues=critical_issues,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
        # Save detailed reports if output directory provided
        if output_dir:
            self._save_detailed_reports(report, individual_reports, output_dir)
        
        logger.info(f"Dataset quality assessment completed in {processing_time:.2f} seconds")
        return report
    
    def _generate_recommendations(self,
                                quality_distribution: Dict[str, int],
                                common_issues: Dict[str, int],
                                avg_quality_score: float) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        # Overall quality recommendations
        if avg_quality_score < 60:
            recommendations.append("Dataset quality is below acceptable threshold. Consider data reacquisition or preprocessing.")
        elif avg_quality_score < 80:
            recommendations.append("Dataset quality is moderate. Preprocessing may improve results.")
        
        # Grade distribution recommendations
        total_samples = sum(quality_distribution.values())
        if quality_distribution.get('F', 0) > total_samples * 0.1:
            recommendations.append("More than 10% of images have failing quality. Review acquisition protocols.")
        
        # Issue-specific recommendations
        for issue, count in common_issues.items():
            if count > total_samples * 0.2:  # Affecting >20% of data
                if "SNR" in issue:
                    recommendations.append("Consider noise reduction preprocessing or acquisition parameter optimization.")
                elif "motion" in issue:
                    recommendations.append("Implement motion correction preprocessing or improve patient positioning.")
                elif "bias field" in issue:
                    recommendations.append("Apply bias field correction preprocessing.")
                elif "metadata" in issue:
                    recommendations.append("Improve metadata collection and DICOM compliance.")
        
        return recommendations
    
    def _save_detailed_reports(self,
                             report: DatasetQualityReport,
                             individual_reports: List[Dict[str, Any]],
                             output_dir: Path):
        """Save detailed quality reports to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary report
        summary_file = output_dir / "quality_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'dataset_name': report.dataset_name,
                'total_samples': report.total_samples,
                'processed_samples': report.processed_samples,
                'failed_samples': report.failed_samples,
                'quality_distribution': report.quality_distribution,
                'avg_quality_score': report.avg_quality_score,
                'min_quality_score': report.min_quality_score,
                'max_quality_score': report.max_quality_score,
                'common_issues': report.common_issues,
                'critical_issues': report.critical_issues,
                'recommendations': report.recommendations,
                'processing_time': report.processing_time,
                'timestamp': report.timestamp.isoformat()
            }, indent=2)
        
        # Save individual reports
        individual_file = output_dir / "individual_quality_reports.json"
        serializable_reports = []
        
        for report_item in individual_reports:
            # Convert metrics to serializable format
            metrics = report_item['metrics']
            serializable_metrics = {
                'shape': metrics.shape,
                'spacing': metrics.spacing,
                'orientation': metrics.orientation,
                'data_type': metrics.data_type,
                'mean_intensity': metrics.mean_intensity,
                'std_intensity': metrics.std_intensity,
                'min_intensity': metrics.min_intensity,
                'max_intensity': metrics.max_intensity,
                'intensity_range': metrics.intensity_range,
                'snr': metrics.snr,
                'cnr': metrics.cnr,
                'sharpness': metrics.sharpness,
                'uniformity': metrics.uniformity,
                'motion_artifacts': metrics.motion_artifacts,
                'noise_level': metrics.noise_level,
                'bias_field': metrics.bias_field,
                'truncation_artifacts': metrics.truncation_artifacts,
                'metadata_completeness': metrics.metadata_completeness,
                'dicom_compliance': metrics.dicom_compliance,
                'quality_score': metrics.quality_score,
                'quality_grade': metrics.quality_grade,
                'issues': metrics.issues,
                'warnings': metrics.warnings
            }
            
            serializable_reports.append({
                'patient_id': report_item['patient_id'],
                'study_id': report_item['study_id'],
                'modality': report_item['modality'],
                'file_path': report_item['file_path'],
                'metrics': serializable_metrics
            })
        
        with open(individual_file, 'w') as f:
            json.dump(serializable_reports, f, indent=2)
        
        logger.info(f"Detailed quality reports saved to {output_dir}")


# Example usage
if __name__ == "__main__":
    # Initialize quality assessment
    qa = DatasetQualityAssessment()
    
    # This would normally use real patient records
    # qa_report = qa.assess_dataset_quality(patient_records, Path("./quality_reports"))
    
    print("Data quality assurance module initialized successfully!")